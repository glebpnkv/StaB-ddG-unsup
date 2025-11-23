import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from tqdm import tqdm

from stabddg.jobs.stability_finetune import finetune
from stabddg.model import StaBddG
from stabddg.mpnn_utils import StructureDataset, ProteinMPNN, parse_PDB
from training import _is_dist_initialized, _is_main_process

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument("--run_name", type=str, default="stability_finetune")
    argparser.add_argument(
        "--checkpoint", type=str, default="model_ckpts/proteinmpnn.pt"
    )
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--num_epochs", type=int, default=70)
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument("--model_save_freq", type=int, default=10)
    argparser.add_argument(
        "--model_save_dir", type=str, default="cache/stability_finetuned"
    )
    argparser.add_argument("--wandb", action="store_true")
    # Sample only one batch of mutants per domain during training.
    argparser.add_argument("--single_batch", action="store_true")
    argparser.add_argument(
        "--val_freq", type=int, default=10
    )  # Train validation frequency
    argparser.add_argument("--noise_level", type=float, default=0.1)  # Backbone noise.
    argparser.add_argument(
        "--dropout", type=float, default=0.0
    )  # Dropout during model training.
    # Do not fix permutation order and backbone noise between mutant and wildtype during decoding.
    argparser.add_argument("--no_antithetic_variates", action="store_true")
    argparser.add_argument(
        "--lam", type=float, default=0.0
    )  # KL regularization strength.
    argparser.add_argument("--pdb_dir", type=str, default="AlphaFold_model_PDBs")
    argparser.add_argument(
        "--stability_data",
        type=str,
        default="Tsuboyama2023_Dataset2_Dataset3_20230416.csv",
    )
    argparser.add_argument("--lr", type=float, default=1e-6)
    argparser.add_argument("--random_init", action="store_true")
    argparser.add_argument("--num_dataloader_workers", type=int, default=4)

    # DDP / multi-GPU arguments
    argparser.add_argument("--distributed", action="store_true",
                           help="Enable DDP if multiple GPUs are available. If launched with torchrun, this is auto-enabled.")
    argparser.add_argument("--ddp_backend", type=str, default="nccl",
                           help="DDP backend to use (default: nccl).")
    # torchrun passes --local_rank automatically
    argparser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by torchrun.")
    args = argparser.parse_args()

    # Initialize distributed if torchrun provides ranks or --distributed is set and multiple GPUs exist
    using_torchrun = args.local_rank != -1 or int(os.environ.get("LOCAL_RANK", -1)) != -1
    if args.local_rank == -1:
        # normalize from env if present
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    enable_ddp = (using_torchrun or args.distributed) and (num_gpus > 0)

    if enable_ddp and not _is_dist_initialized():
        if args.local_rank is None or args.local_rank == -1:
            # Fallback to single-process if no local_rank; still OK for multi-GPU single-process but we prefer torchrun
            pass
        else:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend=args.ddp_backend, init_method="env://")
    device = torch.device(f"cuda:{args.local_rank}") \
        if torch.cuda.is_available() and args.local_rank not in (-1, None) \
        else (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    logger.info(f"Using device: {device}")

    # Seeding per-rank
    rank_for_seed = dist.get_rank() if _is_dist_initialized() else 0
    torch.manual_seed(args.seed + rank_for_seed)
    np.random.seed(args.seed + rank_for_seed)

    # Dataset preprocessing/loading
    # Read split files
    with open("data/rocklin/mega_splits.pkl", "rb") as f:
        splits = pickle.load(f)
    train_names = splits["train"]
    val_names = splits["val"].tolist()
    test_names = splits["test"].tolist()

    # Load AF predicted structures
    pdb_dict_train = []
    for name in tqdm(train_names, desc="Preparing train set") if _is_main_process() else train_names:
        name = name.split(".pdb", 1)[0] + ".pdb"
        name = name.replace("|", ":")
        path = os.path.join(args.pdb_dir, name)
        pdb_dict_train.append(parse_PDB(path)[0])

    pdb_dict_val = []
    if _is_main_process():
        for name in tqdm(val_names, desc="Preparing validation set"):
            name = name.split(".pdb", 1)[0] + ".pdb"
            name = name.replace("|", ":")
            path = os.path.join(args.pdb_dir, name)
            pdb_dict_val.append(parse_PDB(path)[0])
    else:
        # Non-main ranks can skip validation/test parsing to save memory/time if desired,
        # but to keep behavior consistent we still parse. Comment out below to skip.
        for name in val_names:
            name = name.split(".pdb", 1)[0] + ".pdb"
            name = name.replace("|", ":")
            path = os.path.join(args.pdb_dir, name)
            pdb_dict_val.append(parse_PDB(path)[0])

    pdb_dict_test = []
    if _is_main_process():
        for name in tqdm(test_names, desc="Preparing test set"):
            name = name.split(".pdb", 1)[0] + ".pdb"
            name = name.replace("|", ":")
            path = os.path.join(args.pdb_dir, name)
            pdb_dict_test.append(parse_PDB(path)[0])
    else:
        for name in test_names:
            name = name.split(".pdb", 1)[0] + ".pdb"
            name = name.replace("|", ":")
            path = os.path.join(args.pdb_dir, name)
            pdb_dict_test.append(parse_PDB(path)[0])

    # Read ddG data
    ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
    ddG_data = {}

    # Reading stability dataset
    df_stability_raw = pd.read_csv(args.stability_data, low_memory=False)
    df_stability = df_stability_raw[df_stability_raw["ddG_ML"] != "-"]
    df_stability = df_stability.loc[
        ~df_stability.mut_type.str.contains("ins")
        & ~df_stability.mut_type.str.contains("del"),
        :,
    ].reset_index(drop=True)

    for name in train_names + val_names + test_names:
        cleaned_name = name.split(".pdb", 1)[0] + ".pdb"
        cleaned_name = cleaned_name.replace("|", ":")
        ddG_data[cleaned_name] = df_stability[
            (df_stability["WT_name"] == name)
            & (df_stability["mut_type"] != "wt")
        ]

    for name, df_mut in ddG_data.items():
        ddG_data[name] = {
            "mut_seqs": df_mut["aa_seq"].to_list(),
            "ddG": df_mut["ddG_ML"].to_numpy(dtype=np.float32),
        }

    # Featurize mutations as sequences
    for name, df_mut in ddG_data.items():
        index_matrix = []
        for s in df_mut["mut_seqs"]:
            indices = np.asarray([ALPHABET.index(a) for a in s], dtype=np.int64)
            index_matrix.append(indices)
        index_matrix = np.vstack(index_matrix)
        ddG_data[name]["mut_seqs"] = torch.from_numpy(index_matrix)
        ddG_data[name]["ddG"] = torch.tensor(df_mut["ddG"])

    # Mask all input chains
    for cur_dict in pdb_dict_train:
        cur_dict["masked_list"] = ["A"]
        cur_dict["visible_list"] = []
    for cur_dict in pdb_dict_val:
        cur_dict["masked_list"] = ["A"]
        cur_dict["visible_list"] = []
    for cur_dict in pdb_dict_test:
        cur_dict["masked_list"] = ["A"]
        cur_dict["visible_list"] = []

    dataset_train = StructureDataset(pdb_dict_train, truncate=None, max_length=3000)
    dataset_valid = StructureDataset(pdb_dict_val, truncate=None, max_length=3000)
    dataset_test = StructureDataset(pdb_dict_test, truncate=None, max_length=3000)

    # Load pre-trained ProteinMPNN
    # accelerator = Accelerator()
    # device = accelerator.device
    if _is_main_process():
        logger.info(f"Using device: {device}")

    pmpnn = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        dropout=0.0,
        augment_eps=0.0,
    )

    mpnn_checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in mpnn_checkpoint.keys():
        pmpnn.load_state_dict(mpnn_checkpoint["model_state_dict"])
    else:
        pmpnn.load_state_dict(mpnn_checkpoint)
    if _is_main_process():
        logger.info(f"Successfully loaded model at {args.checkpoint}")

    model = StaBddG(
        pmpnn=pmpnn,
        use_antithetic_variates=not args.no_antithetic_variates,
        noise_level=args.noise_level,
        device=device,
    )
    model.compile()
    model.to(device)

    # Wrap with DDP if applicable
    if _is_dist_initialized():
        # find_unused_parameters=True for safety if some parameters aren't used every step
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank] if device.type == "cuda" else None,
            output_device=args.local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    # Initialize wandb logging (main process only)
    if _is_main_process() and args.wandb:
        logger.info("Initializing weights and biases.")
        wandb.init(
            project="",
            entity="",
            name=args.run_name,
        )
        logger.info("Weights and biases initialized.")

    try:
        finetune(
            model,
            dataset_train,
            dataset_valid,
            dataset_test,
            ddG_data,
            args,
            batch_size=args.batch_size,
            device=device,
            num_dataloader_workers=args.num_dataloader_workers,
        )
    finally:
        # Ensure proper cleanup
        if _is_dist_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
