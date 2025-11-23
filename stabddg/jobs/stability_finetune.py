import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from stabddg.model import StaBddG
from stabddg.mpnn_utils import StructureDataset, ProteinMPNN, parse_PDB
from training import _is_dist_initialized, _is_main_process, _unwrap_model, _distributed_concat_1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validation_step(
    model,
    ddG_data,
    dataset_valid,
    batch_size=20000,
    device="cuda"
) -> dict[str, float | None]:
    val_spearman = []
    val_pearson = []
    all_pred = []
    all_labels = []
    for sample in tqdm(dataset_valid):
        pdb_name = sample["name"]
        ddG = ddG_data[f"{pdb_name}.pdb"]["ddG"].to(device)
        mut_seqs = ddG_data[f"{pdb_name}.pdb"]["mut_seqs"]
        N = mut_seqs.shape[0]
        M = (
            batch_size // mut_seqs.shape[1]
        )  # convert the number of tokens to the number of sequences per batch

        sample_pred = []
        # Batching for mutants
        for batch_idx in range(0, N, M):
            B = min(N - batch_idx, M)
            # ddG prediction
            pred = model.folding_ddG(sample, mut_seqs[batch_idx : batch_idx + B])
            sample_pred.append(pred.detach().cpu())

        pred = torch.cat(sample_pred)

        sp, _ = spearmanr(pred.cpu().detach().numpy(), ddG.cpu().detach().numpy())
        val_spearman.append(sp)

        pr, _ = pearsonr(pred.cpu().detach().numpy(), ddG.cpu().detach().numpy())
        val_pearson.append(pr)

        all_pred.append(pred.cpu().detach().numpy())
        all_labels.append(ddG.cpu().detach().numpy())

    sp, _ = spearmanr(np.concatenate(all_pred), np.concatenate(all_labels))
    pr, _ = pearsonr(np.concatenate(all_pred), np.concatenate(all_labels))

    return {
        "spearman": float(np.mean(val_spearman)),
        "pearson": float(np.mean(val_pearson)),
        "all_spearman": float(sp),
        "all_pearson": float(pr),
    }


def finetune(
    model: ProteinMPNN,
    dataset_train: StructureDataset,
    dataset_valid: StructureDataset,
    dataset_test: StructureDataset,
    args,
    batch_size=10000,
    device="cuda",
    num_dataloader_workers: int = 4,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ddG_loss_fn = torch.nn.MSELoss()

    # DataFrame with training, validation and test metrics
    df_metrics = pd.DataFrame()

    # Directory to save model checkpoints
    if _is_main_process() and not os.path.exists(args.model_save_dir):
        logger.info(f"Creating directory {args.model_save_dir}")
        os.makedirs(args.model_save_dir)

    # Creating a logging file logs.txt (main process only)
    if _is_main_process():
        log_path = os.path.join(args.model_save_dir, "logs.txt")
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_path}")

    # DDP state
    world_size = dist.get_world_size() if _is_dist_initialized() else 1

    # Preparing data loaders (batch_size fixed to 1, use a collate_fn that returns the single element)
    train_sampler = DistributedSampler(dataset_train, shuffle=True) if _is_dist_initialized() else None
    valid_sampler = DistributedSampler(dataset_valid, shuffle=False) if _is_dist_initialized() else None
    test_sampler = DistributedSampler(dataset_test, shuffle=False) if _is_dist_initialized() else None

    dl_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_dataloader_workers,
        collate_fn=lambda x: x[0],
        pin_memory=True,
    )
    dl_valid = DataLoader(
        dataset_valid,
        batch_size=1,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=num_dataloader_workers,
        collate_fn=lambda x: x[0],
        pin_memory=True,
    )
    dl_test = DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_dataloader_workers,
        collate_fn=lambda x: x[0],
        pin_memory=True,
    )

    def run_eval(dloader: DataLoader) -> dict[str, float | None]:
        """Validation/test epoch across all ranks with DistributedSampler splitting."""
        _unwrap_model(model).eval()
        rank_spearmans = []
        rank_pearsons = []
        rank_all_pred = []
        rank_all_labels = []

        with torch.no_grad():
            for sample in dloader:
                # sample is a single dict (collate_fn returns x[0])
                ddG = sample["ddG"].to(device)
                mut_seqs = sample["mut_seqs"]

                N = mut_seqs.shape[0]
                M = batch_size // mut_seqs.shape[1]  # convert tokens to sequences per batch

                sample_pred_chunks = []
                for batch_idx in range(0, N, M):
                    B = min(N - batch_idx, M)
                    pred = _unwrap_model(model).folding_ddG(sample, mut_seqs[batch_idx: batch_idx + B])
                    sample_pred_chunks.append(pred.detach().cpu())
                pred = torch.cat(sample_pred_chunks)  # [N]
                ddG_cpu = ddG.detach().cpu()

                sp, _ = spearmanr(pred.numpy(), ddG_cpu.numpy())
                pr, _ = pearsonr(pred.numpy(), ddG_cpu.numpy())
                rank_spearmans.append(sp)
                rank_pearsons.append(pr)

                rank_all_pred.append(pred)
                rank_all_labels.append(ddG_cpu)

        # aggregate per-sample means
        mean_sp = float(np.mean(rank_spearmans)) if len(rank_spearmans) else 0.0
        mean_pr = float(np.mean(rank_pearsons)) if len(rank_pearsons) else 0.0

        if _is_dist_initialized():
            # sum of means weighted by count (counts may differ slightly if sampler sizes differ)
            cnt_tensor = torch.tensor([len(rank_spearmans)], device=device, dtype=torch.float32)
            sp_tensor = torch.tensor([mean_sp * (cnt_tensor.item() if cnt_tensor.item() > 0 else 1.0)], device=device)
            pr_tensor = torch.tensor([mean_pr * (cnt_tensor.item() if cnt_tensor.item() > 0 else 1.0)], device=device)
            dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(sp_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(pr_tensor, op=dist.ReduceOp.SUM)
            mean_sp = (sp_tensor / torch.clamp(cnt_tensor, min=1.0)).item()
            mean_pr = (pr_tensor / torch.clamp(cnt_tensor, min=1.0)).item()

        # aggregate "all_*" metrics by concatenating predictions/labels across ranks
        if len(rank_all_pred) > 0:
            local_pred = torch.cat(rank_all_pred).to(device)
            local_labels = torch.cat(rank_all_labels).to(device)
        else:
            local_pred = torch.empty(0, device=device)
            local_labels = torch.empty(0, device=device)

        all_pred = _distributed_concat_1d(world_size, local_pred)
        all_labels = _distributed_concat_1d(world_size, local_labels)

        # Compute global metrics only on rank 0 to avoid redundant work
        all_sp = None
        all_pr = None
        if (not _is_dist_initialized()) or dist.get_rank() == 0:
            if all_pred.numel() > 0:
                asp, _ = spearmanr(all_pred.cpu().numpy(), all_labels.cpu().numpy())
                apr, _ = pearsonr(all_pred.cpu().numpy(), all_labels.cpu().numpy())
                all_sp = float(asp)
                all_pr = float(apr)

        # Broadcast global numbers to all ranks for consistency
        if _is_dist_initialized():
            if dist.get_rank() == 0:
                buf = torch.tensor([all_sp if all_sp is not None else 0.0,
                                    all_pr if all_pr is not None else 0.0], device=device)
            else:
                buf = torch.zeros(2, device=device)
            dist.broadcast(buf, src=0)
            all_sp = float(buf[0].item())
            all_pr = float(buf[1].item())

        return {
            "spearman": mean_sp,
            "pearson": mean_pr,
            "all_spearman": all_sp,
            "all_pearson": all_pr,
        }


    # TODO complete the logic which writes all train / valid /
    # df_predictions = pd.DataFrame()

    for epoch in tqdm(range(args.num_epochs), desc="Epoch") if _is_main_process() else range(args.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if valid_sampler is not None:
            valid_sampler.set_epoch(epoch)
        if test_sampler is not None:
            test_sampler.set_epoch(epoch)

        _unwrap_model(model).train()
        train_sum = []
        spearmans = []
        print_prefix = f"Epoch {epoch + 1}"

        # Iterate through training batches (each batch is a single sample)
        for sample in dl_train:
            ddG = sample["ddG"].to(device)
            mut_seqs = sample["mut_seqs"]

            N = mut_seqs.shape[0]
            M = batch_size // mut_seqs.shape[1]  # Convert the number of tokens to sequences per batch

            # Random shuffling
            permutation = torch.randperm(ddG.shape[0])
            ddG = ddG[permutation]
            mut_seqs = mut_seqs[permutation]

            sample_pred = []
            for batch_idx in range(0, N, M):
                B = min(N - batch_idx, M)
                optimizer.zero_grad(set_to_none=True)

                # TODO track all predictions for each epoch - create a dataframe for each epoch
                # ddG prediction
                pred = _unwrap_model(model).folding_ddG(sample, mut_seqs[batch_idx: batch_idx + B])

                ddG_loss = ddG_loss_fn(pred, ddG[batch_idx: batch_idx + B])

                ddG_loss.backward()
                optimizer.step()

                train_sum.append(ddG_loss.item())
                sample_pred.append(pred.detach().cpu())
                if args.single_batch:
                    break

            sample_pred = torch.cat(sample_pred)
            sp, _ = spearmanr(
                sample_pred.detach().numpy(),
                ddG.cpu().detach().numpy()[: sample_pred.shape[0]],
            )
            spearmans.append(sp)

        # Aggregate training metrics across processes
        if _is_dist_initialized():
            loss_tensor = torch.tensor([np.mean(train_sum) if len(train_sum) > 0 else 0.0],
                                       device=device, dtype=torch.float32)
            sp_tensor = torch.tensor([np.mean(spearmans) if len(spearmans) > 0 else 0.0],
                                     device=device, dtype=torch.float32)
            cnt_tensor = torch.tensor([1.0 if len(train_sum) > 0 else 0.0],
                                      device=device, dtype=torch.float32)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(sp_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(cnt_tensor, op=dist.ReduceOp.SUM)
            mean_loss_all = (loss_tensor / torch.clamp(cnt_tensor, min=1.0)).item()
            mean_sp_all = (sp_tensor / torch.clamp(cnt_tensor, min=1.0)).item()
        else:
            mean_loss_all = float(np.mean(train_sum)) if len(train_sum) > 0 else 0.0
            mean_sp_all = float(np.mean(spearmans)) if len(spearmans) > 0 else 0.0

        train_metrics = {
            "loss": mean_loss_all,
            "spearman": mean_sp_all,
            "all_spearman": None,
            "all_pearson": None,
        }
        if _is_main_process():
            logger.info(f'{print_prefix}: Training metrics: '
                        f'{ {k: "{0:0.4f}".format(v) for k, v in train_metrics.items() if v is not None} }')

            df_train_metrics = pd.DataFrame(
                {"epoch": epoch + 1, "split": "train"} | train_metrics,
                index=[0]
            )
            df_metrics = pd.concat([df_metrics, df_train_metrics], ignore_index=True)

        _unwrap_model(model).eval()
        # Save checkpoints only on the main process
        if _is_main_process() and (epoch + 1) % args.model_save_freq == 0:
            logger.info(f"Saving model checkpoint at epoch {epoch + 1}")
            torch.save(
                _unwrap_model(model).pmpnn.state_dict(),
                f"{args.model_save_dir}/{args.run_name}_epoch{epoch}.pt",
            )

        # Validation and test across all ranks
        if (epoch + 1) % args.val_freq == 0:
            with torch.no_grad():
                valid_metrics = run_eval(dl_valid)
                test_metrics = run_eval(dl_test)

            if _is_main_process():
                logger.info(f'{print_prefix}: Valid metrics: '
                            f'{ {k: "{0:0.4f}".format(v) for k, v in valid_metrics.items() if v is not None} }')
                logger.info(f'{print_prefix}: Test metrics: '
                            f'{ {k: "{0:0.4f}".format(v) for k, v in test_metrics.items() if v is not None} }')

                df_valid_metrics = pd.DataFrame(
                    {"epoch": epoch + 1, "split": "valid"} | valid_metrics,
                    index=[0]
                )
                df_test_metrics = pd.DataFrame(
                    {"epoch": epoch + 1, "split": "test"} | test_metrics,
                    index=[0]
                )
                df_metrics = pd.concat([df_metrics, df_valid_metrics, df_test_metrics], ignore_index=True)

                # Save metrics CSV
                df_metrics.to_csv(os.path.join(args.model_save_dir, "metrics.csv"), index=False)

                if args.wandb:
                    wandb.log(valid_metrics, step=epoch + 1)
                    wandb.log(test_metrics, step=epoch + 1)
                    wandb.log(
                        {
                            "train_loss": mean_loss_all,
                            "train_spearman": mean_sp_all,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        step=epoch + 1,
                    )

    if _is_main_process():
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        torch.save(
            _unwrap_model(model).pmpnn.state_dict(), f"{args.model_save_dir}/{args.run_name}_final.pt"
        )


if __name__ == "__main__":
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
