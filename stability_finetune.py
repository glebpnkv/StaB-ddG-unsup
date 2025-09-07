import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
import wandb
from accelerate import Accelerator
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from stabddg.model import StaBddG
from stabddg.mpnn_utils import StructureDataset, ProteinMPNN, parse_PDB

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
        f"spearman": float(np.mean(val_spearman)),
        f"pearson": float(np.mean(val_pearson)),
        f"all_spearman": float(sp),
        f"all_pearson": float(pr),
    }


def finetune(
    model,
    dataset_train,
    dataset_valid,
    dataset_test,
    ddG_data,
    args,
    batch_size=10000,
    device="cuda",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ddG_loss_fn = torch.nn.MSELoss()

    # DataFrame with training, validation and test metrics
    df_metrics = pd.DataFrame()

    # Directory to save model checkpoints
    if not os.path.exists(args.model_save_dir):
        logger.info(f"Creating directory {args.model_save_dir}")
        os.makedirs(args.model_save_dir)

    # Creating a logging file logs.txt
    log_path = os.path.join(args.model_save_dir, "logs.txt")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_path}")

    for epoch in tqdm(range(args.num_epochs), desc="Epoch"):
        model.train()
        train_sum = []
        spearmans = []
        print_prefix = f"Epoch {epoch + 1}"

        # Iterate through all training domains.
        for sample in tqdm(dataset_train, desc="Train", unit="sample"):
            pdb_name = sample["name"]
            ddG = ddG_data[f"{pdb_name}.pdb"]["ddG"].to(device)
            mut_seqs = ddG_data[f"{pdb_name}.pdb"]["mut_seqs"]
            N = mut_seqs.shape[0]
            M = (
                batch_size // mut_seqs.shape[1]
            )  # convert number of tokens to number of sequences per batch

            # Random shuffling
            permutation = torch.randperm(ddG.shape[0])
            ddG = ddG[permutation]
            mut_seqs = mut_seqs[permutation]

            sample_pred = []

            # Batching for mutants
            for batch_idx in range(0, N, M):
                B = min(N - batch_idx, M)
                optimizer.zero_grad()

                # ddG prediction
                pred = model.folding_ddG(sample, mut_seqs[batch_idx : batch_idx + B])

                ddG_loss = ddG_loss_fn(pred, ddG[batch_idx : batch_idx + B])

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

        train_metrics = {
            "loss": float(np.mean(train_sum)),
            "spearman": float(np.mean(spearmans)),
            "all_spearman": None,
            "all_pearson": None,
        }
        logger.info(f'{print_prefix}: Training metrics: '
                    f'{ {k: "{0:0.4f}".format(v) for k, v in train_metrics.items() if v is not None} }')

        df_train_metrics = pd.DataFrame(
            {"epoch": epoch + 1, "split": "train"} | train_metrics,
            index=[0]
        )
        df_metrics = pd.concat([df_metrics, df_train_metrics], ignore_index=True)

        model.eval()
        if (epoch + 1) % args.model_save_freq == 0:
            logger.info(f"Saving model checkpoint at epoch {epoch + 1}")
            torch.save(
                model.pmpnn.state_dict(),
                f"{args.model_save_dir}/{args.run_name}_epoch{epoch}.pt",
            )
        if (epoch + 1) % args.val_freq == 0:
            with torch.no_grad():
                valid_metrics = validation_step(model, ddG_data, dataset_valid, batch_size=10000, device=device)
                logger.info(f'{print_prefix}: Valid metrics: '
                            f'{ {k: "{0:0.4f}".format(v) for k, v in valid_metrics.items() if v is not None} }')
                df_valid_metrics = pd.DataFrame(
                    {"epoch": epoch + 1, "split": "valid"} | valid_metrics,
                    index=[0]
                )
                df_metrics = pd.concat([df_metrics, df_valid_metrics], ignore_index=True)

                test_metrics = validation_step(model, ddG_data, dataset_test, batch_size=10000, device=device)
                logger.info(f'{print_prefix}: Test metrics: '
                            f'{ {k: "{0:0.4f}".format(v) for k, v in test_metrics.items() if v is not None} }')
                df_test_metrics = pd.DataFrame(
                    {"epoch": epoch + 1, "split": "test"} | test_metrics,
                    index=[0]
                )
                df_metrics = pd.concat([df_metrics, df_test_metrics], ignore_index=True)

            # Saving training metrics as a CSV file
            df_metrics.to_csv(os.path.join(args.model_save_dir, "metrics.csv"), index=False)

            if args.wandb:
                wandb.log(valid_metrics, step=epoch + 1)
                wandb.log(test_metrics, step=epoch + 1)

                wandb.log(
                    {
                        "train_loss": np.mean(train_sum),
                        "train_spearman": np.mean(spearmans),
                    },
                    step=epoch + 1,
                )
                wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    torch.save(
        model.pmpnn.state_dict(), f"{args.model_save_dir}/{args.run_name}_final.pt"
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
    args = argparser.parse_args()

    torch.manual_seed(args.seed)

    # Dataset preprocessing/loading
    # Read split files
    with open("data/rocklin/mega_splits.pkl", "rb") as f:
        splits = pickle.load(f)
    train_names = splits["train"]
    val_names = splits["val"].tolist()
    test_names = splits["test"].tolist()

    # Load AF predicted structures
    pdb_dict_train = []
    for name in tqdm(train_names, desc="Preparing train set"):
        name = name.split(".pdb", 1)[0] + ".pdb"
        name = name.replace("|", ":")
        path = os.path.join(args.pdb_dir, name)
        pdb_dict_train.append(parse_PDB(path)[0])

    pdb_dict_val = []
    for name in tqdm(val_names, desc="Preparing validation set"):
        name = name.split(".pdb", 1)[0] + ".pdb"
        name = name.replace("|", ":")
        path = os.path.join(args.pdb_dir, name)
        pdb_dict_val.append(parse_PDB(path)[0])

    pdb_dict_test = []
    for name in tqdm(test_names, desc="Preparing test set"):
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
    accelerator = Accelerator()
    device = accelerator.device
    logger.info(f"Using device: {device}", )

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

    mpnn_checkpoint = torch.load(args.checkpoint)
    if "model_state_dict" in mpnn_checkpoint.keys():
        pmpnn.load_state_dict(mpnn_checkpoint["model_state_dict"])
    else:
        pmpnn.load_state_dict(mpnn_checkpoint)
    logger.info(f"Successfully loaded model at {args.checkpoint}")

    model = StaBddG(
        pmpnn=pmpnn,
        use_antithetic_variates=not args.no_antithetic_variates,
        noise_level=args.noise_level,
        device=device,
    )
    model.compile()
    model.to(device)

    # Initialize wandb logging
    if args.wandb:
        logger.info("Initializing weights and biases.")
        wandb.init(
            project="",
            entity="",
            name=args.run_name,
        )
        logger.info("Weights and biases initialized.")

    finetune(
        model,
        dataset_train,
        dataset_valid,
        dataset_test,
        ddG_data,
        args,
        batch_size=args.batch_size,
        device=device,
    )
