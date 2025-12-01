import argparse
import logging
import os

import torch

from stabddg.jobs.skempi_eval import skempi_eval
from stabddg.misc import optional_int
from stabddg.model import StaBddG
from stabddg.mpnn_utils import ProteinMPNN
from stabddg.ppi_dataset import SKEMPIDataset
from stabddg.training import _is_main_process
from stabddg.utils.torch import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    argparser.add_argument("--run_name", type=str, default="skempi-eval")
    argparser.add_argument("--checkpoint", type=str, default="./model_ckpts/stabddg.pt")
    argparser.add_argument("--skempi_path", type=str, default="data/SKEMPI/filtered_skempi.csv")
    argparser.add_argument("--skempi_pdb_dir", type=str, default="")
    argparser.add_argument("--skempi_pdb_cache_path", type=str, default="cache/skempi_full_mask_pdb_dict.pkl")
    argparser.add_argument("--skempi_split_path", type=str, default="data/SKEMPI/test_pdb.pkl")
    argparser.add_argument("--ensemble", type=int, default=20)
    argparser.add_argument("--output_dir", type=str, default="cache")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--noise_level", type=float, default=0.1, help="amount of backbone noise")
    argparser.add_argument("--batch_size", type=int, default=10000)
    argparser.add_argument(
        "--sample_size",
        type=optional_int,
        default=None,
        help="Overall number of SKEMPI data points to use for evaluation (useful for debugging)"
    )
    # torchrun passes --local_rank automatically in distributed runs
    argparser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by torchrun for multi-GPU evaluation.",)

    args = argparser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(local_rank=args.local_rank)
    logger.info(f"Using device: {device}")

    dataset = SKEMPIDataset(
        split_path=args.skempi_split_path,
        pdb_dir=args.skempi_pdb_dir,
        csv_path=args.skempi_path,
        pdb_dict_cache_path=args.skempi_pdb_cache_path,
    )

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

    mpnn_checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in mpnn_checkpoint.keys():
        pmpnn.load_state_dict(mpnn_checkpoint["model_state_dict"])
    else:
        pmpnn.load_state_dict(mpnn_checkpoint)
    print("Successfully loaded model at", args.checkpoint)

    model = StaBddG(pmpnn=pmpnn, noise_level=args.noise_level, device=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        df_pred = skempi_eval(
            model=model,
            dataset=dataset,
            device=device,
            ensemble=args.ensemble,
            batch_size=args.batch_size,
            sample_size=args.sample_size
        )

    # Only the main process will have the full concatenated DataFrame; others may get empty
    if _is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        df_pred.to_csv(
            os.path.join(args.output_dir, f"skempi_predictions.csv"),
            index=False,
        )