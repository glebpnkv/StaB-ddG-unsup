import argparse
import logging
import os

from stabddg.misc import optional_int
from stabddg.jobs.intact_pretrain import intact_pretrain

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Run metadata
    argparser.add_argument(
        "--run_name",
        type=str,
        default="intact-pretrain",
        help="Name for this training run"
    )

    # Data locations
    argparser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("../../data", "intact"),
        help="Root IntAct data directory"
    )
    argparser.add_argument(
        "--proteins_dir",
        type=str,
        default=os.path.join("../../data", "intact", "proteins", "safetensors"),
        help="Directory with protein tensors"
    )
    argparser.add_argument(
        "--assemblies_dir",
        type=str,
        default=os.path.join("../../data", "intact", "assemblies", "safetensors"),
        help="Directory with assembly tensors"
    )

    # Model saving
    argparser.add_argument(
        "--model_save_dir",
        type=str,
        default=os.path.join("../../cache", "intact_pretrain_test_exist_norm"),
        help="Directory to save checkpoints and logs"
    )
    argparser.add_argument("--wandb", action="store_true")

    # Sampling settings for IntActDataset
    argparser.add_argument(
        "--max_length",
        type=int,
        default=400,
        help="Max sequence length / truncation length"
    )
    argparser.add_argument("--k_neutral", type=int, default=20, help="Number of neutral samples per batch")
    argparser.add_argument("--k_pos", type=int, default=5, help="Number of positive samples per batch")
    argparser.add_argument("--k_neg", type=int, default=5, help="Number of negative samples per batch")

    # Core training hyperparameters
    argparser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    argparser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    argparser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # Loss / training controls
    argparser.add_argument(
        "--noise_level",
        type=float,
        default=0.1,
        help="Noise level for StaBddG antithetic variates"
    )
    argparser.add_argument(
        "--normalize_loss",
        dest="normalize_loss",
        action="store_true",
        default=True,
        help="Normalize loss by item counts"
    )

    # Data split settings
    argparser.add_argument(
        "--intact_sample_size",
        type=optional_int,
        default=None,
        help="Overall number of IntAct data to use for training (useful for debugging)"
    )
    argparser.add_argument("--valid_size", type=float, default=0.1, help="Validation split size (fraction)")
    argparser.add_argument("--test_size", type=float, default=0.1, help="Test split size (fraction)")
    argparser.add_argument("--random_state", type=int, default=42, help="Random seed for splits")

    # Dataloader / validation cadence
    argparser.add_argument(
        "--num_dataloader_workers",
        type=int,
        default=1,
        help="Number of workers for DataLoaders"
    )
    argparser.add_argument("--model_val_freq", type=int, default=5, help="Validation frequency in epochs")

    # Model configuration
    argparser.add_argument(
        "--use_antithetic_variates",
        dest="use_antithetic_variates",
        action="store_true",
        default=True,
        help="Use antithetic variates in StaBddG"
    )
    # argparser.set_defaults(use_antithetic_variates=True)

    # Optional: start from an existing model checkpoint (Model Existing Setup)
    argparser.add_argument(
        "--model_existing_checkpoint",
        type=str,
        default=os.path.join("../../model_ckpts", "proteinmpnn.pt"),
        help="Path to an existing ProteinMPNN/StaBddG checkpoint to initialize from; if omitted, start from scratch"
    )

    args = argparser.parse_args()

    intact_pretrain(
        run_name=args.run_name,
        data_dir=args.data_dir,
        proteins_dir=args.proteins_dir,
        assemblies_dir=args.assemblies_dir,
        model_save_dir=args.model_save_dir,
        max_length=args.max_length,
        k_neutral=args.k_neutral,
        k_pos=args.k_pos,
        k_neg=args.k_neg,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        noise_level=args.noise_level,
        normalize_loss=args.normalize_loss,
        intact_sample_size=args.intact_sample_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
        num_dataloader_workers=args.num_dataloader_workers,
        model_val_freq=args.model_val_freq,
        use_antithetic_variates=args.use_antithetic_variates,
        model_existing_checkpoint=args.model_existing_checkpoint,
        use_wandb=args.wandb,
    )
