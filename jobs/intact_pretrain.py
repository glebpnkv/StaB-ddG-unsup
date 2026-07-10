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
    argparser.add_argument("--batch_size", type=int, default=2, help="Number of anchors per optimizer step")
    argparser.add_argument(
        "--micro_batch_size",
        type=int,
        default=0,
        help="If >0, forward each contrast pool (anchor/pos/neg/neutral) in chunks of this size and "
             "concatenate the outputs, to cap peak activation memory. 0 = whole pool at once.",
    )
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
        "--lambda_supcon",
        type=float,
        default=0.0,
        help="Weight for the supervised-contrastive term. 0 = pure sign-direction objective (SupCon on "
             "scalar ΔΔG is finicky); raise once the sign is learning."
    )
    argparser.add_argument(
        "--lambda_sign",
        type=float,
        default=1.0,
        help="Weight for the sign-direction loss. >0 anchors the ΔΔG sign to the weak labels; with 0 "
             "the distance-based SupCon term is sign-invariant so the direction is not identifiable."
    )

    argparser.add_argument(
        "--lambda_neutral",
        type=float,
        default=0.0,
        help="Weight for scale of neutral forecasts"
    )

    argparser.add_argument(
        "--use_neutral_normalizer",
        type=int,
        default=0,
        help="If 1, centre/scale ΔΔG by the neutral pool's robust median/MAD before the contrastive + "
             "sign terms (removes a global offset, sets a common scale; forces the neutral forward).",
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
    argparser.add_argument(
        "--log_every_batches",
        type=int,
        default=1,
        help="Emit a per-batch metrics line to stdout (-> CloudWatch) every N batches (1 = every batch). "
             "Per-epoch summaries are always logged.",
    )

    # Model configuration
    argparser.add_argument(
        "--use_antithetic_variates",
        dest="use_antithetic_variates",
        action="store_true",
        default=True,
        help="Use antithetic variates in StaBddG"
    )
    # argparser.set_defaults(use_antithetic_variates=True)
    argparser.add_argument(
        "--grad_checkpoint",
        type=int,
        default=0,
        help="If 1, recompute the ProteinMPNN forward during backward (gradient checkpointing) to cap "
             "activation memory (~30%% slower). Complements --micro_batch_size for fitting larger pools.",
    )

    # Optional: start from an existing model checkpoint (Model Existing Setup)
    argparser.add_argument(
        "--model_existing_checkpoint",
        type=str,
        default="",
        help="Path to an existing ProteinMPNN/StaBddG checkpoint to initialize from; if omitted, start from scratch"
    )

    args = argparser.parse_args()

    model_existing_checkpoint = None if args.model_existing_checkpoint == "" else args.model_existing_checkpoint

    intact_pretrain(
        run_name=args.run_name,
        data_dir=args.data_dir,
        assemblies_dir=args.assemblies_dir,
        model_save_dir=args.model_save_dir,
        max_length=args.max_length,
        k_neutral=args.k_neutral,
        k_pos=args.k_pos,
        k_neg=args.k_neg,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        epochs=args.epochs,
        lr=args.lr,
        noise_level=args.noise_level,
        lambda_supcon=args.lambda_supcon,
        lambda_sign=args.lambda_sign,
        lambda_neutral=args.lambda_neutral,
        use_neutral_normalizer=bool(args.use_neutral_normalizer),
        intact_sample_size=args.intact_sample_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        random_state=args.random_state,
        num_dataloader_workers=args.num_dataloader_workers,
        model_val_freq=args.model_val_freq,
        log_every_batches=args.log_every_batches,
        use_antithetic_variates=args.use_antithetic_variates,
        use_grad_checkpoint=bool(args.grad_checkpoint),
        model_existing_checkpoint=model_existing_checkpoint,
        use_wandb=args.wandb,
    )
