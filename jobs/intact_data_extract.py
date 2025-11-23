#!/usr/bin/env python3

import argparse
import logging

from stabddg.jobs.intact_data_extract import extract_intact_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract IntAct pretraining data")
    parser.add_argument("--data-dir", default="data", help="Base data directory (default: data)")

    # Worker settings
    parser.add_argument("--alphafold-workers", type=int, default=128, help="Max workers for AlphaFold fetch")
    parser.add_argument("--assemblies-workers", type=int, default=128, help="Max workers for assemblies atoms fetch")
    parser.add_argument("--metadata-workers", type=int, default=32, help="Max workers for assemblies metadata fetch")

    # GCP options
    parser.add_argument("--gcp-upload", action="store_true", help="Upload the intact data dir to GCP after building")
    parser.add_argument("--gcp-download", action="store_true", help="Download the intact data dir from GCP before splitting")
    parser.add_argument("--gcp-bucket", default="stab-ddg-unsup", help="GCP bucket name")
    parser.add_argument("--gcp-prefix", default="data/intact", help="Destination/prefix within the bucket")
    parser.add_argument("--gcp-region", default="europe-west4", help="GCP region for uploads")

    args = parser.parse_args()

    extract_intact_data(
        data_dir=args.data_dir,
        alphafold_workers=args.alphafold_workers,
        metadata_workers=args.metadata_workers,
        assemblies_workers=args.assemblies_workers,
        gcp_bucket=args.gcp_bucket,
        gcp_prefix=args.gcp_prefix,
        gcp_region=args.gcp_region,
        gcp_upload=args.gcp_upload,
        gcp_download=args.gcp_download,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()

