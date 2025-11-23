import argparse
import logging
import os
from urllib.parse import urlparse
from stabddg.utils.gcp import download_dir_from_gcp, upload_dir_to_gcp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parses gs://bucket/prefix/path into (bucket, prefix/path)."""
    p = urlparse(uri)
    if p.scheme != "gs":
        raise ValueError(f"Invalid GCS URI: {uri}")
    return p.netloc, p.path.lstrip("/")


def download(args):
    bucket, prefix = parse_gcs_uri(args.uri)
    logger.info(f"Downloading from gs://{bucket}/{prefix} to {args.dest}...")
    download_dir_from_gcp(bucket, prefix, args.dest)


def upload(args):
    bucket, prefix = parse_gcs_uri(args.uri)

    # Mirror the behavior of 'cp -r src dest':
    # If src is /path/to/folder, we want it to appear at gs://bucket/prefix/folder
    dir_name = os.path.basename(os.path.normpath(args.src))
    final_prefix = os.path.join(prefix, dir_name)

    logger.info(f"Uploading from {args.src} to gs://{bucket}/{final_prefix}...")
    upload_dir_to_gcp(bucket, args.src, final_prefix, region="europe-west4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")

    # Download command
    p_dl = subparsers.add_parser("download")
    p_dl.add_argument("uri", help="Source GCS URI (gs://...)")
    p_dl.add_argument("dest", help="Local destination directory")
    p_dl.set_defaults(func=download)

    # Upload command
    p_up = subparsers.add_parser("upload")
    p_up.add_argument("src", help="Local source directory")
    p_up.add_argument("uri", help="Destination GCS URI (gs://...)")
    p_up.set_defaults(func=upload)

    args = parser.parse_args()
    args.func(args)