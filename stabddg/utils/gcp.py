from pathlib import Path
from google.cloud import storage
from google.cloud.storage import transfer_manager


def upload_dir_to_gcp(
    bucket_name: str,
    local_dir: str,
    dst_prefix: str,
    region: str,
    workers: int = 8,
    skip_existing: bool = False
) -> None:
    """Copies all files under local_dir to gs://bucket_name/dst_prefix/..."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # Ensure the bucket exists; try to create if it doesn't
    try:
        client.get_bucket(bucket_name)
    except Exception:
        try:
            bucket = client.create_bucket(bucket_or_name=bucket_name, location=region)
        except Exception as e:
            raise RuntimeError(f"Failed to create bucket '{bucket_name}' in region '{region}': {e}")

    # Build relative file list
    root = Path(local_dir)
    files = [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()]

    # Ensure the prefix ends with "/" (GCS uses object name prefixes; no real folders)
    prefix = dst_prefix.rstrip("/") + "/"

    results = transfer_manager.upload_many_from_filenames(
        bucket,
        files,
        source_directory=str(root),
        blob_name_prefix=prefix,
        max_workers=workers,
        skip_if_exists=skip_existing,
    )

    # Raise if any failed
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise RuntimeError(f"{len(errors)} uploads failed; first error: {errors[0]}")
