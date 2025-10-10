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


def download_dir_from_gcp(
    bucket_name: str,
    src_prefix: str,
    local_dir: str,
    workers: int = 8,
    skip_existing: bool = False
) -> None:
    """Downloads all objects under gs://bucket_name/src_prefix/** to local_dir, preserving subfolders."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    prefix = src_prefix.rstrip("/") + "/"

    # Collect names *relative to the prefix* so local_dir mirrors the folder contents
    rel_names = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):  # ignore placeholder "directory" objects
            continue
        rel_names.append(blob.name[len(prefix):])

    results = transfer_manager.download_many_to_path(
        bucket,
        rel_names,
        destination_directory=str(Path(local_dir)),
        blob_name_prefix=prefix,
        max_workers=workers,
        skip_if_exists=skip_existing,
    )

    # Raise if any failed
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        raise RuntimeError(f"{len(errors)} downloads failed; first error: {errors[0]}")
