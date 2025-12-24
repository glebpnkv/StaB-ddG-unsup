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


def upload_file_to_gcp(
    bucket_name: str,
    local_path: str,
    dst_blob_name: str,
    region: str,
) -> None:
    """
    Uploads a single file to GCS as gs://bucket_name/dst_blob_name.

    This mirrors `gsutil cp local_path gs://bucket_name/dst_blob_name`.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # Ensure bucket exists (reuse same pattern as upload_dir_to_gcp)
    try:
        client.get_bucket(bucket_name)
    except Exception:
        try:
            bucket = client.create_bucket(bucket_or_name=bucket_name, location=region)
        except Exception as e:
            raise RuntimeError(f"Failed to create bucket '{bucket_name}' in region '{region}': {e}")

    blob_name = dst_blob_name.lstrip("/")
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)


def download_file_from_gcp(
    bucket_name: str,
    src_blob_name: str,
    local_path: str,
) -> None:
    """
    Downloads a single object gs://bucket_name/src_blob_name to local_path.

    local_path may include directories; they will be created if needed.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob_name = src_blob_name.lstrip("/")
    blob = bucket.blob(blob_name)

    # `exists` requires a client in some versions of the library
    if not blob.exists(client):
        raise FileNotFoundError(f"GCS object gs://{bucket_name}/{blob_name} does not exist")

    dest_path = Path(local_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest_path))


def download_dir_from_gcp(
    bucket_name: str,
    src_prefix: str,
    local_dir: str,
    workers: int = 8,
    skip_existing: bool = False
) -> None:
    """
    Downloads from GCS to a local directory.

    Behavior:
      - If gs://bucket_name/src_prefix is a single object, downloads that object
        into local_dir / basename(src_prefix).
      - Otherwise, treats src_prefix as a "directory prefix" and downloads all
        objects under gs://bucket_name/src_prefix/** into local_dir, preserving
        subfolders.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Normalise prefix to a blob-style name (no leading slash)
    norm_prefix = src_prefix.lstrip("/")

    # -------- Case 1: src_prefix is a single object --------
    blob = bucket.blob(norm_prefix)
    if blob.exists(client):
        # Download as a single file into local_dir / basename
        target_dir = Path(local_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = target_dir / Path(norm_prefix).name
        blob.download_to_filename(str(local_path))
        return

    # -------- Case 2: src_prefix is treated as a directory prefix --------
    prefix = norm_prefix.rstrip("/") + "/"

    # Collect names *relative to the prefix* so local_dir mirrors the folder contents
    rel_names: list[str] = []
    for b in client.list_blobs(bucket, prefix=prefix):
        if b.name.endswith("/"):  # ignore placeholder "directory" objects
            continue
        rel_names.append(b.name[len(prefix):])

    if not rel_names:
        raise FileNotFoundError(
            f"No objects found under gs://{bucket_name}/{norm_prefix} "
            f"(neither a file nor a directory prefix)"
        )

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