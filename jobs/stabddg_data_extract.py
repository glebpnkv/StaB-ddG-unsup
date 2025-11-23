#!/usr/bin/env python
import argparse
import os
import pathlib
import subprocess
from typing import Optional


def run_download_data(data_dir: str, script_path: Optional[str] = None) -> None:
    """
    Run download_data.sh with the given data_dir.

    Parameters
    ----------
    data_dir : str
        Local directory where data will be downloaded.
    script_path : str, optional
        Path to download_data.sh. If None, assumes it's in ./scripts.
    """
    if script_path is None:
        # adjust if your script lives somewhere else
        script_path = os.path.join(
            pathlib.Path(__file__).resolve().parents[1],
            "scripts",
            "download_data.sh"
        )

    script_path = pathlib.Path(script_path)
    if not script_path.is_file():
        raise FileNotFoundError(f"download_data.sh not found at {script_path}")

    env = os.environ.copy()
    # data_dir is passed as $1 to the script
    subprocess.run(
        ["bash", str(script_path), data_dir],
        check=True,
        env=env,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory to download data into",
        required=False,
    )
    parser.add_argument(
        "--script-path",
        default=None,
        help="Path to download_data.sh (default: ./scripts/download_data.sh relative to this file)",
        required=False,
    )
    args = parser.parse_args()
    run_download_data(args.data_dir, args.script_path)


if __name__ == "__main__":
    main()