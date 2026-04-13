#!/usr/bin/env python3
"""
download_datasets.py -- Download external benchmark datasets for Engram.

Fetches academic datasets at runtime so they never live in the repo.
Cached in benchmarks/.cache/ (gitignored).

Usage:
    python -m benchmarks.download_datasets              # download all
    python -m benchmarks.download_datasets longmemeval   # just LongMemEval
    python -m benchmarks.download_datasets locomo         # just LoCoMo
"""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent
CACHE_DIR = BENCHMARKS_DIR / ".cache"

# -- Dataset locations --------------------------------------------------------

LONGMEMEVAL_HF_REPO = "xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}
LONGMEMEVAL_DIR = CACHE_DIR / "longmemeval"

LOCOMO_GIT_REPO = "https://github.com/snap-research/locomo.git"
LOCOMO_DIR = CACHE_DIR / "locomo"
LOCOMO_DATA_FILE = LOCOMO_DIR / "data" / "locomo10.json"


# -- Helpers ------------------------------------------------------------------

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _download_hf_file(repo_id: str, filename: str, dest: Path):
    """Download a single file from HuggingFace datasets via direct URL."""
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    dest_file = dest / filename
    if dest_file.exists():
        size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"  OK {filename} already cached ({size_mb:.1f} MB)")
        return dest_file

    print(f"  v Downloading {filename} from HuggingFace...")
    _ensure_dir(dest)
    try:
        urllib.request.urlretrieve(url, str(dest_file))
        size_mb = dest_file.stat().st_size / (1024 * 1024)
        print(f"  OK {filename} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  X Failed to download {filename}: {e}")
        if dest_file.exists():
            dest_file.unlink()
        raise
    return dest_file


def _git_clone_shallow(repo_url: str, dest: Path):
    """Shallow-clone a git repo."""
    if dest.exists() and (dest / ".git").exists():
        print(f"  OK {dest.name} already cloned")
        return
    print(f"  v Cloning {repo_url} (shallow)...")
    _ensure_dir(dest.parent)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"  OK Cloned to {dest}")


# -- Dataset downloaders ------------------------------------------------------

def download_longmemeval(variant: str = "s") -> Path:
    """
    Download LongMemEval dataset (cleaned version).

    Args:
        variant: "s" (small, ~40 sessions), "m" (medium, ~500 sessions),
                 or "oracle" (evidence sessions only)

    Returns:
        Path to the downloaded JSON file.
    """
    filename = LONGMEMEVAL_FILES.get(variant)
    if filename is None:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: s, m, oracle"
        )
    return _download_hf_file(LONGMEMEVAL_HF_REPO, filename, LONGMEMEVAL_DIR)


def download_locomo() -> Path:
    """
    Download LoCoMo dataset via git clone.

    Returns:
        Path to locomo10.json
    """
    _git_clone_shallow(LOCOMO_GIT_REPO, LOCOMO_DIR)
    if not LOCOMO_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Expected {LOCOMO_DATA_FILE} after clone. "
            f"Check if the LoCoMo repo structure changed."
        )
    return LOCOMO_DATA_FILE


def download_all():
    """Download all external datasets."""
    print("=" * 60)
    print("Downloading external benchmark datasets for Engram")
    print("=" * 60)

    print("\n-- LongMemEval (HuggingFace) --")
    for variant in ["s", "oracle"]:
        try:
            download_longmemeval(variant)
        except Exception as e:
            print(f"  ! Skipping longmemeval_{variant}: {e}")

    print("\n-- LoCoMo (GitHub) --")
    try:
        download_locomo()
    except Exception as e:
        print(f"  ! Skipping LoCoMo: {e}")

    print("\nOK Done. Datasets cached in:", CACHE_DIR)


# -- Loaders (used by benchmark scripts) --------------------------------------

def load_longmemeval(variant: str = "s") -> list:
    """
    Load LongMemEval dataset, downloading if needed.

    Returns list of evaluation instances, each containing:
    - question_id, question_type, question, answer, question_date
    - haystack_session_ids, haystack_dates, haystack_sessions
    - answer_session_ids
    """
    path = download_longmemeval(variant)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_locomo() -> list:
    """
    Load LoCoMo dataset, downloading if needed.

    Returns list of conversation samples, each containing:
    - conversation (sessions with dialog turns)
    - observation, session_summary, event_summary
    - qa (question/answer pairs with category and evidence)
    """
    path = download_locomo()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download external benchmark datasets for Engram"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=["longmemeval", "locomo", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--variant",
        default="s",
        help="LongMemEval variant: s, m, oracle (default: s)",
    )
    args = parser.parse_args()

    if args.dataset == "longmemeval":
        print("-- LongMemEval --")
        download_longmemeval(args.variant)
    elif args.dataset == "locomo":
        print("-- LoCoMo --")
        download_locomo()
    else:
        download_all()


if __name__ == "__main__":
    main()
