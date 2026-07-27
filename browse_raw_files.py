"""
APEX-AI — Browse the raw rul_datasets cache folder yourself

Run:
    python browse_raw_files.py

This does two things:
1. Prints the full folder/file tree under the rul_datasets cache directory
   (~/.rul-datasets on Windows: C:\\Users\\<you>\\.rul-datasets) so you can
   see exactly how the data is organized on disk.
2. Opens one specific .npy file and prints its raw contents, since .npy
   files are binary and won't open properly in Notepad.
"""

import os
import numpy as np
from pathlib import Path

CACHE_DIR = Path.home() / ".rul-datasets"


def print_tree(root: Path, max_depth: int = 4, max_files_per_folder: int = 8):
    """Prints a folder/file tree, capped so it doesn't dump thousands of lines."""
    root = Path(root)
    if not root.exists():
        print(f"Folder not found: {root}")
        print("Have you run apex_v1_baseline.py or explore_femto.py at least once yet?")
        return

    for current_root, dirs, files in os.walk(root):
        depth = len(Path(current_root).relative_to(root).parts)
        if depth > max_depth:
            dirs[:] = []  # stop descending further
            continue

        indent = "  " * depth
        print(f"{indent}{Path(current_root).name}/")

        for f in files[:max_files_per_folder]:
            print(f"{indent}  {f}")
        if len(files) > max_files_per_folder:
            print(f"{indent}  ... and {len(files) - max_files_per_folder} more files")


def inspect_one_npy_file(root: Path):
    """Finds the first .npy file it can and prints a peek inside it."""
    npy_files = list(root.rglob("*.npy"))
    if not npy_files:
        print("\nNo .npy files found yet — run the baseline script first to trigger a download.")
        return

    sample_file = npy_files[0]
    print(f"\nOpening a sample file: {sample_file}")
    data = np.load(sample_file)
    print(f"Shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print("First few values:")
    print(data.flatten()[:10])


if __name__ == "__main__":
    print(f"Looking in: {CACHE_DIR}\n")
    print("=== Folder structure ===")
    print_tree(CACHE_DIR)

    print("\n=== Peek inside one file ===")
    inspect_one_npy_file(CACHE_DIR)
