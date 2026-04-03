"""Download and prepare all CFP RETFound benchmark datasets.

Downloads data splits and checkpoints from Google Drive, strips optimizer
state from checkpoints, verifies class structure, and creates zips.

Run from repo root:
    python download_medical_datasets.py
"""

import os
import shutil
import zipfile
from pathlib import Path

import gdown
import torch

DATA_DIR = Path("data/medical_calibration")

# Google Drive links from BENCHMARK.md
DATASETS = {
    "messidor2": {
        "data": "https://drive.google.com/file/d/1vOLBUK9xdzNV8eVkRjVdNrRwhPfaOmda/view?usp=sharing",
        "checkpoint": "https://drive.google.com/drive/folders/1OTBRAHNbaytpwzwMHw9SWrltJouEEuxF?usp=sharing",
    },
    "idrid": {
        "data": "https://drive.google.com/file/d/1c6zexA705z-ANEBNXJOBsk6uCvRnzmr3/view?usp=sharing",
        "checkpoint": "https://drive.google.com/drive/folders/18Ml-B7nhejK4rnNG8upjqIARSlMP5kUc?usp=sharing",
    },
    "papila": {
        "data": "https://drive.google.com/file/d/1JltYs7WRWEU0yyki1CQw5-10HEbqCMBE/view?usp=sharing",
        "checkpoint": "https://drive.google.com/drive/folders/1cHOX6C4NQVi9B6n-7Bxxg7b4-wdI4c73?usp=sharing",
    },
    "glaucoma_fundus": {
        "data": "https://drive.google.com/file/d/18vSazOYDsUGdZ64gGkTg3E6jiNtcrUrI/view?usp=sharing",
        "checkpoint": "https://drive.google.com/drive/folders/10JbanmVxjyX6mghXbxGnGVX1p9nwqsja?usp=sharing",
    },
    "jsiec": {
        "data": "https://drive.google.com/file/d/1q0GFQb-dYwzIx8AwlaFZenUJItix4s8z/view?usp=sharing",
        "checkpoint": "https://drive.google.com/drive/folders/1eosdBXsONUy49cwDO80AOzDHkHiPNJvv?usp=sharing",
    },
    "retina_cataract": {
        "data": "https://drive.google.com/file/d/1vdmjMRDoUm9yk83HMArLiPcLDk_dm92Q/view?usp=sharing",
        "checkpoint": "https://drive.google.com/drive/folders/1n7mXxN-ZUKauOrAlBAiF2E_36F6f0wZD?usp=sharing",
    },
}


def download_and_prepare(name, links):
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")

    work_dir = DATA_DIR / f"_work_{name}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Download data splits
    data_zip = work_dir / "data.zip"
    if not data_zip.exists():
        print(f"Downloading data splits...")
        gdown.download(links["data"], str(data_zip), fuzzy=True)

    # Extract data
    data_dir = work_dir / "data"
    if not data_dir.exists():
        print(f"Extracting data...")
        with zipfile.ZipFile(data_zip) as z:
            z.extractall(data_dir)

    # Find the actual dataset folder (may be nested)
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        dataset_dir = subdirs[0]
    else:
        dataset_dir = data_dir

    dataset_name = dataset_dir.name
    print(f"Dataset folder: {dataset_name}")

    # Check class structure
    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        if split_dir.exists():
            classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
            counts = {c: len(list((split_dir / c).iterdir())) for c in classes}
            total = sum(counts.values())
            print(f"  {split}: {total} images, {len(classes)} classes")
            if split == "train":
                print(f"    Classes (alphabetical): {classes}")
                for c, n in counts.items():
                    print(f"      {c}: {n}")

    # Download checkpoint
    ckpt_dir = work_dir / "checkpoint"
    if not ckpt_dir.exists():
        print(f"Downloading checkpoint...")
        gdown.download_folder(links["checkpoint"], output=str(ckpt_dir))

    # Find checkpoint file
    ckpt_files = list(ckpt_dir.glob("*.pth"))
    if not ckpt_files:
        ckpt_files = list(ckpt_dir.rglob("*.pth"))

    if not ckpt_files:
        print(f"  WARNING: No .pth file found in checkpoint download!")
        return

    ckpt_path = ckpt_files[0]
    print(f"Checkpoint: {ckpt_path.name}")

    # Load and strip optimizer
    print(f"Stripping optimizer state...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head_weight = ckpt["model"]["head.weight"]
    num_classes = head_weight.shape[0]
    print(f"  head.weight shape: {head_weight.shape} -> {num_classes} classes")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")

    # Check class count matches
    train_dir = dataset_dir / "train"
    if train_dir.exists():
        folder_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        if len(folder_classes) != num_classes:
            print(f"  WARNING: {len(folder_classes)} folders but checkpoint has {num_classes} classes!")
        else:
            print(f"  Class count matches: {num_classes}")

    # Save stripped checkpoint
    stripped_path = work_dir / "checkpoint-best.pth"
    torch.save(
        {"model": ckpt["model"], "epoch": ckpt.get("epoch")},
        stripped_path,
    )
    stripped_size = stripped_path.stat().st_size / 1e9
    print(f"  Stripped checkpoint: {stripped_size:.2f} GB")

    # Create final zip: dataset folder + checkpoint
    zip_path = DATA_DIR / f"{name}.zip"
    print(f"Creating {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add dataset folder
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = str(file_path.relative_to(dataset_dir.parent))
                zf.write(file_path, arcname)
        # Add checkpoint
        zf.write(stripped_path, "checkpoint-best.pth")

    zip_size = zip_path.stat().st_size / 1e9
    print(f"  Final zip: {zip_size:.2f} GB")

    # Clean up work dir
    shutil.rmtree(work_dir)
    print(f"  Cleaned up work directory")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, links in DATASETS.items():
        try:
            download_and_prepare(name, links)
        except Exception as e:
            print(f"ERROR processing {name}: {e}")
            import traceback
            traceback.print_exc()

    # Also zip APTOS2019 if it exists unzipped
    aptos_dir = DATA_DIR / "APTOS2019"
    aptos_ckpt = DATA_DIR / "checkpoint-best.pth"
    aptos_zip = DATA_DIR / "aptos2019.zip"
    if aptos_dir.exists() and aptos_ckpt.exists() and not aptos_zip.exists():
        print(f"\n{'='*60}")
        print(f"Zipping existing APTOS2019...")

        # Strip APTOS checkpoint
        ckpt = torch.load(aptos_ckpt, map_location="cpu", weights_only=False)
        stripped_path = DATA_DIR / "_aptos_stripped.pth"
        torch.save(
            {"model": ckpt["model"], "epoch": ckpt.get("epoch")},
            stripped_path,
        )

        with zipfile.ZipFile(aptos_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(aptos_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = str(file_path.relative_to(aptos_dir.parent))
                    zf.write(file_path, arcname)
            zf.write(stripped_path, "checkpoint-best.pth")

        stripped_path.unlink()

        # Remove unzipped files
        shutil.rmtree(aptos_dir)
        aptos_ckpt.unlink()

        zip_size = aptos_zip.stat().st_size / 1e9
        print(f"  aptos2019.zip: {zip_size:.2f} GB")

    print(f"\n{'='*60}")
    print("All done! Final files:")
    for f in sorted(DATA_DIR.glob("*.zip")):
        print(f"  {f.name}: {f.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
