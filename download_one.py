"""Download one dataset. Usage: python download_one.py <name>"""
import os, shutil, sys, zipfile
from pathlib import Path
import gdown, torch

DATA_DIR = Path("data/medical_calibration")
DATASETS = {
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

name = sys.argv[1]
links = DATASETS[name]
work = DATA_DIR / f"_work_{name}"
work.mkdir(parents=True, exist_ok=True)

# Download data
print(f"[{name}] Downloading data...")
data_zip = work / "data.zip"
gdown.download(links["data"], str(data_zip), fuzzy=True)

print(f"[{name}] Extracting data...")
with zipfile.ZipFile(data_zip) as z:
    z.extractall(work / "data")
data_zip.unlink()  # Free space immediately

# Find dataset dir
subdirs = [d for d in (work / "data").iterdir() if d.is_dir()]
dataset_dir = subdirs[0] if len(subdirs) == 1 else work / "data"
print(f"[{name}] Dataset: {dataset_dir.name}")

# Report class structure
for split in ["train", "val", "test"]:
    sd = dataset_dir / split
    if sd.exists():
        classes = sorted([d.name for d in sd.iterdir() if d.is_dir()])
        counts = {c: len(list((sd / c).iterdir())) for c in classes}
        print(f"  {split}: {sum(counts.values())} images, {len(classes)} classes")
        if split == "train":
            for c, n in counts.items():
                print(f"    {c}: {n}")

# Download checkpoint
print(f"[{name}] Downloading checkpoint...")
ckpt_dir = work / "ckpt"
gdown.download_folder(links["checkpoint"], output=str(ckpt_dir))

ckpt_path = list(ckpt_dir.rglob("*.pth"))[0]
print(f"[{name}] Stripping optimizer...")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
hw = ckpt["model"]["head.weight"]
print(f"  head: {hw.shape} = {hw.shape[0]} classes, epoch {ckpt.get('epoch','?')}")
stripped = work / "checkpoint-best.pth"
torch.save({"model": ckpt["model"], "epoch": ckpt.get("epoch")}, stripped)
del ckpt
shutil.rmtree(ckpt_dir)  # Free 3.4GB

# Zip
final_zip = DATA_DIR / f"{name}.zip"
print(f"[{name}] Creating {final_zip}...")
with zipfile.ZipFile(final_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            fp = Path(root) / f
            zf.write(fp, str(fp.relative_to(dataset_dir.parent)))
    zf.write(stripped, "checkpoint-best.pth")

# Clean up
shutil.rmtree(work)
size = final_zip.stat().st_size / 1e9
print(f"[{name}] Done: {size:.2f} GB")
