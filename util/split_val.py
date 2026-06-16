import sys
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic"}
VAL_RATIO = 0.2
SUBFOLDERS = ("real", "scraped", "augmented")


def get_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def split_subfolder(train_sub: Path, val_sub: Path, sub_name: str, class_name: str) -> None:
    train_imgs = get_images(train_sub)
    val_imgs   = get_images(val_sub)

    total = len(train_imgs) + len(val_imgs)
    if total == 0:
        return

    target_val = round(total * VAL_RATIO)
    delta = target_val - len(val_imgs)

    if delta == 0:
        print(f"  {class_name}/{sub_name}: already at {len(val_imgs)}/{total} val ({len(val_imgs)/total:.0%}) — no change")
        return

    if delta > 0:
        # need more in val — move from train
        candidates = train_imgs.copy()
        random.shuffle(candidates)
        to_move = candidates[:delta]
        val_sub.mkdir(parents=True, exist_ok=True)
        for img in to_move:
            shutil.move(str(img), val_sub / img.name)
        print(f"  {class_name}/{sub_name}: moved {len(to_move)} train→val  ({target_val}/{total} val, {target_val/total:.0%})")
    else:
        # too many in val — move excess back to train
        excess = abs(delta)
        candidates = val_imgs.copy()
        random.shuffle(candidates)
        to_move = candidates[:excess]
        train_sub.mkdir(parents=True, exist_ok=True)
        for img in to_move:
            shutil.move(str(img), train_sub / img.name)
        print(f"  {class_name}/{sub_name}: moved {len(to_move)} val→train  ({target_val}/{total} val, {target_val/total:.0%})")


def split_val(data_path: str) -> None:
    data_dir = Path(data_path)
    train_dir = data_dir / "train"

    if not train_dir.is_dir():
        print(f"Error: '{train_dir}' does not exist")
        sys.exit(1)

    classes = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if not classes:
        print("No class folders found in train/")
        return

    print(f"Rebalancing {len(classes)} class(es) to {VAL_RATIO:.0%} val per subfolder:\n")
    for class_dir in classes:
        val_class_dir = data_dir / "val" / class_dir.name
        for sub in SUBFOLDERS:
            split_subfolder(class_dir / sub, val_class_dir / sub, sub, class_dir.name)
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python split_val.py <data_folder>")
        sys.exit(1)

    split_val(sys.argv[1])
