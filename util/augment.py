import sys
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import pillow_heif
pillow_heif.register_heif_opener()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def augment_image(img: Image.Image) -> list[Image.Image]:
    w, h = img.size

    # brightness
    brightness = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))

    # gaussian blur
    blurred = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

    # random crop then resize back to original
    ratio = random.uniform(0.7, 0.95)
    new_w, new_h = int(w * ratio), int(h * ratio)
    left = random.randint(0, w - new_w)
    top  = random.randint(0, h - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.LANCZOS)

    # color jitter
    jittered = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))
    jittered = ImageEnhance.Contrast(jittered).enhance(random.uniform(0.7, 1.3))

    return [brightness, blurred, cropped, jittered]


def augment_class(class_dir: Path) -> None:
    aug_dir = class_dir / "augmented"
    aug_dir.mkdir(exist_ok=True)

    sources = [p for sub in class_dir.iterdir() if sub.is_dir() and sub.name != "augmented"
               for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

    if not sources:
        print(f"  {class_dir.name}: no source images found")
        return

    count = 0
    for img_path in sources:
        try:
            img = Image.open(img_path)
            if img.mode == 'P' and 'transparency' in img.info:
                img = img.convert('RGBA')
            img = img.convert('RGB')
        except Exception as e:
            print(f"  skipping {img_path.name}: {e}")
            continue

        for i, aug in enumerate(augment_image(img)):
            out_name = f"{img_path.stem}_aug{i}{img_path.suffix}"
            aug.save(aug_dir / out_name)
            count += 1

    print(f"  {class_dir.name}: saved {count} augmented images to '{aug_dir}'")


def augment(data_path: str) -> None:
    root = Path(data_path)
    train_dir = root / "train"
    if not train_dir.is_dir():
        print(f"Error: '{train_dir}' does not exist")
        sys.exit(1)

    classes = sorted(d for d in train_dir.iterdir() if d.is_dir())
    if not classes:
        print("No class folders found.")
        return

    print(f"Augmenting {len(classes)} class(es):\n")
    for class_dir in classes:
        augment_class(class_dir)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python augment.py <data_folder>")
        sys.exit(1)

    augment(sys.argv[1])
