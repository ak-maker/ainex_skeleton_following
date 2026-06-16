import sys
from pathlib import Path

KEEP_EXTS = {".webp", ".jpg", ".jpeg", ".png"}


def remove_non_images(folder_path: str) -> None:
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory")
        sys.exit(1)

    files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() not in KEEP_EXTS]
    if not files:
        print("No files to remove.")
        return

    for f in files:
        f.unlink()
        print(f"Removed: {f.name}")

    print(f"\nDeleted {len(files)} file(s).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_non_imgs.py <folder_path>")
        sys.exit(1)

    remove_non_images(sys.argv[1])
