import sys
import hashlib
from pathlib import Path
from PIL import Image
import imagehash

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
NEAR_DUP_THRESHOLD = 8  # hamming distance; 0 = identical, <10 = very similar


def hash_exact(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def deduplicate_folder(folder: Path, dry_run: bool) -> int:
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        return 0

    # --- pass 1: exact duplicates via MD5 ---
    seen_md5: dict[str, Path] = {}
    to_delete: list[tuple[Path, Path, str]] = []

    for path in images:
        h = hash_exact(path)
        if h in seen_md5:
            to_delete.append((path, seen_md5[h], "exact"))
        else:
            seen_md5[h] = path

    # --- pass 2: near-duplicates via perceptual hash ---
    remaining = [p for p in images if p not in {d for d, _, _ in to_delete}]
    phashes: list[tuple[Path, imagehash.ImageHash]] = []

    for path in remaining:
        try:
            ph = imagehash.phash(Image.open(path))
        except Exception:
            continue
        for other_path, other_ph in phashes:
            if ph - other_ph <= NEAR_DUP_THRESHOLD:
                to_delete.append((path, other_path, f"near (dist={ph - other_ph})"))
                break
        else:
            phashes.append((path, ph))

    if to_delete:
        print(f"\n{'[DRY RUN] ' if dry_run else ''}{folder} — removing {len(to_delete)} duplicate(s):")
        for dup, original, reason in to_delete:
            print(f"  {reason}: {dup.name}  (kept: {original.name})")
            if not dry_run:
                dup.unlink()

    return len(to_delete)


def deduplicate(folder_path: str, dry_run: bool = False) -> None:
    root = Path(folder_path)
    if not root.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory")
        sys.exit(1)

    # Collect all dirs that contain images directly (leaf-level class folders)
    dirs = sorted({p.parent for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS})

    total = 0
    for d in dirs:
        total += deduplicate_folder(d, dry_run)

    if total == 0:
        print("No duplicates found.")
    else:
        print(f"\n{'Would remove' if dry_run else 'Removed'} {total} file(s) total.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dedup.py <folder_path> [--dry-run]")
        sys.exit(1)

    dry = "--dry-run" in sys.argv
    deduplicate(sys.argv[1], dry_run=dry)
