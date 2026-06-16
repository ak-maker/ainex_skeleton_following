from pathlib import Path


def count_files(data_path: str = "data") -> None:
    root = Path(data_path)
    if not root.is_dir():
        print(f"Error: '{data_path}' is not a valid directory")
        return

    for folder in sorted(root.rglob("*")):
        if folder.is_dir():
            count = sum(1 for f in folder.iterdir() if f.is_file())
            print(f"{folder}: {count}")


if __name__ == "__main__":
    count_files()
