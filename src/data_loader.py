from pathlib import Path

from config import DATA_DIR


def list_dataset():
    print("=" * 50)
    print("Dataset Directory")
    print("=" * 50)

    if not DATA_DIR.exists():
        print("Dataset folder does not exist yet.")
        return

    for item in DATA_DIR.iterdir():
        print(item.name)


if __name__ == "__main__":
    list_dataset()