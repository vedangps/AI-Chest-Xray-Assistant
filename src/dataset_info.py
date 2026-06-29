from pathlib import Path

from config import DATA_DIR


def count_images(folder: Path) -> int:
    """
    Count JPEG image files inside a folder.
    """

    image_extensions = {".jpeg", ".jpg", ".png"}

    return sum(
        1
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in image_extensions
    )


def inspect_dataset() -> None:
    """
    Display information about the Chest X-ray dataset.
    """

    dataset_path = DATA_DIR / "chest_xray"

    if not dataset_path.exists():
        print("Dataset not found.")
        return

    print("=" * 60)
    print("Chest X-ray Dataset Summary")
    print("=" * 60)

    for split in ["train", "val", "test"]:

        split_path = dataset_path / split

        print(f"\n{split.upper()}")

        total = 0

        for category in ["NORMAL", "PNEUMONIA"]:

            category_path = split_path / category

            count = count_images(category_path)

            total += count

            print(f"  {category:<12}: {count}")

        print(f"  {'TOTAL':<12}: {total}")


if __name__ == "__main__":
    inspect_dataset()