"""
PyTorch DataLoader utilities.
"""

from torch.utils.data import DataLoader

from dataset import load_dataset


from config import BATCH_SIZE, NUM_WORKERS


def create_dataloader(
    split: str,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for a dataset split.
    """

    dataset = load_dataset(split)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=NUM_WORKERS,
    )


if __name__ == "__main__":

    train_loader = create_dataloader("train")

    print("=" * 50)
    print("DataLoader Created")
    print("=" * 50)

    print(f"Number of batches : {len(train_loader)}")

    images, labels = next(iter(train_loader))

    print(f"Batch shape       : {images.shape}")
    print(f"Labels shape      : {labels.shape}")
