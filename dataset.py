"""
Builds PyTorch DataLoaders for train/val/test splits, with augmentation
applied only to the training set.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])

    # No augmentation for val/test -- only resize + normalize, so evaluation
    # reflects real-world performance rather than augmented performance.
    eval_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])

    return train_transform, eval_transform


def get_dataloaders():
    train_transform, eval_transform = get_transforms()

    train_dataset = datasets.ImageFolder(
        root=f"{config.SPLIT_DATA_DIR}/train", transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=f"{config.SPLIT_DATA_DIR}/val", transform=eval_transform
    )
    test_dataset = datasets.ImageFolder(
        root=f"{config.SPLIT_DATA_DIR}/test", transform=eval_transform
    )

    # Sanity check: ImageFolder infers class order alphabetically from
    # folder names -- this should match config.CLASS_NAMES exactly.
    assert train_dataset.classes == config.CLASS_NAMES, (
        f"Class order mismatch!\n"
        f"ImageFolder found: {train_dataset.classes}\n"
        f"config.CLASS_NAMES: {config.CLASS_NAMES}\n"
        "Update config.CLASS_NAMES to match (must be alphabetical)."
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick smoke test
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, images: {len(train_loader.dataset)}")
    print(f"Val batches:   {len(val_loader)}, images: {len(val_loader.dataset)}")
    print(f"Test batches:  {len(test_loader)}, images: {len(test_loader.dataset)}")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, labels shape: {labels.shape}")
