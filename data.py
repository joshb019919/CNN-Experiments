#!/usr/bin/env python3
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
from torch.utils.data import DataLoader, Subset


def get_dataloaders(root="./data", subset_train_samples=40000, batch_size=50, num_workers=4):
    """Builds train and test dataloaders for Fashion-MNIST.

    This uses torchvision datasets as fallback; replace with HF loader 
    if desired.

    Transformations:
        The data is transformed randomly in many ways to make the model
        more robust by not letting it overfit on exact images.  The 
        kinds of transformations are:

        - Horizontal flip
        - Rotation (up to 20 degrees)
        - Scaling (up to 40%)
        - Resizing (up to 36x36)
        - Cropping (up to 40% back to 28 pixels)

    Args:
        root: local source of data and were to download
        subset_train_samples: limits training dataset size
        batch_size: number of samples to run at once
        num_workers: parallel stuff

    Returns:
        (train_loader, val_loader, test_loader):
        The data loaders for training, validation, and testing sets.
    """

    # Transforms per assignment (approximation):
    # - training only: horizontal flip, rotation +/-20 deg, scaling up to 0.4
    # - resizing to 36 followed by a small random crop.
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        # Approximate: scale range chosen so images may be up to 40% larger/smaller
        transforms.RandomResizedCrop(size=28, scale=(0.6, 1.4)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download / load datasets
    train_full = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform_train)
    testset = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform_test)

    # Build index list for training data (optionally subset), then split 80/20 -> train/val
    indices = list(range(len(train_full)))
    random.shuffle(indices)

    if subset_train_samples is not None and subset_train_samples < len(train_full):
        indices = indices[:subset_train_samples]

    split_point = int(len(indices) * 0.8)
    train_idx = indices[:split_point]
    val_idx = indices[split_point:]

    train_subset = Subset(train_full, train_idx)
    val_subset = Subset(train_full, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader