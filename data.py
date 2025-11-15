#!/usr/bin/env python3
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
from torch.utils.data import DataLoader, Subset


def get_dataloaders(root="./data", subset_train_samples=10000, batch_size=50, num_workers=4):
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
        (train_loader, test_loader):
        The data loaders for training and testing sets.
    """

    # Transforms per assignment:
    # - training only: horizontal flip, rotation +/-20 deg, scaling up to 0.4 (random resized crop),
    #   resize to 36, random crop 5 pixels (-> final 28)
    #   We'll approximate by RandomResizedCrop(28, scale=(0.6,1.4))
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(size=28, scale=(0.6, 1.4)),  # scale ~ 0.6 -> up to 0.4 scaling
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

    # Subset training to e.g., 10k samples per assignment
    if subset_train_samples is not None and subset_train_samples < len(train_full):
        indices = list(range(len(train_full)))
        random.shuffle(indices)
        sub_idx = indices[:subset_train_samples]
        train_subset = Subset(train_full, sub_idx)
    else:
        train_subset = train_full

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader