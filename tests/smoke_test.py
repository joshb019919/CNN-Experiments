#!/usr/bin/env python3
"""Smoke test: run a single validation pass on Fashion-MNIST test set.

This script loads the test dataset, creates a model (BaseNet or ResNet),
and runs the `validate` function to compute loss and accuracy.
"""
import argparse
import torch
import torch.nn as nn

from data import get_dataloaders
from models.basenet import BaseNet
from models.resnet import ResNetCustom
from train import validate
from utils import set_seed


def main():
    parser = argparse.ArgumentParser(description='Smoke test for models on test set')
    parser.add_argument('--model', choices=['base', 'resnet'], default='base')
    parser.add_argument('--variant', type=int, choices=[10, 16], default=10,
                        help='For BaseNet: 10 -> 2 convs/module, 16 -> 4 convs/module')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # We only need the test loader for this smoke test. Keep training subset default (full).
    _, _, test_loader = get_dataloaders(root=args.data_root, subset_train_samples=None,
                                        batch_size=args.batch_size, num_workers=0)

    if args.model == 'base':
        convs = 2 if args.variant == 10 else 4
        model = BaseNet(convs_per_module=convs, num_classes=10)
        print(f'Instantiated BaseNet variant {args.variant}')
    else:
        model = ResNetCustom(blocks_per_module=2, num_classes=10)
        print('Instantiated ResNet-18 analog')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    val_loss, val_acc = validate(model, device, test_loader, criterion)
    print(f'Validation loss: {val_loss:.4f}  acc: {val_acc:.2f}%')


if __name__ == '__main__':
    main()
