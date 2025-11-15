#!/usr/bin/env python3
"""
train_cnn.py

Scaffolded PyTorch implementation for CSC 737 Assignment 3:
- Data loading + augmentation (training only)
- BaseNet (configurable: 2 or 4 convs per module -> BaseNet-10 or BaseNet-16)
- ResNet (manual residual connections)
- Training + validation loops, scheduler: start lr=0.1, divide by 10 every 30 epochs
- Compare optimizers (SGD+momentum vs Adam)
- Save models and produce accuracy/loss plots

Usage examples:
    python train_cnn.py --model base --variant 10 --optimizer sgd
    python train_cnn.py --model resnet --optimizer adam
"""

from tqdm import tqdm
import torch

def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc='train', leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)
    return running_loss / total, 100. * correct / total

def validate(model, device, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += images.size(0)
    return running_loss / total, 100. * correct / total
