#!/usr/bin/env python3
import os
import torch
import torch.nn as nn

from data import get_dataloaders
from models.basenet import BaseNet
from models.resnet import ResNetCustom
from plot import plot_metrics
from train import train_one_epoch, validate
from utils import set_seed


def run_experiment(args):
    """Runs a convolutional neural net (CNN) experiment.
    
    Uses Fashion-MNIST dataset of clothing items and shoes.  Models are
    BaseNet and ResNet, attention-based convolutional nets.

    Args:
        args: command line arguments (see parse_args() for options)
    
    Returns:
        history: training and validation history

    """
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader, test_loader = get_dataloaders(root=args.data_root,
                                               subset_train_samples=args.subset_train_samples,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

    # model selection
    if args.model == 'base':
        convs = 2 if args.variant == 10 else 4
        model = BaseNet(convs_per_module=convs, num_classes=10)
        model_name = f'BaseNet-{args.variant}'
    elif args.model == 'resnet':
        # ResNet-18 analog: 2 blocks per module -> blocks_per_module=2
        model = ResNetCustom(blocks_per_module=2, num_classes=10)
        model_name = 'ResNet-18'
    else:
        raise ValueError("unknown model")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # optimizer selection
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        raise ValueError("unknown optimizer")

    # Scheduler: divide LR by 10 every 30 epochs
    # (if epochs <= 30, scheduler will only step after epoch 30)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} - lr: {optimizer.param_groups[0]['lr']:.4f}")
        train_loss, train_acc = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, device, test_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f" Train loss: {train_loss:.4f}  acc: {train_acc:.2f}%")
        print(f" Val   loss: {val_loss:.4f}  acc: {val_acc:.2f}%")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(save_dir, f'{model_name}_{args.optimizer}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(" Saved best model:", save_path)

    # final save and plots
    final_path = os.path.join(save_dir, f'{model_name}_{args.optimizer}_final.pth')
    torch.save({'model_state': model.state_dict()}, final_path)
    print("Saved final model:", final_path)

    plot_metrics(history, save_dir, f"{model_name}_{args.optimizer}")

    # also return history for programmatic use
    return history
