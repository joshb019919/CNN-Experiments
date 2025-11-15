#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os


def plot_metrics(history, out_dir, name_prefix):
    """Plots training and validation accuracy and loss."""
    # history: dict { 'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[] }
    epochs = range(1, len(history['train_loss'])+1)
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'{name_prefix} Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f'{name_prefix}_accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{name_prefix} Loss')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, f'{name_prefix}_loss.png'))
    plt.close()