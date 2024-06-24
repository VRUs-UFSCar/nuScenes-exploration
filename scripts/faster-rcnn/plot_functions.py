
import os
import matplotlib.pyplot as plt
from configs import OUT_DIR


def _save_loss_plot(loss_list, epoch, label, type):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label=label)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title(f'{label} plot for epoch {epoch}')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, f'{type}_loss_plot_epoch_{epoch}.png'))
    plt.close()


def save_val_train_loss_plot(train_loss_list, val_loss_list, epoch):
    _save_loss_plot(train_loss_list, epoch, 'Train loss', 'train')
    _save_loss_plot(val_loss_list, epoch, 'Validation loss', 'val')

def _save_mean_loss_plot(loss_list, label, type):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{label} mean plot per epoch')
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, f'{type}_loss_plot_mean.png'))
    plt.close()

def save_mean_val_train_loss_plot(train_loss_list, val_loss_list):
    _save_mean_loss_plot(train_loss_list, 'Train loss', 'train')
    _save_mean_loss_plot(val_loss_list, 'Validation loss', 'val')