
import os
import matplotlib.pyplot as plt


def _save_loss_plot(loss_list, epoch, label, type, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label=label)
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title(f'{label} plot for epoch {epoch}')
    plt.legend()
    plt.savefig(os.path.join(out_dir, f'{type}_loss_plot_epoch_{epoch}.png'))
    plt.close()


def save_val_train_loss_plot(train_loss_list, val_loss_list, epoch, out_dir):
    _save_loss_plot(train_loss_list, epoch, 'Train loss', 'train', out_dir)
    _save_loss_plot(val_loss_list, epoch, 'Validation loss', 'val', out_dir)

def _save_mean_loss_plot(loss_list, label, type, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{label} mean plot per epoch')
    plt.legend()
    plt.savefig(os.path.join(out_dir, f'{type}_loss_plot_mean.png'))
    plt.close()

def save_mean_val_train_loss_plot(train_loss_list, val_loss_list, out_dir):
    _save_mean_loss_plot(train_loss_list, 'Train loss', 'train', out_dir)
    _save_mean_loss_plot(val_loss_list, 'Validation loss', 'val', out_dir)

def save_val_train_mean_loss_joined_plot(train_loss_list, val_loss_list, out_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label='Train loss')
    plt.plot(val_loss_list, label='Validation loss')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title(f'Train and Validation loss plot per epoch')
    plt.legend()
    plt.savefig(os.path.join(out_dir, f'loss_plot.png'))
    plt.close()