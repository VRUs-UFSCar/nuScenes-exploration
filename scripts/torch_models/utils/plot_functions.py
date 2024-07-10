
import os
import matplotlib.pyplot as plt


def save_val_train_mean_loss_joined_plot(train_loss_list, val_loss_list, model_name, out_dir):
    '''
    Salva um gráfico com as losses de treino e validação por época. O eixo x é a época e o eixo y é a loss média.

    Args:
    - train_loss_list: lista com as losses médios de treino por época.
    - val_loss_list: lista com as losses médios de validação por época.
    - model_name: nome do modelo que está sendo treinado. Esse nome será usado no título do gráfico.
    - out_dir: diretório onde o gráfico será salvo.
    '''
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label='Train loss')
    plt.plot(val_loss_list, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train and Validation loss plot per epoch for {model_name} model')
    plt.legend()
    plt.savefig(os.path.join(out_dir, f'loss_plot.png'))
    plt.close()