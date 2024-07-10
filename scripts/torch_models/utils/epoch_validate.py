
from tqdm import tqdm
import torch


def validate(valid_data_loader, model, device):
    '''
    Valida um modelo qualquer de detções de objetos 2D da biblioteca torchvision.
    Para a base de validação, um processo semelhante ao treinamento será feito, calculando as losses somadas de cada batch. Porém, diferentemente do treinamento, aqui os pesos não serão atualizados.

    Args:
    - valid_data_loader: DataLoader com os dados de validação.
    - model: modelo que será validado.
    - device: dispositivo onde o modelo será validado.

    Returns:
    - val_loss_list: lista com os valores de loss de cada batch. O loss guardado é o loss total obtido em cada batch, e não a média dos losses.
    '''
    val_loss_list = []
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))  # Usado para mostrar o progresso da validação
    
    for _, data in enumerate(prog_bar):
        images, targets = data
        
        # Adapta os dados para o dispositivo que está sendo usado
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():  # Desliga o cálculo do gradiente, pois não é necessário para a validação
            loss_dict = model(images, targets)  # Faz um forward pass, calculando a perda média obtida no batch.

        losses = sum(loss for loss in loss_dict.values())   # Os modelos retornam várias losses diferentes ao fazer um passo para frente. Para deixar esse código genérico, todas as losses calculadas são somadas.
        loss_value = losses.item()
        val_loss_list.append(loss_value * len(images))  # A loss é multiplicada pelo tamanho do batch para que seja possível calcular a média dos losses por época depois

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    return val_loss_list