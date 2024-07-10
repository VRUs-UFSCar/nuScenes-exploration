
from tqdm import tqdm


def train(train_data_loader, model, optimizer, device):
    '''
    Treina um modelo qualquer de detções de objetos 2D da biblioteca torchvision.

    Args:
    - train_data_loader: DataLoader com os dados de treino.
    - model: modelo que será treinado.
    - optimizer: otimizador que será usado para treinar o modelo.
    - device: dispositivo onde o modelo será treinado.

    Returns:
    - train_loss_list: lista com os valores de loss de cada batch. O loss guardado é o loss total obtido em cada batch, e não a média dos losses.
    '''
    train_loss_list = []

    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))  # Usado para mostrar o progresso do treino
    
    for _, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        # Adapta os dados para o dispositivo que está sendo usado
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Faz um forward pass, calculando a perda média obtida no batch.
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())  # Os modelos retornam várias losses diferentes ao fazer um passo para frente. Para deixar esse código genérico, todas as losses calculadas são somadas.
        loss_value = losses.item()
        train_loss_list.append(loss_value * len(images))  # A loss é multiplicada pelo tamanho do batch para que seja possível calcular a média dos losses por época depois

        # Faz um passo para trás, atualizando os pesos do modelo
        losses.backward()
        optimizer.step()
    
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return train_loss_list