
from tqdm import tqdm


def train(train_data_loader, model, optimizer, device):
    train_loss_list = []

    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value * len(images))
        losses.backward()
        optimizer.step()
    
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return train_loss_list