
from configs import DEVICE
from tqdm import tqdm
import torch


def validate(valid_data_loader, model):
    val_loss_list = []
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)

        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    return val_loss_list