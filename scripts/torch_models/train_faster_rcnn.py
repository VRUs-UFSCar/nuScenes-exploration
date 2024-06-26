
# -------- CONFIGURAÇÕES ---------------

import torch
import os
import nuscenes

IN_DIR = '../../data'

nusc = nuscenes.NuScenes(version='v1.0-mini', dataroot=IN_DIR, verbose=True)

BATCH_SIZE = 10 # increase / decrease according to GPU memory
RESIZE_PERCENT = 1 # resize the image for training and transforms
NUM_EPOCHS = 10 # number of epochs to train for


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_NAME = 'mini_train'
VALIDATION_NAME = 'mini_val'
# classes: 0 index is reserved for background
CLASSES = [
    'background', *[category['name'] for category in nusc.category]
]
NUM_CLASSES = len(CLASSES)


OUT_DIR = '../../outputs/faster-rcnn/mini'  # location to save model and plots

os.makedirs(OUT_DIR, exist_ok=True)

SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs





# -------- FASTERRCNN MODEL ---------------


import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights  # Essa rede foi escolhida por ser mais leve e mais rápida, não é igual ao faster-rcnn original
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):
    # Carrega o modelo com os pesos pré-treinados
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
    )
    
    # Altera a rede para que ela tenha o número de classes correto
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model




# ------------- TRAINING ---------------

import torch
from utils.NuScenesDataset import NuScenesDataset
from torch.utils.data import DataLoader
from utils.epoch_train import train
from utils.epoch_validate import validate
from utils.plot_functions import save_val_train_mean_loss_joined_plot
import time
from tqdm import tqdm
import json


# initialize the model and move to the computation device
model = create_model(num_classes=NUM_CLASSES)
model = model.to(DEVICE)
# define the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
# train and validation loss lists to store loss values of all...
# ... iterations till ena and plot graphs for all iterations
train_loss_mean_list = []
val_loss_mean_list = []
# name to save the trained model with
MODEL_NAME = 'model'

train_dataset = NuScenesDataset(nusc, IN_DIR, TRAIN_NAME, CLASSES, ['1', '2', '3', '4'])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

valid_dataset = NuScenesDataset(nusc, IN_DIR, VALIDATION_NAME, CLASSES, ['1', '2', '3', '4'])
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# start the training epochs
for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
    # start timer and carry out training and validation
    start = time.time()
    epoch_train_loss_list = train(train_loader, model, optimizer, DEVICE)
    epoch_val_loss_list = validate(valid_loader, model, DEVICE)

    train_loss_mean_list.append(sum(epoch_train_loss_list) / len(train_dataset))
    val_loss_mean_list.append(sum(epoch_val_loss_list) / len(valid_dataset))

    print(f"Epoch #{epoch} train loss average: {train_loss_mean_list[-1]:.3f}")   
    print(f"Epoch #{epoch} validation loss average: {val_loss_mean_list[-1]:.3f}")

    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
    
    if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
        torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
    
    if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
        save_val_train_mean_loss_joined_plot(train_loss_mean_list, val_loss_mean_list, OUT_DIR)
        torch.save(model.state_dict(), f"{OUT_DIR}/{MODEL_NAME}.pth")




# -------------- GENERATING DETECTIONS ---------------

model.eval()

valid_dataset = NuScenesDataset(nusc, IN_DIR, VALIDATION_NAME, CLASSES, ['1', '2', '3', '4'], return_tokens=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

print('Generating detections...')
prog_bar = tqdm(valid_loader, total=len(valid_loader))
json_detections = {}
score_threshold = 0.5
    
for i, data in enumerate(prog_bar):
    images, targets, tokens = data
    
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    
    with torch.no_grad():
        outputs = model(images)

    for output, token in zip(outputs, tokens):
        boxes = output['boxes'][output['scores'] > score_threshold]
        boxes = boxes.cpu().numpy().tolist()

        labels = output['labels'][output['scores'] > score_threshold]
        labels = [CLASSES[label] for label in labels.cpu().numpy().tolist()]
        
        json_detections[token] = {
            'boxes': boxes,
            'labels': labels
        }

json.dump(json_detections, open(f"{OUT_DIR}/detections.json", 'w'), indent=4)