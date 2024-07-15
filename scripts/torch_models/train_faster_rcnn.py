
from utils.clear_memory import clear_memory
clear_memory('/home/gregorio/.cache/torch')




# -------- CONFIGURAÇÕES ---------------

import torch
import os
import nuscenes

IN_DIR = '../../data'

nusc = nuscenes.NuScenes(version='v1.0-trainval', dataroot=IN_DIR, verbose=True)  # v1.0-mini | v1.0-trainval

BATCH_SIZE = 32 # increase / decrease according to GPU memory
RESIZE_PERCENT = 1 # resize the image for training and transforms
NUM_EPOCHS = 100 # number of epochs to train for


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_NAME = 'train'  # mini_train | train
VALIDATION_NAME = 'val'  # mini_val | val
# classes: 0 index is reserved for background
CLASSES_TO_IGNORE = [
    'animal',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'static_object.bicycle_rack',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
]

CLASSES_GROUP_MAPPING = {
    'background': 'background',
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bycicle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
}

CLASSES = ['background', *[category['name'] for category in nusc.category]]
CLASSES = [c for c in CLASSES if c not in CLASSES_TO_IGNORE]
CLASSES = [CLASSES_GROUP_MAPPING[c] for c in CLASSES]
CLASSES = list(set(CLASSES))
NUM_CLASSES = len(CLASSES)


OUT_DIR = '../../outputs/filtered/full'  # location to save model and plots

os.makedirs(OUT_DIR, exist_ok=True)

SAVE_MODEL_EPOCH = 20 # save model after these many epochs





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


# Inicializa o modelo
model = create_model(num_classes=NUM_CLASSES)
model = model.to(DEVICE)
# Inicializa o otimizador
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)
train_loss_mean_list = []
val_loss_mean_list = []
# Nome que será usado para salvar os pesos aprendidos pelo modelo
MODEL_NAME = 'model'

train_dataset = NuScenesDataset(nusc, IN_DIR, TRAIN_NAME, CLASSES, ['1', '2', '3', '4'], classes_map=CLASSES_GROUP_MAPPING)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

valid_dataset = NuScenesDataset(nusc, IN_DIR, VALIDATION_NAME, CLASSES, ['1', '2', '3', '4'], classes_map=CLASSES_GROUP_MAPPING)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Começa o treinamento
for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
    # start timer and carry out training and validation
    start = time.time()
    epoch_train_loss_list = train(train_loader, model, optimizer, DEVICE)  # Treina o modelo
    epoch_val_loss_list = validate(valid_loader, model, DEVICE)

    train_loss_mean_list.append(sum(epoch_train_loss_list) / len(train_dataset))
    val_loss_mean_list.append(sum(epoch_val_loss_list) / len(valid_dataset))  # Avalia o modelo, calculando a loss de validação

    print(f"Epoch #{epoch} train loss average: {train_loss_mean_list[-1]:.3f}")   
    print(f"Epoch #{epoch} validation loss average: {val_loss_mean_list[-1]:.3f}")

    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
    
    if (epoch+1) % SAVE_MODEL_EPOCH == 0: # Salva o modelo a cada SAVE_MODEL_EPOCH épocas
        torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
    
    if (epoch+1) == NUM_EPOCHS: # Salva o modelo e o gráfico de loss ao final do treinamento
        save_val_train_mean_loss_joined_plot(train_loss_mean_list, val_loss_mean_list, 'Fater R-CNN', OUT_DIR)
        torch.save(model.state_dict(), f"{OUT_DIR}/{MODEL_NAME}.pth")




# -------------- GENERATING DETECTIONS ---------------

model.eval()

# Gera um dataloader para a validação
valid_dataset = NuScenesDataset(nusc, IN_DIR, VALIDATION_NAME, CLASSES, ['1', '2', '3', '4'], classes_map=CLASSES_GROUP_MAPPING, return_tokens=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))


print('Generating detections...')
prog_bar = tqdm(valid_loader, total=len(valid_loader))
json_detections = {}
score_threshold = 0.5  # Threshold de score para considerar uma detecção válida
    
for i, data in enumerate(prog_bar):
    images, targets, tokens = data
    
    # Adapta os dados para o dispositivo que está sendo usado
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    
    with torch.no_grad():
        outputs = model(images)  # Faz um forward pass, obtendo as detecções

    for output, token in zip(outputs, tokens):
        # Filtra as detecções com score menor que o threshold e adapta as bounding boxes para serem salvas
        boxes = output['boxes'][output['scores'] > score_threshold]
        boxes = boxes.cpu().numpy().tolist()

        # Traduz os labels com score menor que o threshold e coleta os nomes das classes previstas
        labels = output['labels'][output['scores'] > score_threshold]
        labels = [CLASSES[label] for label in labels.cpu().numpy().tolist()]
        
        # Salva a detecção de uma imagem
        json_detections[token] = {
            'boxes': boxes,
            'labels': labels
        }

# Salva as detecções em um arquivo json
json.dump(json_detections, open(f"{OUT_DIR}/detections.json", 'w'), indent=4)



clear_memory('/home/gregorio/.cache/torch')