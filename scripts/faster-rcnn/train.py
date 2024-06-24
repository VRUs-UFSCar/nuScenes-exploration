
from create_model import create_model
from configs import NUM_CLASSES, DEVICE, TRAIN_NAME, VALIDATION_NAME, CLASSES, BATCH_SIZE, NUM_EPOCHS, SAVE_MODEL_EPOCH, SAVE_PLOTS_EPOCH, OUT_DIR, nusc, IN_DIR
import torch
from NuScenesDataset import NuScenesDataset
from torch.utils.data import DataLoader
from epoch_train import train
from epoch_validate import validate
from plot_functions import save_val_train_loss_plot, save_mean_val_train_loss_plot
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
train_loss_list = []
val_loss_list = []

train_loss_mean_list = []
val_loss_mean_list = []
# name to save the trained model with
MODEL_NAME = 'model'

train_dataset = NuScenesDataset(nusc, IN_DIR, TRAIN_NAME, CLASSES, ['1', '2', '3', '4'])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

valid_dataset = NuScenesDataset(nusc, IN_DIR, VALIDATION_NAME, CLASSES, ['1', '2', '3', '4'])
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

# start the training epochs
for epoch in range(NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
    # start timer and carry out training and validation
    start = time.time()
    epoch_train_loss_list = train(train_loader, model, optimizer)
    epoch_val_loss_list = validate(valid_loader, model)

    train_loss_list.extend(epoch_train_loss_list)
    val_loss_list.extend(epoch_val_loss_list)

    train_loss_mean_list.append(sum(epoch_train_loss_list) / len(epoch_train_loss_list))
    val_loss_mean_list.append(sum(epoch_val_loss_list) / len(epoch_val_loss_list))

    print(f"Epoch #{epoch} train loss average: {train_loss_mean_list[-1]:.3f}")   
    print(f"Epoch #{epoch} validation loss average: {val_loss_mean_list[-1]:.3f}")

    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
    
    if (epoch+1) % SAVE_MODEL_EPOCH == 0: # save model after every n epochs
        torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch+1}.pth")
        print('SAVING MODEL COMPLETE...\n')
    
    if (epoch+1) % SAVE_PLOTS_EPOCH == 0: # save loss plots after n epochs
        save_val_train_loss_plot(train_loss_list, val_loss_list, epoch+1)
        print('SAVING PLOTS COMPLETE...\n')
    
    if (epoch+1) == NUM_EPOCHS: # save loss plots and model once at the end
        save_mean_val_train_loss_plot(train_loss_mean_list, val_loss_mean_list)
        torch.save(model.state_dict(), f"{OUT_DIR}/{MODEL_NAME}.pth")

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