import torch
import os
import nuscenes

IN_DIR = '../../data'

nusc = nuscenes.NuScenes(version='v1.0-trainval', dataroot=IN_DIR, verbose=True)

BATCH_SIZE = 25 # increase / decrease according to GPU memory
RESIZE_PERCENT = 1 # resize the image for training and transforms
NUM_EPOCHS = 10 # number of epochs to train for


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_NAME = 'train'
VALIDATION_NAME = 'val'
# classes: 0 index is reserved for background
CLASSES = [
    'background', *[category['name'] for category in nusc.category]
]
NUM_CLASSES = len(CLASSES)


OUT_DIR = '../../outputs/faster-rcnn/full'  # location to save model and plots

os.makedirs(OUT_DIR, exist_ok=True)

SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs