#@ Configuration for YOLO image detection:

import os
import torchvision.transforms as T


DATA_PATH='data'
CLASSES_PATH=os.path.join(DATA_PATH, 'classes.json')

BATCH_SIZE = 64
EPOCHS = 135
WARMUP_EPOCHS = 0
LEARNING_RATE = 1E-4

EPSILON=1E-4
IMAGE_SIZE=(448, 448)

# the predictions are encoded as S x S x (B * 5 + C):
S=7 #dividing image into S x S grid
B=2 # Number of bounding box to predict
C=20 # Number of classes in dataset 