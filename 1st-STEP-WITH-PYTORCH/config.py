import torch

MODEL_NAME = "resnet50"
NUM_CLASSES = 28

K_FOLDS = 5
RANDOM_SEED = 42
NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

DATA_DIR = '../data/data-raw'
MODEL_SAVE_DIR = f'./models/domino_classifier_{MODEL_NAME}_final.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")