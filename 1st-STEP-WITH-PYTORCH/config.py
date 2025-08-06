import torch

K_FOLDS = 5
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

DATA_DIR = '../data/data-raw'
MODEL_SAVE_DIR = './models/domino_classifier_final.pth'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")