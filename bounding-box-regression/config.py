import os
import torch

class Config:

    BASE_PATH = "dataset"
    IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
    ANNOTS_PATH = os.path.sep.join([BASE_PATH, "airplanes.csv"])
    MODELS_OUTPUT_PATH = "models"
    MODEL_PATH = os.path.sep.join([MODELS_OUTPUT_PATH, "localizator.pth"])

    # Hyper parameters
    INIT_LR = 0.01
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    IMAGE_SIZE = 224
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()
