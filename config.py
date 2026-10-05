import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "best_car_damage_cnn.pth")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")

IMAGE_SIZE = (160, 160)
INPUT_CHANNELS = 3

DEFAULT_CAR_CLASSES = [
    "Sedan",
    "SUV",
    "Hatchback",
    "Pickup Truck",
    "Coupe",
    "Van / Minivan"
]

DEFAULT_DAMAGE_CLASSES = [
    "No Damage",
    "Scratch",
    "Dent",
    "Bumper Damage",
    "Glass Crack / Shatter",
    "Headlight / Taillight Damage",
    "Major Smash / Collision"
]

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
WEIGHT_DECAY = 1e-4

LOSS_WEIGHT_CAR = 1.0
LOSS_WEIGHT_DAMAGE = 1.5
LOSS_WEIGHT_PERCENT = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
