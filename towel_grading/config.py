"""Configuration settings for the towel grading model."""

import os

# Dataset configuration
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data')
GRADES = ['Grade_A', 'Grade_B', 'Grade_C']
GRADE_LABELS = {'Grade_A': 0, 'Grade_B': 1, 'Grade_C': 2}
LABEL_TO_GRADE = {0: 'Grade A', 1: 'Grade B', 2: 'Grade C'}

# Image preprocessing configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3
NORMALIZE_RANGE = (0, 1)

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.3
DENSE_UNITS = 128

# Data split configuration
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Augmentation configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 15,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': [0.8, 1.2],
    'contrast_range': [0.8, 1.2],
}

# Model paths
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'models')
RESULTS_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'results')

# Create directories if they don't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
