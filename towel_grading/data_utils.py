"""Data loading, preprocessing, and augmentation utilities."""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
import cv2
from tqdm import tqdm
from config import (
    DATASET_PATH, GRADES, GRADE_LABELS, IMG_HEIGHT, IMG_WIDTH,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, AUGMENTATION_CONFIG
)


class DataLoader:
    """Load and preprocess towel images from disk."""
    
    def __init__(self, dataset_path=DATASET_PATH):
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
    
    def load_images(self):
        """
        Load all images from the dataset directory structure.
        
        Expected directory structure:
        data/
        ├── Grade_A/
        ├── Grade_B/
        └── Grade_C/
        
        Returns:
            tuple: (images array, labels array)
        """
        print("Loading images from dataset...")
        
        for grade_folder in GRADES:
            folder_path = os.path.join(self.dataset_path, grade_folder)
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist. Please add images.")
                continue
            
            label = GRADE_LABELS[grade_folder]
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            print(f"Loading {len(image_files)} images from {grade_folder}...")
            
            for image_file in tqdm(image_files, desc=grade_folder):
                image_path = os.path.join(folder_path, image_file)
                try:
                    # Load and preprocess image
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
                    
                    self.images.append(img)
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
        
        self.images = np.array(self.images, dtype='float32')
        self.labels = np.array(self.labels)
        
        # Normalize images to [0, 1] range
        self.images = self.images / 255.0
        
        print(f"Loaded {len(self.images)} images with shape {self.images[0].shape}")
        print(f"Label distribution: {np.bincount(self.labels)}")
        
        return self.images, self.labels
    
    def split_data(self):
        """
        Split data into training, validation, and test sets.
        
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if len(self.images) == 0:
            raise ValueError("No images loaded. Call load_images() first.")
        
        # First split: 70% train+val, 15% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.images, self.labels,
            test_size=TEST_SPLIT,
            random_state=42,
            stratify=self.labels
        )
        
        # Second split: 70/15 = 0.933, so val_size = 15/85 ≈ 0.176
        val_size = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class DataAugmentor:
    """Apply data augmentation to increase dataset diversity."""
    
    @staticmethod
    def get_augmentation_generator():
        """
        Create a data augmentation generator using Keras ImageDataGenerator.
        
        Returns:
            ImageDataGenerator: Configured augmentation generator
        """
        return keras.preprocessing.image.ImageDataGenerator(
            rotation_range=AUGMENTATION_CONFIG['rotation_range'],
            zoom_range=AUGMENTATION_CONFIG['zoom_range'],
            horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
            vertical_flip=AUGMENTATION_CONFIG['vertical_flip'],
            brightness_range=AUGMENTATION_CONFIG['brightness_range'],
            fill_mode='nearest',
            rescale=None  # Already normalized
        )


def get_data(batch_size=32, validation_split=0.15, test_split=0.15):
    """
    Load and prepare data for training.
    
    Args:
        batch_size (int): Batch size for training
        validation_split (float): Fraction for validation
        test_split (float): Fraction for test
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_weights)
    """
    # Load images
    loader = DataLoader()
    images, labels = loader.load_images()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data()
    
    # Convert labels to one-hot encoding
    y_train_encoded = keras.utils.to_categorical(y_train, num_classes=3)
    y_val_encoded = keras.utils.to_categorical(y_val, num_classes=3)
    y_test_encoded = keras.utils.to_categorical(y_test, num_classes=3)
    
    # Get augmentation generator
    augmentor = DataAugmentor()
    train_augmenter = augmentor.get_augmentation_generator()
    
    # Create tf.data.Dataset for efficient training
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_encoded))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val_encoded))
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_encoded))
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Calculate class weights for imbalanced data
    class_weights = {
        0: len(y_train) / (3 * np.sum(y_train == 0)),
        1: len(y_train) / (3 * np.sum(y_train == 1)),
        2: len(y_train) / (3 * np.sum(y_train == 2))
    }
    
    return train_dataset, val_dataset, (X_test, y_test_encoded), class_weights


import tensorflow as tf

if __name__ == '__main__':
    # Test data loading
    loader = DataLoader()
    images, labels = loader.load_images()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data()
