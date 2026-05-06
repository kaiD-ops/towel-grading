"""Model architecture for towel quality grading using MobileNetV2 transfer learning."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import (
    IMG_HEIGHT, IMG_WIDTH, CHANNELS, DROPOUT_RATE, DENSE_UNITS
)


def create_model(num_classes=3, weights='imagenet'):
    """
    Create a CNN model using MobileNetV2 transfer learning.
    
    Args:
        num_classes (int): Number of output classes (3 for Grade A, B, C)
        weights (str): Pre-trained weights to load ('imagenet' or None)
    
    Returns:
        keras.Model: Compiled model ready for training
    """
    
    # Load pre-trained MobileNetV2 model without the top classification layer
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        include_top=False,
        weights=weights
    )
    
    # Freeze the base model layers initially to preserve learned features
    base_model.trainable = False
    
    # Build the custom classification head
    model = keras.Sequential([
        # Input layer
        keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        
        # Base MobileNetV2 model
        base_model,
        
        # Global Average Pooling to reduce spatial dimensions
        layers.GlobalAveragePooling2D(),
        
        # Dropout to prevent overfitting
        layers.Dropout(DROPOUT_RATE),
        
        # Dense layer with ReLU activation
        layers.Dense(DENSE_UNITS, activation='relu'),
        
        # Output layer with Softmax for multi-class classification
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model, base_model


def compile_model(model, learning_rate=1e-4):
    """
    Compile the model with optimizer and loss function.
    
    Args:
        model (keras.Model): The model to compile
        learning_rate (float): Learning rate for the optimizer
    
    Returns:
        keras.Model: Compiled model
    """
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(),
            keras.metrics.Recall(),
            keras.metrics.AUC()
        ]
    )
    
    return model


def get_model_summary(model):
    """
    Get a summary of the model architecture.
    
    Args:
        model (keras.Model): The model to summarize
    
    Returns:
        str: Model summary
    """
    model.summary()
    return model.count_params()


if __name__ == '__main__':
    # Test model creation
    model, base = create_model()
    compile_model(model)
    print(f"\nTotal trainable parameters: {model.count_params():,}")
    model.summary()
