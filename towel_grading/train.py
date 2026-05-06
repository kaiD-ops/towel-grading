"""Training script for the towel quality grading model."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from model import create_model, compile_model
from data_utils import get_data
from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH, RESULTS_SAVE_PATH
)


class TrainingCallback(keras.callbacks.Callback):
    """Custom callback to monitor training progress."""
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch + 1}/{self.params['epochs']}")
            print(f"  Loss: {logs.get('loss', 0):.4f} | "
                  f"Accuracy: {logs.get('accuracy', 0):.4f}")
            print(f"  Val Loss: {logs.get('val_loss', 0):.4f} | "
                  f"Val Accuracy: {logs.get('val_accuracy', 0):.4f}")


def train_model():
    """
    Train the towel grading CNN model.
    
    Steps:
    1. Load and prepare data
    2. Create and compile model
    3. Train with early stopping
    4. Save model and training history
    """
    
    print("=" * 60)
    print("TOWEL QUALITY GRADING CNN - TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading and preparing data...")
    try:
        train_dataset, val_dataset, (X_test, y_test), class_weights = get_data(
            batch_size=BATCH_SIZE
        )
    except ValueError as e:
        print(f"Error: {e}")
        print("Please add training images to the data directories:")
        print("  - data/Grade_A/")
        print("  - data/Grade_B/")
        print("  - data/Grade_C/")
        print("\nMinimum 300 images per class recommended (900 total).")
        return None
    
    # Create model
    print("\n[2/4] Creating model architecture...")
    model, base_model = create_model(num_classes=3, weights='imagenet')
    model = compile_model(model, learning_rate=LEARNING_RATE)
    
    print(f"Model has {model.count_params():,} total parameters")
    print(f"Base model (frozen): {base_model.count_params():,} parameters")
    
    # Setup callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    print("\n[3/4] Training model...")
    print(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping, reduce_lr, model_checkpoint, TrainingCallback()],
        class_weight=class_weights,
        verbose=0
    )
    
    # Save final model
    print("\n[4/4] Saving model and history...")
    model.save(os.path.join(MODEL_SAVE_PATH, 'towel_grading_model.h5'))
    
    # Save training history
    history_dict = {
        'loss': history.history['loss'],
        'accuracy': history.history['accuracy'],
        'val_loss': history.history['val_loss'],
        'val_accuracy': history.history['val_accuracy'],
    }
    
    with open(os.path.join(RESULTS_SAVE_PATH, 'training_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    # Plot training history
    plot_training_history(history, RESULTS_SAVE_PATH)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Models saved to: {MODEL_SAVE_PATH}")
    print(f"Results saved to: {RESULTS_SAVE_PATH}")
    
    return model, history


def plot_training_history(history, save_path):
    """
    Plot and save training/validation metrics.
    
    Args:
        history: Training history object
        save_path (str): Path to save plots
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy over Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss over Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to: {os.path.join(save_path, 'training_history.png')}")


if __name__ == '__main__':
    model, history = train_model()
