"""Model evaluation, predictions, and performance metrics."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelBinarizer
import cv2
from config import (
    LABEL_TO_GRADE, MODEL_SAVE_PATH, RESULTS_SAVE_PATH, 
    IMG_HEIGHT, IMG_WIDTH
)


class ModelEvaluator:
    """Evaluate model performance and generate predictions."""
    
    def __init__(self, model_path=None):
        """
        Initialize evaluator with a trained model.
        
        Args:
            model_path (str): Path to saved model. If None, loads best model.
        """
        if model_path is None:
            model_path = os.path.join(MODEL_SAVE_PATH, 'best_model.h5')
        
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set and compute metrics.
        
        Args:
            X_test (np.array): Test images
            y_test (np.array): Test labels (one-hot encoded)
        
        Returns:
            dict: Dictionary containing all metrics
        """
        # Get predictions
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_probs': y_pred_probs
        }
        
        return metrics
    
    def print_classification_report(self, metrics):
        """
        Print detailed classification report.
        
        Args:
            metrics (dict): Metrics dictionary from evaluate()
        """
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        
        report = classification_report(
            y_true, y_pred,
            target_names=[LABEL_TO_GRADE[i] for i in range(3)],
            digits=4
        )
        print(report)
    
    def plot_confusion_matrix(self, metrics, save_path=RESULTS_SAVE_PATH):
        """
        Plot and save confusion matrix.
        
        Args:
            metrics (dict): Metrics dictionary from evaluate()
            save_path (str): Path to save plot
        """
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[LABEL_TO_GRADE[i] for i in range(3)],
            yticklabels=[LABEL_TO_GRADE[i] for i in range(3)],
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        save_file = os.path.join(save_path, 'confusion_matrix.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_file}")
        plt.close()
    
    def plot_metrics(self, metrics, save_path=RESULTS_SAVE_PATH):
        """
        Plot performance metrics.
        
        Args:
            metrics (dict): Metrics dictionary from evaluate()
            save_path (str): Path to save plot
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylim([0, 1.0])
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Metrics')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_file = os.path.join(save_path, 'performance_metrics.png')
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Performance metrics plot saved to: {save_file}")
        plt.close()
    
    def predict_single_image(self, image_path):
        """
        Make prediction on a single image.
        
        Args:
            image_path (str): Path to image file
        
        Returns:
            tuple: (predicted_grade, confidence_scores)
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Make prediction
        pred_probs = self.model.predict(img_batch, verbose=0)
        pred_class = np.argmax(pred_probs[0])
        
        return LABEL_TO_GRADE[pred_class], pred_probs[0]
    
    def predict_and_visualize(self, image_path, save_path=RESULTS_SAVE_PATH):
        """
        Make prediction on image and visualize result.
        
        Args:
            image_path (str): Path to image file
            save_path (str): Path to save visualization
        """
        # Load image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_HEIGHT, IMG_WIDTH))
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Make prediction
        img_batch = np.expand_dims(img_normalized, axis=0)
        pred_probs = self.model.predict(img_batch, verbose=0)
        pred_class = np.argmax(pred_probs[0])
        confidence = pred_probs[0][pred_class]
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Image
        axes[0].imshow(img_rgb)
        axes[0].set_title(f'Input Image')
        axes[0].axis('off')
        
        # Predictions
        grades = [LABEL_TO_GRADE[i] for i in range(3)]
        colors = ['#2ecc71' if i == pred_class else '#95a5a6' for i in range(3)]
        
        bars = axes[1].barh(grades, pred_probs[0], color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Confidence')
        axes[1].set_title(f'Prediction: {LABEL_TO_GRADE[pred_class]} ({confidence:.2%})')
        axes[1].set_xlim([0, 1])
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, pred_probs[0])):
            axes[1].text(prob, bar.get_y() + bar.get_height()/2.,
                        f'{prob:.4f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'prediction_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Prediction visualization saved to: {os.path.join(save_path, 'prediction_visualization.png')}")
        plt.close()
        
        return LABEL_TO_GRADE[pred_class], pred_probs[0]


def evaluate_model():
    """Main function to evaluate the trained model."""
    
    from data_utils import get_data
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Load data
    print("\nLoading test data...")
    try:
        _, _, (X_test, y_test), _ = get_data(batch_size=32)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Load and evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(X_test, y_test)
    
    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    evaluator.print_classification_report(metrics)
    evaluator.plot_confusion_matrix(metrics)
    evaluator.plot_metrics(metrics)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {RESULTS_SAVE_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    evaluate_model()
