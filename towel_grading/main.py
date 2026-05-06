"""
Main script to run the entire towel grading pipeline.
"""

import os
import sys
from train import train_model
from evaluate import evaluate_model, ModelEvaluator
from config import MODEL_SAVE_PATH


def main():
    
    
            
            image_path = input("Enter path to image: ").strip()
            if os.path.exists(image_path):
                evaluator = ModelEvaluator()
                grade, probs = evaluator.predict_single_image(image_path)
                print(f"\nPredicted Grade: {grade}")
                print("Confidence scores:")
                print(f"  Grade A: {probs[0]:.4f}")
                print(f"  Grade B: {probs[1]:.4f}")
                print(f"  Grade C: {probs[2]:.4f}")
            else:
                print("Image not found!")
                
        

if __name__ == '__main__':
    main()
