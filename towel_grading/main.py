"""
Main script to run the entire towel grading pipeline.
"""

import os
import sys
from train import train_model
from evaluate import evaluate_model, ModelEvaluator
from config import MODEL_SAVE_PATH


def main():
    """Main pipeline runner."""
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  AUTOMATED TOWEL QUALITY GRADING SYSTEM  ".center(58) + "║")
    print("║" + "  Using Convolutional Neural Networks  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    print("Available options:")
    print("  1. Train model (requires dataset)")
    print("  2. Evaluate model (requires trained model)")
    print("  3. Predict on single image (requires trained model)")
    print("  4. Exit")
    print()
    
    while True:
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            model, history = train_model()
            
        elif choice == '2':
            evaluate_model()
            
        elif choice == '3':
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
                
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid option. Please select 1-4.")
        
        print()


if __name__ == '__main__':
    main()
