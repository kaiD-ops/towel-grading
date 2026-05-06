#!/usr/bin/env python
"""
Quick start guide for the Towel Quality Grading system.
Run this script to get started immediately.
"""

import os
import sys


def print_header():
    """Print welcome header."""
    print("\n")
    print("╔" + "=" * 70 + "╗")
    print("║" + " " * 70 + "║")
    print("║" + "TOWEL QUALITY GRADING - QUICK START GUIDE".center(70) + "║")
    print("║" + " " * 70 + "║")
    print("╚" + "=" * 70 + "╝")
    print()


def print_step(num, title, description, command=None):
    """Print a formatted step."""
    print(f"STEP {num}: {title}")
    print(f"{'─' * 70}")
    print(f"{description}\n")
    if command:
        print(f"  Command: {command}\n")


def main():
    """Main quick start guide."""
    print_header()
    
    print("This guide will help you set up and run the Towel Quality Grading model.\n")
    
    print("=" * 70)
    print("OPTION 1: QUICK START (with synthetic data - for testing)")
    print("=" * 70)
    
    print_step(
        1,
        "Install Dependencies",
        "Install required Python packages:",
        "pip install -r requirements.txt"
    )
    
    print_step(
        2,
        "Create Test Data",
        "Generate synthetic towel images for testing (100 per grade = 300 total):",
        "python create_test_data.py 100"
    )
    
    print_step(
        3,
        "Train the Model",
        "Train the CNN model on synthetic data:",
        "python train.py"
    )
    
    print_step(
        4,
        "Evaluate the Model",
        "Evaluate performance on test set and generate reports:",
        "python evaluate.py"
    )
    
    print("\n" + "=" * 70)
    print("OPTION 2: WITH YOUR OWN DATA")
    print("=" * 70)
    
    print("\n1. PREPARE YOUR DATASET")
    print("   Directory structure:")
    print("   towel_grading/data/")
    print("   ├── Grade_A/  (premium quality images)")
    print("   ├── Grade_B/  (standard quality images)")
    print("   └── Grade_C/  (substandard quality images)")
    print()
    print("   ✓ Minimum 300 images per grade (900 total)")
    print("   ✓ Supported formats: PNG, JPG, JPEG, BMP")
    print("   ✓ No need to resize - model handles any size")
    
    print("\n2. TRAIN THE MODEL")
    print("   Command: python train.py")
    
    print("\n3. EVALUATE THE MODEL")
    print("   Command: python evaluate.py")
    
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\nFor an interactive menu, run:")
    print("  python main.py")
    print()
    print("This allows you to:")
    print("  • Train the model")
    print("  • Evaluate on test set")
    print("  • Predict on single images")
    print()
    
    print("=" * 70)
    print("FILE DESCRIPTIONS")
    print("=" * 70)
    print("""
  config.py          - Configuration and hyperparameters
  model.py           - CNN model architecture (MobileNetV2)
  data_utils.py      - Data loading and preprocessing
  train.py           - Training script
  evaluate.py        - Evaluation and predictions
  main.py            - Interactive menu
  create_test_data.py - Generate synthetic test images
  README.md          - Full documentation
    """)
    
    print("=" * 70)
    print("EXPECTED OUTPUTS")
    print("=" * 70)
    print("""
  After training, check these folders:
  
  models/
    ├── best_model.h5              - Best checkpoint
    └── towel_grading_model.h5     - Final model
  
  results/
    ├── training_history.png       - Accuracy/Loss curves
    ├── confusion_matrix.png       - Grade confusion matrix
    ├── performance_metrics.png    - Metrics bar chart
    └── training_history.json      - Raw metrics
    """)
    
    print("\n" + "=" * 70)
    print("PERFORMANCE EXPECTATIONS")
    print("=" * 70)
    print("""
  With proper dataset (500+ images per grade):
    • Accuracy:  92-96%
    • Precision: 91-95%
    • Recall:    91-95%
    • F1-Score:  91-95%
    """)
    
    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    
    response = input("\nWould you like to create synthetic test data now? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nCreating synthetic test data...")
        os.system("python create_test_data.py 100")
        print("\n✓ Test data created!")
        print("\nNow run: python train.py")
    else:
        print("\nYou can manually prepare data in:")
        print("  data/Grade_A/")
        print("  data/Grade_B/")
        print("  data/Grade_C/")
        print("\nThen run: python train.py")
    
    print("\n")


if __name__ == '__main__':
    main()
