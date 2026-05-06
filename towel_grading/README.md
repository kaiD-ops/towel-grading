# Automated Towel Quality Grading Using CNNs

A complete implementation of an automated towel quality grading system using Convolutional Neural Networks (CNNs) with transfer learning via MobileNetV2.

## Project Overview

This project implements the system described in the research paper to automatically classify towels into three quality grades:
- **Grade A**: Premium quality (uniform color, even pile, smooth texture, zero defects)
- **Grade B**: Standard quality (minor variations, 1-3 minor defects)
- **Grade C**: Substandard quality (significant defects, >3 defects)

## Features

✓ Transfer learning with pre-trained MobileNetV2 model  
✓ Data augmentation (rotation, zoom, flip, brightness/contrast)  
✓ Early stopping and learning rate reduction  
✓ Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)  
✓ Confusion matrix and performance visualization  
✓ Single image prediction capability  
✓ Class weight balancing for imbalanced datasets  

## Project Structure

```
towel_grading/
├── data/                          # Dataset directory
│   ├── Grade_A/                  # Premium quality towel images
│   ├── Grade_B/                  # Standard quality towel images
│   └── Grade_C/                  # Substandard quality towel images
├── models/                        # Saved trained models
├── results/                       # Training history, plots, metrics
├── config.py                      # Configuration and hyperparameters
├── model.py                       # MobileNetV2 model architecture
├── data_utils.py                  # Data loading and preprocessing
├── train.py                       # Training script
├── evaluate.py                    # Evaluation and prediction
├── main.py                        # Main pipeline runner
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

### Directory Structure
Organize your towel images as follows:

```
towel_grading/data/
├── Grade_A/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Grade_B/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Grade_C/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Minimum Dataset Requirements
- **300 images per class** (900 total)
- **Recommended: 500-1000 images per class** for better accuracy
- Supported formats: PNG, JPG, JPEG, BMP
- Image resolution: Any size (will be resized to 224×224)

### Image Collection Tips
- Capture under consistent lighting conditions
- Use a camera mounted above a flat inspection surface
- Ensure representative samples for each grade
- Include various angles and positions

## Usage

### Option 1: Using the Interactive Menu

```bash
python main.py
```

This launches an interactive menu with options to:
1. Train the model
2. Evaluate on test set
3. Predict on single images
4. Exit

### Option 2: Direct Training

```bash
python train.py
```

Trains the model and saves:
- `models/best_model.h5` - Best model checkpoint
- `models/towel_grading_model.h5` - Final trained model
- `results/training_history.json` - Training metrics
- `results/training_history.png` - Accuracy/Loss plots

### Option 3: Evaluation

```bash
python evaluate.py
```

Evaluates the model and generates:
- Classification report
- Confusion matrix
- Performance metrics visualization
- Test set results

### Option 4: Single Image Prediction

```python
from evaluate import ModelEvaluator

evaluator = ModelEvaluator()
grade, confidence_scores = evaluator.predict_single_image('path/to/image.jpg')
print(f"Predicted Grade: {grade}")
print(f"Confidence scores: {confidence_scores}")
```

## Configuration

Edit `config.py` to customize:

```python
# Image preprocessing
IMG_HEIGHT = 224          # Image height in pixels
IMG_WIDTH = 224           # Image width in pixels

# Training
BATCH_SIZE = 32           # Training batch size
EPOCHS = 50               # Maximum epochs
LEARNING_RATE = 1e-4      # Initial learning rate
DROPOUT_RATE = 0.3        # Dropout rate
DENSE_UNITS = 128         # Hidden dense layer units

# Data split
TRAIN_SPLIT = 0.70        # Training set percentage
VAL_SPLIT = 0.15          # Validation set percentage
TEST_SPLIT = 0.15         # Test set percentage

# Augmentation
AUGMENTATION_CONFIG = {
    'rotation_range': 15,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'brightness_range': [0.8, 1.2],
    'contrast_range': [0.8, 1.2],
}
```

## Model Architecture

The model uses **MobileNetV2** transfer learning with a custom classification head:

```
Input (224×224×3)
    ↓
MobileNetV2 (pretrained on ImageNet, frozen)
    ↓
Global Average Pooling
    ↓
Dropout (rate=0.3)
    ↓
Dense (128, ReLU)
    ↓
Dense (3, Softmax)
    ↓
Output (Grade A, B, or C)
```

### Training Strategy
1. **Phase 1**: Train with frozen base model (MobileNetV2)
2. **Phase 2**: Fine-tune top layers of base model
3. **Regularization**: Early stopping, learning rate reduction

## Expected Performance

Based on the research paper and transfer learning approach:
- **Accuracy**: 92-96%
- **Precision**: 91-95%
- **Recall**: 91-95%
- **F1-Score**: 91-95%
- **Inference Time**: <50ms per image

## Results and Outputs

After training, the following files are generated:

### In `models/` directory:
- `best_model.h5` - Best performing model checkpoint
- `towel_grading_model.h5` - Final trained model

### In `results/` directory:
- `training_history.json` - Training metrics (JSON format)
- `training_history.png` - Accuracy and loss curves
- `confusion_matrix.png` - Confusion matrix heatmap
- `performance_metrics.png` - Bar chart of metrics
- `prediction_visualization.png` - Sample prediction visualization

## Metrics Explained

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted Grade X, how many were actually Grade X
- **Recall**: Of actual Grade X samples, how many were correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Shows which grades are confused with each other

## Troubleshooting

### No images found error
```
Error: No images loaded. Call load_images() first.
```
**Solution**: Ensure images are in `data/Grade_A`, `data/Grade_B`, `data/Grade_C` directories

### Out of Memory error
**Solutions**:
- Reduce `BATCH_SIZE` in `config.py`
- Use fewer images
- Run on GPU if available

### Low accuracy
**Solutions**:
- Add more training images (minimum 300-500 per class)
- Ensure images are properly labeled
- Check image quality and consistency
- Increase `EPOCHS` in `config.py`
- Adjust augmentation parameters

### Model not found
```
Model loaded from: models/best_model.h5
```
**Solution**: Train the model first using `python train.py`

## Files Description

| File | Purpose |
|------|---------|
| `config.py` | Central configuration file with all hyperparameters |
| `model.py` | MobileNetV2 model architecture and compilation |
| `data_utils.py` | Data loading, preprocessing, and augmentation |
| `train.py` | Training loop with callbacks and checkpointing |
| `evaluate.py` | Model evaluation and prediction utilities |
| `main.py` | Interactive command-line interface |

## Advanced Usage

### Fine-tuning the model
```python
from model import create_model, compile_model

model, base_model = create_model(num_classes=3)

# Unfreeze last layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = compile_model(model, learning_rate=1e-5)
```

### Using different base models
```python
# In model.py, replace MobileNetV2 with:
# keras.applications.ResNet50
# keras.applications.VGG16
# keras.applications.EfficientNetB0
# keras.applications.InceptionV3
```

### Batch prediction
```python
from evaluate import ModelEvaluator
import os

evaluator = ModelEvaluator()

image_folder = 'path/to/images'
for image_file in os.listdir(image_folder):
    grade, probs = evaluator.predict_single_image(
        os.path.join(image_folder, image_file)
    )
    print(f"{image_file}: {grade}")
```

## Future Improvements

- [ ] Integrate with real-time conveyor belt system
- [ ] Deploy on edge devices (Raspberry Pi, NVIDIA Jetson)
- [ ] Add more towel types and textures
- [ ] Extend to other textile products (bedsheets, bathrobes)
- [ ] Web API for predictions
- [ ] Mobile app integration
- [ ] Explainability (attention maps, saliency)

## References

[1] Seker et al. (2019). "Fabric Defect Detection Using Convolutional Neural Networks"

[2] Jing et al. (2020). "Mobile-Unet: An Efficient Convolutional Neural Network for Fabric Defect Detection"

[3] Li et al. (2021). "Deformable Patterned Fabric Defect Detection with Fisher Criterion-Based Deep Learning"

## License

This project is for educational purposes.

## Authors

- Huzaifa Awais (31774)
- Ammar Danish (28304)
- Mustafa Mehmood (28426)

SMCS IBA, Karachi

---

**Last Updated**: May 2026  
**Python Version**: 3.8+  
**Framework**: TensorFlow/Keras 2.13+
