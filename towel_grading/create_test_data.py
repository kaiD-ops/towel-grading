"""
Utility script to create synthetic test images for demonstration purposes.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import random
from config import DATASET_PATH, GRADES, IMG_HEIGHT, IMG_WIDTH


def create_synthetic_grade_a_image():
    """
    Create synthetic Grade A (premium quality) image:
    - Uniform color
    - Even texture
    - No visible defects
    - Smooth appearance
    """
    # Create base image with uniform color (light beige - typical towel color)
    img = Image.new('RGB', (IMG_HEIGHT, IMG_WIDTH), color=(220, 200, 180))
    
    # Add subtle texture with fine lines
    pixels = img.load()
    for i in range(IMG_HEIGHT):
        for j in range(IMG_WIDTH):
            if random.random() < 0.02:  # 2% noise
                r, g, b = pixels[j, i]
                pixels[j, i] = (r + random.randint(-5, 5), 
                               g + random.randint(-5, 5), 
                               b + random.randint(-5, 5))
    
    # Add subtle pile texture
    img_array = np.array(img)
    for _ in range(100):
        y = random.randint(0, IMG_HEIGHT - 5)
        x = random.randint(0, IMG_WIDTH - 5)
        img_array[y:y+3, x:x+3] += np.random.randint(-3, 3)
    
    img = Image.fromarray(np.uint8(np.clip(img_array, 0, 255)))
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    return np.array(img)


def create_synthetic_grade_b_image():
    """
    Create synthetic Grade B (standard quality) image:
    - Minor color variations
    - Slightly uneven pile
    - 1-3 minor defects
    - Small texture irregularities
    """
    # Create base with slight color variations
    img = Image.new('RGB', (IMG_HEIGHT, IMG_WIDTH), color=(215, 195, 175))
    
    # Add color patches (minor variations)
    draw = ImageDraw.Draw(img, 'RGBA')
    for _ in range(3):
        x1 = random.randint(0, IMG_WIDTH - 40)
        y1 = random.randint(0, IMG_HEIGHT - 40)
        color = (random.randint(210, 230), random.randint(190, 210), random.randint(170, 190), 100)
        draw.rectangle([x1, y1, x1 + 40, y1 + 40], fill=color)
    
    # Add some defects (small spots or marks)
    img_array = np.array(img)
    for _ in range(3):
        y = random.randint(10, IMG_HEIGHT - 20)
        x = random.randint(10, IMG_WIDTH - 20)
        size = random.randint(3, 8)
        img_array[y:y+size, x:x+size] -= np.random.randint(10, 20)
    
    img = Image.fromarray(np.uint8(np.clip(img_array, 0, 255)))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    
    return np.array(img)


def create_synthetic_grade_c_image():
    """
    Create synthetic Grade C (substandard quality) image:
    - Significant color patches/variations
    - Rough/uneven texture
    - Visible defects or marks
    - >3 defects visible
    """
    # Create base with noticeable color variations
    img = Image.new('RGB', (IMG_HEIGHT, IMG_WIDTH), color=(200, 180, 160))
    
    # Add multiple color patches
    draw = ImageDraw.Draw(img, 'RGBA')
    for _ in range(6):
        x1 = random.randint(0, IMG_WIDTH - 50)
        y1 = random.randint(0, IMG_HEIGHT - 50)
        color = (random.randint(170, 220), random.randint(150, 200), random.randint(130, 180), 120)
        draw.rectangle([x1, y1, x1 + 50, y1 + 50], fill=color)
    
    # Add visible defects
    img_array = np.array(img)
    for _ in range(6):
        y = random.randint(5, IMG_HEIGHT - 15)
        x = random.randint(5, IMG_WIDTH - 15)
        size = random.randint(5, 15)
        img_array[y:y+size, x:x+size] -= np.random.randint(20, 40)
    
    # Add scratches
    for _ in range(3):
        y1 = random.randint(0, IMG_HEIGHT)
        x1 = random.randint(0, IMG_WIDTH)
        y2 = y1 + random.randint(-50, 50)
        x2 = x1 + random.randint(-50, 50)
        cv2.line(img_array, (x1, y1), (x2, y2), 
                (random.randint(80, 120), random.randint(60, 100), random.randint(50, 90)), 2)
    
    img = Image.fromarray(np.uint8(np.clip(img_array, 0, 255)))
    
    return np.array(img)


def create_synthetic_dataset(images_per_class=100):
    """
    Create synthetic dataset for testing.
    
    Args:
        images_per_class (int): Number of images to generate per grade
    """
    
    print("Creating synthetic towel images for testing...")
    print(f"Generating {images_per_class} images per grade (300 total)...")
    
    functions = {
        'Grade_A': create_synthetic_grade_a_image,
        'Grade_B': create_synthetic_grade_b_image,
        'Grade_C': create_synthetic_grade_c_image,
    }
    
    for grade_name, create_func in functions.items():
        grade_path = os.path.join(DATASET_PATH, grade_name)
        os.makedirs(grade_path, exist_ok=True)
        
        print(f"Generating {grade_name} images...")
        for i in range(images_per_class):
            img = create_func()
            img_path = os.path.join(grade_path, f'{grade_name}_{i:03d}.jpg')
            cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{images_per_class} images")
    
    print(f"\n✓ Synthetic dataset created successfully!")
    print(f"Location: {DATASET_PATH}")
    print(f"Total images: {images_per_class * 3}")


def create_small_test_dataset(images_per_class=10):
    """
    Create a small test dataset for quick testing.
    
    Args:
        images_per_class (int): Number of images per grade (default 10)
    """
    create_synthetic_dataset(images_per_class=images_per_class)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        num_images = int(sys.argv[1])
    else:
        num_images = 100
    
    create_synthetic_dataset(images_per_class=num_images)
    print(f"\nYou can now run: python train.py")
