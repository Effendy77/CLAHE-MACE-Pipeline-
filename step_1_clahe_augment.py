import os
import cv2
import numpy as np
from glob import glob

# Define paths for MACE and No MACE image directories
mace_dir = "D:/EXPERIMENT/40-CLAHE/MACE"
no_mace_dir = "D:/EXPERIMENT/40-CLAHE/NOMACE"
aug_mace_dir = "D:/EXPERIMENT/40-CLAHE/AUGMACE"
aug_no_mace_dir = "D:/EXPERIMENT/40-CLAHE/AUGNOMACE"

# Ensure output directories exist
os.makedirs(aug_mace_dir, exist_ok=True)
os.makedirs(aug_no_mace_dir, exist_ok=True)

# Function to apply CLAHE
def apply_CLAHE(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

# Function to apply augmentations (flip, rotation, blur)
def apply_augmentations(image):
    flipped = cv2.flip(image, 1)  # Horizontal flip
    angle = np.random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    blurred = cv2.GaussianBlur(rotated, (5,5), 0)
    return blurred

# Function to process and save augmented images
def process_images(input_dir, output_dir, augment_factor=3, apply_clahe=True):
    image_paths = glob(os.path.join(input_dir, "*.png"))
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply CLAHE if enabled
        if apply_clahe:
            img = apply_CLAHE(img)

        # Save original (CLAHE processed) image
        base_name = os.path.basename(img_path)
        cv2.imwrite(os.path.join(output_dir, f"CLAHE_{base_name}"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Generate augmented images
        for i in range(augment_factor):
            aug_img = apply_augmentations(img)
            cv2.imwrite(os.path.join(output_dir, f"CLAHE_AUG_{i}_{base_name}"), cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

# Process MACE (minority class) with higher augmentation factor
process_images(mace_dir, aug_mace_dir, augment_factor=4, apply_clahe=True)

# Process No MACE (majority class) with lower augmentation factor
process_images(no_mace_dir, aug_no_mace_dir, augment_factor=2, apply_clahe=True)

print("CLAHE augmentation completed for both MACE and No MACE datasets.")
