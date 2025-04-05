import os
import random
import shutil
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Paths to original and augmented image folders
cv_event_folder = "D:/EXPERIMENT/40-CLAHE/MACE"
non_cv_event_folder = "D:/EXPERIMENT/40-CLAHE/NOMACE"
aug_cv_event_folder = "D:/EXPERIMENT/40-CLAHE/AUGMACE"
aug_non_cv_event_folder = "D:/EXPERIMENT/40-CLAHE/AUGNOMACE"
base_folder = "D:/EXPERIMENT/40-CLAHE/k-fold-splits"

# Function to create directories
def create_dir_if_not_exists(base_folder, fold, subset, class_name):
    folder = os.path.join(base_folder, f"fold_{fold}", subset, class_name)
    os.makedirs(folder, exist_ok=True)
    return folder

# Function to copy images
def copy_image(image, source_folder, destination_folder):  # Single image copy
    src_path = os.path.join(source_folder, image)
    if os.path.exists(src_path):
        shutil.copy(src_path, destination_folder)
    else:
        print(f"Warning: {image} not found in {source_folder}")

# Group images by patient ID (first 7 characters for original, adjusted for augmented)
def group_images_by_patient(image_list):
    patient_dict = defaultdict(list)
    for image in image_list:
        if image.startswith("CLAHE_"):  # Augmented image
            parts = image.split("_")
            if "AUG" in parts:
                #CLAHE_AUG_0_1008966_21015_0_0.png format
                patient_id = parts[3] #Extract eid
            else:
                #CLAHE_1008966_21015_0_0.png format
                patient_id = parts[1] #Extract eid
        else:  # Original image
            patient_id = image.split("_")[0]  # Extract patient ID
        patient_dict[patient_id].append(image)
    return patient_dict


# Load all images from original and augmented folders
cv_event_images = os.listdir(cv_event_folder)
non_cv_event_images = os.listdir(non_cv_event_folder)
aug_cv_event_images = os.listdir(aug_cv_event_folder)
aug_non_cv_event_images = os.listdir(aug_non_cv_event_folder)

# Combine the lists
all_cv_event_images = cv_event_images + aug_cv_event_images
all_non_cv_event_images = non_cv_event_images + aug_non_cv_event_images

# Group images by patient ID
cv_event_patients = group_images_by_patient(all_cv_event_images)
non_cv_event_patients = group_images_by_patient(all_non_cv_event_images)

# Shuffle patient IDs
cv_event_patient_ids = list(cv_event_patients.keys())
non_cv_event_patient_ids = list(non_cv_event_patients.keys())
random.shuffle(cv_event_patient_ids)
random.shuffle(non_cv_event_patient_ids)

# Combine and create labels (1 for MACE, 0 for No MACE)
all_patients = cv_event_patient_ids + non_cv_event_patient_ids
labels = np.array([1] * len(cv_event_patient_ids) + [0] * len(non_cv_event_patient_ids))

# Stratified K-Fold with 6 splits (Fold 6 for test set)
skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)

# Separate test set first (Fold 6)
train_val_patients_idx, test_patients_idx = next(skf.split(all_patients, labels)) #fixed error
test_patient_ids = [all_patients[i] for i in test_patients_idx]
train_val_patient_ids = [all_patients[i] for i in train_val_patients_idx]

# Ensure test images are completely exclusive
test_images = []
for pid in test_patient_ids:
    if pid in cv_event_patients:
        test_images.extend(cv_event_patients[pid])
    elif pid in non_cv_event_patients:
        test_images.extend(non_cv_event_patients[pid])

# Now apply 5-fold Stratified K-Fold on the remaining train/val set
remaining_labels = [1 if pid in cv_event_patient_ids else 0 for pid in train_val_patient_ids]
skf_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf_5fold.split(train_val_patient_ids, remaining_labels), start=1):
    train_patient_ids = [train_val_patient_ids[i] for i in train_idx]
    val_patient_ids = [train_val_patient_ids[i] for i in val_idx]

    train_images = []
    val_images = []

    for pid in train_patient_ids:
        if pid in cv_event_patients:
            train_images.extend(cv_event_patients[pid])
        elif pid in non_cv_event_patients:
            train_images.extend(non_cv_event_patients[pid])

    for pid in val_patient_ids:
        if pid in cv_event_patients:
            val_images.extend(cv_event_patients[pid])
        elif pid in non_cv_event_patients:
            val_images.extend(non_cv_event_patients[pid])

    print(f"Processing Fold {fold} - Training: {len(train_images)}, Validation: {len(val_images)}")

    # Create directories and copy images
    train_cv_event_dir = create_dir_if_not_exists(base_folder, fold, "train", "cv_event")
    train_non_cv_event_dir = create_dir_if_not_exists(base_folder, fold, "train", "non_cv_event")
    val_cv_event_dir = create_dir_if_not_exists(base_folder, fold, "val", "cv_event")
    val_non_cv_event_dir = create_dir_if_not_exists(base_folder, fold, "val", "non_cv_event")

    for image in train_images:
        if image in cv_event_images:
            copy_image(image, cv_event_folder, train_cv_event_dir)
        elif image in aug_cv_event_images:
            copy_image(image, aug_cv_event_folder, train_cv_event_dir)
        elif image in non_cv_event_images:
            copy_image(image, non_cv_event_folder, train_non_cv_event_dir)
        elif image in aug_non_cv_event_images:
            copy_image(image, aug_non_cv_event_folder, train_non_cv_event_dir)
        else:
            print(f"Warning: {image} not found in any source folder.")

    for image in val_images:
        if image in cv_event_images:
            copy_image(image, cv_event_folder, val_cv_event_dir)
        elif image in aug_cv_event_images:
            copy_image(image, aug_cv_event_folder, val_cv_event_dir)
        elif image in non_cv_event_images:
            copy_image(image, non_cv_event_folder, val_non_cv_event_dir)
        elif image in aug_non_cv_event_images:
            copy_image(image, aug_non_cv_event_folder, val_non_cv_event_dir)
        else:
            print(f"Warning: {image} not found in any source folder.")

# Process Fold 6 (Test set only)
print(f"Processing Fold 6 - Test set: {len(test_images)}")
test_cv_event_dir = create_dir_if_not_exists(base_folder, 6, "test", "cv_event")
test_non_cv_event_dir = create_dir_if_not_exists(base_folder, 6, "test", "non_cv_event")

for image in test_images:
    if image in cv_event_images:
        copy_image(image, cv_event_folder, test_cv_event_dir)
    elif image in aug_cv_event_images:
        copy_image(image, aug_cv_event_folder, test_cv_event_dir)
    elif image in non_cv_event_images:
        copy_image(image, non_cv_event_folder, test_non_cv_event_dir)
    elif image in aug_non_cv_event_images:
        copy_image(image, aug_non_cv_event_folder, test_non_cv_event_dir)
    else:
        print(f"Warning: {image} not found in any source folder.")

print("Data splitting complete. No test images are in train/val sets.")