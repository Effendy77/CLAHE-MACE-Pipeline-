import os

# Define correct dataset paths
base_path = "D:/EXPERIMENT/40-CLAHE/k-fold-splits"
folds = [1, 2, 3, 4, 5, 6]  # 6th is test set

# Load all images in each fold, grouped by patient ID
def load_filenames_by_patient(fold, subset):
    folder = os.path.join(base_path, f"fold_{fold}", subset, "cv_event")
    if not os.path.exists(folder):
        print(f"Warning: {folder} not found!")
        return {}

    images = os.listdir(folder)
    patient_dict = {}
    for img in images:
        if img.startswith("CLAHE_"):  # Augmented image
            parts = img.split("_")
            if "AUG" in parts:
                #CLAHE_AUG_0_1008966_21015_0_0.png format
                patient_id = parts[3] #Extract eid
            else:
                #CLAHE_1008966_21015_0_0.png format
                patient_id = parts[1] #Extract eid
        else:  # Original image
            patient_id = img.split("_")[0]  # Extract patient ID

        if patient_id not in patient_dict:
            patient_dict[patient_id] = set()
        patient_dict[patient_id].add(img)

    return patient_dict

# Load test set (Fold 6)
test_patient_images = load_filenames_by_patient(6, "test")

# Load train/validation sets (Folds 1-5)
train_val_patient_images = {}
for fold in folds[:-1]:  # Skip test set (Fold 6)
    train_val_patient_images.update(load_filenames_by_patient(fold, "train"))
    train_val_patient_images.update(load_filenames_by_patient(fold, "val"))

# Check for patient ID contamination
overlapping_patients = set(test_patient_images.keys()).intersection(set(train_val_patient_images.keys()))

# Print results
if overlapping_patients:
    print("❌ WARNING: Overlapping patient IDs found between test set (Fold 6) and train/val sets (Folds 1-5)!")
    for patient in overlapping_patients:
        print(f"Patient {patient} appears in both training and test sets.")
else:
    print("✅ No patient ID contamination detected. Data split is correct.")