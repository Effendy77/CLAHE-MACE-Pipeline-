import os
import pandas as pd

# Base directory containing all k-fold splits
base_dir = "D:/EXPERIMENT/40-CLAHE/k-fold-splits"

# Paths to CSV files containing patient EIDs
cv_event_csv = "D:/EXPERIMENT/40-CLAHE/10-year_diabetesprevalence_MACEincidence_bothfundus.csv"
no_cv_event_csv ="D:/EXPERIMENT/40-CLAHE/10-year_diabetesprevalence_NoMACE_bothfundus.csv"

# Read CSV files into Pandas DataFrames
cv_event_df = pd.read_csv(cv_event_csv)
no_cv_event_df = pd.read_csv(no_cv_event_csv)

# List of folds (1-6, where fold 6 is test only)
folds = [1, 2, 3, 4, 5, 6]

# Function to count images in a given directory (including augmented files)
def count_images(directory):
    total_left_eye = 0
    total_right_eye = 0
    aug_left_eye = 0
    aug_right_eye = 0

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            if "21016" in filename:  # Right eye
                if "CLAHE_AUG" in filename or "CLAHE_" in filename:
                    aug_right_eye += 1  # Augmented
                else:
                    total_right_eye += 1  # Original
            elif "21015" in filename:  # Left eye
                if "CLAHE_AUG" in filename or "CLAHE_" in filename:
                    aug_left_eye += 1  # Augmented
                else:
                    total_left_eye += 1  # Original

    return total_left_eye, total_right_eye, aug_left_eye, aug_right_eye

# Function to count matched and unmatched images
def count_matched_unmatched(df, directory):
    matched_left_eye = 0
    matched_right_eye = 0

    for _, row in df.iterrows():
        eid = str(row['eid'])  # Extract full EID

        # Construct filenames for left and right eyes
        left_eye_filename = f"{eid}_21015_0_0.png"
        right_eye_filename = f"{eid}_21016_0_0.png"

        # Check if files exist
        left_eye_path = os.path.join(directory, left_eye_filename)
        right_eye_path = os.path.join(directory, right_eye_filename)

        if os.path.isfile(left_eye_path):
            matched_left_eye += 1
        if os.path.isfile(right_eye_path):
            matched_right_eye += 1

    return matched_left_eye, matched_right_eye

# Iterate through each fold and count images
for fold in folds:
    print(f"\n📂 Processing Fold {fold}...")

    # Define paths for train, val, and test sets
    if fold == 6:
        directories = {
            "test_cv": f"{base_dir}/fold_{fold}/test/cv_event",
            "test_non_cv": f"{base_dir}/fold_{fold}/test/non_cv_event",
        }
    else:
        directories = {
            "train_cv": f"{base_dir}/fold_{fold}/train/cv_event",
            "train_non_cv": f"{base_dir}/fold_{fold}/train/non_cv_event",
            "val_cv": f"{base_dir}/fold_{fold}/val/cv_event",
            "val_non_cv": f"{base_dir}/fold_{fold}/val/non_cv_event",
        }

    for dataset_name, directory in directories.items():
        if not os.path.exists(directory):
            print(f"⚠️ Warning: {directory} does not exist, skipping...")
            continue

        print(f"\n📁 Checking directory: {dataset_name} ({directory})")

        # Count total images
        total_left, total_right, aug_left, aug_right = count_images(directory)

        # Count matched images for CV and No CV events
        if "cv" in dataset_name:
            matched_left, matched_right = count_matched_unmatched(cv_event_df, directory)
        else:
            matched_left, matched_right = count_matched_unmatched(no_cv_event_df, directory)

        # Calculate unmatched images
        unmatched_left = total_left - matched_left
        unmatched_right = total_right - matched_right

        # Print results for this directory
        print(f"🟢 Total Left Eye: {total_left} (Original) | {aug_left} (Augmented)")
        print(f"🔵 Total Right Eye: {total_right} (Original) | {aug_right} (Augmented)")
        print(f"✅ Matched Left Eye: {matched_left}")
        print(f"✅ Matched Right Eye: {matched_right}")
        print(f"❌ Unmatched Left Eye: {unmatched_left}")
        print(f"❌ Unmatched Right Eye: {unmatched_right}")

# Final Summary
print("\n✅ Image counting complete for all folds.")
