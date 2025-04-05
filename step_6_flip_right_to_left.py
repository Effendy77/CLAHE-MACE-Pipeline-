from PIL import Image
import os

# Define the directories for each fold
directories = {
    "fold_1_train": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_1/train/cv_event",
    "fold_1_val": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_1/val/non_cv_event",
    "fold_2_train": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_2/train/non_cv_event",
    "fold_2_val": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_2/val/non_cv_event",
    "fold_3_train": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_3/train/non_cv_event",
    "fold_3_val": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_3/val/non_cv_event",
    "fold_4_train": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_4/train/non_cv_event",
    "fold_4_val": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_4/val/non_cv_event",
    "fold_5_train": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_5/train/non_cv_event",
    "fold_5_val": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_5/val/non_cv_event",
    "fold_6_test": "D:/EXPERIMENT/38-10year-TestXclusive+R&Lpaired_T2diabetes/k-fold-splits/fold_6/test/non_cv_event",
}

# Initialize counters for the images
flipped_count = 0
removed_count = 0
maintained_count = 0
left_eye_images = []

# Loop through each directory and process the images
for fold_name, image_dir in directories.items():
    print(f"Processing images in {fold_name}...")

    # Iterate through each image file in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):  # Assuming all images are PNG format
            # Extract the EID from the filename
            eid = filename.split("_")[0]

            # Check if it's a left eye image
            if "21015" in filename:
                left_eye_images.append(filename)
                maintained_count += 1

            # Check if it's a right eye image
            elif "21016" in filename:
                # Load the image
                img = Image.open(os.path.join(image_dir, filename))

                # Flip the image horizontally
                flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)

                # Generate the new filename for the flipped image
                new_filename = filename.replace("_0_0.png", "_flipped.png")

                # Save the flipped image
                flipped_img.save(os.path.join(image_dir, new_filename))
                flipped_count += 1
                print(f"Flipped {filename} and saved as {new_filename}")

                # Remove the original unflipped right eye image
                os.remove(os.path.join(image_dir, filename))
                removed_count += 1
                print(f"Removed original {filename}")

# Print counts
print("Conversion complete.")
print(f"Number of right eye images flipped: {flipped_count}")
print(f"Number of original right eye images removed: {removed_count}")
print(f"Number of left eye images maintained: {maintained_count}")

# Print left eye images for reference
print("Left eye images:")
for filename in left_eye_images:
    print(filename)
