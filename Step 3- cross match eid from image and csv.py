import os
import pandas as pd

# Define the directory containing the original images
original_image_dir = "D:/EXPERIMENT/40-CLAHE/AUGNOMACE"

# Read the CSV files containing the image filenames and categories
cv_event_csv ="D:/EXPERIMENT/40-CLAHE/10-year_diabetesprevalence_MACEincidence_bothfundus.csv"
no_cv_event_csv = "D:/EXPERIMENT/40-CLAHE/10-year_diabetesprevalence_NoMACE_bothfundus.csv"

# Read CSV files into Pandas DataFrames
cv_event_df = pd.read_csv(cv_event_csv)
no_cv_event_df = pd.read_csv(no_cv_event_csv)

# Get list of image EIDs
image_eids = [image_file.split('_')[0] for image_file in os.listdir(original_image_dir)]

# Check how many MACE EIDs match with images
matching_mace_images = cv_event_df['eid'].astype(str).isin(image_eids)
print(f"Number of matching MACE EIDs with images: {matching_mace_images.sum()}")
