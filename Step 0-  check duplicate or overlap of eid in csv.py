import pandas as pd

# Load the CSV files
file1 = pd.read_csv("D:/EXPERIMENT/39-10year-TestXclusive+R&Lpaired_Alldiabetes/10-year_diabetesprevalence_MACEincidence_bothfundus.csv")
file2 = pd.read_csv("D:/EXPERIMENT/39-10year-TestXclusive+R&Lpaired_Alldiabetes/10-year_diabetesprevalence_NoMACE_bothfundus.csv")
# Check for duplicate eids in both files
print("Duplicates in File 1:", file1[file1.duplicated(subset=['eid'])])
print("Duplicates in File 2:", file2[file2.duplicated(subset=['eid'])])

# Check for common eids between file1 and file2
common_eids = pd.merge(file1, file2, on='eid', how='inner')
print(f"Number of common eids: {common_eids.shape[0]}")
