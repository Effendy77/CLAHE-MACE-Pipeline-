# CLAHE-Based Fundus Image Augmentation Pipeline for MACE Prediction

This repository provides a fully reproducible Python pipeline for preprocessing and augmenting retinal fundus images using CLAHE (Contrast Limited Adaptive Histogram Equalization), tailored for deep learning prediction of major adverse cardiovascular events (MACE) in diabetic patients using UK Biobank data.

---

## 📁 Repository Structure

```
.
├── step_0_check_overlap.py
├── step_1_clahe_augment.py
├── step_2_stratified_split.py
├── step_2_1_verify_split.py
├── step_3_cross_match.py
├── step_4_check_fold_consistency.py
├── step_5_count_augmented_images.py
├── step_6_flip_right_to_left.py
├── /images/         # Original and augmented image folders (MACE, NOMACE, AUGMACE, AUGNOMACE)
└── /k-fold-splits/  # Contains fold_1 to fold_6 split data
```

---

## 🧪 Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Pandas
- Pillow (PIL)

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🧬 Steps Overview

| Step | Description |
|------|-------------|
| Step 0 | Remove overlapping `eid` across MACE and NOMACE cohorts |
| Step 1 | Apply CLAHE enhancement and generate augmented variants |
| Step 2 | Perform 6-fold patient-stratified split |
| Step 2.1 | Verify stratification integrity and fold balance |
| Step 3 | Ensure image and label `eid` alignment |
| Step 4 | Validate original + augmented images stay in same fold |
| Step 5 | Count and compare images across MACE/NOMACE classes |
| Step 6 | Flip all right-eye images (`21016`) to left-eye orientation (`21015`) |

---

## 🧾 Output

- CLAHE-enhanced and augmented images saved with standard naming conventions.
- Consistent folder structure per fold.
- Final image-level metadata used for deep learning evaluation and test-time augmentation (TTA).

---

## 🔁 Reproducibility

- All folds are patient-exclusive.
- Random seeds are fixed in splitting scripts.
- Scripts can be modified for other datasets with minimal changes.

---

## 📜 Citation

Please cite our associated manuscript if you use this pipeline in your research:
> Effendy Bin Hashim et al., *“Retinal Biomarker-Based Deep Learning for Predicting 10-Year Cardiovascular Risk in Diabetes: A Comparative Evaluation with QRISK3”*, The Lancet Digital Health, 2025.

---

## License

MIT License


## Data Availability

No UK Biobank participant-level data (including eid) is included in this repository.
All analyses were conducted under UK Biobank approval using secure local data.
