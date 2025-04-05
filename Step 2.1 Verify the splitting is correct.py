import os

# Paths (Make sure these match your data splitting script)
base_folder = "D:/EXPERIMENT/40-CLAHE/k-fold-splits"

# Verification Script
def verify_image_counts(base_folder):
    """
    Verifies the image counts in each fold/subset created by the data splitting script.
    """
    fold_summary = {}

    for fold in range(1, 7):  # Loop through folds 1 to 6
        fold_summary[fold] = {}
        for subset in ["train", "val"] if fold < 6 else ["test"]:  # No val in fold 6

            cv_event_dir = os.path.join(base_folder, f"fold_{fold}", subset, "cv_event")
            non_cv_event_dir = os.path.join(base_folder, f"fold_{fold}", subset, "non_cv_event")

            num_cv_event = len(os.listdir(cv_event_dir)) if os.path.exists(cv_event_dir) else 0
            if os.path.exists(non_cv_event_dir):
                num_non_cv_event = len(os.listdir(non_cv_event_dir))
            else:
                num_non_cv_event = 0
            total = num_cv_event + num_non_cv_event

            fold_summary[fold][subset] = {
                "cv_event": num_cv_event,
                "non_cv_event": num_non_cv_event,
                "total": total
            }

            print(f"Fold {fold}, {subset}: cv_event = {num_cv_event}, non_cv_event = {num_non_cv_event}, total = {total}")

    return fold_summary

def calculate_expected_counts(cv_event_total, non_cv_event_total, train_prop=0.64, val_prop=0.16, test_prop=0.20):
    """
    Calculates the expected image counts for each split (train, val, test)
    based on the given proportions.
    """
    expected = {}

    # Test set (Fold 6)
    expected["test"] = {
        "cv_event": round(cv_event_total * test_prop),
        "non_cv_event": round(non_cv_event_total * test_prop),
        "total": round((cv_event_total + non_cv_event_total) * test_prop)
    }

    # Train + Validation (Folds 1-5)
    train_val_cv_event = cv_event_total - expected["test"]["cv_event"]
    train_val_non_cv_event = non_cv_event_total - expected["test"]["non_cv_event"]

    expected["train"] = {
        "cv_event": round(train_val_cv_event * train_prop / (train_prop + val_prop)),
        "non_cv_event": round(train_val_non_cv_event * train_prop / (train_prop + val_prop)),
        "total": round((train_val_cv_event + train_val_non_cv_event) * train_prop / (train_prop + val_prop))
    }

    expected["val"] = {
        "cv_event": round(train_val_cv_event * val_prop / (train_prop + val_prop)),
        "non_cv_event": round(train_val_non_cv_event * val_prop / (train_prop + val_prop)),
        "total": round((train_val_cv_event + train_val_non_cv_event) * val_prop / (train_prop + val_prop))
    }

    return expected

def compare_counts(fold_summary, expected_counts):
  """Compares the observed image counts from the folds against the expected counts."""
  print("\n--- Comparison with Expected Counts ---")
  for subset in ["train", "val", "test"]:
    print(f"Subset: {subset}")
    observed_cv_event = sum([fold_summary[f][subset]['cv_event'] for f in range(1, 6)]) if subset != "test" else fold_summary[6][subset]['cv_event']
    observed_non_cv_event = sum([fold_summary[f][subset]['non_cv_event'] for f in range(1, 6)]) if subset != "test" else fold_summary[6][subset]['non_cv_event']
    observed_total = sum([fold_summary[f][subset]['total'] for f in range(1, 6)]) if subset != "test" else fold_summary[6][subset]['total']

    print(f"  Observed: cv_event={observed_cv_event}, non_cv_event={observed_non_cv_event}, total={observed_total}")
    print(f"  Expected: cv_event={expected_counts[subset]['cv_event']}, non_cv_event={expected_counts[subset]['non_cv_event']}, total={expected_counts[subset]['total']}")
    print(f"  cv_event Diff: {observed_cv_event - expected_counts[subset]['cv_event']}, non_cv_event Diff: {observed_non_cv_event - expected_counts[subset]['non_cv_event']}, total Diff: {observed_total - expected_counts[subset]['total']}")
    print("-" * 40)


# Main execution
if __name__ == "__main__":
    # 1. Define the total number of original and augmented images for each class:
    cv_event_total = 567 + 2835 # original MACE + augmented MACE
    non_cv_event_total = 2593 + 7779 # original No MACE + augmented No MACE

    # 2. Calculate the expected counts based on the proportions:
    expected_counts = calculate_expected_counts(cv_event_total, non_cv_event_total)

    # 3. Verify image counts in each folder:
    fold_summary = verify_image_counts(base_folder)

    #4. Compare results with the expected counts
    compare_counts(fold_summary, expected_counts)

    print("\nVerification complete.")