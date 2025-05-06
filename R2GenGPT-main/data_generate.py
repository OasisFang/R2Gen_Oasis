# generate_label_report_cleaned.py
import json
import pandas as pd
from collections import defaultdict
import numpy as np
import os
from tqdm import tqdm # For progress bar

# --- Parameters ---
# Input CSV file with CheXpert labels
csv_file_path = '/root/autodl-tmp/mimic-cxr-2.0.0-chexpert.csv'

# Input JSON file with structure, report, image path, split
input_json_path = '/root/autodl-tmp/mimic_cxr/annotation.json'

# Output JSON file path with corrected label_report strings
output_json_path = '/root/autodl-tmp/mimic_cxr/mimic_cxr_label_report_corrected.json' # New name

# Column names for matching
id_column = 'study_id'
subj_id_column = 'subject_id'

# Define the 14 disease labels IN THE ORDER required for the string
disease_labels_ordered = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices"
]

# How to handle CSV labels: 1.0 -> "Positive", Others (0.0, -1.0, NaN) -> "Negative"
def map_label_to_string(csv_value):
    if csv_value == 1.0:
        return "Positive"
    else:
        return "Negative"
# --- End Parameters ---

def generate_label_report_annotations():
    # --- Load CSV Labels and get Study IDs ---
    print(f"Loading CSV labels from: {csv_file_path}")
    try:
        df_csv = pd.read_csv(csv_file_path)
        required_csv_cols = [id_column, subj_id_column] + disease_labels_ordered
        if not all(col in df_csv.columns for col in required_csv_cols):
             missing_cols = [col for col in required_csv_cols if col not in df_csv.columns]
             print(f"Error: Required columns {missing_cols} not found in CSV.")
             return None, None
        csv_study_ids = set(df_csv[id_column].unique())
        print(f"Found {len(csv_study_ids)} unique study IDs in CSV.")
        df_csv = df_csv[required_csv_cols].drop_duplicates(subset=[id_column], keep='first')
        df_csv.set_index(id_column, inplace=True)
        print(f"Indexed {len(df_csv)} unique study entries from CSV for label lookup.")
    except Exception as e:
        print(f"An unexpected error occurred while loading CSV: {e}")
        return None, None

    # --- Load Input JSON and get Study IDs ---
    print(f"Loading input JSON structure from: {input_json_path}")
    json_study_ids = set()
    input_data = {}
    try:
        with open(input_json_path, 'r') as f: input_data = json.load(f)
        print("Input JSON structure loaded successfully.")
        for split in ["train", "val", "test"]:
            if split in input_data:
                for entry in input_data[split]:
                    study_id = entry.get(id_column)
                    if study_id is not None: json_study_ids.add(study_id)
        print(f"Found {len(json_study_ids)} unique study IDs in input JSON.")
    except Exception as e:
        print(f"An unexpected error occurred while loading input JSON: {e}")
        return None, None

    # --- Find Common Study IDs ---
    common_study_ids = csv_study_ids.intersection(json_study_ids)
    print(f"Found {len(common_study_ids)} common study IDs between CSV and JSON.")
    if not common_study_ids: print("Error: No common study IDs found."); return None, None

    # --- Process Only Common Entries and Generate Output Data ---
    output_data = {"train": [], "val": [], "test": []}
    processed_count = 0
    included_count = 0
    label_lookup_errors = 0

    print("Processing entries and generating corrected 'label_report' strings...")
    for split in ["train", "val", "test"]:
        if split not in input_data: continue
        print(f"Processing split: {split}")
        for entry in tqdm(input_data[split], desc=f"Split {split}"):
            processed_count += 1
            study_id = entry.get(id_column)
            if study_id is None or study_id not in common_study_ids: continue

            entry_id = entry.get('id')
            subject_id_json = entry.get(subj_id_column)
            report = entry.get('report', '')
            image_path = entry.get('image_path', [])
            if entry_id is None: print(f"Warning: Skipping entry {study_id} missing 'id'."); continue

            # --- Get Labels from CSV and Create label_report String ---
            try:
                csv_row = df_csv.loc[study_id]
                label_parts = []
                for disease in disease_labels_ordered:
                    label_str_val = map_label_to_string(csv_row[disease])
                    label_parts.append(f"{disease}:{label_str_val}")
                label_report_string = ", ".join(label_parts) # Join with comma and space

                # --- Create New Entry ---
                new_entry = {
                    "id": entry_id,
                    "study_id": study_id,
                    "subject_id": subject_id_json,
                    "report": report,
                    "image_path": image_path,
                    "split": split,
                    "label_report": label_report_string # Store the new string
                }
                output_data[split].append(new_entry)
                included_count += 1

            except KeyError: print(f"Error: Common study_id {study_id} not in CSV index. Skipping."); label_lookup_errors += 1
            except Exception as e: print(f"Error processing entry {study_id} (id: {entry_id}): {e}. Skipping."); label_lookup_errors += 1

    print("\nProcessing finished.")
    print(f"Total entries checked in JSON: {processed_count}")
    print(f"Entries included in output (common study_id found): {included_count}")
    # ... (print other counts) ...

    # --- Save Output JSON ---
    print(f"Saving corrected annotations to: {output_json_path}")
    try:
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print("Successfully saved corrected annotations.")
    except Exception as e:
        print(f"Error saving output JSON file: {e}")

if __name__ == "__main__":
    generate_label_report_annotations()