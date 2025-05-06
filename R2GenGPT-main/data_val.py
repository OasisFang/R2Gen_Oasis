# validate_data_distribution.py
import json
from collections import defaultdict

# --- Parameters ---
# !!! 请确保这个路径指向你用于模型验证的、包含 label_report 字段的 JSON 文件 !!!
annotation_file_path = '/root/autodl-tmp/mimic_cxr/mimic_cxr_label_report_corrected.json'
# 指定要分析的数据分割 ('train', 'val', or 'test')
# 根据你的日志，验证是在 'val' split 上进行的
split_key = 'test'
# 定义你的模型使用的所有疾病类别，顺序需要和模型输出一致
diseases = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
    "Pneumonia", "Pneumothorax", "Support Devices"
]
# --- End Parameters ---

def parse_label_report(label_report_str, disease_list):
    """
    Parses the label_report string and returns a dictionary of labels.
    Handles potential errors during parsing.
    """
    labels = {disease: 0 for disease in disease_list} # Default to Negative (0)
    if not isinstance(label_report_str, str) or not label_report_str:
        print(f"Warning: Empty or invalid label_report string found.")
        return labels # Return default labels if input is invalid

    parts = label_report_str.split(',')
    parsed_diseases = set()

    for part in parts:
        part = part.strip()
        if not part: continue # Skip empty parts if any

        key_value = part.split(':')
        if len(key_value) != 2:
            print(f"Warning: Malformed label part '{part}' in report '{label_report_str[:50]}...'. Skipping.")
            continue

        disease_name, value = key_value[0].strip(), key_value[1].strip().lower()

        if disease_name not in disease_list:
            # print(f"Warning: Unrecognized disease '{disease_name}' found in report. Ignoring.")
            continue # Ignore if disease name is not in our predefined list

        parsed_diseases.add(disease_name)

        # Assuming 'Positive' means 1, others (Negative, Uncertain) mean 0 for binary presence
        if value == 'positive':
            labels[disease_name] = 1
        elif value == 'negative':
            labels[disease_name] = 0
        else:
            # Handle 'uncertain' or other values if necessary, here we treat them as negative (0)
            # print(f"Info: Treating label value '{value}' for '{disease_name}' as Negative (0).")
            labels[disease_name] = 0

    # Check if all expected diseases were found in the report string
    # missing_diseases = set(disease_list) - parsed_diseases
    # if missing_diseases:
    #     print(f"Warning: Missing diseases {missing_diseases} in report '{label_report_str[:50]}...'. Assuming Negative (0).")

    return labels


def analyze_data_distribution(file_path, split, disease_list):
    """
    Loads annotation data and analyzes the distribution of positive labels
    for the specified split.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print(f"Successfully loaded annotation file: {file_path}")
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return

    if split not in data:
        print(f"Error: Split '{split}' not found in the annotation file. Available splits: {list(data.keys())}")
        return

    split_data = data[split]
    total_samples = len(split_data)

    if total_samples == 0:
        print(f"Warning: The '{split}' split contains no samples.")
        return

    print(f"\nAnalyzing '{split}' split with {total_samples} samples...")

    positive_counts = defaultdict(int)
    label_parse_errors = 0

    for i, sample in enumerate(split_data):
        label_report = sample.get('label_report', None)
        if label_report is None:
            print(f"Warning: Sample {i} (ID: {sample.get('id', 'N/A')}) is missing 'label_report'. Skipping.")
            label_parse_errors += 1
            continue

        try:
            labels = parse_label_report(label_report, disease_list)
            for disease in disease_list:
                if labels.get(disease, 0) == 1: # Count positive labels
                    positive_counts[disease] += 1
        except Exception as e:
            print(f"Error parsing label report for sample {i} (ID: {sample.get('id', 'N/A')}): {e}")
            label_parse_errors += 1


    print(f"\n--- Label Distribution in '{split}' Split ---")
    print(f"Total Samples Processed: {total_samples}")
    if label_parse_errors > 0:
         print(f"Samples skipped due to missing/invalid label_report: {label_parse_errors}")

    print(f"{'Disease':<30} | {'Positive Count':<15} | {'Percentage':<10}")
    print("-" * 60)

    sorted_diseases = sorted(disease_list, key=lambda d: positive_counts[d], reverse=True)

    for disease in sorted_diseases:
        count = positive_counts[disease]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        print(f"{disease:<30} | {count:<15} | {percentage:.2f}%")

    print("-" * 60)

    zero_positive_classes = [d for d in disease_list if positive_counts[d] == 0]
    if zero_positive_classes:
        print("\nWarning: The following classes have ZERO positive samples in this split:")
        for d in zero_positive_classes:
            print(f"- {d}")
        print("This directly explains why Precision, Recall, and F1 are zero for these classes.")

    low_positive_classes = [d for d in disease_list if 0 < positive_counts[d] <= (total_samples * 0.01)] # Example: <1%
    if low_positive_classes:
         print("\nWarning: The following classes have very FEW (<1%) positive samples:")
         for d in low_positive_classes:
              print(f"- {d} ({positive_counts[d]} samples)")
         print("Low sample count makes it hard for the model to learn and perform well.")


if __name__ == "__main__":
    analyze_data_distribution(annotation_file_path, split_key, diseases)