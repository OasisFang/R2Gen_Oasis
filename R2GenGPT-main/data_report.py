import pandas as pd
import json
import os

csv_file = "/root/autodl-tmp/mimic-cxr-2.0.0-chexpert.csv"
json_file = "/root/autodl-tmp/mimic_cxr_mini/p10_annotation.json"
output_json_file = "/root/autodl-tmp/mimic_cxr_mini/p10_annotation_label_report.json"

disease_columns = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]

# 检查文件是否存在
if not os.path.exists(csv_file) or not os.path.exists(json_file):
    print("文件不存在！")
    exit()

# 读取 CSV 文件
df = pd.read_csv(csv_file)
print("CSV 中的 study_id 示例：", df['study_id'].unique()[:10])

# 生成 label_report
def generate_label_report(row):
    labels = [f"{disease}:{'Positive' if row[disease] == 1.0 else 'Negative'}" for disease in disease_columns]
    return ", ".join(labels) + "."

# 创建 csv_dict
csv_dict = {}
for _, row in df.iterrows():
    study_id = str(row['study_id']).strip()
    if '.' in study_id:
        study_id = study_id.split('.')[0]
    csv_dict[study_id] = generate_label_report(row)

print("csv_dict keys 示例：", list(csv_dict.keys())[:10])
print("是否存在 50414267：", "50414267" in csv_dict)

# 读取并更新 JSON
with open(json_file, 'r') as f:
    json_data = json.load(f)

for split in ["train", "val", "test"]:
    if split in json_data:
        for sample in json_data[split]:
            study_id = str(sample['study_id']).strip()
            sample['label_report'] = csv_dict.get(study_id, "No label available")
            if study_id == "50414267":
                print(f"处理 50414267，找到的 label_report: {sample['label_report']}")

# 保存结果
with open(output_json_file, 'w') as f:
    json.dump(json_data, f, indent=4)
print(f"已生成新文件：{output_json_file}")