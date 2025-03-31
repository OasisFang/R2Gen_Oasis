import pandas as pd
import json

# 文件路径
json_file_path = "/root/autodl-tmp/mimic_cxr_mini/p10_annotation.json"    # 原始 JSON 文件路径
csv_file_path = "/root/autodl-tmp/mimic-cxr-2.0.0-chexpert.csv"          # CSV 文件路径
output_json_path = "/root/autodl-tmp/mimic_cxr_mini/p10_annotation_v10.json"  # 输出 JSON 文件路径

# 读取原始 JSON 文件
with open(json_file_path, "r", encoding="utf-8") as json_file:
    json_data = json.load(json_file)

# 读取 CSV 文件
df = pd.read_csv(csv_file_path)

# 定义固定的标签顺序
label_columns = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# 辅助函数：将数值转换为状态值
def value_to_state(value):
    if pd.isna(value):
        return "0"
    elif value == 1.0:
        return "1"
    else:
        return "0"

# 处理 JSON 数据
for split in ["train", "val", "test"]:  # 假设 JSON 有这些分割
    if split in json_data:  # 检查分割是否存在
        for entry in json_data[split]:
            # 获取 JSON 中的 subject_id 和 study_id
            subject_id = entry.get("subject_id")
            study_id = entry.get("study_id")
            
            # 在 CSV 中查找匹配的行
            matching_row = df[(df["subject_id"] == subject_id) & (df["study_id"] == study_id)]
            
            if not matching_row.empty:
                row = matching_row.iloc[0]
                # 提取标签列的数据
                labels_dict = {col: row[col] for col in label_columns if col in df.columns}
                
                # 如果所有标签都是 NaN，则设置 "No Finding" 为 1
                if all(pd.isna(value) for value in labels_dict.values()):
                    labels_dict["No Finding"] = 1.0
                
                # 生成标签字符串
                descriptions = []
                for label in label_columns:
                    value = labels_dict.get(label, None)
                    state = value_to_state(value)
                    description = f"{label}:{state}"
                    descriptions.append(description)
                
                # 更新 JSON 条目的 labels 字段
                entry["labels"] = ", ".join(descriptions)
            else:
                # 如果没有匹配的行，设置默认值
                entry["labels"] = "No matching data found"

# 保存更新后的 JSON 文件
with open(output_json_path, "w", encoding="utf-8") as updated_json_file:
    json.dump(json_data, updated_json_file, indent=4, ensure_ascii=False)

print(f"✅ 处理完成，更新后的 JSON 文件已保存至: {output_json_path}")