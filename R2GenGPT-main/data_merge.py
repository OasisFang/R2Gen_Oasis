import pandas as pd
import json
import numpy as np

# 文件路径
csv_file_path = "/root/autodl-tmp/mimic-cxr-2.0.0-chexpert.csv"
json_file_path = "/root/autodl-tmp/mimic_cxr/mimic_annotation_all.json"
updated_json_path = "/root/autodl-tmp/mimic_cxr/mimic_cxr_final.json"

# 读取 CSV 文件，并将 NaN 和 -1.0 替换为 None
csv_data = pd.read_csv(csv_file_path)
csv_data.replace({np.nan: None, -1.0: None}, inplace=True)

# 读取 JSON 文件
with open(json_file_path, "r") as json_file:
    json_data = json.load(json_file)

# 以 (subject_id, study_id) 作为键，构建字典存储 CSV 数据中的所有标签信息
csv_data.set_index(["subject_id", "study_id"], inplace=True)
study_label_map = csv_data.to_dict(orient="index")

# 遍历 JSON 数据，对 train、val 和 test 分片进行标签信息的添加
for split in ["train", "val", "test"]:
    for entry in json_data[split]:
        key = (entry["subject_id"], entry["study_id"])
        if key in study_label_map:
            labels = study_label_map[key]
            # 检查所有标签是否都是 None
            if all(value is None for value in labels.values()):
                labels["No Finding"] = 1.0
            entry["labels"] = labels
        else:
            entry["labels"] = {} 

# 保存更新后的 JSON 文件，并确保 NaN 变为 null
with open(updated_json_path, "w", encoding="utf-8") as updated_json_file:
    json.dump(json_data, updated_json_file, indent=4, ensure_ascii=False)

print(f"✅ 处理完成，更新后的 JSON 文件已保存至: {updated_json_path}")