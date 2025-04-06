import json

# 1. 读取 JSON 文件
json_file_path = "/root/autodl-tmp/mimic_cxr_mini/p10_annotation_label_report.json"
with open(json_file_path, "r") as file:
    data = json.load(file)

# 2. 处理数据（例如保留每个 split 的前 11 条）
for split in ["train", "test", "val"]:
    if split in data:
        data[split] = data[split][:11]  # 截取前 11 条

# 3. 保存新的 JSON 文件
filtered_json_path = "/root/autodl-tmp/mimic_cxr_mini/p10_annotation_label_report_filtered_1.json"
with open(filtered_json_path, "w") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

# 4. 输出结果
print(f"处理完成，文件保存至：{filtered_json_path}")