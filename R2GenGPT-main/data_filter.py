import json

# 读取 JSON 文件
json_file_path = "/root/autodl-tmp/mimic_cxr_mini/p10_annotation_updated_2.json"
with open(json_file_path, "r") as file:
    data = json.load(file)

# 只保留每个 split（train, test, valid）中的前 10 条数据
for split in ["train", "test", "val"]:
    if split in data:
        data[split] = data[split][:11]  # 截取前 10 条

# 保存新的 JSON 文件
filtered_json_path = "/root/autodl-tmp/mimic_cxr_mini/p10_annotation_filtered_Final_1.json"
with open(filtered_json_path, "w") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

# 返回处理完成的 JSON 文件路径
filtered_json_path
