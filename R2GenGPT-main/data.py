import json

# 读取 JSON 文件
json_file_path = '/root/autodl-tmp/mimic_cxr/annotation.json'
output_file_path = '/root/autodl-tmp/mimic_cxr/filtered_annotation.json'

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 过滤包含特定路径部分的条目
filtered_data = {}
target_folder = "p10"  # 目标路径部分

# 遍历 JSON 文件的每个键（例如 'train', 'valid', 'test'）
for key, entries in data.items():
    filtered_entries = []
    # 确保 entries 是列表
    if isinstance(entries, list):
        for entry in entries:
            # 确保 entry 是字典并且包含 'image_path' 键
            if isinstance(entry, dict) and 'image_path' in entry:
                image_paths = entry['image_path']
                # 如果 image_paths 是列表，检查其中是否包含 target_folder
                if isinstance(image_paths, list):
                    if any(target_folder in path for path in image_paths):
                        filtered_entries.append(entry)
                elif isinstance(image_paths, str):
                    if target_folder in image_paths:
                        filtered_entries.append(entry)
    
    # 只在找到匹配条目的情况下保存到 filtered_data
    if filtered_entries:
        filtered_data[key] = filtered_entries

# 保存过滤后的数据到新的 JSON 文件
if filtered_data:
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    print(f"Filtered data saved to {output_file_path}")
else:
    print("No matching data found.")
