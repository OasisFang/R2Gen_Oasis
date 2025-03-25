import json

# 读取 JSON 文件
json_file_path = '/root/autodl-tmp/mimic_cxr/filtered_annotation.json'

with open(json_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 查看 JSON 文件的总体结构
def inspect_json(data, level=0):
    indent = '  ' * level
    if isinstance(data, dict):
        print(f"{indent}字典 ({len(data)} 键):")
        for key, value in data.items():
            print(f"{indent}键: {key} -> 类型: {type(value).__name__}")
            inspect_json(value, level + 1)
            break  # 只打印第一个键值对的内容以示范
    elif isinstance(data, list):
        print(f"{indent}列表 (长度: {len(data)}) -> 元素类型: {type(data[0]).__name__}")
        inspect_json(data[0], level + 1)
    else:
        print(f"{indent}{type(data).__name__}: {data}")

# 开始检查 JSON 文件
print("检查 JSON 文件结构:")
inspect_json(data)

# 打印 JSON 文件中的前50段内容
print("\nJSON 文件中的前50段内容:")
if isinstance(data, list):
    preview_data = data[:50]  # 获取前50个元素
else:
    preview_data = {k: data[k] for k in list(data.keys())[:50]}  # 获取前50个键

print(json.dumps(preview_data, indent=4, ensure_ascii=False))

# # 打印 JSON 文件中的前50段内容
# print("\nJSON 文件中的前50段内容:")
# if isinstance(data, list):
#     preview_data = data[:50]  # 获取前50个元素
# else:
#     preview_data = {k: data[k] for k in list(data.keys())[:50]}  # 获取前50个键

# print(json.dumps(preview_data, indent=4, ensure_ascii=False))
