import json

with open('/root/autodl-tmp/mimic_cxr_mini/p10_annotation_updated.json') as f:
    data = json.load(f)

for split in ['train', 'val', 'test']:
    print(f"Checking {split} split...")
    for idx, item in enumerate(data[split]):
        if 'labels' not in item:
            print(f"Missing labels in {split} split, index {idx}")
        if len(item['image_path']) == 0:
            print(f"Empty image path in {split} split, index {idx}")