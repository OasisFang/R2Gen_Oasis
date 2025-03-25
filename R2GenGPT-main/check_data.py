# check_data.py
import json

def validate_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    
    required_keys = ["train", "val", "test"]
    for split in required_keys:
        print(f"检查 {split} 分割...")
        for idx, item in enumerate(data[split]):
            # 必须包含的字段
            assert "image_path" in item, f"{split}[{idx}] 缺少 image_path"
            assert len(item["image_path"]) > 0, f"{split}[{idx}] 图像路径为空"
            
            # 验证集/测试集必须有labels字段
            if split != "train":
                assert "labels" in item, f"{split}[{idx}] 缺少 labels 字段"
                
    print("数据校验通过！")

if __name__ == "__main__":
    validate_json("/root/autodl-tmp/mimic_cxr_mini/p10_annotation_updated.json")