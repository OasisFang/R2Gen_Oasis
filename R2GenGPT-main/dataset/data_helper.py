# dataset/data_helper.py
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import AutoImageProcessor

class FieldParser:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.vis_processor = AutoImageProcessor.from_pretrained(args.vision_model)
        
    def clean_report(self, report):
        # 保持原有清洗逻辑不变
        if self.dataset == "mimic_cxr":
            report = re.sub(r'\n', ' ', report)
            report = re.sub(r'\s{2,}', ' ', report)
            report = report.strip().lower()
        return report

    def parse(self, features):
        return {
            "id": features["id"],
            "input_text": self.clean_report(features.get("report", "")),
            "image": [self._process_image(p) for p in features["image_path"]],
            "image_paths": features["image_path"]  # 新增原始路径字段
        }

    def _process_image(self, image_path):
        with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
            img = np.array(pil.convert("RGB"), dtype=np.uint8)
            return self.vis_processor(img, return_tensors="pt").pixel_values[0]

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        with open(args.annotation, 'r') as f:
            self.meta = json.load(f)[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.parse(self.meta[index])

def create_datasets(args):
    return (
        ParseDataset(args, 'train'),
        ParseDataset(args, 'val'),
        ParseDataset(args, 'test')
    )