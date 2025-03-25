# data_helper_multilabel.py
import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor

# 定义14个标签的固定顺序
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
    "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
    "Pneumothorax", "Support Devices"
]

class FieldParser:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0]

    def create_label_text(self, label_dict):
        """
        将14个疾病标签转换为一串固定格式文本，格式如:
        "Atelectasis: absent; Cardiomegaly: present; ..."
        """
        texts = []
        for disease in DISEASES:
            val = label_dict.get(disease, None)
            # val == 1.0 => present, 否则 => absent
            if val == 1.0:
                state = "present"
            else:
                state = "absent"
            texts.append(f"{disease}: {state}")
        # 用分号分隔
        label_text = "; ".join(texts)
        return label_text

    def parse(self, features):
        to_return = {}
        to_return['id'] = features['id']

        # 1) 生成标签文本
        label_text = self.create_label_text(features.get("labels", {}))

        # 2) 直接用 label_text，忽略报告部分
        to_return['input_text'] = label_text

        # 3) 处理图像
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path)) as pil:
                array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images

        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class ParseDataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])


def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset
