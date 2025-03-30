import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor

class FieldParser:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)

    def _parse_image(self, img):
        """Convert image to tensor"""
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0]

    def clean_report(self, report):
        """Clean medical report text"""
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
                .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
                .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '') \
                                            .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '') \
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        return report

    def parse(self, features):
        """Parse features of a single sample"""
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        images = []
        for image_path in features['image_path']:
            full_path = os.path.join(self.args.base_dir, image_path)
            try:
                with Image.open(full_path) as pil:
                    array = np.array(pil, dtype=np.uint8)
                    if array.shape[-1] != 3 or len(array.shape) != 3:
                        array = np.array(pil.convert("RGB"), dtype=np.uint8)
                    image = self._parse_image(array)
                    images.append(image)
            except Exception as e:
                print(f"Error loading image {full_path}: {e}")
                default_image = np.zeros((224, 224, 3), dtype=np.uint8)
                image = self._parse_image(default_image)
                images.append(image)
        # Only return the first image
        to_return["image"] = images[0] if images else None
        if to_return["image"] is None:
            print(f"Warning: No image for sample {features['id']}")
            default_image = np.zeros((224, 224, 3), dtype=np.uint8)
            to_return["image"] = self._parse_image(default_image)
        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        """Get a single sample and add debug information"""
        features = self.meta[index]
        parsed_data = self.parser.transform_with_parse(features)
        
        report = parsed_data['input_text']
        labels_dict = features.get('labels', {})
        labels = [disease for disease, value in labels_dict.items() if value == 1.0]
        
        if self.args.task == 'report':
            target_text = report
            ref = report
        elif self.args.task == 'classification':
            target_text = ', '.join(labels) + '.' if labels else 'No diseases detected.'
            ref = labels
        
        image = parsed_data["image"]
        if image is None:
            print(f"Warning: No image for sample {features['id']}")
            default_image = np.zeros((224, 224, 3), dtype=np.uint8)
            image = self.parser._parse_image(default_image)
        
        # Debug: Print sample information
        print(f"Sample {parsed_data['id']}: image shape={image.shape}, ref={ref}")
        
        return {
            'id': parsed_data['id'],
            'image': image,
            'target_text': target_text,
            'ref': ref
        }

def create_datasets(args):
    """Create train, validation, and test datasets"""
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset