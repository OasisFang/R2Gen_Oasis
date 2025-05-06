import os
import json
import re
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
from transformers import AutoImageProcessor
from torchvision import transforms # <-- 引入 transforms
import random # <-- 引入 random

class FieldParser:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        try:
            # Keep the feature extractor for final processing
            self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)
            # Get expected image size from the processor config if possible
            self.image_size = (
                self.vit_feature_extractor.size['height'],
                self.vit_feature_extractor.size['width']
            ) if isinstance(self.vit_feature_extractor.size, dict) and 'height' in self.vit_feature_extractor.size else (224, 224)
            print(f"Using image size from feature extractor: {self.image_size}")
        except Exception as e:
            print(f"Error loading AutoImageProcessor for {args.vision_model}: {e}")
            self.image_size = (224, 224) # Default fallback size
            print(f"Falling back to default image size: {self.image_size}")
            # Raise only if feature extractor loading truly fails
            if 'self.vit_feature_extractor' not in locals():
                 raise

    def _parse_image(self, pil_image):
        """Processes a PIL image using the Hugging Face processor."""
        try:
            # The processor handles resizing, normalization, and conversion to tensor
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert("RGB")
            # Note: The processor applies its own normalization based on the pretrained model
            pixel_values = self.vit_feature_extractor(pil_image, return_tensors="pt").pixel_values
            return pixel_values[0] # Remove batch dimension
        except Exception as e:
            print(f"Error parsing image with feature extractor: {e}")
            # Return a zero tensor with the correct expected shape
            return torch.zeros((3, self.image_size[0], self.image_size[1]))

    def clean_report(self, report):
        if not isinstance(report, str): return ""
        report = report.replace('\n', ' ')
        report = re.sub(r'\s+', ' ', report)
        report = re.sub(r'\.{2,}', '.', report)
        report = re.sub(r'FINDINGS:', '', report, flags=re.IGNORECASE)
        report = re.sub(r'IMPRESSION:', '.', report, flags=re.IGNORECASE)
        report = report.strip().lower()
        sentences = report.split('.')
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            sent = re.sub(r'^\d+\.\s*', '', sent)
            # Keep '.' in allowed characters if reports use it meaningfully
            sent = re.sub(r'[^\w\s\.]', '', sent) # Allow letters, numbers, whitespace, periods
            if sent: cleaned_sentences.append(sent)
        report = ' . '.join(cleaned_sentences)
        # Only add final period if report isn't empty and doesn't already end with one
        if report and not report.endswith('.'): report += ' .'
        return report

    # Removed parse method, its logic is moved to ParseDataset.__getitem__
    # def parse(self, features): ...

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.split = split
        self.meta = []
        try:
            with open(args.annotation, 'r') as f: full_data = json.load(f)
            if split not in full_data: raise KeyError(f"Split '{split}' not found in {args.annotation}")
            self.meta = full_data[split]
            print(f"Loaded {split} dataset with {len(self.meta)} samples from {args.annotation}")
        except FileNotFoundError:
             print(f"Error: Annotation file not found at {args.annotation}")
             raise
        except Exception as e: print(f"Error loading annotations for {split} from {args.annotation}: {e}"); raise
        if not self.meta: print(f"Warning: Dataset split '{split}' is empty.")

        self.parser = FieldParser(args) # Keep parser for its _parse_image and clean_report methods
        self.diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
        self.num_classes = len(self.diseases)

        # --- Data Augmentation Setup ---
        # Define the classes to augment based on user input
        self.minority_classes = [
            "Edema", "Pneumonia", "Pneumothorax", "Lung Lesion",
            "Consolidation", "Fracture", "Enlarged Cardiomediastinum", "Pleural Other"
        ]
        # Get indices of these classes in the self.diseases list
        self.minority_class_indices = {i for i, disease in enumerate(self.diseases) if disease in self.minority_classes}
        print(f"Will apply augmentation to classes: {self.minority_classes} (Indices: {self.minority_class_indices}) for '{split}' split.")

        # Define augmentation transforms for TRAIN set
        # Note: Resizing/Normalization is handled by the vit_feature_extractor later
        self.train_transform = transforms.Compose([
            transforms.RandomRotation(15), # Rotate by up to 15 degrees
            transforms.RandomHorizontalFlip(p=0.5), # Flip horizontally with 50% probability
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Adjust brightness/contrast
            # Add more transforms if needed, e.g., RandomAffine for small shifts/shear
            # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])

        # Define augmentation transforms for VAL set (less aggressive)
        self.val_transform = transforms.Compose([
             transforms.RandomHorizontalFlip(p=0.5),
             # Potentially add a very mild rotation or keep it simple
             # transforms.RandomRotation(5),
        ])
        # --- End Data Augmentation Setup ---

    def __len__(self): return len(self.meta)

    def parse_label_report(self, label_report_str):
        labels_vector = [0] * self.num_classes
        if not isinstance(label_report_str, str) or not label_report_str.strip(): return labels_vector
        labels_dict = {}
        try:
            # Handle potential variations like extra spaces or missing commas
            parts = re.split(r'\s*,\s*', label_report_str.strip())
            for part in parts:
                if ':' in part:
                    disease, label = part.split(':', 1)
                    disease, label = disease.strip(), label.strip().lower()
                    if disease in self.diseases and label == 'positive': labels_dict[disease] = 1
        except Exception as e: print(f"Warning: Error parsing label_report '{label_report_str[:50]}...': {e}")
        for i, disease in enumerate(self.diseases): labels_vector[i] = labels_dict.get(disease, 0)
        return labels_vector

    def __getitem__(self, index):
        if index >= len(self.meta): raise IndexError("Index out of bounds")
        features = self.meta[index]
        item_id = features.get('id', f"{self.split}_{index}") # Get ID or create one

        # --- Image Loading ---
        pil_img = None
        image_paths = features.get('image_path', [])
        if not isinstance(image_paths, list): image_paths = [image_paths]

        for image_path in image_paths:
            if not image_path: continue
            full_path = os.path.join(self.args.base_dir, image_path)
            try:
                pil_img = Image.open(full_path)
                # Ensure image has 3 channels (RGB) before potential augmentation
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert("RGB")
                break # Use the first valid image found
            except FileNotFoundError: print(f"Warning: Image file not found {full_path}")
            except Exception as e: print(f"Error loading image {full_path}: {e}")

        # Handle case where no valid image was loaded
        if pil_img is None:
            print(f"Warning: No valid image found for sample {item_id}. Returning default tensor.")
            # Use the parser's method to get a consistent default tensor
            h, w = self.parser.image_size
            image_tensor = torch.zeros((3, h, w))
            # Set dummy values for other fields to avoid errors in collate_fn
            label_report_str = ""
            labels_vector = [0] * self.num_classes
            llm_target_text = ""
            ref_text = ""
            return {'id': item_id, 'image': image_tensor, 'target_text': llm_target_text, 'labels': labels_vector, 'ref': ref_text}
        # --- End Image Loading ---

        # --- Label Processing ---
        label_report_str = features.get('label_report', '')
        labels_vector = self.parse_label_report(label_report_str)
        # --- End Label Processing ---

        # --- Apply Conditional Augmentation ---
        is_minority = any(labels_vector[i] == 1 for i in self.minority_class_indices)

        augmented_pil_img = pil_img # Start with the original image

        if is_minority:
            if self.split == 'train':
                # print(f"Applying TRAIN augmentation to {item_id} (minority)") # Optional: for debugging
                augmented_pil_img = self.train_transform(pil_img)
            elif self.split == 'val':
                # print(f"Applying VAL augmentation to {item_id} (minority)") # Optional: for debugging
                augmented_pil_img = self.val_transform(pil_img)
        # Note: No augmentation applied if not minority or if split is 'test'
        # --- End Apply Conditional Augmentation ---

        # --- Final Image Parsing (Tensor Conversion, Normalization) ---
        image_tensor = self.parser._parse_image(augmented_pil_img)
        # --- End Final Image Parsing ---

        # --- Text Processing ---
        if self.args.task == 'classification':
            # Use the original label report string for classification target/ref
            llm_target_text = label_report_str
            ref_text = label_report_str
        elif self.args.task == 'report':
            report_str = features.get("report", "")
            llm_target_text = self.parser.clean_report(report_str)
            ref_text = llm_target_text # Use cleaned report for generation tasks
        else:
            llm_target_text = ""
            ref_text = ""
        # --- End Text Processing ---

        # Return the final dictionary
        return {
            'id': item_id,
            'image': image_tensor,
            'target_text': llm_target_text,
            'labels': labels_vector, # Return the vector of 0s and 1s
            'ref': ref_text
        }

def create_datasets(args):
    """Creates train, validation, and test datasets using ParseDataset."""
    print("Creating datasets using ParseDataset...")
    try:
        train_dataset = ParseDataset(args, 'train')
        dev_dataset = ParseDataset(args, 'val')
        test_dataset = ParseDataset(args, 'test')
        print("Datasets created successfully.")
        return train_dataset, dev_dataset, test_dataset
    except Exception as e:
        print(f"Failed to create datasets: {e}")
        raise