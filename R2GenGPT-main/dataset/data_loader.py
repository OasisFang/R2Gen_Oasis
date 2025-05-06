from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import json
import os
import torch
from PIL import Image
from torchvision import transforms

class MimicCXRClassificationDataset(Dataset):
    """
    NOTE: This Dataset class appears unused by the main train.py which uses ParseDataset.
    Keeping it here for completeness, but augmentations were added to ParseDataset.
    """
    def __init__(self, annotation_file, base_dir, transforms=None):
        super().__init__()
        self.base_dir = base_dir
        self.transforms = transforms
        self.samples = []
        try:
            with open(annotation_file, 'r') as f: self.samples = json.load(f)
            print(f"Loaded {len(self.samples)} samples from {annotation_file} (Simple Classification Dataset)")
        except Exception as e: print(f"Error loading annotation {annotation_file} for MimicCXRClassificationDataset: {e}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        if idx >= len(self.samples): raise IndexError(f"Index {idx} out of bounds.")
        data = self.samples[idx]
        image_path = data["image_path"]
        label = data["label"] # Assumes label is a single class index
        try: label = int(label)
        except ValueError: print(f"Error: Cannot convert label '{label}' to int. Using 0."); label = 0
        full_path = os.path.join(self.base_dir, image_path)
        try:
            img = Image.open(full_path).convert('RGB')
            if self.transforms: img = self.transforms(img)
            return img, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error processing image {full_path} for sample {idx} in MimicCXRClassificationDataset: {e}. Returning dummy data.")
            # Ensure dummy tensor matches expected input dimensions
            # This assumes a standard 224x224 input size
            dummy_img_tensor = torch.zeros((3, 224, 224))
            if self.transforms:
                # Apply transforms to dummy tensor if they exist (e.g., normalization)
                 try: dummy_img_tensor = self.transforms(dummy_img_tensor)
                 except: pass # Ignore errors applying transforms to zero tensor
            return dummy_img_tensor, torch.tensor(0, dtype=torch.long)

class SimpleClassificationDataModule(LightningDataModule):
    """
    NOTE: This DataModule appears unused by the main train.py which uses DataModule (from data_module.py).
    Keeping it here for completeness.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Assuming 'annotation' file contains splits or separate files are needed
        # This setup might need adjustment based on how annotations are structured
        self.train_annotation = getattr(args, 'train_annotation', args.annotation)
        self.val_annotation = getattr(args, 'val_annotation', args.annotation)
        self.test_annotation = getattr(args, 'test_annotation', args.annotation)
        self.base_dir = args.base_dir
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = args.num_workers

    def setup(self, stage=None):
        # Standard ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Basic transforms for a classification task
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        # Assuming annotation file points to a JSON containing image paths and single labels
        if stage == 'fit' or stage is None:
            # These paths might need to point to train/val specific annotation files
            self.train_dataset = MimicCXRClassificationDataset(self.train_annotation, self.base_dir, transform_train)
            self.val_dataset = MimicCXRClassificationDataset(self.val_annotation, self.base_dir, transform_eval)
        if stage == 'validate' and not hasattr(self, 'val_dataset'):
            self.val_dataset = MimicCXRClassificationDataset(self.val_annotation, self.base_dir, transform_eval)
        if stage == 'test' or stage is None:
             if not hasattr(self, 'test_dataset'):
                 self.test_dataset = MimicCXRClassificationDataset(self.test_annotation, self.base_dir, transform_eval)

    def train_dataloader(self):
        if not hasattr(self, 'train_dataset'): self.setup(stage='fit')
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True)
    def val_dataloader(self):
        if not hasattr(self, 'val_dataset'): self.setup(stage='fit')
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)
    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'): self.setup(stage='test')
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True)