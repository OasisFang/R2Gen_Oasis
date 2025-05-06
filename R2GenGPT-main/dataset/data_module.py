from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import torch
import sys

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable data types and potential errors.
    """
    # Filter out potential problematic items (e.g., None or those missing keys)
    batch = [item for item in batch if item is not None and all(k in item for k in ['id', 'image', 'target_text', 'labels', 'ref'])]

    if not batch:
        # print("Warning: custom_collate_fn received an empty or fully filtered batch.", file=sys.stderr)
        # Return empty tensors with expected shapes/types
        return {
            'id': [],
            'image': torch.empty(0, 3, 224, 224), # Assuming 3x224x224 image tensors
            'target_text': [],
            'labels': torch.empty(0, 14, dtype=torch.float), # Assuming 14 classes, float for BCE loss
            'ref': []
        }

    ids = [item['id'] for item in batch]
    images = [item['image'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    # Labels should already be vectors/tensors from the dataset
    labels = [item['labels'] for item in batch]
    refs = [item['ref'] for item in batch]

    # Stack images and labels
    try:
        # Ensure all images are tensors before stacking
        if all(isinstance(img, torch.Tensor) for img in images):
            images = torch.stack(images)
        else:
             # Find non-tensor elements for debugging
             non_tensors = [type(img) for img in images if not isinstance(img, torch.Tensor)]
             print(f"Error: Not all images in batch are tensors. Found types: {non_tensors}. Skipping batch.", file=sys.stderr)
             # Return empty structure consistent with the 'if not batch' case
             return {
                'id': [], 'image': torch.empty(0, 3, 224, 224), 'target_text': [],
                'labels': torch.empty(0, 14, dtype=torch.float), 'ref': []
             }

        # Ensure all label items are list or tensor before converting
        if all(isinstance(lbl, (list, torch.Tensor)) for lbl in labels):
             # Convert list of lists/tensors to a single tensor
             # Ensure consistent dtype (float for BCEWithLogitsLoss)
             labels = torch.tensor(labels, dtype=torch.float)
        else:
             non_list_tensors = [type(lbl) for lbl in labels if not isinstance(lbl, (list, torch.Tensor))]
             print(f"Error: Not all labels in batch are lists or tensors. Found types: {non_list_tensors}. Skipping batch.", file=sys.stderr)
             return {
                 'id': [], 'image': torch.empty(0, 3, 224, 224), 'target_text': [],
                 'labels': torch.empty(0, 14, dtype=torch.float), 'ref': []
              }

    except Exception as e:
        # Catch potential errors during stacking or tensor conversion
        print(f"Error during collate: {e}. Skipping batch.", file=sys.stderr)
        # Provide details about the batch items that might have caused the error
        # for i, item in enumerate(batch):
        #     print(f"  Item {i}: id={item.get('id')}, image_type={type(item.get('image'))}, label_type={type(item.get('labels'))}")

        return {
            'id': [], 'image': torch.empty(0, 3, 224, 224), 'target_text': [],
            'labels': torch.empty(0, 14, dtype=torch.float), 'ref': []
        }

    # Return the collated batch
    return {'id': ids, 'image': images, 'target_text': target_texts, 'labels': labels, 'ref': refs}


class DataModule(LightningDataModule):
    def __init__(self, dataset, args):
        super().__init__()
        if not isinstance(dataset, dict) or not all(k in dataset for k in ['train', 'val', 'test']):
             raise ValueError("DataModule expects 'dataset' dict with 'train', 'val', 'test' keys.")
        self.dataset = dataset
        self.args = args
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.test_batch_size = args.test_batch_size
        self.num_workers = args.num_workers
        self.prefetch_factor = getattr(args, 'prefetch_factor', 2) # Default prefetch factor

    def train_dataloader(self):
        print(f"Creating train dataloader with batch_size={self.batch_size}, num_workers={self.num_workers}")
        return DataLoader(
            self.dataset["train"], batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=custom_collate_fn, pin_memory=True,
            # Use prefetch_factor only if num_workers > 0
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            # Use persistent_workers only if num_workers > 0
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        print(f"Creating val dataloader with batch_size={self.val_batch_size}, num_workers={self.num_workers}")
        return DataLoader(
            self.dataset["val"], batch_size=self.val_batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=custom_collate_fn, pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        print(f"Creating test dataloader with batch_size={self.test_batch_size}, num_workers={self.num_workers}")
        return DataLoader(
            self.dataset["test"], batch_size=self.test_batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=custom_collate_fn, pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=True if self.num_workers > 0 else False
        )