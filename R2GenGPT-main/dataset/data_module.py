from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import torch

def custom_collate_fn(batch):
    """Custom collate function to stack data and add debug information"""
    ids = [item['id'] for item in batch]
    images = [item['image'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    refs = [item['ref'] for item in batch]
    
    # Stack images into a tensor
    images = torch.stack(images)  # e.g., (batch_size, 3, 224, 224)
    
    # Debug: Print batch information
    print(f"Batch image shape: {images.shape}")
    print(f"Batch refs: {refs}")
    
    return {
        'id': ids,
        'image': images,
        'target_text': target_texts,
        'ref': refs
    }

class DataModule(LightningDataModule):
    def __init__(self, dataset, args):
        super().__init__()
        self.dataset = dataset
        self.args = args

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["val"],
            batch_size=self.args.val_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=custom_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.args.test_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=custom_collate_fn
        )