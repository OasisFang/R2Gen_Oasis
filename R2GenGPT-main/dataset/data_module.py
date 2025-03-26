from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
import torch

def custom_collate_fn(batch):
    """自定义批处理函数，堆叠数据并添加调试信息"""
    ids = [item['id'] for item in batch]
    images = [item['image'] for item in batch]
    target_texts = [item['target_text'] for item in batch]
    refs = [item['ref'] for item in batch]
    
    # 堆叠图像为张量
    images = torch.stack(images)  # 例如 (batch_size, 3, 224, 224)
    
    # 调试：打印批次信息
    print(f"批次图像形状: {images.shape}")
    print(f"批次参考标签: {refs}")
    
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