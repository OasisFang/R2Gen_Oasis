# data_module_multilabel.py
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from data_helper_multilabel import create_datasets

class MultiLabelDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset, self.test_dataset = create_datasets(
            annotation_file=self.args.annotation,
            base_dir=self.args.base_dir,
            vision_model=self.args.vision_model
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.val_batch_size,
            num_workers=self.args.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            shuffle=False
        )
