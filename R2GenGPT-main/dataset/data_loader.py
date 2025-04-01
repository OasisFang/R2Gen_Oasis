from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from dataset.data_helper import create_datasets

############### 2. 数据集与DataModule ###############
class MimicCXRClassificationDataset(Dataset):
    """
    假设注释文件结构大致如下:
    [
      {"image_path": "p10/xxxx.png", "label": 0},
      {"image_path": "p11/yyyy.png", "label": 1},
      ...
    ]
    """
    def __init__(self, annotation_file, base_dir, transforms=None):
        super().__init__()
        self.base_dir = base_dir
        self.transforms = transforms

        with open(annotation_file, 'r') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        image_path = data["image_path"]
        label = data["label"]  # int

        full_path = os.path.join(self.base_dir, image_path)
        # 这里简单用PIL读取图像做示例，也可自行换成其他读取方式
        from PIL import Image
        img = Image.open(full_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img, torch.tensor(label, dtype=torch.long)


class MimicCXRDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage=None):
        """
        在这里初始化训练、验证、测试集
        """
        from torchvision import transforms
        # 简单的图像增强和预处理
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        # 训练集
        if stage == 'fit' or stage is None:
            self.train_dataset = MimicCXRClassificationDataset(
                annotation_file=self.args.train_annotation,
                base_dir=self.args.base_dir,
                transforms=transform_train
            )
            self.val_dataset = MimicCXRClassificationDataset(
                annotation_file=self.args.val_annotation,
                base_dir=self.args.base_dir,
                transforms=transform_eval
            )

        # 验证集
        if stage == 'validate' or stage is None:
            self.val_dataset = MimicCXRClassificationDataset(
                annotation_file=self.args.val_annotation,
                base_dir=self.args.base_dir,
                transforms=transform_eval
            )

        # 测试集
        if stage == 'test' or stage is None:
            self.test_dataset = MimicCXRClassificationDataset(
                annotation_file=self.args.test_annotation,
                base_dir=self.args.base_dir,
                transforms=transform_eval
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.args.batch_size,
                          num_workers=self.args.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.args.val_batch_size,
                          num_workers=self.args.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.args.test_batch_size,
                          num_workers=self.args.num_workers,
                          shuffle=False)