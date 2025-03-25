import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from transformers import SwinModel
from torchvision import transforms
import argparse

# ============================ Model Definition ============================
class SwinClassifier(pl.LightningModule):
    """
    Pure Swin Transformer Classifier
    """
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # 初始化Swin模型
        print(f'Loading vision encoder: {args.vision_model}')
        self.swin = SwinModel.from_pretrained(args.vision_model)
        
        # 分类头
        self.num_classes = args.num_classes
        self.classifier = nn.Linear(self.swin.config.hidden_size, self.num_classes)
        
        # 冻结配置
        if args.freeze_backbone:
            self._freeze_swin()
        else:
            print("Training full Swin Transformer model")
            
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 打印可训练参数
        self._print_trainable_params()

    def _freeze_swin(self):
        """冻结Swin主干网络"""
        for param in self.swin.parameters():
            param.requires_grad = False
        print("Frozen Swin backbone parameters")
        
        # 解冻分类头
        for param in self.classifier.parameters():
            param.requires_grad = True

    def _print_trainable_params(self):
        """打印可训练参数"""
        print("\n===== Trainable Parameters =====")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"[Trainable] {name}")
        print("===============================\n")

    def forward(self, x):
        # Swin前向传播
        outputs = self.swin(x)
        pooled_output = outputs.pooler_output  # 使用全局池化输出
        
        # 分类
        logits = self.classifier(pooled_output)
        return logits

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.hparams.max_epochs
        )
        return [optimizer], [scheduler]

# ============================ Data Modules ============================
class MimicCXRClassificationDataset(Dataset):
    def __init__(self, annotation_file, base_dir, transform=None):
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        img_path = os.path.join(self.base_dir, item["image_path"])
        label = item["label"]
        
        # 使用PIL读取图像
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

class MimicCXRDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MimicCXRClassificationDataset(
                annotation_file=os.path.join(self.args.annotation, "train.json"),
                base_dir=self.args.base_dir,
                transform=self.train_transform
            )
            self.val_dataset = MimicCXRClassificationDataset(
                annotation_file=os.path.join(self.args.annotation, "val.json"),
                base_dir=self.args.base_dir,
                transform=self.eval_transform
            )

        if stage == "test" or stage is None:
            self.test_dataset = MimicCXRClassificationDataset(
                annotation_file=os.path.join(self.args.annotation, "test.json"),
                base_dir=self.args.base_dir,
                transform=self.eval_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.args.val_batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True
        )

# ============================ Configuration ============================
def parse_args():
    parser = argparse.ArgumentParser(description="Swin Transformer Classifier")
    
    # Dataset
    parser.add_argument('--annotation', type=str, required=True, 
                       help="Path to annotation directory containing train/val/test.json")
    parser.add_argument('--base_dir', type=str, required=True,
                       help="Base directory for images")
    parser.add_argument('--num_classes', type=int, required=True,
                       help="Number of target classes")
    
    # Model
    parser.add_argument('--vision_model', type=str, default='microsoft/swin-base-patch4-window7-224',
                       help="Pretrained Swin model name")
    parser.add_argument('--freeze_backbone', action='store_true',
                       help="Freeze Swin backbone and only train classifier")
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--max_epochs', type=int, default=50)
    
    # Hardware
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    
    return parser.parse_args()

# ============================ Main ============================
if __name__ == "__main__":
    args = parse_args()
    
    # 初始化数据模块
    dm = MimicCXRDataModule(args)
    
    # 初始化模型
    model = SwinClassifier(args)
    
    # 训练配置
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision="16-mixed",
        enable_checkpointing=True,
        default_root_dir="./checkpoints",
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=3,
                filename="best-{epoch}-{val_acc:.2f}"
            )
        ]
    )
    
    # 开始训练
    trainer.fit(model, datamodule=dm)