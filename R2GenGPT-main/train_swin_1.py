import os
import json
import torch
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SwinModel
from sklearn.metrics import roc_auc_score
import argparse
import torch.nn as nn

# ========================== 数据预处理配置 ==========================
LABEL_MAP = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Enlarged Cardiomediastinum": 4,
    "Fracture": 5,
    "Lung Lesion": 6,
    "Lung Opacity": 7,
    "No Finding": 8,
    "Pleural Effusion": 9,
    "Pleural Other": 10,
    "Pneumonia": 11,
    "Pneumothorax": 12,
    "Support Devices": 13
}

class MimicCXRDataset(Dataset):
    """多标签胸部X光分类数据集"""
    def __init__(self, annotations, base_dir, transform=None):
        self.annotations = annotations
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # 加载图像
        img_path = os.path.join(
            self.base_dir, 
            item["image_path"][0]  # 取第一个图像路径
        )
        img = Image.open(img_path).convert('RGB')
        
        # 转换标签
        labels = self._process_labels(item["labels"])
        
        # 数据增强
        if self.transform:
            img = self.transform(img)
            
        return img, labels

    def _process_labels(self, labels_dict):
        """处理多标签数据（支持不确定标签）"""
        target = torch.zeros(len(LABEL_MAP), dtype=torch.float32)
        for label_name, value in labels_dict.items():
            idx = LABEL_MAP[label_name]
            if value == 1.0:    # 阳性
                target[idx] = 1
            elif value == -1.0: # 不确定
                target[idx] = 0.5
            # null保持0
        return target

# ========================== 数据模块 ==========================
class MimicCXRDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # 数据增强配置
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        # 加载标注文件
        with open(self.args.annotation_file) as f:
            full_data = json.load(f)
        
        # 分割数据集
        if stage == "fit" or stage is None:
            self.train_dataset = MimicCXRDataset(
                full_data["train"],
                self.args.image_dir,
                self.train_transform
            )
            self.val_dataset = MimicCXRDataset(
                full_data["val"],
                self.args.image_dir,
                self.eval_transform
            )
            
        if stage == "test" or stage is None:
            self.test_dataset = MimicCXRDataset(
                full_data["test"],
                self.args.image_dir,
                self.eval_transform
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
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

# ========================== 模型定义 ==========================
class SwinClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        
        # 用于保存验证步骤输出的列表
        self.val_outputs = []
        
        # Swin Transformer主干网络
        self.swin = SwinModel.from_pretrained(args.vision_model)
        
        # 冻结主干网络（可选）
        if args.freeze_backbone:
            for param in self.swin.parameters():
                param.requires_grad = False
            print("冻结Swin主干网络参数")
        
        # 修正后的分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.swin.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, len(LABEL_MAP))  # 输出层
        )
        
        # 损失函数
        self.pos_weight = torch.tensor([2.0]*len(LABEL_MAP))
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def forward(self, x):
        features = self.swin(x).pooler_output
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # 计算AUC
        preds = torch.sigmoid(logits)
        auc = self._calculate_auc(preds, labels)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_auc", auc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.sigmoid(logits)
        result = {"val_loss": loss, "preds": preds, "labels": labels}
        self.val_outputs.append(result)
        return result

    # 使用 on_validation_epoch_end 替代 validation_epoch_end
    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return
        
        preds = torch.cat([x["preds"] for x in self.val_outputs]).cpu().numpy()
        labels = torch.cat([x["labels"] for x in self.val_outputs]).cpu().numpy()
        
        auc = roc_auc_score(labels, preds, average='macro')
        avg_loss = torch.mean(torch.stack([x["val_loss"] for x in self.val_outputs]))
        
        self.log("val_auc", auc, prog_bar=True)
        self.log("val_loss", avg_loss)
        
        # 清空保存的输出
        self.val_outputs.clear()

    def _calculate_auc(self, preds, labels):
        try:
            return roc_auc_score(
                labels.cpu().numpy(),
                preds.cpu().numpy(),
                average='macro'
            )
        except ValueError:
            return 0.5  # 处理全正样本或全负样本的情况

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            ),
            'monitor': 'val_auc'
        }
        return [optimizer], [scheduler]

# ========================== 配置参数 ==========================
def parse_args():
    parser = argparse.ArgumentParser(description="Swin Transformer多标签分类训练")
    
    # 数据参数
    parser.add_argument('--annotation_file', type=str, 
                        default='/root/autodl-tmp/mimic_cxr_mini/p10_annotation_filtered.json')
    parser.add_argument('--image_dir', type=str,
                        default='/root/autodl-tmp/mimic_cxr_mini/images')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # 模型参数
    parser.add_argument('--vision_model', type=str,
                        default='microsoft/swin-base-patch4-window7-224')
    parser.add_argument('--freeze_backbone', action='store_true')
    
    # 训练参数
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=30)
    
    # 训练器参数
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    
    return parser.parse_args()

# ========================== 主程序 ==========================
if __name__ == "__main__":
    args = parse_args()
    
    # 初始化数据模块
    dm = MimicCXRDataModule(args)
    
    # 初始化模型
    model = SwinClassifier(args)
    
    # 训练回调
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_auc",
        mode="max",
        save_top_k=3,
        filename="best-{epoch}-{val_auc:.2f}"
    )
    
    # 配置训练器
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        log_every_n_steps=10,
        deterministic=True
    )
    
    # 开始训练
    trainer.fit(model, datamodule=dm)
