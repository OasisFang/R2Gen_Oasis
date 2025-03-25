# model_multilabel.py
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import SwinModel
from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy

class MultiLabelClassifier(pl.LightningModule):
    def __init__(self, args, num_labels=14):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # 1) 加载视觉编码器
        print(f"Loading vision encoder: {args.vision_model}")
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        hidden_dim = self.visual_encoder.num_features  # 一般是1024或768

        # 2) LLaMA模型 (语言生成)
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        self.llama_model = LlamaForCausalLM.from_pretrained(args.llama_model)

        # 3) 分类头 (直接跟LLM输出的Logits对接)
        self.classifier = nn.Linear(hidden_dim, num_labels)

        # 损失函数 (多标签 -> BCE)
        self.criterion = nn.BCEWithLogitsLoss()

        # 评估指标
        self.train_f1 = MultilabelF1Score(num_labels=num_labels, threshold=0.5)
        self.val_f1 = MultilabelF1Score(num_labels=num_labels, threshold=0.5)
        self.test_f1 = MultilabelF1Score(num_labels=num_labels, threshold=0.5)

        self.train_acc = MultilabelAccuracy(num_labels=num_labels, threshold=0.5)
        self.val_acc = MultilabelAccuracy(num_labels=num_labels, threshold=0.5)
        self.test_acc = MultilabelAccuracy(num_labels=num_labels, threshold=0.5)

    def forward(self, x, input_text):
        # 1) 视觉编码器处理图像
        outputs = self.visual_encoder(x)
        pooled_feat = outputs.pooler_output  # shape [batch, hidden_dim]
        
        # 将图像特征与输入的文本拼接
        text_inputs = self.llama_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(x.device)
        text_embeds = self.llama_model.get_input_embeddings()(text_inputs["input_ids"])

        # 拼接图像和文本的embedding作为输入
        combined_inputs = torch.cat([pooled_feat.unsqueeze(1), text_embeds], dim=1)

        # 2) 使用 LLaMA 生成诊断报告
        outputs = self.llama_model(input_ids=text_inputs["input_ids"], 
                                   inputs_embeds=combined_inputs)
        logits = outputs.logits

        return logits

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["labels"]
        input_text = batch["input_text"]

        logits = self(images, input_text)
        loss = self.criterion(logits, labels)

        # 计算指标
        preds = torch.sigmoid(logits)
        self.train_f1.update(preds, labels.int())
        self.train_acc.update(preds, labels.int())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        f1 = self.train_f1.compute()
        acc = self.train_acc.compute()
        self.log("train_f1", f1)
        self.log("train_acc", acc)
        self.train_f1.reset()
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["labels"]
        input_text = batch["input_text"]

        logits = self(images, input_text)
        loss = self.criterion(logits, labels)

        preds = torch.sigmoid(logits)
        self.val_f1.update(preds, labels.int())
        self.val_acc.update(preds, labels.int())

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute()
        acc = self.val_acc.compute()
        self.log("val_f1", f1)
        self.log("val_acc", acc)
        self.val_f1.reset()
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["labels"]
        input_text = batch["input_text"]

        logits = self(images, input_text)
        preds = torch.sigmoid(logits)
        self.test_f1.update(preds, labels.int())
        self.test_acc.update(preds, labels.int())

    def on_test_epoch_end(self):
        f1 = self.test_f1.compute()
        acc = self.test_acc.compute()
        self.log("test_f1", f1)
        self.log("test_acc", acc)
        self.test_f1.reset()
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
