import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import precision_recall_fscore_support
import torchvision.utils  # 用于保存图像

class R2GenGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        
        # 加载视觉编码器
        print(f'Loading vision encoder: {args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loaded vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loaded frozen vision encoder: {args.vision_model} -- Done')
        else:
            print(f'Loaded trainable vision encoder: {args.vision_model} -- Done')
        
        # 加载 LLaMA 模型
        print('Loading LLAMA model')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )
        
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loaded LLAMA model with LoRA -- Done')
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loaded LLAMA model -- Done')
        
        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        
        # 根据任务选择提示词
        if args.task == 'classification':
            self.prompt = 'This is a 14-class classification task related to medical imaging, where the 14 disease categories are as follows: Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax,Support Devices. Please identify the dieases with the name'
        else:
            self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        
        # 加载检查点（如果提供）
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Loaded checkpoint from {args.delta_file}')

    def score(self, ref, hypo):
        """计算评估指标"""
        if self.hparams.task == 'report':
            scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr")
            ]
            final_scores = {}
            for scorer, method in scorers:
                score, scores = scorer.compute_score(ref, hypo)
                if type(score) == list:
                    for m, s in zip(method, score):
                        final_scores[m] = s
                else:
                    final_scores[method] = score
            return final_scores
        elif self.hparams.task == 'classification':
            disease_list = ['Atelectasis", "Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
                            'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
                            'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding']
            all_true = [self._parse_labels(r[0]) for r in ref.values()]
            all_predicted = [self._parse_labels(h[0]) for h in hypo.values()]
            true_vectors = [[1 if d in true else 0 for d in disease_list] for true in all_true]
            pred_vectors = [[1 if d in pred else 0 for d in disease_list] for pred in all_predicted]
            precision, recall, f1, _ = precision_recall_fscore_support(true_vectors, pred_vectors, average='micro', zero_division=0)
            return {'Precision': precision, 'Recall': recall, 'F1': f1}

    def _parse_labels(self, text):
        """解析疾病标签"""
        text = text.strip('.').replace('No diseases detected', '')
        labels = [label.strip() for label in text.split(',') if label.strip()]
        valid_labels = {
                        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                        "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
                        "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
                        }
        return [label for label in labels if label in valid_labels]

    def encode_img(self, images):
        """编码图像"""
        device = images.device  
        if self.hparams.global_only:
            image_embeds = self.visual_encoder(images)['pooler_output'].unsqueeze(1).to(device)
        else:
            image_embeds = self.visual_encoder(images)['last_hidden_state'].to(device)
        inputs_llama = self.llama_proj(image_embeds)  
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        """将图像嵌入与提示词结合"""
        prompt = f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        """前向传播"""
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["target_text"]]
        
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image.device)
        
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )
        
        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)
        
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        
        # 保存训练集中的图片和 ref text（每个 epoch 的第一个 batch）
        if batch_idx == 0:
            image_folder = os.path.join(self.hparams.savedmodel_path, 'images', 'train', f'epoch_{self.current_epoch}')
            os.makedirs(image_folder, exist_ok=True)
            for idx, (image, ref) in enumerate(zip(batch['image'], batch['ref'])):
                if idx >= 10:  # 只保存前 10 个样本
                    break
                image_path = os.path.join(image_folder, f"image_{idx}.png")
                # 归一化图像到 [0, 1]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                torchvision.utils.save_image(image, image_path)
                with open(os.path.join(image_folder, f"text_{idx}.txt"), 'w') as f:
                    f.write(f"Reference: {ref}\n")
        
        return result

    def save_checkpoint(self, eval_res):
        """保存检查点"""
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        if self.hparams.task == 'report':
            save_to = os.path.join(
                self.hparams.savedmodel_path, 'checkpoints',
                "checkpoint_epoch{}_step{}_bleu{:.3f}_cider{:.3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr'])
            )
        else:
            save_to = os.path.join(
                self.hparams.savedmodel_path, 'checkpoints',
                "checkpoint_epoch{}_step{}_f1{:.3f}.pth".format(current_epoch, global_step, eval_res['F1'])
            )
        self.print("Saving checkpoint at step {} to {}".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        """验证步骤"""
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['target_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )
        
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)
        
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"], "image": samples["image"]})
        return hypo, ref

    def decode(self, output_token):
        """解码输出"""
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        if self.hparams.task == 'classification':
            valid_labels = {
                                "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                                "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
                                "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
                            }

            labels = [word.strip() for word in output_text.split(',') if word.strip() in valid_labels]
            return ', '.join(labels) if labels else 'No finding'
        return output_text

    def on_validation_epoch_end(self):
        """验证结束"""
        ref, hypo, ids, images = [], [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])
            images.extend(i['image'])
        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)
        
        # 显示 Precision, Recall, F1
        if self.hparams.task == 'classification':
            precision = eval_res['Precision']
            recall = eval_res['Recall']
            f1 = eval_res['F1']
            self.print(f"Epoch {self.trainer.current_epoch} - Validation Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)
        
        # 保存评估指标
        with open(os.path.join(result_folder, f"metrics_epoch_{current_epoch}.json"), 'w') as f:
            json.dump(eval_res, f)
        
        val_score = 0
        if self.hparams.task == 'report':
            for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
                val_score += eval_res[score_type] * weight
        else:
            val_score = eval_res['F1']
        
        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        
        # 保存图像和文本对比（每个 epoch 保存 10 个样本）
        if self.hparams.save_images:
            image_folder = os.path.join(self.hparams.savedmodel_path, 'images', f'epoch_{current_epoch}')
            os.makedirs(image_folder, exist_ok=True)
            for idx, (image, r, h) in enumerate(zip(images[:10], list(ref.values())[:10], list(hypo.values())[:10])):
                image_path = os.path.join(image_folder, f"image_{idx}.png")
                # 归一化图像到 [0, 1]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                torchvision.utils.save_image(image, image_path)
                with open(os.path.join(image_folder, f"text_{idx}.txt"), 'w') as f:
                    f.write(f"Reference: {r[0]}\n")
                    f.write(f"Generated: {h[0]}\n")
        
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        """测试步骤"""
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['target_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )
        
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)
        
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"], "image": samples["image"]})
        return hypo, ref

    def on_test_epoch_end(self):
        """测试结束"""
        ref, hypo, ids, images = [], [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])
            images.extend(i['image'])
        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self._save_test_results(hypo, ref, eval_res)
        
        if self.hparams.save_images:
            result_folder = os.path.join(self.hparams.savedmodel_path, 'images', 'test')
            os.makedirs(result_folder, exist_ok=True)
            for idx, (image, r, h) in enumerate(zip(images[:10], list(ref.values())[:10], list(hypo.values())[:10])):
                image_path = os.path.join(result_folder, f"image_{idx}.png")
                # 归一化图像到 [0, 1]
                image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                torchvision.utils.save_image(image, image_path)
                with open(os.path.join(result_folder, f"text_{idx}.txt"), 'w') as f:
                    f.write(f"Reference: {r[0]}\n")
                    f.write(f"Generated: {h[0]}\n")
        
        self.test_step_outputs.clear()

    def _save_test_results(self, hypo, ref, eval_res):
        """保存测试结果"""
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, "test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        """获取进度条信息"""
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        """优化器梯度清零"""
        optimizer.zero_grad()