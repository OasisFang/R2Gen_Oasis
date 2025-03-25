import argparse

parser = argparse.ArgumentParser(description="Hyper-parameters for Classification Task")



# ========================= Dataset Configs ==========================
parser.add_argument('--test', action='store_true', help="only run test set")
parser.add_argument('--validate', action='store_true', help="only run validation set")
parser.add_argument('--dataset', type=str, default='mimic_cxr', help="iu-xray or mimic-cxr")
parser.add_argument('--annotation', type=str, default=r'/root/autodl-tmp/mimic_cxr_mini/p10_annotation.json', help="annotation file of the dataset")
parser.add_argument('--base_dir', type=str, default=r'/root/autodl-tmp/mimic_cxr_mini/images', help="base dir to help find images")
parser.add_argument('--batch_size', default=6, type=int, help="use for training duration per worker")
parser.add_argument('--val_batch_size', default=16, type=int, help="use for validation duration per worker")
parser.add_argument('--test_batch_size', default=16, type=int, help="use for testing duration per worker")
parser.add_argument('--prefetch_factor', default=4, type=int, help="use for training duration per worker")
parser.add_argument('--num_workers', default=8, type=int, help="Cpu num for dataloaders")

# ========================= Model Settings ============================
parser.add_argument('--vision_model', default='microsoft/swin-base-patch4-window7-224', type=str, help="vision model to use")
# parser.add_argument('--llama_model', default='meta-llama/Llama-2-7b-chat-hf', type=str, help="LLM model to use")
parser.add_argument('--llama_model', default='meta-llama/Llama-3.2-3B-Instruct', type=str, help="LLM model to use")
parser.add_argument('--freeze_vm', default=True, type=lambda x: (str(x).lower() == 'true'), help='freeze vision model')
parser.add_argument('--llm_use_lora', default=False, type=lambda x: (str(x).lower() == 'true'), help="whether use lora for LLM model")
parser.add_argument('--llm_r', default=16, type=int, help='The dimension used by the LoRA update matrices')
parser.add_argument('--llm_alpha', default=16, type=int, help='Scaling factor.')
parser.add_argument('--vis_use_lora', default=False, type=lambda x: (str(x).lower() == 'true'), help="whether use lora for vision model")
parser.add_argument('--vis_r', default=16, type=int, help='The dimension used by the LoRA update matrices')
parser.add_argument('--vis_alpha', default=16, type=int, help='Scaling factor.')
parser.add_argument('--lora_dropout', default=0.1, type=float, help='lora dropout')
parser.add_argument('--global_only', default=False, type=lambda x: (str(x).lower() == 'true'), help='use global embedding only')
parser.add_argument('--low_resource', default=False, type=bool)
parser.add_argument('--end_sym', default='</s>', type=str)

# ======================== SavedModel Configs ===========================
parser.add_argument('--savedmodel_path', type=str, default='/root/autodl-tmp/save/mimic/v1')
parser.add_argument('--ckpt_file', type=str, default=None, help='the checkpoint file to load')
parser.add_argument('--delta_file', type=str, default=None, help='the delta file to load')
parser.add_argument('--weights', type=list, default=[0.5, 0.5])
parser.add_argument('--scorer_types', type=list, default=['Bleu_4', 'CIDEr'])

# ========================= Learning Configs ==========================
parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--gradient_clip_val', default=None, type=int, help='gradient clip value')

# ========================= Decoding Settings ==========================
parser.add_argument('--beam_size', type=int, default=3)
parser.add_argument('--do_sample', type=bool, default=False)
parser.add_argument('--no_repeat_ngram_size', type=int, default=2)
parser.add_argument('--num_beam_groups', type=int, default=1)
parser.add_argument('--min_new_tokens', type=int, default=80)
parser.add_argument('--max_new_tokens', type=int, default=120)
parser.add_argument('--max_length', type=int, default=100)
parser.add_argument('--repetition_penalty', type=float, default=2.0)
parser.add_argument('--length_penalty', type=float, default=2.0)
parser.add_argument('--diversity_penalty', type=float, default=0)
parser.add_argument('--temperature', type=float, default=0)

# ====================== Pytorch Lightning ===========================
parser.add_argument('--devices', type=int, default=2, help='how many gpus to use')
parser.add_argument('--num_nodes', type=int, default=1, help='Number of GPU nodes for distributed training.')
parser.add_argument('--accelerator', type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps"], help='accelerator types')
parser.add_argument('--strategy', type=str, default="ddp", help='default ddp for multi-gpus')
parser.add_argument('--precision', type=str, default='bf16-mixed', help='16 or 32 bf16-mixed, using for original pytorch amp auto cast')
parser.add_argument('--limit_val_batches', type=float, default=1.0, help='How much of validation dataset to check (float = fraction, int = num_batches).')
parser.add_argument('--limit_test_batches', type=float, default=1.0, help='How much of test dataset to check (float = fraction, int = num_batches).')
parser.add_argument('--limit_train_batches', type=float, default=1.0, help='How much of training dataset to check (float = fraction, int = num_batches)')
parser.add_argument('--max_epochs', type=int, default=3, help='Stop training once this number of epochs is reached')
parser.add_argument('--every_n_train_steps', type=int, default=0, help='How many training steps to save a checkpoint')
parser.add_argument('--val_check_interval', type=float, default=1.0, help='How often to check the validation set')
parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates gradients over k batches before stepping the optimizer')
parser.add_argument("--num_sanity_val_steps", type=int, default=2, help='Sanity check runs n validation batches before starting the training routine')

############### 1. 配置文件 ###############


# ============ 数据集相关 ============
parser.add_argument('--train_annotation', type=str, default='train_annotation.json', 
                    help="训练集的注释文件，文件内容形如 [{'image_path':..., 'label':...},...]")
parser.add_argument('--val_annotation', type=str, default='val_annotation.json', 
                    help="验证集的注释文件")
parser.add_argument('--test_annotation', type=str, default='test_annotation.json', 
                    help="测试集的注释文件")
parser.add_argument('--base_dir', type=str, default='/path/to/images', help="图像根目录")
parser.add_argument('--batch_size', type=int, default=8, help="训练batch大小")
parser.add_argument('--val_batch_size', type=int, default=8, help="验证batch大小")
parser.add_argument('--test_batch_size', type=int, default=8, help="测试batch大小")
parser.add_argument('--num_workers', type=int, default=4, help="dataloader的线程数")

# ============ 模型相关 ============
# parser.add_argument('--vision_model', type=str, default='microsoft/swin-base-patch4-window7-224',
#                     help="Swin Transformer模型名称或路径")

parser.add_argument('--freeze_vm', action='store_true', default=False,
                    help='是否冻结视觉模型（不训练Swin）')
parser.add_argument('--num_classes', type=int, default=14, help="分类数")

# ============ 优化器和训练超参数 ============
parser.add_argument('--max_epochs', type=int, default=5, help="最大训练轮数")
parser.add_argument('--learning_rate', type=float, default=1e-4, help="初始学习率")

# parser.add_argument('--savedmodel_path', type=str, default='./checkpoints',
#                     help="保存模型的路径")

# parser.add_argument('--precision', type=str, default='32', help='训练精度(16, 32, bf16等)')

# parser.add_argument('--accelerator', type=str, default='gpu', help='加速器类型(cpu, gpu, etc.)')

# parser.add_argument('--devices', type=int, default=1, help='GPU数量')

# parser.add_argument('--strategy', type=str, default='auto', help='DDP或单机单卡等分布式策略')

# ============ 测试和验证开关 ============

parser.add_argument('--test', action='store_true', help="是否只运行测试集")

parser.add_argument('--validate', action='store_true', help="是否只运行验证集")

parser.add_argument('--ckpt_file', type=str, default=None, help="是否加载已有模型权重")

# args = parser.parse_args()
    
    # return args