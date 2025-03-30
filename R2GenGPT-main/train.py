import os
from pprint import pprint
from configs.config import parser
from dataset.data_helper import create_datasets
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from models.R2GenGPT import R2GenGPT
from lightning.pytorch import seed_everything
import lightning.pytorch as pl

def train(args):
    # Create datasets
    train_dataset, dev_dataset, test_dataset = create_datasets(args)
    dataset = {
        "train": train_dataset,
        "val": dev_dataset,
        "test": test_dataset
    }
    
    # Create data module
    dm = DataModule(dataset, args)
    
    # Add callbacks
    callbacks = add_callbacks(args)
    
    # Create trainer
    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        max_epochs=args.max_epochs,
        num_sanity_val_steps=args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )
    
    # Load model
    if args.ckpt_file is not None:
        model = R2GenGPT.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = R2GenGPT(args)
    

    if args.test:
        print("正在运行测试模式...")

    if args.save_images:
        print("正在保存图像...")
        # 示例保存逻辑（根据实际需求替换）
        save_path = "/root/autodl-tmp/save/mimic_cxr/v5_test"
        import os
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "example_image.txt"), "w") as f:
            f.write("这是一个示例图像保存文件\n")

    # 其他训练或测试逻辑
    print(f"批量大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
        
def main():
    args = parser.parse_args()
    # 添加 save_images 参数到 args
    if not hasattr(args, 'save_images'):
        args.save_images = False
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)

if __name__ == '__main__':
    main()