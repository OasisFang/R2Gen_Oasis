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
    # 创建数据集
    train_dataset, dev_dataset, test_dataset = create_datasets(args)
    dataset = {
        "train": train_dataset,
        "val": dev_dataset,
        "test": test_dataset
    }
    
    # 创建数据模块
    dm = DataModule(dataset, args)
    
    # 添加回调函数
    callbacks = add_callbacks(args)
    
    # 创建训练器
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
    
    # 加载模型
    if args.ckpt_file is not None:
        model = R2GenGPT.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = R2GenGPT(args)
    
    # 训练或测试模型
    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)

if __name__ == '__main__':
    main()