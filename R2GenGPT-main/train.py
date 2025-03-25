# train_multilabel.py
import argparse
import lightning.pytorch as pl
from model_multilabel import MultiLabelClassifier
from data_module.data_module_multilabel import MultiLabelDataModule

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', type=str, required=True)
    parser.add_argument('--base_dir', type=str, required=True)
    parser.add_argument('--vision_model', type=str, default='microsoft/swin-base-patch4-window7-224')
    parser.add_argument('--freeze_vm', default=False, type=lambda x: (str(x).lower()=='true'))

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    # 其它Lightning参数
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    # ...
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    dm = MultiLabelDataModule(args)
    dm.setup()

    model = MultiLabelClassifier(args)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
    )

    # 训练
    trainer.fit(model, datamodule=dm)
    # 测试
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    main()
