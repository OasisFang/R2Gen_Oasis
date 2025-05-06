from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger  # Import TensorBoardLogger
import os

def add_callbacks(args):
    """Add callbacks and loggers for training."""
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/F1' if args.task == 'classification' else 'val/CIDEr',
        dirpath=os.path.join(args.savedmodel_path, 'checkpoints'),
        filename='checkpoint-{epoch:02d}-{val/F1:.4f}' if args.task == 'classification' else 'checkpoint-{epoch:02d}-{val/CIDEr:.4f}',
        save_top_k=1,
        mode='max',
    )
    early_stop_callback = EarlyStopping(
        monitor='val/F1' if args.task == 'classification' else 'val/CIDEr',
        patience=3,
        mode='max',
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')  # Monitor learning rate per step
    
    # Define logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(args.savedmodel_path, 'logs'),
        name='training_logs'
    )
    
    # Return both callbacks and logger
    return {
        "callbacks": [checkpoint_callback, early_stop_callback, lr_monitor],
        "loggers": logger
    }