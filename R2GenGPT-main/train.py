import os
import sys
from pprint import pprint
import traceback # Import traceback for detailed error printing

# --- Configuration Import ---
# Try importing from root first, then from potential subdirectories
try:
    from config import parser
except ModuleNotFoundError:
    try:
        # Example: if config.py is in a 'configs' subdirectory
        from configs.config import parser
        print("Imported parser from configs.config")
    except ModuleNotFoundError:
        print("Error: Cannot find 'config.py'. Make sure it's in the project root or accessible in PYTHONPATH.", file=sys.stderr)
        sys.exit(1)

# --- Dataset Modules Import ---
try:
    # Assuming data helper/module are in a 'dataset' subdirectory
    from dataset.data_helper import create_datasets
    from dataset.data_module import DataModule
except ModuleNotFoundError:
    # Try importing from root if not found in 'dataset'
    try:
        from data_helper import create_datasets
        from data_module import DataModule
        print("Imported dataset modules from project root.")
    except ModuleNotFoundError:
        print("Error: Cannot find dataset modules ('data_helper.py', 'data_module.py'). Ensure they are in the root or a 'dataset' directory.", file=sys.stderr)
        sys.exit(1)


# --- Model Import ---
try:
    # Assuming model is in a 'models' subdirectory
    from models.R2GenGPT import R2GenGPT
except ModuleNotFoundError:
    # Try importing from root if not found in 'models'
    try:
        from R2GenGPT import R2GenGPT
        print("Imported R2GenGPT model from project root.")
    except ModuleNotFoundError:
        print("Error: Cannot find model module ('R2GenGPT.py'). Ensure it's in the root or a 'models' directory.", file=sys.stderr)
        sys.exit(1)

# --- PyTorch Lightning Imports ---
try:
    from lightning.pytorch import seed_everything, Trainer
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
    # Optional: Check for specific PL version if needed
    # import lightning as pl
    # print(f"Using PyTorch Lightning version: {pl.__version__}")
except ImportError:
     print("Error: PyTorch Lightning is not installed or accessible. Please install it (`pip install lightning`).", file=sys.stderr)
     sys.exit(1)


def train(args):
    """Main function to run training, validation, or testing."""
    print("Starting training/validation/testing process...")
    print("Arguments:")
    pprint(vars(args)) # Print all arguments for reproducibility

    # --- Seed for reproducibility ---
    # Set seed before dataset/model creation
    seed = getattr(args, 'seed', 42) # Default seed is 42
    seed_everything(seed, workers=True)
    print(f"Global random seed set to: {seed}")

    # --- Dataset and DataModule Setup ---
    print("\nCreating datasets...")
    try:
        # create_datasets expects args object
        train_dataset, dev_dataset, test_dataset = create_datasets(args)
        dataset = {"train": train_dataset, "val": dev_dataset, "test": test_dataset}
        print("\nCreating DataModule...")
        # DataModule expects the dataset dictionary and args
        dm = DataModule(dataset, args)
        print("DataModule created successfully.")
    except Exception as e:
        print(f"Error creating datasets or DataModule: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr) # Print detailed traceback
        return # Stop execution if data cannot be loaded

    # --- Callbacks and Loggers Setup ---
    print("\nSetting up callbacks and loggers...")
    callbacks = []
    loggers = []
    # Define base directory for logs and checkpoints
    base_save_dir = args.savedmodel_path or "lightning_logs" # Default if not provided
    os.makedirs(base_save_dir, exist_ok=True)
    print(f"Base save directory: {base_save_dir}")

    # TensorBoard Logger
    tb_logger = TensorBoardLogger(save_dir=base_save_dir, name="tb_logs")
    # CSV Logger
    csv_logger = CSVLogger(save_dir=base_save_dir, name="csv_logs")
    loggers = [tb_logger, csv_logger]
    print(f"Logging to TensorBoard: {tb_logger.log_dir}")
    print(f"Logging to CSV: {csv_logger.log_dir}")

    # Model Checkpoint Callback
    # Monitor the primary validation metric defined in the model
    primary_metric_key = 'F1_mac' if args.task == 'classification' else 'CIDEr' # Consistent with model logic
    monitor_metric = f"val/{primary_metric_key}_primary" # Metric name logged by the model
    filename_format = f"ckpt-{{epoch:02d}}-{{{monitor_metric}:.4f}}"
    # Sanitize filename format in case metric name has slashes
    safe_filename_format = filename_format.replace("/", "_")

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=os.path.join(base_save_dir, 'checkpoints'),
        filename=safe_filename_format,
        save_top_k=getattr(args, 'save_top_k', 1), # Save top K models based on monitored metric
        mode='max', # 'max' because higher F1/CIDEr is better
        save_last=False, # Optionally save the last checkpoint
        verbose=True # Print messages when checkpoints are saved
    )
    callbacks.append(checkpoint_callback)
    print(f"ModelCheckpoint monitoring: '{monitor_metric}' (mode: 'max'). Saving to: {checkpoint_callback.dirpath}")

    # Learning Rate Monitor Callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch') # Log LR every epoch
    callbacks.append(lr_monitor)

    # Optional: Early Stopping Callback
    # early_stop_callback = EarlyStopping(
    #    monitor=monitor_metric,
    #    patience=getattr(args, 'early_stopping_patience', 3), # Number of epochs with no improvement
    #    verbose=True,
    #    mode="max"
    # )
    # callbacks.append(early_stop_callback)
    # print(f"EarlyStopping enabled: Monitoring '{monitor_metric}', Patience: {early_stop_callback.patience}")

    # --- Trainer Setup ---
    print("\nCreating Trainer...")
    try:
        # Determine devices and strategy
        num_devices = args.devices
        strategy = args.strategy
        # Auto-select strategy if devices=1
        if isinstance(num_devices, int) and num_devices <= 1 and strategy != 'auto':
             print(f"Note: Setting strategy to 'auto' as devices={num_devices}.")
             strategy = 'auto'
        # If devices is a list/tuple, get the count
        if isinstance(num_devices, (list, tuple)):
            num_devices = len(num_devices)

        trainer = Trainer(
            devices=args.devices,               # Int (num GPUs), list[int] (specific GPUs), or string (e.g., "auto")
            num_nodes=args.num_nodes,           # For multi-node training
            strategy=strategy,                  # e.g., "ddp", "fsdp", "auto"
            accelerator=args.accelerator,       # "gpu", "cpu", etc.
            precision=args.precision,           # e.g., "16-mixed", "bf16-mixed", "32-true"
            val_check_interval=args.val_check_interval, # Float (fraction of epoch) or Int (steps)
            limit_val_batches=args.limit_val_batches, # Float (fraction) or Int (num batches)
            limit_train_batches=args.limit_train_batches,
            limit_test_batches=args.limit_test_batches,
            max_epochs=args.max_epochs,         # Max training epochs (-1 for infinite)
            num_sanity_val_steps=args.num_sanity_val_steps, # Steps to run validation before training
            accumulate_grad_batches=args.accumulate_grad_batches, # Gradient accumulation
            gradient_clip_val=args.gradient_clip_val, # Gradient clipping value
            callbacks=callbacks,                # List of callbacks
            logger=loggers,                     # List of loggers
            deterministic=True,                 # Ensure reproducibility (might impact performance)
            benchmark=False                     # Set True if input sizes don't vary, for performance
            # enable_progress_bar=True,         # Default is True
            # profiler="simple",                # Optional: for performance profiling
        )
        print("Trainer created successfully.")
        print(f"  Using Devices: {args.devices}, Strategy: {trainer.strategy}, Precision: {args.precision}")

    except Exception as e:
        print(f"Error creating Trainer: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return

    # --- Model Loading / Initialization ---
    print("\nLoading/Initializing model...")
    model = None
    ckpt_path_to_load = args.ckpt_file   # Full PL checkpoint (.ckpt)
    delta_path_to_load = args.delta_file # Adapter/projection weights (.pth)

    # Scenario 1: Load from a full PyTorch Lightning checkpoint (.ckpt)
    if ckpt_path_to_load and os.path.exists(ckpt_path_to_load):
        print(f"Attempting to load model from PL checkpoint: {ckpt_path_to_load}")
        try:
            # load_from_checkpoint handles hyperparameter loading etc.
            # Pass current args to potentially override loaded hparams if needed
            # strict=False allows loading even if some weights mismatch (e.g., projection layer added later)
            model = R2GenGPT.load_from_checkpoint(ckpt_path_to_load, args=args, strict=False)
            print(f"Successfully loaded model state from {ckpt_path_to_load}")
            # If a full checkpoint is loaded, usually ignore delta file unless specifically intended
            if delta_path_to_load and os.path.exists(delta_path_to_load):
                print(f"Warning: Both PL checkpoint ({ckpt_path_to_load}) and delta file ({delta_path_to_load}) provided. Prioritizing PL checkpoint. Delta file will NOT be loaded here.")
                delta_path_to_load = None # Prevent delta loading after full ckpt load
        except Exception as e:
            print(f"Error loading from PL checkpoint '{ckpt_path_to_load}': {e}. Will attempt to initialize a new model.", file=sys.stderr)
            model = None # Reset model variable
            ckpt_path_to_load = None # Don't use this path for resume/test later

    # Scenario 2: Initialize a new model (or if .ckpt loading failed)
    if model is None:
        print("Initializing a new model instance...")
        try:
            # Pass arguments to the model constructor
            # The model's __init__ should handle loading delta weights if delta_path_to_load is set
            original_delta_arg = args.delta_file # Store original arg
            args.delta_file = delta_path_to_load # Pass the potentially valid delta path
            model = R2GenGPT(args)
            args.delta_file = original_delta_arg # Restore original arg

            if delta_path_to_load and os.path.exists(delta_path_to_load):
                # The model __init__ or a separate method inside R2GenGPT should have loaded the delta
                print(f"Initialized new model and requested loading delta weights from: {delta_path_to_load}")
                # Re-check if loading actually happened (e.g., by checking parameter values or a flag in model)
            else:
                print("Initialized new model from scratch (no checkpoint or valid delta file provided/loaded).")
        except Exception as e:
            print(f"Error initializing R2GenGPT model: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            return # Stop if model cannot be initialized

    # Check if model is successfully loaded/initialized
    if model is None:
        print("Fatal: Failed to load or initialize the model.", file=sys.stderr)
        return

    # --- Determine Checkpoint Path for Trainer Actions ---
    # Use PL checkpoint for resuming training if loaded successfully
    ckpt_for_resume_fit = ckpt_path_to_load if ckpt_path_to_load else None
    # For testing/validation, prioritize delta file if no full checkpoint was loaded
    # The model should already have delta weights loaded if delta_path_to_load was valid
    ckpt_for_eval_weights = None # Trainer uses the model state directly, delta loaded internally

    # --- Execute Trainer Action ---
    try:
        if args.test:
            print("\nRunning test mode...")
            # The model passed should already have weights loaded (either from ckpt or delta)
            trainer.test(model, datamodule=dm) # No ckpt_path needed here
        elif args.validate:
            print("\nRunning validation mode...")
            # Validate the currently loaded model state
            trainer.validate(model, datamodule=dm) # No ckpt_path needed here
        else:
            print("\nRunning training mode...")
            # trainer.fit will resume from ckpt_for_resume_fit if it's a valid PL checkpoint path
            trainer.fit(model, datamodule=dm, ckpt_path=ckpt_for_resume_fit)
    except Exception as e:
        # Catch errors during Trainer execution (e.g., OOM, CUDA errors)
        print(f"\n--- An error occurred during Trainer execution ---", file=sys.stderr)
        print(f"Error type: {type(e).__name__}", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        print("Traceback:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print("----------------------------------------------------\n", file=sys.stderr)

    print("\nProcess finished.")


def main():
    """Parses arguments and calls the train function."""
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    if args.savedmodel_path:
        os.makedirs(args.savedmodel_path, exist_ok=True)
        print(f"Using save directory: {args.savedmodel_path}")
    else:
        print("Warning: --savedmodel_path not specified. Checkpoints and logs will go to default 'lightning_logs'.")

    # Call the main training/testing function
    train(args)

if __name__ == '__main__':
    main()