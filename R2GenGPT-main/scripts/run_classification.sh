#!/bin/bash

# Set dataset and paths
dataset="mimic_cxr"
# --- 确认你的标注文件路径正确 ---
# This should point to the JSON file containing 'train', 'val', 'test' splits
annotation="/root/autodl-tmp/mimic_cxr/mimic_cxr_label_report_corrected.json" # <--- 使用修正后的文件名
base_dir="/root/autodl-tmp/mimic_cxr/images"
# --- 如果有预训练的delta权重，指定路径，否则留空或注释掉 ---
# delta_file="/path/to/your/pretrained_weights.pth" # Example if you have one
delta_file="" # No delta file initially

# Set version and save path
# Choose a meaningful version name
version="classification_L32_3B_Aug_v1" # Indicate augmentation in version name
savepath="/root/autodl-tmp/save/$version"

# Ensure save path exists
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

# Run Python training script
# Use python -u for unbuffered output (logs appear immediately)
python -u train.py \
  --dataset "$dataset" \
  --annotation "$annotation" \
  --base_dir "$base_dir" \
  --savedmodel_path "$savepath" \
  --image_save_path "$savepath" \
  --vision_model "microsoft/swinv2-base-patch4-window16-256" \
  --llama_model "meta-llama/Llama-3.2-3B-Instruct" \
  --end_sym "<|eot_id|>" \
  --vis_use_lora True \
  --vis_r 16 \
  --vis_alpha 16 \
  --batch_size 8 \
  --val_batch_size 8 \
  --accumulate_grad_batches 2 \
  --learning_rate 2e-4 \
  --precision "bf16-mixed" \
  --max_epochs 5 \
  --num_workers 6 \
  --devices 1 \
  --accelerator "gpu" \
  --strategy "auto" \
  --freeze_vm False \
  ${delta_file:+--delta_file "$delta_file"} \
  --max_length 512 `# Max sequence length for tokenizer` \
  --min_new_tokens 70 `# Min tokens for LLM generation (classification output)` \
  --max_new_tokens 180 `# Max tokens for LLM generation (needs to be enough for all labels)` \
  --repetition_penalty 1.0 \
  --length_penalty 1.0 \
  --limit_val_batches 1.0 `# Use all validation data` \
  --val_check_interval 0.25 `# Validate 4 times per epoch` \
  --num_sanity_val_steps 2 \
  --task classification `# Specify task as classification` \
  --save_images `# Save qualitative examples` \
  --seed 42 \
  --pth_save_filename "" \
  2>&1 | tee -a "$savepath/train_log.txt" # Log output to file and console