#!/bin/bash

# --- Configuration for Testing ---

# Dataset details (should match training)
dataset="mimic_cxr"
annotation="/root/autodl-tmp/mimic_cxr/mimic_cxr_label_report_corrected.json" # <--- 使用修正后的文件名
base_dir="/root/autodl-tmp/mimic_cxr/images"

# --- Path to the weights to test ---
# This should point to the specific .pth file containing the trained adapter/projection weights
# Usually found within the training save path's 'pth_weights' subfolder
# Example: delta_file="/root/autodl-tmp/save/classification_L32_3B_Aug_v1/pth_weights/best_classification_weights.pth"
delta_file="/root/autodl-tmp/save/classification_L32_3B_Aug_v1/pth_weights/best_classification_weights.pth" # <--- UPDATE THIS PATH

# Set version/name for this test run and the results save path
test_version="test_on_Aug_v1_best" # Describe which weights are being tested
savepath="/root/autodl-tmp/save/$test_version"

# Ensure save path exists
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Test results folder '$savepath' created."
else
  echo "Test results folder '$savepath' already exists."
fi

# Check if the delta file exists
if [ ! -f "$delta_file" ]; then
    echo "Error: Delta file not found at $delta_file"
    exit 1
fi

# Run Python test script using train.py with --test flag
python -u train.py \
    --test `# Activate test mode` \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} `# Load the specific weights` \
    --vision_model "microsoft/swinv2-base-patch4-window16-256" `# Vision model used during training` \
    --llama_model "meta-llama/Llama-3.2-3B-Instruct" `# LLM used during training` \
    --end_sym "<|eot_id|>" \
    --task classification `# Specify the task` \
    --max_length 120 `# Max sequence length (consistent with training)` \
    --min_new_tokens 100 `# Min generation length` \
    --max_new_tokens 150 `# Max generation length (ensure sufficient for output)` \
    --repetition_penalty 1.0 `# Generation penalty (adjust if needed)` \
    --length_penalty 1.0 `# Generation penalty (adjust if needed)` \
    --test_batch_size 4 `# Batch size for testing (adjust based on GPU memory)` \
    --vis_use_lora True `# Vision LoRA setting (must match training)` \
    --vis_r 16 \
    --vis_alpha 16 \
    --llm_use_lora False `# LLM LoRA setting (must match training)` \
    --precision "bf16-mixed" `# Precision for loading/inference` \
    --savedmodel_path ${savepath} `# Directory to save test results/logs` \
    --image_save_path ${savepath} `# Directory to save qualitative examples` \
    --num_workers 4 \
    --devices 1 \
    --accelerator "gpu" \
    --strategy "auto" \
    --save_images `# Save qualitative examples during testing` \
    --seed 42 \
    2>&1 | tee -a ${savepath}/test_log.txt # Log output to file and console