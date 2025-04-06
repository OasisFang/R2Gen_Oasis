#!/bin/bash

dataset="mimic_cxr"
annotation="/root/autodl-tmp/mimic_cxr_mini/p10_annotation_label_report.json"
base_dir="/root/autodl-tmp/mimic_cxr_mini/images"

version="classification_on_main_dataset"
savepath="/root/autodl-tmp/save/classification_v5/$version"
# delta_file="/root/autodl-tmp/save/classification_v3/checkpoints/checkpoint_epoch2_step7284_f10.222.pth"

# Ensure the folder exists
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 8 \
    # --delta_file ${delta_file} \
    --val_batch_size 8 \
    --freeze_vm True \
    --vis_use_lora True \
    --vis_r 8 \
    --vis_alpha 32 \
    --savedmodel_path ${savepath} \
    --image_save_path ${savepath} \
    --max_length 50 \
    --min_new_tokens 50 \
    --max_new_tokens 100 \
    --repetition_penalty 1.5 \
    --length_penalty -1.0 \
    --num_workers 4 \
    --devices 1 \
    --max_epochs 3 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.1 \
    --num_sanity_val_steps 2 \
    --task report \
    --save_images \
    2>&1 | tee -a ${savepath}/log.txt