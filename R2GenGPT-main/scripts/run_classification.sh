#!/bin/bash

dataset="mimic_cxr"
annotation="/root/autodl-tmp/mimic_cxr_mini/p10_annotation_final.json"
base_dir="/root/autodl-tmp/mimic_cxr_mini/images"

version="classification_v10_save_image"
savepath="/root/autodl-tmp/save/$dataset/$version"

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
    --batch_size 10 \
    --val_batch_size 10 \
    --freeze_vm True \
    --vis_use_lora True \
    --vis_r 8 \
    --vis_alpha 32 \
    --savedmodel_path ${savepath} \
    --image_save_path ${savepath} \
    --max_length 10 \
    --min_new_tokens 5 \
    --max_new_tokens 10 \
    --repetition_penalty 2.0 \
    --length_penalty -1.0 \
    --num_workers 4 \
    --devices 1 \
    --max_epochs 3 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.25 \
    --num_sanity_val_steps 2 \
    --task classification \
    --save_images \
    2>&1 | tee -a ${savepath}/log.txt