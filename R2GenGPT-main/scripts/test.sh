#!/bin/bash

dataset="mimic_cxr"
annotation="/root/autodl-tmp/mimic_cxr_mini/p10_annotation_final.json"
base_dir="/root/autodl-tmp/mimic_cxr_mini/images"
delta_file="/root/autodl-tmp/mimic_checkpoint/delta_checkpoint_epoch4_step42310.pth"

version="v5_test"
savepath="/root/autodl-tmp/save/$dataset/$version"

# 确保保存路径存在
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --max_length 5 \
    --min_new_tokens 1 \
    --max_new_tokens 10 \
    --repetition_penalty 2.0 \
    --length_penalty 1.0 \
    --test_batch_size 4 \
    --freeze_vm True \
    --vis_use_lora True \
    --vis_r 8 \
    --vis_alpha 32 \
    --savedmodel_path ${savepath} \
    --image_save_path ${savepath} \
    --num_workers 4 \
    --devices 1 \
    --save_images \
    2>&1 | tee -a ${savepath}/log.txt