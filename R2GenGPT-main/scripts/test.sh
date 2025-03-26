#!/bin/bash

dataset="mimic_cxr"
annotation="/root/autodl-tmp/mimic_cxr_mini/p10_annotation_final.json"
base_dir="/root/autodl-tmp/mimic_cxr_mini/images"
delta_file="/root/autodl-tmp/mimic_checkpoint/delta_checkpoint_epoch4_step42310.pth"

version="v1_delta"
savepath="/root/autodl-tmp/save/$dataset/$version"

python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --max_length 50 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --test_batch_size 2 \
    --freeze_vm True \
    --vis_use_lora True \
    --vis_r 16 \
    --vis_alpha 16 \
    --savedmodel_path ${savepath} \
    #
    --image_save_path ${savepath} \
    #
    --num_workers 2 \
    --devices 1 \

    2>&1 |tee -a ${savepath}/log.txt