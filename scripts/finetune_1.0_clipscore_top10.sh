#!/bin/bash

# IMPORTANT: this is the training script for the original LLaVA, NOT FOR LLaVA V1.5!

# Uncomment and set the following variables correspondingly to run this script:

################## VICUNA ##################
PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
################## VICUNA ##################

################## LLaMA-2 ##################
# PROMPT_VERSION="llava_llama_2"
# MODEL_VERSION="llama-2-7b-chat"
################## LLaMA-2 ##################
export NCCL_P2P_DISABLE=1

deepspeed --include localhost:1,2,3 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path path/to/model/vicuna-7b-v0 \
    --version $PROMPT_VERSION \
    --data_path /home/huangp/code/LLaVA/IT_data_select/llava_instruct_clipscore_top10.json \
    --image_folder path/to/data/LLaVA/finetune/image/train2017 \
    --vision_tower path/to/model/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter path/to/codes/LLaVA/checkpoints/llava-v1.0-pretain-v0/LLaVA-7b-pretrain-projector-v0-CC3M-595K-original_caption.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-1.0-clipscore10-finetune-epoch5-v0 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
