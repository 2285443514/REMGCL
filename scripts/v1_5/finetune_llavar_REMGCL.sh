#!/bin/bash
export NCCL_P2P_DISABLE=1

deepspeed --include localhost:1 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path path/to/codes/LLaVA/checkpoints/llavar-v1.5-7b \
    --version v1 \
    --data_path path/to/codes/deita/output/LLaVAR/gptqa_15k_llavar_5k_random.json \
    --image_folder path/to/data/LLaVA/finetune/image/train2017 \
    --vision_tower path/to/model/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavar-v1.5-7b-15K+5K-random-contra-global-sw-refined-local-noweight-4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 6 \
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
    --report_to none
