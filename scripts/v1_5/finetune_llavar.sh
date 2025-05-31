#!/bin/bash
export NCCL_P2P_DISABLE=1

deepspeed --include localhost:1,2,3 --master_port 25600 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path path/to/model/vicuna-7b-v1.5 \
    --version v1 \
    --data_path path/to/codes/deita/output/LLaVAR/llava_instruct_150k_llavar_16k_pretrain_1of16.json \
    --image_folder path/to/data/LLaVA/finetune/image/train2017 \
    --vision_tower path/to/model/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter ./checkpoints/llavar-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llavar-v1.5-7b-5K \
    --num_train_epochs 6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 50000 \
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
