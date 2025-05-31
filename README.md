# REMGCL

**Reasoning Elicitation and Multi-Granularity Contrastive Learning for Text-Rich Image Understanding in Large Vision-Language Models**

## Environment Setup

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/2285443514/REMGCL
cd REMGCL
```

2. Install Package
```Shell
conda create -n REMGCL python=3.10 -y
conda activate REMGCL
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Training Data ([Huggingface](https://huggingface.co/datasets/SALT-NLP/LLaVAR))

Our image data is from [LLaVAR](https://github.com/SALT-NLP/LLaVAR), which is already transformed into the format of LLaVA pretraining/finetuning (They have "fake" file names in the format of CC3M and COCO). You can download them and merge them into the LLaVA training sets.

Our instructions, on the other hand, already contain LLaVA's instructions.

Pretraining Images: [Google Drive](https://drive.google.com/file/d/1zWpqnAcaG_dUwkJJUvP9FH9zq__c-ODY/view?usp=sharing)

Pretraining Instructions (595K + 422K): [Google Drive](https://drive.google.com/file/d/1_GCHFwrPGjp-9tZlDBwVkdz-L1ymchKY/view?usp=sharing)

Finetuning Images: [Google Drive](https://drive.google.com/file/d/1Ms7OCjcFQ18Whmujszpc9bTp0Jy0Dye4/view?usp=sharing)

Finetuning Instructions (158K + 16K): [Google Drive](https://drive.google.com/file/d/1ISdKOV1wwVkLHf5FNutctpOBa-CmNRFv/view?usp=sharing)

REMGCL Instructuons: [Google Drive](https://drive.google.com/file/d/1K41VSx6gyvLLZ-WQVegFZ0oXufTKMKgI/view?usp=sharing)


## Training Script

You should merge our pretraining images into the cc3m folder.

### Pre-train

```Shell
deepspeed --include localhost:3,4 llava/train/train_mem.py \
    --deepspeed ./scripts/zero3_offload.json \
    --model_name_or_path path/to/model/vicuna-7b-v1.5 \
    --version plain \
    --data_path path/to/data/LLaVAR/pretrain/chat_llavar.json \
    --image_folder path/to/data/LLaVAR/pretrain/image \
    --vision_tower path/to/model/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
```

### Fine-tuning
You should merge our finetuning images into the coco2017 folder.


```Shell
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
```

### REMGCL
```Shell
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
```

## Evaluation Script

Instruction-following on COCO images.

```
python /path/to/LLaVA/llava/eval/model_vqa.py \
    --model-name /path/to/checkpoint \
    --question-file \
    /path/to/LLaVA/playground/data/coco2014_val_qa_eval/qa90_questions.jsonl \
    --image-folder \
    /path/to/coco2014/val2014 \
    --answers-file \
    /path/to/qa90-answer-file.jsonl \
    --conv-mode "llava_v1"
```

Instruction-following on a given image URL.
```
python -m llava.eval.run_llava \
    --model-name /path/to/checkpoint \
    --image-file "https://cdn.shopify.com/s/files/1/0057/3728/3618/products/a-man-called-otto_ezrjr0pm_480x.progressive.jpg" \
    --query "Who starred in the movie?"
```

For text-based VQA (from [MultimodalOCR](https://github.com/Yuliang-Liu/MultimodalOCR)): after cloning their repo and preparing the data, you can put the `./MultimodalOCR/Eval_LLaVAR.py` in `/your/path/to/MultimodalOCR/models/LLaVA/` and add our model to `/your/path/to/MultimodalOCR/eval.py` for evaluation.


## Acknowledgement
The code base is mainly from the [LLaVA](https://github.com/haotian-liu/LLaVA) project. Our evaluation is also built on the [MultimodalOCR](https://github.com/Yuliang-Liu/MultimodalOCR) project. Traing data is from [LLaVAR](https://github.com/SALT-NLP/LLaVAR).