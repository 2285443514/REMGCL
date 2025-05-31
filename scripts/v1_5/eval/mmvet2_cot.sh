#!/bin/bash
MODEL="llavar-v1.5-7b-15K+5K-random-new-3epo"
CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa \
    --model-path path/to/codes/LLaVA/checkpoints/$MODEL \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$MODEL.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$MODEL.json

echo $MODEL
echo $CHECKPOINT