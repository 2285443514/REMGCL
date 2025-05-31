#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path path/to/codes/LLaVA/checkpoints/llavar-v1.5-7b-5k-mean/checkpoint-315 \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/llavar-v1.5-7b-5k-mean-3epoch.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/llavar-v1.5-7b-5k-mean-3epoch.jsonl \
    --dst ./playground/data/eval/mm-vet/results/llavar-v1.5-7b-5k-mean-3epoch.json

