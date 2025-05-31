#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path path/to/codes/LLaVA/checkpoints/llava-1.0-finetune-v0 \
    --question-file path/to/codes/LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder path/to/codes/LLaVA/playground/data/eval/scienceqa/images/test \
    --answers-file path/to/codes/LLaVA/playground/data/eval/scienceqa/answers/llava-1.0-finetune-v0.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir path/to/codes/LLaVA/playground/data/eval/scienceqa \
    --result-file path/to/codes/LLaVA/playground/data/eval/scienceqa/answers/llava-1.0-finetune-v0.jsonl \
    --output-file path/to/codes/LLaVA/playground/data/eval/scienceqa/answers/llava-1.0-finetune-v0_output.jsonl \
    --output-result path/to/codes/LLaVA/playground/data/eval/scienceqa/answers/llava-1.0-finetune-v0_result.json
