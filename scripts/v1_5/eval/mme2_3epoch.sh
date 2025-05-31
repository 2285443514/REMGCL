#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path path/to/codes/LLaVA/checkpoints/llavar-v1.5-7b-4k+1k-mean/checkpoint-1155 \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llavar-v1.5-7b-4k+1k-mean-11epoch.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llavar-v1.5-7b-4k+1k-mean-11epoch

cd eval_tool

python calculation.py --results_dir answers/llavar-v1.5-7b-4k+1k-mean-11epoch
