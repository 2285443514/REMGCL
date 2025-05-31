#!/bin/bash
MODEL="llavar-v1.5-7b-4+1k-mean-image"
python -m llava.eval.model_vqa_loader \
    --model-path path/to/codes/LLaVA/checkpoints/$MODEL\
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$MODEL.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $MODEL

cd eval_tool

python calculation.py --results_dir answers/$MODEL

echo $MODEL