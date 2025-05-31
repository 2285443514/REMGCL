#!/bin/bash
MODEL="llavar-v1.5-7b-5k-sum"
CHECKPOINT="checkpoint-315"
python -m llava.eval.model_vqa_loader \
    --model-path path/to/codes/LLaVA/checkpoints/$MODEL/$CHECKPOINT \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$MODEL-$CHECKPOINT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $MODEL-$CHECKPOINT

cd eval_tool

python calculation.py --results_dir answers/$MODEL-$CHECKPOINT

echo $MODEL
echo $CHECKPOINT