#!/bin/bash
MODEL="llavar-v1.5-7b-15k+5k-random"
CHECKPOINT="checkpoint-1251"
python -m llava.eval.model_vqa \
    --model-path path/to/codes/LLaVA/checkpoints/$MODEL/$CHECKPOINT \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/$MODEL-$CHECKPOINT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/$MODEL-$CHECKPOINT.jsonl \
    --dst ./playground/data/eval/mm-vet/results/$MODEL-$CHECKPOINT.json

echo $MODEL
echo $CHECKPOINT