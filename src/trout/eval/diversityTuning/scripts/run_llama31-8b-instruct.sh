#!/bin/bash

python src/trout/eval/diversityTuning/evaluate.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --completions_output_file data/diversityTuning/llama31-8b-instruct.jsonl \
    --results_output_file results/diversityTuning/llama31-8b-instruct.json \
    --n_completions 4 \
    --n_batch_size 4 \
    --max_new_tokens 2048 \
    --temperature 1.0 \
    --top_p 0.95 \