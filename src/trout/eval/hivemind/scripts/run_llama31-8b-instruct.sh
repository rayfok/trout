#!/bin/bash

python src/trout/eval/hivemind/evaluate.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --completions_output_file data/hivemind/llama31-8b-instruct.jsonl \
    --results_output_file results/hivemind/llama31-8b-instruct.json \
    --n_completions 50