#!/bin/bash

python src/trout/eval/hivemind/evaluate.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --completions_output_file data/hivemind/llama3-8b-instruct.jsonl \
    --results_output_file results/hivemind/llama3-8b-instruct.json \
    --n_completions 50