#!/bin/bash

python src/trout/eval/hivemind/evaluate.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --completions_output_file data/hivemind/qwen25-7b-instruct.jsonl \
    --results_output_file results/hivemind/qwen25-7b-instruct.json \
    --n_completions 50