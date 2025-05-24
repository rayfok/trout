#!/bin/bash

python src/trout/eval/hivemind/evaluate.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --completions_output_file data/hivemind/qwen3-8b-instruct.jsonl \
    --results_output_file results/hivemind/qwen3-8b-instruct.json \
    --n_completions 50