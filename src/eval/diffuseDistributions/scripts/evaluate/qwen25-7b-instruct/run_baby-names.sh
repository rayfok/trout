#!/bin/bash

python src/eval/diffuseDistributions/src/evaluate.py \
    --generations_file data/diffuseDistributions/baby-names/qwen25-7b-instruct.json \
    --output_file results/diffuseDistributions/baby-names/qwen25-7b-instruct.json