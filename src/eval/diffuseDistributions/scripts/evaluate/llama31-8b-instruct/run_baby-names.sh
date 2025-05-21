#!/bin/bash

python src/eval/diffuseDistributions/src/evaluate.py \
    --generations_file data/diffuseDistributions/baby-names/llama31-8b-instruct.json \
    --output_file results/diffuseDistributions/baby-names/llama31-8b-instruct.json