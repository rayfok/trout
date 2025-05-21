#!/bin/bash

python src/eval/diffuseDistributions/src/generate.py \
    --config src/eval/diffuseDistributions/configs/trout/fruits/qwen25-7b-instruct.yaml \
    --output_file data/diffuseDistributions/fruits/qwen25-7b-instruct.json \
    --mode untuned