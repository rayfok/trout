#!/bin/bash

python src/eval/diffuseDistributions/src/generate.py \
    --config src/eval/diffuseDistributions/configs/trout/occupations/qwen25-7b-instruct.yaml \
    --output_file data/diffuseDistributions/occupations/qwen25-7b-instruct.json \
    --mode untuned