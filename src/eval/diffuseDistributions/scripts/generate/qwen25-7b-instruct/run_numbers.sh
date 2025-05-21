#!/bin/bash

python src/eval/diffuseDistributions/src/generate.py \
    --config src/eval/diffuseDistributions/configs/trout/numbers/qwen25-7b-instruct.yaml \
    --output_file data/diffuseDistributions/numbers/qwen25-7b-instruct.json \
    --mode untuned