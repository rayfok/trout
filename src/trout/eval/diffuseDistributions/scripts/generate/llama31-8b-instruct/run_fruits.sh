#!/bin/bash

python src/trout/eval/diffuseDistributions/src/generate.py \
    --config src/trout/eval/diffuseDistributions/configs/trout/fruits/llama31-8b-instruct.yaml \
    --output_file data/diffuseDistributions/fruits/llama31-8b-instruct.json \
    --mode untuned