#!/bin/bash

python src/eval/diffuseDistributions/src/generate.py \
    --config src/eval/diffuseDistributions/configs/trout/baby-names/llama31-8b-instruct.yaml \
    --output_file data/diffuseDistributions/baby-names/llama31-8b-instruct.json \
    --mode untuned