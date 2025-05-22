#!/bin/bash

python src/trout/eval/diffuseDistributions/src/generate.py \
    --config src/trout/eval/diffuseDistributions/configs/trout/fictitious-people/qwen25-7b-instruct.yaml \
    --output_file data/diffuseDistributions/fictitious-people/qwen25-7b-instruct.json \
    --mode untuned