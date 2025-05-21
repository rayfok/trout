#!/bin/bash

python src/eval/diffuseDistributions/src/generate.py \
    --config src/eval/diffuseDistributions/configs/trout/fictitious-people/llama31-8b-instruct.yaml \
    --output_file data/diffuseDistributions/fictitious-people/llama31-8b-instruct.json \
    --mode untuned