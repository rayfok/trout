#!/bin/bash

python src/trout/eval/diffuseDistributions/src/evaluate.py \
    --generations_file data/diffuseDistributions/continent-countries/llama31-8b-instruct.json \
    --output_file results/diffuseDistributions/continent-countries/llama31-8b-instruct.json