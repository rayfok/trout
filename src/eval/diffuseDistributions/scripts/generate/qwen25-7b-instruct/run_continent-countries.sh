#!/bin/bash

python src/eval/diffuseDistributions/src/generate.py \
    --config src/eval/diffuseDistributions/configs/trout/continent-countries/qwen25-7b-instruct.yaml \
    --output_file data/diffuseDistributions/continent-countries/qwen25-7b-instruct.json \
    --mode untuned