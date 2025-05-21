#!/bin/bash

MODEL_NAME="qwen25-7b-instruct"
BASE_DIR="./src/eval/diffuseDistributions/scripts/generate/${MODEL_NAME}"
TASKS=(
  baby-names
  continent-countries
  day-date
  fictitious-people
  fruits
  numbers
  occupations
)

for task in "${TASKS[@]}"; do
  "${BASE_DIR}/run_${MODEL_NAME}_${task}.sh"
done
