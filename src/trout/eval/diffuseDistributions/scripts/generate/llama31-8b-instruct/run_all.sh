#!/bin/bash

MODEL_NAME="llama31-8b-instruct"
BASE_DIR="./src/trout/eval/diffuseDistributions/scripts/generate/${MODEL_NAME}"
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
  "${BASE_DIR}/run_${task}.sh"
done
