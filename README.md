# Trout: Improving LM Output Diversity

## Tasks

### diffuseDistributions

(From [Forcing Diffuse Distributions out of Language Models](https://arxiv.org/abs/2404.10859))

Generate with `./src/trout/eval/diffuseDistributions/scripts/generate/<model_name>/run_all.sh`

Evaluate with `./src/trout/eval/diffuseDistributions/scripts/evaluate/<model_name>/run_all.sh`

- Generations saved to `data/diffuseDistributions/<task>/<model_name>.json`
- Results saved to `results/diffuseDistributions/<task>/<model_name>.json`


### diversityTuning

(From [Modifying Large Language Model Post-Training for Diverse
Creative Writing](https://arxiv.org/pdf/2503.17126))


### hivemind

(From [Artificial Hiveminds]() (TBA))

Generate and evaluate with `./src/trout/eval/hivemind/scripts/run_<model_name>.sh`

- Generations saved to `data/hivemind/<model_name>.jsonl`
- Results saved to `results/hivemind/<model_name>.json`