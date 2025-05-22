import argparse
import json
import os

import numpy as np
from datasets import load_dataset

from trout.metrics import SimilarityScorer
from trout.utils.chat_models import get_chat_model


def load_prompts():
    ds = load_dataset("liweijiang/infinite-chats-eval")
    eval_set = ds["train"]
    prompts = eval_set["query"]
    return prompts


def generate(
    prompts: list[str],
    model_name: str,
    n_completions: int,
    output_file: str,
    batch_size: int,
):
    model = get_chat_model(
        model_name=model_name,
        config={
            "max_num_seqs": batch_size,
            "gpu_memory_utilization": 0.8,
            "temperature": 1.0,
            "top_p": 0.9,
            "num_tokens": 2048,
        },
    )

    results = []
    for prompt in prompts:
        duplicated_prompts = [prompt for _ in range(n_completions)]
        completions = model.batch_generate(duplicated_prompts)
        results.append(
            {
                "prompt": prompt,
                "completions": completions,
            }
        )

    # Write to JSONL file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def evaluate(completions_file: str, results_output_file: str):
    sim_scorer = SimilarityScorer(model_name="all-MiniLM-L6-v2")

    with open(completions_file, "r") as f:
        prompt_completions = [json.loads(line.strip()) for line in f.readlines()]
    sims = []
    for c in prompt_completions:
        completions = c["completions"]
        sim = sim_scorer.average_pairwise_cosine_similarity(completions)
        sims.append(sim)

    os.makedirs(os.path.dirname(results_output_file), exist_ok=True)
    with open(results_output_file, "w") as f:
        json.dump({"macro_avg_cos_sim": float(np.mean(sims))}, f, indent=2)


def generate_and_evaluate(
    prompts: list[str],
    model_name: str,
    n_completions: int,
    output_file: str,
    results_output_file: str,
    batch_size: int = 10,
):
    generate(prompts, model_name, n_completions, output_file, batch_size)
    evaluate(output_file, results_output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="model name or path to the model to evaluate",
    )
    parser.add_argument(
        "--completions_output_file", type=str, help="where to write the completions"
    )
    parser.add_argument(
        "--results_output_file",
        type=str,
        help="where to write the evaluation results",
    )
    parser.add_argument(
        "--n_completions",
        type=int,
        default=20,
        help="number of samples to generate per prompt",
    )
    args = parser.parse_args()

    prompts = load_prompts()

    generate_and_evaluate(
        prompts=prompts,
        model_name=args.model_name_or_path,
        n_completions=args.n_completions,
        output_file=args.completions_output_file,
        results_output_file=args.results_output_file,
    )


if __name__ == "__main__":
    main()
