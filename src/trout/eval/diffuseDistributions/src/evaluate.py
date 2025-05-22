import argparse
import json
import os

from trout.metrics import (
    compute_coverage,
    compute_entropy,
    get_probability_distribution,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--generations_file", type=str, help="where to read the generations"
    )
    parser.add_argument("--output_file", type=str, help="where to write the results")
    args = parser.parse_args()

    if not os.path.exists(args.generations_file):
        raise FileNotFoundError(f"File {args.generations_file} does not exist.")

    with open(args.generations_file, "r") as f:
        all_generations = json.load(f)

    results = {}
    for prompts_fp, generations in all_generations.items():
        results[prompts_fp] = {}

        probs = get_probability_distribution(generations)
        entropy = compute_entropy(probs)
        results[prompts_fp]["entropy"] = entropy

        if not os.path.exists(prompts_fp):
            print(
                f"Prompts file ({prompts_fp}) not found. Skipping coverage calculation."
            )
            results[prompts_fp]["coverage"] = None
        else:
            with open(prompts_fp, "r") as f:
                prompt_cfg = json.load(f)
            targets = prompt_cfg["targets"]
            coverage = compute_coverage(generations, targets)
            results[prompts_fp]["coverage"] = coverage

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
