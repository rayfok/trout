import argparse
import json
import os
from collections import Counter

import numpy as np


def get_probability_distribution(strings: list[str]) -> np.ndarray:
    """
    Converts a list of strings into a probability distribution.

    Returns:
        probs (np.ndarray): Array of probabilities corresponding to each unique string.
    """
    counts = Counter(strings)
    total = sum(counts.values())
    probs = np.array([count / total for count in counts.values()])
    return probs


def compute_entropy(probs: np.ndarray, base: float = np.e) -> float:
    """
    Computes Shannon entropy from a probability distribution.

    Args:
        probs (np.ndarray): Probability distribution (must sum to ~1).
        base (float): Log base. Use `np.e` for nats, 2 for bits.

    Returns:
        float: Entropy value.
    """
    probs = probs[probs > 0]  # Remove zero entries to avoid log(0)
    return -np.sum(probs * np.log(probs)) / np.log(base)


def compute_coverage(preds: list[str], targets: list[str]):
    """
    Computes coverage of predictions over targets.

    Args:
        preds (list[str]): List of predicted strings.
        targets (list[str]): List of target strings.

    Returns:
        float: Coverage value.
    """
    if not targets:
        return None

    preds_set = set(preds)
    targets_set = set(targets)
    return len(preds_set.intersection(targets_set)) / len(targets_set)


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
