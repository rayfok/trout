import argparse
import json
import os

import datasets
import numpy as np
from tqdm import tqdm

from trout.metrics import DiversityScorer
from trout.utils.chat_models import get_chat_model


def generate(
    prompts: list[str],
    model_name: str,
    batch_size: int,
    n: int,
    num_tokens: int,
    temperature: float,
    top_p: float,
    output_file: str,
):
    params = {
        "model_name": model_name,
        "batch_size": batch_size,
        "n": n,
        "num_tokens": num_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "output_file": output_file,
        "num_prompts": len(prompts),
    }
    print("Generation parameters:", json.dumps(params, indent=2))

    model = get_chat_model(
        model_name=model_name,
        config={
            "use_tqdm": False,
            "max_num_seqs": batch_size,
            "gpu_memory_utilization": 0.8,
            "temperature": temperature,
            "top_p": top_p,
            "num_tokens": num_tokens,
            "n_devices": 2,
        },
    )

    # Adding a simple instruction to give context to the model
    prompts_with_instr = [
        f"Write a creative writing response based on the user-given writing prompt: {prompt}"
        for prompt in prompts
    ]

    generations = []
    for prompt in tqdm(prompts_with_instr):
        duplicated_prompts = [prompt for _ in range(n)]
        completions = model.batch_generate(duplicated_prompts)
        generations.append(
            {
                "prompt": prompt,
                "completions": completions,
            }
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Writing generations to {output_file}")
    with open(output_file, "w") as f:
        for generation in generations:
            f.write(json.dumps(generation) + "\n")
    print("Done!")


def evaluate(completions_file: str, results_output_file: str):
    semantic_diversity_model_name = "jinaai/jina-embeddings-v3"
    style_diversity_model_name = "AnnaWegmann/Style-Embedding"

    semantic_model = DiversityScorer(model_name=semantic_diversity_model_name)
    style_model = DiversityScorer(model_name=style_diversity_model_name)

    with open(completions_file, "r") as f:
        prompt_completions = [json.loads(line.strip()) for line in f.readlines()]

    semantic_diversity_scores = []
    style_diversity_scores = []
    for c in prompt_completions:
        completions = c["completions"]
        sem_div_score = semantic_model.average_pairwise_diversity(completions)
        style_div_score = style_model.average_pairwise_diversity(completions)
        semantic_diversity_scores.append(sem_div_score)
        style_diversity_scores.append(style_div_score)

    os.makedirs(os.path.dirname(results_output_file), exist_ok=True)
    with open(results_output_file, "w") as f:
        avg_sem_div = float(np.mean(semantic_diversity_scores))
        avg_style_div = float(np.mean(style_diversity_scores))
        results = {
            "avg_semantic_diversity": avg_sem_div,
            "avg_style_diversity": avg_style_div,
        }
        json.dump(results, f, indent=2)


def generate_and_evaluate(
    prompts: list[str],
    model_name_or_path: str,
    completions_output_file: str,
    results_output_file: str,
    n_completions: int,
    n_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    generate(
        prompts=prompts,
        model_name=model_name_or_path,
        batch_size=n_batch_size,
        n=n_completions,
        num_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        output_file=completions_output_file,
    )
    evaluate(
        completions_file=completions_output_file,
        results_output_file=results_output_file,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="pretrained model name or path",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--completions_output_file",
        type=str,
        help="where to dump the generations (should be a jsonl file)",
        default="data/diversityTuning/completions.jsonl",
    )
    parser.add_argument(
        "--results_output_file",
        type=str,
        help="where to dump the results (should be a json file)",
        default="results/diversityTuning/results.json",
    )
    parser.add_argument(
        "--n_completions",
        type=int,
        help="the number of completions for each prompt. should be divisible by batch size",
        default=50,
    )
    parser.add_argument(
        "--n_batch_size", type=int, help="batch size of generation", default=10
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="max number of tokens to sample",
        default=2048,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="temperature for sampling",
        default=1.0,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        help="top_p for sampling",
        default=0.9,
    )

    args = parser.parse_args()

    # Take the cleaned test set
    ds = datasets.load_from_disk("data/diversityTuning/writingPrompt_cleaned")
    val_dataset = ds["test"]

    # replicating same sampled data as in original paper
    val_dataset = val_dataset.select(range(1000, 2000))
    post_titles = val_dataset["post_title"]

    generate_and_evaluate(
        prompts=post_titles,
        model_name_or_path=args.model_name_or_path,
        completions_output_file=args.completions_output_file,
        results_output_file=args.results_output_file,
        n_completions=args.n_completions,
        n_batch_size=args.n_batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
