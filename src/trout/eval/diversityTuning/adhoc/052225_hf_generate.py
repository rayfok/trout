from collections import defaultdict
from typing import Any

import torch
from peft import PeftModelForCausalLM
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PRECISIONS = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@torch.inference_mode()
def generate(
    model: AutoModelForCausalLM | PeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],  # updated: now list of strings
    reps: int,
    batch_size: int,
    device: str = "cuda",
    sampling_config: dict[str, Any] = None,
) -> dict[str, list[str]]:
    generation_args = {
        "do_sample": True,
        "temperature": 1.0,
        "top_k": None,
        "top_p": 0.95,
        "max_new_tokens": 2048,
    }
    if sampling_config:
        generation_args.update(sampling_config)

    generations: dict[str, list[str]] = defaultdict(list)

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with tqdm(total=batch_size * reps * len(prompts), desc="sampling..") as pbar:
        for i, prompt in enumerate(prompts):
            prompt_input_ids = input_ids[i].unsqueeze(0)
            prompt_attention_mask = attention_mask[i].unsqueeze(0)

            for _ in range(reps):
                # Repeat prompt `batch_size` times
                batch_input_ids = prompt_input_ids.repeat(batch_size, 1)
                batch_attention_mask = prompt_attention_mask.repeat(batch_size, 1)

                try:
                    outputs = model.generate(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        **generation_args,
                    )
                    decoded = tokenizer.batch_decode(
                        outputs[:, prompt_input_ids.shape[1] :],
                        skip_special_tokens=True,
                        # clean_up_tokenization_spaces=True,
                    )
                    generations[prompt].extend(decoded)
                except RuntimeError as e:
                    print(f"Generation failed for prompt [{prompt[:30]}...]: {e}")
                pbar.update(batch_size)

    return generations


def test_generate_with_hf():
    pretrained_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name, use_fast=False, legacy=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    precision = "bf16"
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name,
        device_map="auto",
        torch_dtype=PRECISIONS[precision],
        trust_remote_code=True,
    )
    model = torch.compile(model)

    prompts = [
        "Write a haiku about snow.",
        "Explain quantum entanglement in simple terms.",
    ]

    n_generations = 20
    n_batch_size = 10
    reps = n_generations // n_batch_size

    generations = generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        reps=reps,
        batch_size=n_batch_size,
        sampling_config={"temperature": 1.0, "top_p": 0.9, "max_new_tokens": 256},
    )

    for prompt, responses in generations.items():
        print(f"\n\n\n ===Prompt===: {prompt}")
        for r in responses:
            print(f"- {r}\n")


if __name__ == "__main__":
    test_generate_with_hf()
