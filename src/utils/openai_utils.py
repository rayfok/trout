import json
from json import JSONDecodeError
from typing import Any

import numpy as np
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

client = OpenAI()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(JSONDecodeError),
)
def prompt_openai_model(
    user_prompt: str,
    model_name: str,
    system_instruction: str = "You are a helpful assistant.",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    reasoning_effort: str = "low",
    parse_mode: str = "json",  # or "text"
) -> Any:
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt},
    ]

    kwargs = {
        "model": model_name,
        "messages": messages,
    }

    if any(x in model_name for x in ["o3", "o4"]):
        kwargs["max_completion_tokens"] = max_tokens
        kwargs["reasoning_effort"] = reasoning_effort
        kwargs["top_p"] = top_p
    else:
        kwargs["temperature"] = temperature
        kwargs["max_tokens"] = max_tokens
        kwargs["top_p"] = top_p

    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content

    if parse_mode == "json":
        try:
            content_cleaned = strip_json_fences(content)
            return json.loads(content_cleaned)
        except JSONDecodeError:
            print("Invalid JSON in LLM response:")
            print(content)
            raise
    return content


def strip_json_fences(content: str) -> str:
    """
    Removes common markdown-style JSON code fences (e.g., ```json\n...\n```) from a string.
    """
    content = content.strip()
    if content.startswith("```json") or content.startswith("```"):
        # Remove starting and ending code block lines
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)
    return content.strip()


def get_openai_embeddings(texts, model="text-embedding-3-small"):
    response = client.embeddings.create(input=texts, model=model)
    return np.array([e.embedding for e in response.data])
Ø