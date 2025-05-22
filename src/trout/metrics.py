from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer


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


class SimilarityScorer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def average_pairwise_cosine_similarity(self, strings: list[str]) -> float:
        if len(strings) < 2:
            return 1.0
        embeddings = self.model.encode(
            strings, convert_to_numpy=True, normalize_embeddings=True
        )
        sim_matrix = embeddings @ embeddings.T
        n = len(strings)
        upper_indices = np.triu_indices(n, k=1)
        return sim_matrix[upper_indices].mean()
