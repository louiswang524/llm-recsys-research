import numpy as np


def hit_at_k(ranked: list, target: str, k: int) -> float:
    return float(target in ranked[:k])


def ndcg_at_k(ranked: list, target: str, k: int) -> float:
    if target not in ranked[:k]:
        return 0.0
    return 1.0 / np.log2(ranked.index(target) + 2)


def mrr(ranked: list, target: str) -> float:
    if target not in ranked:
        return 0.0
    return 1.0 / (ranked.index(target) + 1)


def compute_metrics(ranked_lists: list[list], targets: list[str], k_values: list[int]) -> dict[str, float]:
    results: dict[str, list] = {f"HR@{k}": [] for k in k_values}
    results.update({f"NDCG@{k}": [] for k in k_values})
    results["MRR"] = []

    for ranked, target in zip(ranked_lists, targets):
        for k in k_values:
            results[f"HR@{k}"].append(hit_at_k(ranked, target, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(ranked, target, k))
        results["MRR"].append(mrr(ranked, target))

    return {k: float(np.mean(v)) for k, v in results.items()}
