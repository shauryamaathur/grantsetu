"""Ranking-quality metrics computed against the manually labelled evaluation set
(blueprint Section G). No metric here is estimated -- all are computed from
data/evaluation/labels.csv, which must be created by a human reviewer first."""


def precision_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    if k <= 0 or not ranked_ids:
        return 0.0
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def recall_at_k(ranked_ids: list, relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(ranked_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(ranked_ids: list, relevant_ids: set) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def duplicate_rate(ranked_ids: list) -> float:
    if not ranked_ids:
        return 0.0
    return 1 - (len(set(ranked_ids)) / len(ranked_ids))
