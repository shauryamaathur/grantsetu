"""Build the evaluation candidate set (Section G.1) and, once labels exist,
compute ranking-quality metrics (Section G.2).

Honest two-step process, matching the blueprint:
  1. `--generate-candidates`: for each synthetic NGO profile in
     data/evaluation/ngo_profiles.json, run the actual ranking engine and
     write its top-K candidates to data/evaluation/candidates.csv for a human
     to review and label (relevant / partially_relevant / not_relevant).
  2. `--score`: once data/evaluation/labels.csv exists (human-labelled copy of
     candidates.csv with a `label` column filled in), compute Precision@K, MRR,
     and duplicate rate per profile and overall.

No metric is printed or claimed without labels.csv actually existing --
running --score without labels raises rather than fabricating a number.
"""
import argparse
import csv
import json
from pathlib import Path

from src.evaluation.metrics import duplicate_rate, mean_reciprocal_rank, precision_at_k
from src.ranking.corpus import get_corpus
from src.ranking.score import score_opportunities

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = PROJECT_ROOT / "data" / "evaluation" / "ngo_profiles.json"
CANDIDATES_PATH = PROJECT_ROOT / "data" / "evaluation" / "candidates.csv"
LABELS_PATH = PROJECT_ROOT / "data" / "evaluation" / "labels.csv"

TOP_K_FOR_LABELLING = 20
EVAL_K = 10


def generate_candidates():
    profiles = json.loads(PROFILES_PATH.read_text())
    corpus = get_corpus()

    rows = []
    for profile in profiles:
        query_text = " ".join(profile["focus_areas"] + profile["beneficiaries"])
        tfidf_scores = corpus.index.query(query_text, top_k=200)
        scored = score_opportunities(
            ngo_focus_areas=profile["focus_areas"],
            ngo_registration_type=profile["registration_type"],
            tfidf_scores=tfidf_scores,
            opportunities_df=corpus.df,
        )
        for rank, s in enumerate(scored[:TOP_K_FOR_LABELLING], start=1):
            row = corpus.get_opportunity(s.opportunity_id)
            rows.append(
                {
                    "profile_id": profile["profile_id"],
                    "archetype": profile["archetype"],
                    "rank": rank,
                    "opportunity_id": s.opportunity_id,
                    "opportunity_title": row["opportunity_title"],
                    "opportunity_category": row.get("opportunity_category"),
                    "total_score": s.total_score,
                    "label": "",  # to be filled in by a human reviewer: relevant | partially_relevant | not_relevant
                }
            )

    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} candidate rows to {CANDIDATES_PATH}")
    print("Next step: copy this file to labels.csv and fill in the 'label' column by hand, "
          "then re-run with --score.")


def score():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"{LABELS_PATH} does not exist. Run --generate-candidates first, then manually "
            "label data/evaluation/candidates.csv and save it as labels.csv before scoring. "
            "Metrics are never computed against unlabelled or fabricated data."
        )

    by_profile: dict[str, list[dict]] = {}
    with open(LABELS_PATH) as f:
        for row in csv.DictReader(f):
            by_profile.setdefault(row["profile_id"], []).append(row)

    overall_precisions, overall_mrrs, overall_dup_rates = [], [], []
    for profile_id, rows in by_profile.items():
        rows.sort(key=lambda r: int(r["rank"]))
        ranked_ids = [int(r["opportunity_id"]) for r in rows]
        relevant_ids = {int(r["opportunity_id"]) for r in rows if r["label"] == "relevant"}

        p_at_k = precision_at_k(ranked_ids, relevant_ids, EVAL_K)
        mrr = mean_reciprocal_rank(ranked_ids, relevant_ids)
        dup = duplicate_rate(ranked_ids)

        overall_precisions.append(p_at_k)
        overall_mrrs.append(mrr)
        overall_dup_rates.append(dup)

        print(f"{profile_id}: Precision@{EVAL_K}={p_at_k:.2f}  MRR={mrr:.2f}  DuplicateRate={dup:.2f}")

    n = len(overall_precisions) or 1
    print("\n--- Overall (mean across profiles) ---")
    print(f"Precision@{EVAL_K}: {sum(overall_precisions) / n:.3f}")
    print(f"MRR: {sum(overall_mrrs) / n:.3f}")
    print(f"Duplicate rate: {sum(overall_dup_rates) / n:.3f}")
    print(f"(Computed from {len(by_profile)} labelled profiles in {LABELS_PATH})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-candidates", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    if args.generate_candidates:
        generate_candidates()
    elif args.score:
        score()
    else:
        parser.print_help()
