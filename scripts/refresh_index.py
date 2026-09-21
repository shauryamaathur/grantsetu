"""Standalone entrypoint to re-run cleaning/feature-engineering and rebuild the
TF-IDF index from the raw dataset. Intended for cron/scheduled use (blueprint
Section J.1) or manual invocation after a raw data update.

Usage: python -m scripts.refresh_index
"""
import json
import logging

from src.ranking.corpus import reset_corpus

logging.basicConfig(level=logging.INFO)


def main():
    corpus = reset_corpus(force_rebuild=True)
    print(f"Index refreshed: {len(corpus.df):,} opportunities indexed")
    if corpus.pipeline_stats:
        print(json.dumps(corpus.pipeline_stats, indent=2, default=str))


if __name__ == "__main__":
    main()
