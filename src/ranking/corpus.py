"""In-process opportunity corpus + TF-IDF index, loaded once and reused across
requests (blueprint J.1: "start with an in-process TF-IDF/BM25 index").

Loading ~75k rows and fitting TF-IDF takes a few seconds, so this module
caches the cleaned+featured dataframe to data/processed/ and only rebuilds
from data/raw/ when the cache is missing or force_rebuild=True.
"""
import logging
from pathlib import Path

import pandas as pd

from src.cleaning.pipeline import clean_dataset
from src.features.build_features import build_features
from src.features.tfidf_index import TfidfIndex, build_tfidf_index
from src.ingestion.load_data import load_raw_grants
from src.ingestion.validate_schema import validate_schema

logger = logging.getLogger("grantsetu.corpus")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "opportunities_clean.pkl"


class Corpus:
    """Holds the cleaned+featured opportunities dataframe and its TF-IDF index."""

    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.index: TfidfIndex | None = None
        self.pipeline_stats: dict = {}

    def build(self, force_rebuild: bool = False):
        if not force_rebuild and PROCESSED_PATH.exists():
            logger.info("Loading cached processed dataset from %s", PROCESSED_PATH)
            self.df = pd.read_pickle(PROCESSED_PATH)
        else:
            logger.info("Building processed dataset from raw source")
            raw = load_raw_grants()
            report = validate_schema(raw)
            logger.info("Schema validation: %s", report.summary().replace("\n", " | "))

            cleaned, stats = clean_dataset(raw)
            self.pipeline_stats = stats
            featured = build_features(cleaned)

            PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
            featured.to_pickle(PROCESSED_PATH)
            self.df = featured

        self.index = build_tfidf_index(self.df)
        logger.info("Corpus ready: %d opportunities indexed", len(self.df))
        return self

    def get_opportunity(self, opportunity_id: int) -> pd.Series | None:
        matches = self.df[self.df["opportunity_id"] == opportunity_id]
        if matches.empty:
            return None
        return matches.iloc[0]

    def search(
        self, query_text: str, category: str | None = None, funder: str | None = None, top_k: int = 50
    ) -> pd.DataFrame:
        """Plain keyword/filtered search (baseline, no NGO profile required)."""
        df = self.df
        if category:
            df = df[df["opportunity_category"].fillna("").str.lower().str.contains(category.lower())]
        if funder:
            df = df[df["agency_name"].fillna("").str.lower().str.contains(funder.lower())]
        if not query_text or not query_text.strip():
            return df.head(top_k)
        scores = self.index.query(query_text, top_k=None)
        # `valid_ids` MUST be computed once, outside the comprehension below --
        # writing `set(df["opportunity_id"])` directly inside the `if` clause
        # re-evaluates (rebuilds the whole set from all ~75k rows) on EVERY
        # iteration of the outer loop, turning an O(n) filter into an O(n^2)
        # one that can take minutes on the full corpus. This was a real,
        # measured hang, not a hypothetical -- found via a live smoke test.
        valid_ids = set(df["opportunity_id"])
        ranked_ids = [oid for oid in scores.index if oid in valid_ids][:top_k]
        ranked = df.set_index("opportunity_id").loc[ranked_ids].reset_index()
        return ranked


_corpus_singleton: Corpus | None = None


def get_corpus() -> Corpus:
    global _corpus_singleton
    if _corpus_singleton is None:
        _corpus_singleton = Corpus().build()
    return _corpus_singleton


def reset_corpus(force_rebuild: bool = True) -> Corpus:
    """Used by the admin refresh endpoint and by scripts/refresh_index.py."""
    global _corpus_singleton
    _corpus_singleton = Corpus().build(force_rebuild=force_rebuild)
    return _corpus_singleton
