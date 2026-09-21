import pandas as pd

from src.features.tfidf_index import build_tfidf_index
from src.ranking.score import score_opportunities


def _sample_df():
    return pd.DataFrame({
        "opportunity_id": [1, 2, 3],
        "combined_text": [
            "environmental research grant for marine habitat conservation",
            "youth education literacy program for rural schools",
            "urban infrastructure road construction funding",
        ],
        "opportunity_category": ["Environment", "Education", "Infrastructure"],
        "category_of_funding_activity": ["Environment", "Education", "Infrastructure"],
        "is_recent": [True, True, True],
        "is_expired": [False, False, False],
        "eligibility_text": ["nonprofit organizations", "nonprofit organizations", "nonprofit organizations"],
        "close_date": pd.to_datetime(["2030-01-01", "2030-01-01", "2030-01-01"]),
    })


def test_exact_category_match_scores_higher_than_no_match():
    df = _sample_df()
    index = build_tfidf_index(df)
    tfidf_scores = index.query("environmental research marine habitat")

    scored = score_opportunities(
        ngo_focus_areas=["environment"],
        ngo_registration_type="trust",
        tfidf_scores=tfidf_scores,
        opportunities_df=df,
    )

    scores_by_id = {s.opportunity_id: s.total_score for s in scored}
    assert scores_by_id.get(1, 0) > scores_by_id.get(3, 0)


def test_results_below_threshold_are_excluded():
    df = _sample_df()
    index = build_tfidf_index(df)
    tfidf_scores = index.query("completely unrelated query about astrophysics telescopes")

    scored = score_opportunities(
        ngo_focus_areas=["astrophysics"],
        ngo_registration_type="trust",
        tfidf_scores=tfidf_scores,
        opportunities_df=df,
    )
    # low text relevance + no category overlap should drop below MIN_RELEVANCE_THRESHOLD
    assert all(s.total_score >= 0.03 for s in scored)


def test_expired_opportunity_scores_lower_than_active_one():
    df = _sample_df()
    df.loc[df["opportunity_id"] == 2, "is_expired"] = True
    df.loc[df["opportunity_id"] == 2, "is_recent"] = False

    index = build_tfidf_index(df)
    tfidf_scores = index.query("education literacy program")

    scored = score_opportunities(
        ngo_focus_areas=["education"],
        ngo_registration_type="trust",
        tfidf_scores=tfidf_scores,
        opportunities_df=df,
    )
    # opportunity 2 matches best textually but is expired, so its recency component is 0
    match = next(s for s in scored if s.opportunity_id == 2)
    assert match.recency_score == 0.0


def test_empty_query_text_returns_zero_scores():
    df = _sample_df()
    index = build_tfidf_index(df)
    scores = index.query("")
    assert (scores == 0.0).all()


def test_corpus_search_completes_quickly_on_a_moderately_large_corpus():
    """Regression test for a real bug found via a live smoke test: writing
    `if oid in set(df["opportunity_id"])` directly inside a list
    comprehension's `if` clause re-builds that set on EVERY outer iteration,
    turning an O(n) filter into O(n^2) -- fine at the 3-row size every other
    test in this file uses, but a measured 90+ second hang on the real
    ~75,640-row corpus. A 3-row fixture can never catch an O(n^2) bug, so
    this test uses thousands of rows and asserts on wall-clock time, not
    just correctness, specifically to catch this class of regression."""
    import time

    from src.ranking.corpus import Corpus

    n = 4000
    df = pd.DataFrame({
        "opportunity_id": list(range(n)),
        "opportunity_title": [f"Opportunity {i} about health and education" for i in range(n)],
        "opportunity_category": ["Health" if i % 2 == 0 else "Education" for i in range(n)],
        "agency_name": [f"Agency {i % 50}" for i in range(n)],
        "combined_text": [f"Opportunity {i} about health and education programs" for i in range(n)],
    })
    corpus = Corpus()
    corpus.df = df
    corpus.index = build_tfidf_index(df)

    start = time.monotonic()
    results = corpus.search("health education", top_k=10)
    elapsed = time.monotonic() - start

    assert len(results) == 10
    # Measured: the correct O(n) version takes ~0.004s at this size; the
    # O(n^2) bug (reintroduced and re-measured to confirm this gate actually
    # catches it) took ~1.4s at n=4000, and 90+ seconds (and still not done)
    # on the real ~75,640-row corpus. 0.5s is well above normal noise but
    # far below the buggy version's measured time.
    assert elapsed < 0.5, f"corpus.search() took {elapsed:.2f}s on {n} rows -- likely an O(n^2) regression"
