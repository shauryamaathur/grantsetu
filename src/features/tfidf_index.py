"""TF-IDF index over the opportunity corpus, used as the core text-relevance signal."""
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Boilerplate tokens common to US government grant listings that add noise
# to similarity scoring without carrying topical meaning.
BOILERPLATE_STOPWORDS = [
    "program", "grant", "grants", "funding", "opportunity", "opportunities",
    "fy", "fiscal", "year", "notice", "announcement", "federal", "award",
    "application", "applications", "funds", "agency", "government",
]


@dataclass
class TfidfIndex:
    vectorizer: TfidfVectorizer
    matrix: object  # scipy sparse matrix, shape (n_opportunities, n_terms)
    opportunity_ids: list

    def query(self, text: str, top_k: int | None = None) -> pd.Series:
        """Return cosine similarity scores between `text` and every indexed
        opportunity, as a Series indexed by opportunity_id, sorted descending."""
        if not text or not text.strip():
            return pd.Series(0.0, index=self.opportunity_ids)
        query_vec = self.vectorizer.transform([text])
        sims = cosine_similarity(query_vec, self.matrix).flatten()
        scores = pd.Series(sims, index=self.opportunity_ids).sort_values(ascending=False)
        if top_k is not None:
            scores = scores.head(top_k)
        return scores


def build_tfidf_index(df: pd.DataFrame, text_column: str = "combined_text") -> TfidfIndex:
    """Fit a TF-IDF vectorizer over the opportunity corpus. Expects a dataframe
    with `opportunity_id` and a combined text column (see build_features.py)."""
    texts = df[text_column].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(TfidfVectorizer(stop_words="english").get_stop_words()) + BOILERPLATE_STOPWORDS,
        max_features=20000,
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix = vectorizer.fit_transform(texts)

    return TfidfIndex(
        vectorizer=vectorizer,
        matrix=matrix,
        opportunity_ids=df["opportunity_id"].tolist(),
    )


if __name__ == "__main__":
    from src.cleaning.pipeline import clean_dataset
    from src.features.build_features import build_features
    from src.ingestion.load_data import load_raw_grants

    raw = load_raw_grants()
    cleaned, _ = clean_dataset(raw)
    featured = build_features(cleaned)
    index = build_tfidf_index(featured)
    print(f"TF-IDF matrix shape: {index.matrix.shape}")
    top = index.query("environmental education for youth in rural communities", top_k=5)
    print(top)
