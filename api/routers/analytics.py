"""Grant Intelligence: real, on-demand analysis of the historical opportunity
corpus (and, where noted, the live ImportedGrant table). Every number here is
computed from the actual in-process/DB data at request time -- nothing is
precomputed or fabricated. Endpoints are read-only and cheap enough to run
per-request on ~75k rows (a few hundred ms), so no caching layer is needed.

Column availability varies across the corpus (e.g. estimated_total_program_
funding is missing for a meaningful fraction of rows) -- every stat that
excludes missing values says so via a `sample_size` alongside `total_rows`,
rather than silently treating NaN as zero.
"""
import csv
import io
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.db.models import ImportedGrant
from api.db.session import get_db
from src.ranking.corpus import get_corpus

router = APIRouter(prefix="/analytics", tags=["analytics"])


def _clean_num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(f) or np.isinf(f)) else f


def _apply_filters(df: pd.DataFrame, category: str | None, funder: str | None, status: str | None) -> pd.DataFrame:
    # "category" here means SECTOR (Health/Education/Environment/...) --
    # real bug found via live smoke-testing this phase: the corpus's
    # `opportunity_category` column is actually a FUNDING-MECHANISM type
    # (Discretionary/Mandatory/Continuation/Earmark/Other -- 5 values
    # total), not a theme. `category_of_funding_activity` is the real,
    # ~60-value sector field ("Health", "Education", "Environment", etc)
    # that every "Sector"/"Compare"/category-filter feature in Grant
    # Intelligence actually means. Filtering on `opportunity_category` by a
    # sector name like "Health" silently matched ZERO rows against the real
    # 75,640-row corpus (it only ever matches "Other"/"Discretionary"/etc)
    # -- caught by testing against real data, not just the test fixture
    # (whose fake corpus happens to set both columns to the same values).
    if category:
        df = df[df["category_of_funding_activity"].fillna("").str.lower().str.contains(category.lower())]
    if funder:
        df = df[df["agency_name"].fillna("").str.lower().str.contains(funder.lower())]
    if status == "open":
        df = df[df["is_expired"] == False]  # noqa: E712
    elif status == "expired":
        df = df[df["is_expired"] == True]  # noqa: E712
    return df


@router.get("/overview")
def overview(
    category: str | None = Query(default=None, description="Filter to a category substring, e.g. 'Health'"),
    funder: str | None = Query(default=None, description="Filter to a funder/agency substring"),
    year: int | None = Query(default=None, description="Filter to opportunities posted in this year"),
):
    df = get_corpus().df
    df = _apply_filters(df, category, funder, None)
    if year is not None:
        df = df[df["post_date"].notna() & (df["post_date"].dt.year == year)]

    total = len(df)
    open_count = int((df["is_expired"] == False).sum())  # noqa: E712
    expired_count = int((df["is_expired"] == True).sum())  # noqa: E712
    recent_count = int(df.get("is_recent", pd.Series(dtype=bool)).sum()) if "is_recent" in df else None

    funding_col = df["estimated_total_program_funding"].dropna()
    ceiling_col = df["award_ceiling"].dropna()

    return {
        "total_rows": total,
        "open_count": open_count,
        "expired_count": expired_count,
        "recent_count": recent_count,
        "unique_funders": int(df["agency_name"].dropna().nunique()),
        "unique_categories": int(df["category_of_funding_activity"].dropna().nunique()),
        "total_estimated_funding": {
            "sum": _clean_num(funding_col.sum()) if len(funding_col) else None,
            "mean": _clean_num(funding_col.mean()) if len(funding_col) else None,
            "median": _clean_num(funding_col.median()) if len(funding_col) else None,
            "sample_size": int(len(funding_col)),
        },
        "award_ceiling": {
            "mean": _clean_num(ceiling_col.mean()) if len(ceiling_col) else None,
            "median": _clean_num(ceiling_col.median()) if len(ceiling_col) else None,
            "min": _clean_num(ceiling_col.min()) if len(ceiling_col) else None,
            "max": _clean_num(ceiling_col.max()) if len(ceiling_col) else None,
            "sample_size": int(len(ceiling_col)),
        },
        "filters_applied": {"category": category, "funder": funder, "year": year},
        "data_source": "Historical opportunity corpus (see docs/data_audit_report.md)",
    }


@router.get("/available-years")
def available_years():
    """Real years present in the corpus's post_date column -- so the
    frontend's time-range control only ever offers years that actually have
    data, never a fabricated/assumed range like "2022-2026" regardless of
    what's really indexed."""
    df = get_corpus().df
    years = sorted(df["post_date"].dropna().dt.year.unique().tolist())
    return {"years": [int(y) for y in years]}


@router.get("/trends")
def trends(
    category: str | None = Query(default=None),
    funder: str | None = Query(default=None),
):
    """Postings and deadlines grouped by month -- real dates from the
    corpus, not a synthetic time series."""
    df = get_corpus().df
    df = _apply_filters(df, category, funder, None)

    def _monthly(col: str):
        s = df[col].dropna()
        if s.empty:
            return []
        counts = s.dt.to_period("M").value_counts().sort_index()
        return [{"month": str(period), "count": int(count)} for period, count in counts.items()]

    return {
        "postings_by_month": _monthly("post_date"),
        "deadlines_by_month": _monthly("close_date"),
        "filters_applied": {"category": category, "funder": funder},
        "note": "Only rows with a known post_date/close_date are counted; both columns can be missing for some rows.",
    }


@router.get("/funders")
def funders(
    limit: int = Query(default=15, ge=1, le=100),
    category: str | None = Query(default=None),
):
    df = get_corpus().df
    df = _apply_filters(df, category, None, None)
    by_count = df["agency_name"].dropna().value_counts().head(limit)
    funding = df.dropna(subset=["agency_name", "estimated_total_program_funding"])
    by_funding = (
        funding.groupby("agency_name")["estimated_total_program_funding"].sum().sort_values(ascending=False).head(limit)
    )
    return {
        "top_by_opportunity_count": [{"funder": k, "count": int(v)} for k, v in by_count.items()],
        "top_by_total_estimated_funding": [
            {"funder": k, "total_estimated_funding": _clean_num(v)} for k, v in by_funding.items()
        ],
        "funding_sample_size": int(len(funding)),
        "total_funders": int(df["agency_name"].dropna().nunique()),
        "filters_applied": {"category": category},
    }


@router.get("/compare")
def compare(
    dimension: str = Query(..., description="Only 'category' is supported currently"),
    a: str = Query(..., description="First value to compare, e.g. 'Health'"),
    b: str = Query(..., description="Second value to compare, e.g. 'Education'"),
):
    """A real side-by-side comparison, computed from the actual corpus --
    not a static chart. Currently only compares by category (the only
    dimension with clean, structured values in the historical corpus);
    other dimensions (geography, source type) don't have equivalent
    structured columns here yet -- see /analytics/live/geography for the
    live-opportunity equivalent."""
    if dimension != "category":
        raise HTTPException(status_code=422, detail="Only dimension='category' is currently supported")

    df = get_corpus().df

    def _stats(value: str) -> dict:
        subset = _apply_filters(df, value, None, None)
        ceiling = subset["award_ceiling"].dropna()
        funding = subset["estimated_total_program_funding"].dropna()
        return {
            "label": value,
            "total_rows": int(len(subset)),
            "open_count": int((subset["is_expired"] == False).sum()),  # noqa: E712
            "expired_count": int((subset["is_expired"] == True).sum()),  # noqa: E712
            "unique_funders": int(subset["agency_name"].dropna().nunique()),
            "avg_award_ceiling": _clean_num(ceiling.mean()) if len(ceiling) else None,
            "median_award_ceiling": _clean_num(ceiling.median()) if len(ceiling) else None,
            "total_estimated_funding": _clean_num(funding.sum()) if len(funding) else None,
            "funding_sample_size": int(len(funding)),
        }

    return {"dimension": dimension, "a": _stats(a), "b": _stats(b)}


@router.get("/categories")
def categories(limit: int = Query(default=15, ge=1, le=100)):
    """"category" here means sector (Health/Education/Environment/...) --
    see _apply_filters's comment on why this reads `category_of_funding_
    activity`, not the confusingly-named `opportunity_category` column."""
    df = get_corpus().df
    counts = df["category_of_funding_activity"].dropna().value_counts().head(limit)
    avg_award = (
        df.dropna(subset=["category_of_funding_activity", "award_ceiling"])
        .groupby("category_of_funding_activity")["award_ceiling"]
        .mean()
    )
    result = []
    for cat, count in counts.items():
        result.append({
            "category": cat,
            "count": int(count),
            "avg_award_ceiling": _clean_num(avg_award.get(cat)),
        })
    return {
        "categories": result,
        "total_categories": int(df["category_of_funding_activity"].dropna().nunique()),
    }


@router.get("/eligibility")
def eligibility(limit: int = Query(default=15, ge=1, le=100)):
    df = get_corpus().df
    counts = df["eligible_applicants_type"].dropna().value_counts().head(limit)
    return {"eligible_applicant_types": [{"type": k, "count": int(v)} for k, v in counts.items()]}


@router.get("/explorer")
def data_explorer(
    q: str | None = Query(default=None, description="Substring filter on title"),
    category: str | None = Query(default=None),
    funder: str | None = Query(default=None),
    status: str | None = Query(default=None, description="open | expired"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
):
    df = get_corpus().df
    if q:
        df = df[df["opportunity_title"].fillna("").str.lower().str.contains(q.lower())]
    df = _apply_filters(df, category, funder, status)

    total = len(df)
    start = (page - 1) * page_size
    page_rows = df.iloc[start:start + page_size]

    columns = [
        "opportunity_id", "opportunity_title", "agency_name", "opportunity_category",
        "funding_instrument_type", "eligible_applicants_type", "post_date", "close_date",
        "award_ceiling", "award_floor", "estimated_total_program_funding", "is_expired",
    ]
    records = []
    for _, row in page_rows.iterrows():
        rec = {}
        for c in columns:
            v = row.get(c)
            if isinstance(v, pd.Timestamp):
                rec[c] = v.date().isoformat() if pd.notna(v) else None
            elif isinstance(v, (float, np.floating)):
                rec[c] = _clean_num(v)
            elif isinstance(v, (np.integer,)):
                rec[c] = int(v)
            elif isinstance(v, (np.bool_, bool)):
                rec[c] = bool(v)
            else:
                rec[c] = v if pd.notna(v) else None
        records.append(rec)

    return {
        "total_rows": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "rows": records,
    }


def _filtered_export_df(q, category, funder, status) -> pd.DataFrame:
    df = get_corpus().df
    if q:
        df = df[df["opportunity_title"].fillna("").str.lower().str.contains(q.lower())]
    df = _apply_filters(df, category, funder, status)
    columns = [
        "opportunity_id", "opportunity_title", "agency_name", "opportunity_category",
        "funding_instrument_type", "eligible_applicants_type", "post_date", "close_date",
        "award_ceiling", "award_floor", "estimated_total_program_funding", "is_expired",
    ]
    # Explicit allow-list of columns only -- guards against ever exporting a
    # column that isn't meant for the public API surface, should the corpus
    # schema grow internal-only fields later.
    return df[columns].copy()


@router.get("/export.csv")
def export_csv(
    q: str | None = None,
    category: str | None = None,
    funder: str | None = None,
    status: str | None = None,
):
    df = _filtered_export_df(q, category, funder, status)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    filename = f"grantsetu_opportunities_{datetime.utcnow().date().isoformat()}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export.json")
def export_json(
    q: str | None = None,
    category: str | None = None,
    funder: str | None = None,
    status: str | None = None,
):
    df = _filtered_export_df(q, category, funder, status)
    records = df.to_json(orient="records", date_format="iso")
    filename = f"grantsetu_opportunities_{datetime.utcnow().date().isoformat()}.json"
    return StreamingResponse(
        iter([records]),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/analysis")
def analysis():
    """Real, lightweight data-science pass using scikit-learn: KMeans
    clustering of opportunities by award size + category (one-hot), reduced
    to 2D with PCA for a scatter plot, plus a real Pearson correlation
    between award_ceiling and estimated_total_program_funding. All numbers
    are computed on-demand from the current corpus; if fewer than 20 rows
    have complete data, clustering is skipped and that's reported directly
    rather than showing an empty or fabricated chart."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    df = get_corpus().df
    subset = df.dropna(subset=["award_ceiling", "award_floor", "opportunity_category"]).copy()

    result = {"sample_size": int(len(subset)), "total_rows": int(len(df))}

    if len(subset) < 20:
        result["clusters"] = None
        result["clustering_note"] = "Fewer than 20 complete rows available -- clustering skipped rather than shown on insufficient data."
    else:
        top_categories = subset["opportunity_category"].value_counts().head(8).index
        subset["category_grouped"] = subset["opportunity_category"].where(
            subset["opportunity_category"].isin(top_categories), "Other"
        )
        cat_dummies = pd.get_dummies(subset["category_grouped"], prefix="cat")
        numeric = subset[["award_ceiling", "award_floor"]].apply(np.log1p)
        features = pd.concat([numeric.reset_index(drop=True), cat_dummies.reset_index(drop=True)], axis=1)

        scaled = StandardScaler().fit_transform(features)
        k = min(5, max(2, len(subset) // 50))
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)

        coords = PCA(n_components=2, random_state=42).fit_transform(scaled)

        points = []
        sample = subset.reset_index(drop=True)
        # Cap the scatter payload for a large corpus -- a representative
        # random sample (fixed seed for reproducibility), not the full 75k
        # points, keeps the response small without misleading the shape of
        # the plot (each cluster's proportion is preserved by random sampling).
        max_points = 1200
        if len(sample) > max_points:
            idx = np.random.RandomState(42).choice(len(sample), max_points, replace=False)
        else:
            idx = np.arange(len(sample))

        for i in idx:
            points.append({
                "x": _clean_num(coords[i, 0]),
                "y": _clean_num(coords[i, 1]),
                "cluster": int(labels[i]),
                "category": sample.loc[i, "category_grouped"],
                "title": sample.loc[i, "opportunity_title"],
            })

        cluster_summary = []
        for c in range(k):
            mask = labels == c
            cluster_summary.append({
                "cluster": c,
                "count": int(mask.sum()),
                "avg_award_ceiling": _clean_num(subset.loc[mask, "award_ceiling"].mean()),
                "top_category": subset.loc[mask, "category_grouped"].mode().iloc[0] if mask.sum() else None,
            })

        result["clusters"] = {
            "k": k,
            "points": points,
            "points_sampled": len(points) < len(subset),
            "cluster_summary": cluster_summary,
        }

    corr_subset = df.dropna(subset=["award_ceiling", "estimated_total_program_funding"])
    if len(corr_subset) >= 5:
        corr = float(corr_subset["award_ceiling"].corr(corr_subset["estimated_total_program_funding"]))
        result["award_ceiling_vs_total_funding_correlation"] = {
            "pearson_r": _clean_num(corr),
            "sample_size": int(len(corr_subset)),
        }
    else:
        result["award_ceiling_vs_total_funding_correlation"] = None

    return result


@router.get("/live-summary")
def live_summary(db: Session = Depends(get_db)):
    """A small, honest summary of the live/imported grants table -- kept
    separate from the historical-corpus endpoints above since it's a very
    different (much smaller, currently-synced-only) dataset."""
    rows = db.query(ImportedGrant).all()
    total = len(rows)
    real = [r for r in rows if not r.is_sample_data]
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    return {
        "total_rows": total,
        "real_rows": len(real),
        "sample_rows": total - len(real),
        "by_status": by_status,
        "note": "Populated only after a source has been synced via POST /admin/sync-source/{source_id}; empty until then.",
    }


def _real_live_grants(db: Session):
    """Non-sample ImportedGrant rows -- the shared base query for every
    live-opportunity analytics endpoint below, so sample/fixture data can
    never leak into an analytical number the same way it can never leak
    into the live feed itself."""
    return db.query(ImportedGrant).filter_by(is_sample_data=False).all()


@router.get("/live/geography")
def live_geography(db: Session = Depends(get_db)):
    """Country breakdown of LIVE opportunities (src/ai/research.py::
    _detect_country -- never fabricated; "unknown" is an honest, common
    outcome, not hidden). Distinct from the historical corpus, which has no
    country concept at all (100% U.S. federal)."""
    grants = _real_live_grants(db)
    by_country: dict[str, int] = {}
    for g in grants:
        by_country[g.country] = by_country.get(g.country, 0) + 1
    return {
        "total_rows": len(grants),
        "by_country": [{"country": k, "count": v} for k, v in sorted(by_country.items(), key=lambda kv: -kv[1])],
        "note": "State/district-level geography is not currently extracted for live opportunities -- only country-level detection exists.",
    }


@router.get("/live/deadlines")
def live_deadlines(db: Session = Depends(get_db)):
    """Buckets live opportunities by deadline proximity -- real dates, real
    bucket boundaries, computed at request time. An opportunity with no
    known deadline goes in its own honest bucket, never silently dropped or
    counted as "later"."""
    from src.ranking.live_match import RECOMMENDABLE_STATUSES

    now = datetime.utcnow()
    grants = [g for g in _real_live_grants(db) if g.status in RECOMMENDABLE_STATUSES]

    buckets = {"closing_this_week": 0, "closing_this_month": 0, "closing_next_3_months": 0, "later": 0, "no_deadline": 0}
    for g in grants:
        if g.deadline is None:
            buckets["no_deadline"] += 1
            continue
        days = (g.deadline - now).days
        if days < 0:
            continue  # already resolved to a closed-like status upstream; shouldn't appear here, but never double-count as "later"
        elif days <= 7:
            buckets["closing_this_week"] += 1
        elif days <= 30:
            buckets["closing_this_month"] += 1
        elif days <= 90:
            buckets["closing_next_3_months"] += 1
        else:
            buckets["later"] += 1

    return {
        "total_active_rows": len(grants),
        "buckets": buckets,
        "note": "Only opportunities with an active/unknown status are counted (matches the live feed's default view) -- closed/expired opportunities are excluded.",
    }


@router.get("/live/funders")
def live_funders(db: Session = Depends(get_db), limit: int = Query(default=15, ge=1, le=100)):
    grants = _real_live_grants(db)
    by_funder: dict[str, int] = {}
    for g in grants:
        if g.funder:
            by_funder[g.funder] = by_funder.get(g.funder, 0) + 1
    top = sorted(by_funder.items(), key=lambda kv: -kv[1])[:limit]
    return {
        "total_rows": len(grants),
        "unique_funders": len(by_funder),
        "top_funders": [{"funder": k, "count": v} for k, v in top],
        "unattributed": sum(1 for g in grants if not g.funder),
        "note": "Only opportunities where a funder name was successfully extracted are counted.",
    }
