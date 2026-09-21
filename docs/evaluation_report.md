# GrantSetu — Evaluation Report

## Status: scaffold built, labelling not yet performed

Per blueprint Section G, no ground-truth relevance labels exist for this
dataset. This MVP build includes the full scaffold to create and score an
honest, manually curated evaluation set, but **does not fabricate the labels
themselves** -- that step requires a human reviewer.

## What exists now

- `data/evaluation/ngo_profiles.json` -- 10 synthetic NGO profiles spanning
  distinct focus-area archetypes (education, health, disaster relief, girls'
  education, conservation, youth employment, agriculture, mental
  health/disability, water/sanitation, arts/culture), per blueprint G.1.
- `src/evaluation/run_eval.py --generate-candidates` -- runs the actual ranking
  engine for each profile and writes its top-20 candidates to
  `data/evaluation/candidates.csv` (profile, rank, opportunity, score, and an
  empty `label` column).
- `src/evaluation/metrics.py` -- Precision@K, Recall@K, MRR, and duplicate-rate
  implementations, unit-tested independently of any specific dataset.
- `src/evaluation/run_eval.py --score` -- computes and prints these metrics,
  but only once `data/evaluation/labels.csv` exists (a human-reviewed copy of
  candidates.csv with the `label` column filled in as `relevant` /
  `partially_relevant` / `not_relevant`). Running `--score` without labels
  raises an error rather than printing an invented number.

## To complete this section

1. `python -m src.evaluation.run_eval --generate-candidates`
2. Open `data/evaluation/candidates.csv`, review each row against its
   opportunity's actual title/category, and fill in `label`.
3. Save as `data/evaluation/labels.csv`.
4. `python -m src.evaluation.run_eval --score`
5. Paste the resulting numbers into this report, replacing this section.

## Do not

Do not report Precision@K, MRR, or any other ranking-quality number in the
README, resume, or portfolio materials until step 4 above has actually been
run and its output is pasted here.
