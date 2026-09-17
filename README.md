# Hiver SDE Intern Assignment — AI Support Agent

An AI support agent that classifies customer messages by intent, retrieves
similar past resolutions, drafts a grounded reply, and decides whether to
auto-handle or escalate.

## Setup

```bash
pip install -r requirements.txt
```

## Get the data

1. Download **`thoughtvector/customer-support-on-twitter`** from Kaggle.
2. Place `twcs.csv` at `data/raw/twcs.csv`.
   *(Don't point the pipeline at the full ~3M-row file directly for
   exploration — `run_all.py` takes a subsample itself via `SAMPLE_SIZE`,
   but the initial load still reads the whole CSV into memory once, so a
   pre-trimmed subsample, e.g. 100k–300k rows, is recommended if memory is
   limited.)*

## Run

```bash
python run_all.py
```

This runs the entire pipeline once and prints/saves results. Then:

```bash
# 1. Open and hand-correct the labels:
open data/golden/golden_set_TO_REVIEW.csv

# 2. Save your corrected copy as:
data/golden/golden_set_reviewed.csv

# 3. Re-run for trustworthy numbers:
python run_all.py
```

**Why the two-pass workflow matters:** the first pass drafts golden-set
labels using simple keyword rules. Until a human corrects them, any
"accuracy" number is really just measuring "does the model agree with an
if/else keyword function" — not real intent-classification quality. The
pipeline refuses to pretend otherwise: it prints an explicit ⚠ warning on
every run until it finds `golden_set_reviewed.csv`.

## Run tests

```bash
pytest tests/
```

## Project structure

```
hiver-support-agent/
├── data/
│   ├── raw/twcs.csv                  # you provide this
│   ├── processed/                    # sample, conversation pairs, training pool
│   └── golden/                       # draft + human-reviewed golden set
├── models/
│   ├── intent_model.joblib
│   └── retriever.joblib
├── src/
│   ├── data_loader.py                # Steps 1-3: load, analyze, sample
│   ├── preprocessing.py              # cleaning, brand selection, pairing
│   ├── intent_classifier.py          # Steps 4-5, 9: intents, weak labels, golden set, training
│   ├── retriever.py                  # Step 6: TF-IDF (default) or sentence-transformer retrieval
│   ├── response_generator.py         # Step 7: grounded reply drafting
│   ├── escalation.py                 # Step 8: auto-handle vs escalate rules
│   └── pipeline.py                   # ties it all together per message
├── evaluation/
│   ├── baselines.py                  # majority-class + keyword baselines
│   ├── evaluate_intent.py            # Step 10: model vs baselines, no leakage
│   ├── evaluate_replies.py           # Step 10: reply groundedness/relevance + optional LLM judge
│   ├── failure_analysis.py           # Step 11: top confusion patterns
│   ├── metrics.csv / metrics_full.json / confusion_matrix.csv
│   ├── failure_analysis.csv
│   ├── reply_quality.csv
│   └── REPORT.txt                    # Step 12: generated summary report
├── tests/test_pipeline.py            # fast unit tests, no dataset needed
├── run_all.py                        # single entry point, runs every step in order
├── requirements.txt
└── README.md
```

## Non-obvious decisions (and why)

1. **The golden set is excluded from training data at creation time**, not
   after. Sampling it first and dropping those rows from the training pool
   is what makes "no leakage" actually true, rather than just documented.
2. **Weak (keyword) labels are used to bootstrap a large training pool
   (thousands of rows), never as the final evaluation ground truth.**
   Evaluating a model against the same labeling function that trained it
   just measures how well it imitates that function.
3. **"Other" is capped during training** (default: at most 2x the largest
   real-intent count). Keyword rules miss most real messages by design, so
   uncapped weak labeling produces a training set that's mostly "Other" —
   the model then just learns to predict "Other" most of the time.
4. **Model selection is by cross-validated macro-F1, not accuracy.**
   With imbalanced intents, a model that only ever predicts the majority
   class can still post high accuracy; macro-F1 penalizes that.
5. **Retrieval defaults to TF-IDF, not sentence-transformer embeddings.**
   It needs no internet access or model download, is deterministic, and is
   still a legitimate "embeddings + cosine similarity" implementation.
   Swapping `RETRIEVAL_BACKEND = "sentence-transformer"` in `run_all.py`
   is a one-line upgrade if paraphrase-level matching is needed.
6. **Replies are template + evidence based, not LLM-generated**, so the
   whole pipeline runs in seconds with no API key and is easy to grade line
   by line. `evaluate_replies.py` includes an optional LLM-as-judge hook
   (`ANTHROPIC_API_KEY`) for scoring reply quality without needing an LLM
   to generate the replies themselves.
7. **Sensitive intents (Payment Issue, Account Problem, Complaint) always
   escalate**, regardless of model confidence — the cost of a wrong
   auto-handled reply on these is asymmetric (money, account access, an
   already-frustrated customer).
8. **Confidence and retrieval-evidence strength are checked together**,
   not independently. A confident intent prediction with no similar past
   case to ground a reply in is still routed to a human.
9. **Failure analysis groups by (true → predicted) intent pairs**, not by
   individual wrong examples — that's what's actionable (merge two
   intents? add contrasting examples? adjust keywords?), whereas a flat
   list of wrong predictions mostly isn't.
10. **`select_brand` picks the account with the most outbound replies**,
    not the most inbound complaints — richer, more consistent reply data
    to learn support-response patterns from.
11. **Training pool size (20k) is deliberately much larger than the golden
    set (200).** More noisy data reliably beats less "clean" data for this
    kind of text classification; the golden set's job is trustworthy
    *evaluation*, not training volume.
12. **`run_all.py` degrades gracefully on tiny data** (prints a warning and
    continues instead of crashing) so the code can be sanity-checked on a
    small sample before pointing it at the full dataset.

## Known limitation

With a small input sample, several steps (golden set size, training pool
size, cross-validation folds) auto-shrink to fit — useful for verifying the
code runs, but not meaningful for real accuracy. Point `data/raw/twcs.csv`
at a proper Kaggle subsample (tens of thousands of rows) for numbers worth
reporting.
