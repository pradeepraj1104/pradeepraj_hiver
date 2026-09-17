"""
Step 4 (define intents), Step 5 (build intent model), and the Golden Set
(Step 9) live here.

DESIGN NOTE - why weak labels are split into two separate uses
-----------------------------------------------------------------
`weak_label()` is a keyword classifier. It is used for two DIFFERENT jobs
that must never be mixed:

  (a) Bootstrapping labels for a LARGE training pool (noisy but plentiful -
      this is fine, it's standard weak supervision).
  (b) Acting as the "keyword baseline" that gets judged against ground
      truth it did NOT produce - the human-reviewed Golden Set.

If you ever train a model on weak labels and then evaluate it against
those SAME weak labels, you get a model that appears to be near-perfect at
imitating an if/else keyword function - not a validated intent classifier.
That bug is deliberately designed out of this codebase: the Golden Set is
excluded from the training pool at creation time, and evaluation always
reads from the (ideally human-corrected) Golden Set only.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

INTENTS = [
    "Refund",
    "Payment Issue",
    "Account Problem",
    "Login Problem",
    "Delivery Problem",
    "Cancellation",
    "Subscription",
    "Technical Issue",
    "Complaint",
    "Other",
]

# Step 4: Define Intents (edit these keywords if you rename/merge intents
# after actually reading a batch of real customer messages, as the
# assignment instructs).
KEYWORDS = {
    "Refund": ["refund", "money back", "reimburse", "return my money"],
    "Payment Issue": ["charged", "charge", "payment", "billing", "bill", "card", "transaction", "duplicate charge"],
    "Account Problem": ["account", "profile", "account locked", "account suspended"],
    "Login Problem": ["login", "log in", "sign in", "password", "can't access", "cannot access"],
    "Delivery Problem": ["delivery", "delivered", "package", "parcel", "shipping", "shipment", "where is my order", "not arrived", "late"],
    "Cancellation": ["cancel", "cancellation", "stop order"],
    "Subscription": ["subscription", "subscribe", "unsubscribe", "membership", "plan"],
    "Technical Issue": ["bug", "error", "crash", "broken", "website", "app", "technical"],
    "Complaint": ["complaint", "terrible", "worst", "angry", "unhappy", "disappointed"],
}


def weak_label(text: str) -> str:
    """Keyword-rule intent guess. See module docstring for correct usage."""
    text = text.lower()
    scores = {intent: sum(1 for kw in kws if kw in text) for intent, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"


def create_golden_set(pairs: pd.DataFrame, golden_dir: str, golden_size: int = 200, seed: int = 42):
    """Step 9: sample held-out examples, weak-label them as a DRAFT, and
    look for a human-reviewed version before trusting them as ground truth.

    Returns (golden_df, remaining_pairs_df, is_human_reviewed: bool)
    """
    golden_dir = Path(golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)

    n = min(golden_size, len(pairs))
    golden_source = pairs.sample(n, random_state=seed).copy()
    remaining_pairs = pairs.drop(golden_source.index).reset_index(drop=True)

    golden_source["text"] = golden_source["customer_text"]
    golden_source["true_intent"] = golden_source["text"].apply(weak_label)
    golden_source["expected_handling"] = "escalate"
    golden_source["expected_reply"] = ""

    golden_draft = golden_source[["text", "true_intent", "expected_handling", "expected_reply"]]

    draft_path = golden_dir / "golden_set_TO_REVIEW.csv"
    golden_draft.to_csv(draft_path, index=False)
    print(f"\u2713 Draft golden set ({len(golden_draft)} rows) -> {draft_path}")

    reviewed_path = golden_dir / "golden_set_reviewed.csv"
    if reviewed_path.exists():
        golden = pd.read_csv(reviewed_path)
        print(f"\u2713 Using human-reviewed golden set: {reviewed_path}")
        is_reviewed = True
    else:
        golden = golden_draft
        print(
            "\u26a0 No reviewed golden set found yet.\n"
            f"  ACTION: open {draft_path}, hand-correct 'true_intent' for every row,\n"
            f"  save as {reviewed_path}, then re-run. Until then, every accuracy\n"
            "  number printed is PROVISIONAL (keyword baseline vs. its own labels)."
        )
        is_reviewed = False

    return golden, remaining_pairs, is_reviewed


def build_training_pool(remaining_pairs: pd.DataFrame, out_path: str, pool_size: int = 20000,
                         max_other_ratio: float = 2.0, seed: int = 42) -> pd.DataFrame:
    """Step 5 data prep: weak-label a LARGE pool (excludes the Golden Set
    entirely) so the real model gets more signal than 150-200 rows would give.
    Caps the 'Other' bucket so it doesn't drown out real intents.
    """

    n = min(pool_size, len(remaining_pairs))
    pool = remaining_pairs.sample(n, random_state=seed).copy()
    pool["text"] = pool["customer_text"]
    pool["intent"] = pool["text"].apply(weak_label)

    print("\nRaw weak-label distribution:")
    print(pool["intent"].value_counts())

    counts = pool["intent"].value_counts()
    non_other = counts.drop("Other", errors="ignore")

    if "Other" in counts and len(non_other) > 0:
        cap = max(int(non_other.max() * max_other_ratio), 1)
        other_rows = pool[pool["intent"] == "Other"]
        if len(other_rows) > cap:
            keep_other = other_rows.sample(cap, random_state=seed)
            pool = pd.concat([pool[pool["intent"] != "Other"], keep_other])
            pool = pool.sample(frac=1, random_state=seed).reset_index(drop=True)
            print("\nBalanced distribution (Other capped):")
            print(pool["intent"].value_counts())

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pool[["text", "intent"]].to_csv(out_path, index=False)
    print(f"\u2713 Training pool saved: {out_path} ({len(pool)} rows)")

    return pool[["text", "intent"]]


def _candidate_models():
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
        "linear_svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=1.0), cv=3),
        "naive_bayes": MultinomialNB(alpha=0.3),
    }


def train_intent_model(training_pool: pd.DataFrame, model_path: str, cv_splits: int = 5):
    """Step 5: cross-validate a few model families, keep the best by
    macro-F1 (robust to class imbalance, unlike raw accuracy), fit on all
    training data, and save it.
    """

    print("\n" + "=" * 60)
    print("TRAINING INTENT MODEL")
    print("=" * 60)

    X, y = training_pool["text"], training_pool["intent"]

    if y.nunique() < 2:
        raise ValueError("Training pool has fewer than 2 intent classes - not enough data/variety yet.")

    n_splits = max(2, min(cv_splits, y.value_counts().min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = {}
    for name, clf in _candidate_models().items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", clf),
        ])
        try:
            scores = cross_val_score(pipe, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
            results[name] = scores.mean()
            print(f"  {name:15s} CV macro-F1 = {scores.mean():.4f}")
        except ValueError as e:
            print(f"  {name:15s} skipped ({e})")

    if not results:
        raise RuntimeError("No candidate model could be cross-validated - check your training pool size/variety.")

    best_name = max(results, key=results.get)
    print(f"\n>> Best model: {best_name}")

    best_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("classifier", _candidate_models()[best_name]),
    ])
    best_pipe.fit(X, y)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, model_path)
    print(f"\u2713 Model saved: {model_path}")

    return best_pipe, best_name, results
