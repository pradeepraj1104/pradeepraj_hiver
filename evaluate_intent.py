"""
Step 10 (intent portion): evaluate the trained model AND both baselines on
the held-out Golden Set - never on training rows, never on labels the
model or baseline itself produced.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from evaluation.baselines import keyword_baseline_predict, majority_baseline_predict


def _score(y_true, preds, name):
    acc = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="macro", zero_division=0
    )
    print(f"\n{name}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Macro F1:  {f1:.4f}")
    return {"accuracy": acc, "macro_precision": precision, "macro_recall": recall, "macro_f1": f1}


def evaluate_intent_model(model, training_pool: pd.DataFrame, golden: pd.DataFrame,
                           golden_is_reviewed: bool, out_dir: str):

    print("\n" + "=" * 60)
    print("INTENT EVALUATION (held-out golden set)")
    print("=" * 60)

    if not golden_is_reviewed:
        print(
            "\u26a0 Golden set is NOT human-reviewed yet. Every number below is "
            "provisional - see data/golden/golden_set_TO_REVIEW.csv.\n"
        )

    y_true = golden["true_intent"]
    texts = golden["text"]

    majority_preds = majority_baseline_predict(training_pool["intent"], len(golden))
    keyword_preds = keyword_baseline_predict(texts)
    model_preds = model.predict(texts)

    results = {
        "golden_set_human_reviewed": golden_is_reviewed,
        "n_golden_examples": len(golden),
        "baseline_majority_class": _score(y_true, majority_preds, "Baseline 1: Majority class"),
        "baseline_keyword_rules": _score(y_true, keyword_preds, "Baseline 2: Keyword rules"),
        "your_model": _score(y_true, model_preds, "Your model"),
    }

    print("\nPer-class report (your model):")
    print(classification_report(y_true, model_preds, zero_division=0))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([results["your_model"]]).to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "metrics_full.json", "w") as f:
        json.dump(results, f, indent=2)

    labels_sorted = sorted(y_true.unique())
    cm = confusion_matrix(y_true, model_preds, labels=labels_sorted)
    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(out_dir / "confusion_matrix.csv")

    print(f"\n\u2713 Saved metrics.csv, metrics_full.json, confusion_matrix.csv -> {out_dir}")

    if results["your_model"]["accuracy"] < 0.90:
        print(
            "\nNOTE: below 90%. Once the golden set is genuinely human-reviewed, "
            "this is almost always fixed by (1) merging intents that overlap in "
            "meaning, (2) adding more real examples for whichever intents show "
            "up worst in the confusion matrix, or (3) checking golden-label "
            "consistency."
        )

    return results, model_preds
