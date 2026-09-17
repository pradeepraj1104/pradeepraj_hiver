"""
Step 11: Failure Analysis - find the top failure modes on the golden set.

Groups misclassifications by (true_intent -> predicted_intent) pair, since
that's almost always more informative than looking at individual wrong
predictions one at a time: it tells you WHICH intents are actually
confusable, which is what you fix (by merging intents, adding keywords, or
adding more training examples for that pair).
"""

from pathlib import Path

import pandas as pd


def analyze_failures(golden: pd.DataFrame, model_preds, out_path: str, top_n: int = 5):

    print("\n" + "=" * 60)
    print("FAILURE ANALYSIS")
    print("=" * 60)

    df = golden.copy().reset_index(drop=True)
    df["predicted_intent"] = list(model_preds)
    wrong = df[df["true_intent"] != df["predicted_intent"]]

    if len(wrong) == 0:
        print("No misclassifications on the golden set - nothing to analyze "
              "(or the golden set is too small/still unreviewed to be meaningful).")
        return pd.DataFrame()

    confusion_counts = (
        wrong.groupby(["true_intent", "predicted_intent"])
        .size()
        .sort_values(ascending=False)
        .head(top_n)
    )

    rows = []
    for (true_i, pred_i), count in confusion_counts.items():
        example_row = wrong[
            (wrong["true_intent"] == true_i) & (wrong["predicted_intent"] == pred_i)
        ].iloc[0]

        hypothesis = _hypothesize(true_i, pred_i)

        rows.append({
            "problem": f"'{true_i}' misclassified as '{pred_i}'",
            "count": count,
            "example_text": example_row["text"],
            "hypothesis": hypothesis,
            "suggested_improvement": _suggest_fix(true_i, pred_i),
        })

    result = pd.DataFrame(rows)

    print(f"\nTop {len(result)} failure modes:\n")
    for _, r in result.iterrows():
        print(f"- {r['problem']}  (x{r['count']})")
        print(f"    example: {r['example_text'][:100]}")
        print(f"    hypothesis: {r['hypothesis']}")
        print(f"    fix: {r['suggested_improvement']}\n")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\u2713 Saved failure analysis -> {out_path}")

    return result


def _hypothesize(true_intent: str, pred_intent: str) -> str:
    return (
        f"Messages that are genuinely about '{true_intent}' apparently share "
        f"enough vocabulary with typical '{pred_intent}' messages that the "
        "model (and/or the keyword rules used to bootstrap training labels) "
        "can't reliably tell them apart from wording alone."
    )


def _suggest_fix(true_intent: str, pred_intent: str) -> str:
    return (
        f"Review whether '{true_intent}' and '{pred_intent}' should be merged "
        f"into one intent; if not, add more labeled examples that contrast the "
        f"two, and add/adjust keywords that are specific to '{true_intent}' "
        "only."
    )
