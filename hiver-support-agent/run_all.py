"""
Hiver SDE Intern Assignment - single entry point.

Runs every step from the assignment brief in order:
  1  Get Dataset            -> src.data_loader.load_dataset
  2  Analyze Data           -> src.data_loader.analyze_data
  3  Select Brand           -> src.preprocessing.select_brand
  4  Define Intents         -> src.intent_classifier.INTENTS / KEYWORDS
  5  Build Intent Model     -> src.intent_classifier.train_intent_model (+ baselines)
  6  Historical Retrieval   -> src.retriever.build_retrieval_index
  7  Generate Reply         -> src.response_generator.generate_reply
  8  Auto-handle/Escalate   -> src.escalation.escalation_decision
  9  Create Golden Set      -> src.intent_classifier.create_golden_set
  10 Evaluate               -> evaluation.evaluate_intent / evaluate_replies
  11 Failure Analysis       -> evaluation.failure_analysis
  12 Report + README        -> writes evaluation/REPORT.txt (README.md is separate, static)
  13 GitHub Repo            -> this folder structure IS the repo; see README.md

USAGE
-----
1. Put your Kaggle subsample at data/raw/twcs.csv (or edit DATA_FILE below
   to point at a smaller sample while you're testing).
2. python run_all.py
3. Open data/golden/golden_set_TO_REVIEW.csv, hand-correct 'true_intent'
   for every row, save it as data/golden/golden_set_reviewed.csv.
4. Run python run_all.py again - it will now report trustworthy accuracy.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from src.data_loader import load_dataset, analyze_data, create_sample
from src.preprocessing import clean_text, select_brand, create_pairs
from src.intent_classifier import create_golden_set, build_training_pool, train_intent_model
from src.retriever import build_retrieval_index
from src.response_generator import generate_reply
from src.escalation import escalation_decision
from src.pipeline import run_agent
from evaluation.evaluate_intent import evaluate_intent_model
from evaluation.failure_analysis import analyze_failures
from evaluation.evaluate_replies import evaluate_replies_heuristic


# ============================================================
# CONFIG - the only section you should need to edit
# ============================================================
DATA_FILE = "data/raw/twcs.csv"          # point this at your Kaggle subsample
SAMPLE_SIZE = 150000                     # Step 3: subsample size from the raw file
GOLDEN_SIZE = 200                        # Step 9: golden set size (150-250 per assignment)
TRAINING_POOL_SIZE = 20000               # Step 5: weak-labeled training pool size
MIN_PAIRS_REQUIRED = 20                  # sanity floor before continuing
RETRIEVAL_BACKEND = "tfidf"              # "tfidf" (offline) or "sentence-transformer" (needs internet + pip install)


def main():
    print("=" * 70)
    print("        HIVER SDE AI CUSTOMER SUPPORT AGENT")
    print("=" * 70)

    # ---- Step 1: Get Dataset ----
    df = load_dataset(DATA_FILE)

    # ---- Step 2: Analyze Data ----
    analyze_data(df)

    # ---- Step 3: subsample + clean ----
    create_sample(df, SAMPLE_SIZE, "data/processed/sample.csv", clean_text)

    # ---- Step 3b: Select Brand ----
    brand = select_brand(df)

    # ---- Step 4: build conversation pairs for that brand ----
    pairs = create_pairs(df, brand, "data/processed/conversation_pairs.csv")

    if len(pairs) < MIN_PAIRS_REQUIRED:
        print(
            f"\nOnly {len(pairs)} conversation pairs found for '{brand}' - "
            f"need at least {MIN_PAIRS_REQUIRED}. This usually means your "
            "input file is a tiny sample, not the full Kaggle subsample. "
            "Replace data/raw/twcs.csv with a bigger subsample and re-run."
        )
        if len(pairs) == 0:
            return
        print("Continuing anyway with what's available, for demonstration purposes only.\n")

    # ---- Step 9: Golden Set (held out BEFORE training - no leakage) ----
    golden_size = min(GOLDEN_SIZE, max(len(pairs) // 3, 1))
    golden, remaining_pairs, golden_is_reviewed = create_golden_set(
        pairs, "data/golden", golden_size=golden_size
    )

    # ---- Step 5: weak-labeled training pool (excludes golden set) ----
    pool_size = min(TRAINING_POOL_SIZE, len(remaining_pairs)) if len(remaining_pairs) else 0
    if pool_size == 0:
        print("\nNo data left to build a training pool after holding out the golden set.")
        return

    training_pool = build_training_pool(
        remaining_pairs, "data/processed/training_pool.csv", pool_size=pool_size
    )

    # ---- Step 5: train + select best model ----
    model, best_model_name, cv_results = train_intent_model(
        training_pool, "models/intent_model.joblib"
    )

    # ---- Step 10: Evaluate intent model + baselines on golden set ----
    eval_results, model_preds = evaluate_intent_model(
        model, training_pool, golden, golden_is_reviewed, "evaluation"
    )

    # ---- Step 11: Failure Analysis ----
    analyze_failures(golden, model_preds, "evaluation/failure_analysis.csv")

    # ---- Step 6: Historical Retrieval index ----
    retriever = build_retrieval_index(
        pairs, "models/retriever.joblib", backend=RETRIEVAL_BACKEND
    )

    # ---- Steps 7-8: run the full agent on a few sample messages ----
    print("\n" + "=" * 70)
    print("RUNNING THE FULL AGENT ON SAMPLE MESSAGES")
    print("=" * 70)

    sample_messages = golden["text"].head(5).tolist() if len(golden) else [
        "My payment was charged twice and I need help with the duplicate charge."
    ]

    agent_records = []
    for msg in sample_messages:
        result = run_agent(msg, model, retriever, generate_reply, escalation_decision)
        agent_records.append(result)

        print(f"\nCUSTOMER: {result['customer_message']}")
        print(f"INTENT: {result['intent']}  (confidence {result['confidence']})")
        print(f"DECISION: {result['decision']}  ({result['reason']})")
        print(f"DRAFT REPLY: {result['draft_reply']}")

    # ---- Step 10 (reply side): heuristic reply-quality scoring ----
    evaluate_replies_heuristic(agent_records, "evaluation/reply_quality.csv")

    # ---- Step 12: Report ----
    write_report(df, pairs, brand, best_model_name, eval_results, golden_is_reviewed)

    print("\n" + "=" * 70)
    print("PROJECT COMPLETE - see evaluation/REPORT.txt and README.md")
    print("=" * 70)


def write_report(df, pairs, brand, best_model_name, eval_results, golden_is_reviewed):
    report = f"""HIVER SDE INTERN ASSIGNMENT - RESULTS
======================================

Dataset
-------
Total tweets in input file: {len(df)}
Conversation pairs for selected brand: {len(pairs)}
Selected support account: {brand}

Intent Model
------------
Best model (selected by cross-validated macro-F1): {best_model_name}
Trained on a weak-labeled pool that EXCLUDES the golden set (no leakage).

Evaluation (held-out golden set, human-reviewed: {golden_is_reviewed})
------------------------------------------------------------------
Baseline 1 (majority class):   accuracy = {eval_results['baseline_majority_class']['accuracy']:.4f}
Baseline 2 (keyword rules):    accuracy = {eval_results['baseline_keyword_rules']['accuracy']:.4f}
Your model ({best_model_name}): accuracy = {eval_results['your_model']['accuracy']:.4f}
                                macro-F1  = {eval_results['your_model']['macro_f1']:.4f}

{'NOTE: golden set is still WEAK-LABELED (not human-reviewed). These numbers are provisional - see data/golden/golden_set_TO_REVIEW.csv.' if not golden_is_reviewed else 'Golden set has been human-reviewed - these numbers are trustworthy.'}

System Flow
-----------
Customer Message -> Intent Classifier -> Historical Retrieval ->
Grounded Reply -> Confidence + Risk Check -> AUTO-HANDLE / ESCALATE

Safety Logic
------------
Sensitive intents (Payment Issue, Account Problem, Complaint) always escalate.
High confidence (>=0.80) + strong historical evidence (>=0.60) -> AUTO-HANDLE.
Everything else -> ESCALATE.

See evaluation/failure_analysis.csv for the top failure modes and
evaluation/reply_quality.csv for heuristic reply scoring.
"""
    Path("evaluation").mkdir(parents=True, exist_ok=True)
    with open("evaluation/REPORT.txt", "w", encoding="utf8") as f:
        f.write(report)
    print("\n\u2713 Report written to evaluation/REPORT.txt")


if __name__ == "__main__":
    main()
