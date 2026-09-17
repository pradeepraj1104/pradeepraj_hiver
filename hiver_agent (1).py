import os
import re
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "twcs.csv"

BASE = Path("hiver_project")

RAW = BASE / "data" / "raw"
PROCESSED = BASE / "data" / "processed"
GOLDEN = BASE / "data" / "golden"
MODELS = BASE / "models"
EVALUATION = BASE / "evaluation"

SAMPLE_SIZE = 150000
GOLDEN_SIZE = 200
RETRIEVAL_SIZE = 20000

# How many weak-labeled examples to train the "real" model on.
# This is separate from, and much larger than, the Golden Set.
TRAINING_POOL_SIZE = 20000

# Cap how dominant the "Other" class is allowed to be in training data.
# Weak labeling almost always leaves most messages unmatched -> "Other"
# floods the training set and the model just learns to predict it.
MAX_OTHER_RATIO = 2.0  # "Other" capped at 2x the size of the largest real intent

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
    "Other"
]


# ============================================================
# CREATE FOLDERS
# ============================================================

def create_folders():

    folders = [RAW, PROCESSED, GOLDEN, MODELS, EVALUATION]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print("\u2713 Project folders created")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# STEP 1 - LOAD DATA
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_FILE):
        print("\nERROR:\nPut twcs.csv in the same folder as this Python file.\n")
        return None

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_FILE)

    print("Original dataset:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ============================================================
# STEP 2 - DATA ANALYSIS
# ============================================================

def analyze_data(df):

    print("\n" + "=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)

    print("\nNumber of tweets:", len(df))
    print("Unique authors:", df["author_id"].nunique())
    print("Missing values:")
    print(df.isnull().sum())

    print("\nInbound / Customer messages:")
    print(df["inbound"].value_counts())

    print("\nAverage message length:", df["text"].fillna("").str.len().mean())
    print("\nNumber of conversations:", df["in_response_to_tweet_id"].notna().sum())

    print("\nTop company accounts:")
    company_messages = df[df["inbound"] == False]
    companies = (
        company_messages.groupby("author_id").size()
        .sort_values(ascending=False).head(20)
    )
    print(companies)

    return companies


# ============================================================
# STEP 3 - SAMPLE DATA
# ============================================================

def create_sample(df):

    print("\nCreating sample...")

    size = min(SAMPLE_SIZE, len(df))
    sample = df.sample(size, random_state=42)
    sample["text"] = sample["text"].apply(clean_text)
    sample = sample[sample["text"].str.len() > 3]

    path = PROCESSED / "sample.csv"
    sample.to_csv(path, index=False)

    print(f"\u2713 Saved {len(sample)} rows to {path}")

    return sample


# ============================================================
# STEP 4 - SELECT BRAND
# ============================================================

def select_brand(df):

    print("\n" + "=" * 60)
    print("BRAND ANALYSIS")
    print("=" * 60)

    outbound = df[df["inbound"] == False]
    brands = outbound.groupby("author_id").size().sort_values(ascending=False)

    print("\nTop 20 support accounts:")
    print(brands.head(20))

    selected = brands.index[0]

    print("\nSelected brand/account:")
    print(selected)

    print(
        "\nNOTE:"
        "\nThe dataset anonymizes account IDs."
        "\nThe selected author_id represents the support account."
    )

    return selected


# ============================================================
# STEP 5 - CREATE CONVERSATION PAIRS
# ============================================================

def create_pairs(df, brand):

    print("\nCreating customer -> company response pairs...")

    customers = df[df["inbound"] == True].copy()
    responses = df[df["inbound"] == False].copy()

    pairs = customers.merge(
        responses,
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_customer", "_company")
    )

    pairs = pairs[pairs["author_id_company"] == brand]

    pairs = pairs[[
        "tweet_id_customer", "text_customer",
        "tweet_id_company", "text_company"
    ]]

    pairs.columns = ["customer_id", "customer_text", "response_id", "response_text"]

    pairs["customer_text"] = pairs["customer_text"].apply(clean_text)
    pairs["response_text"] = pairs["response_text"].apply(clean_text)

    pairs = pairs.drop_duplicates().reset_index(drop=True)

    path = PROCESSED / "conversation_pairs.csv"
    pairs.to_csv(path, index=False)

    print("Conversation pairs:", len(pairs))
    print("Saved:", path)

    return pairs


# ============================================================
# STEP 6 - WEAK INTENT LABELING (keyword rules)
# ============================================================
# NOTE: this function has TWO legitimate uses that must stay separate:
#   (a) bootstrapping labels for TRAINING data (large volume, noisy - OK)
#   (b) acting as the "keyword baseline" for EVALUATION (must be compared
#       against ground truth it did NOT produce, i.e. the human-reviewed
#       golden set - never against labels it generated itself)
# Mixing these two uses is what caused the circular-accuracy bug.

KEYWORDS = {
    "Refund": ["refund", "money back", "reimburse", "return my money"],
    "Payment Issue": ["charged", "charge", "payment", "billing", "bill", "card", "transaction", "duplicate charge"],
    "Account Problem": ["account", "profile", "account locked", "account suspended"],
    "Login Problem": ["login", "log in", "sign in", "password", "can't access", "cannot access"],
    "Delivery Problem": ["delivery", "delivered", "package", "parcel", "shipping", "shipment", "where is my order", "not arrived", "late"],
    "Cancellation": ["cancel", "cancellation", "stop order"],
    "Subscription": ["subscription", "subscribe", "unsubscribe", "membership", "plan"],
    "Technical Issue": ["bug", "error", "crash", "broken", "website", "app", "technical"],
    "Complaint": ["complaint", "terrible", "worst", "angry", "unhappy", "disappointed"]
}


def weak_label(text):

    text = text.lower()
    scores = {}

    for intent, keywords in KEYWORDS.items():
        scores[intent] = sum(1 for word in keywords if word in text)

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "Other"

    return best_intent


# ============================================================
# STEP 7 - GOLDEN SET (held out, never used for training)
# ============================================================

def create_golden_set(pairs):

    print("\nCreating Golden Set (held-out evaluation data)...")

    golden_source = pairs.sample(
        min(GOLDEN_SIZE, len(pairs)), random_state=42
    ).copy()

    # Everything NOT chosen for the golden set is safe to use for training.
    remaining_pairs = pairs.drop(golden_source.index).reset_index(drop=True)

    golden_source["text"] = golden_source["customer_text"]
    golden_source["true_intent"] = golden_source["text"].apply(weak_label)
    golden_source["expected_handling"] = "escalate"
    golden_source["expected_reply"] = ""

    golden_draft = golden_source[["text", "true_intent", "expected_handling", "expected_reply"]]

    draft_path = GOLDEN / "golden_set_TO_REVIEW.csv"
    golden_draft.to_csv(draft_path, index=False)

    print(f"\u2713 Draft golden set created: {len(golden_draft)} examples -> {draft_path}")
    print(
        "\nACTION REQUIRED:"
        f"\n  1. Open {draft_path}"
        "\n  2. Manually correct the 'true_intent' column for every row"
        "     (these are only keyword-based guesses, not real labels)."
        f"\n  3. Save your corrected file as: {GOLDEN / 'golden_set_reviewed.csv'}"
        "\n  4. Re-run this script - it will automatically pick up the"
        "     reviewed file and report TRUSTWORTHY accuracy numbers."
    )

    reviewed_path = GOLDEN / "golden_set_reviewed.csv"

    if reviewed_path.exists():
        golden = pd.read_csv(reviewed_path)
        print(f"\n\u2713 Using human-reviewed golden set: {reviewed_path}")
        golden_is_reviewed = True
    else:
        golden = golden_draft
        print(
            "\n\u26a0 WARNING: no reviewed golden set found. Proceeding with WEAK "
            "(keyword-based) labels for now. Any accuracy numbers below are "
            "PROVISIONAL and will look better than the model actually is, "
            "because the 'keyword baseline' and the golden 'ground truth' "
            "are currently the same function. Review the file above before "
            "trusting any reported metric."
        )
        golden_is_reviewed = False

    return golden, remaining_pairs, golden_is_reviewed


# ============================================================
# STEP 8 - BUILD A LARGE WEAK-LABELED TRAINING POOL
# ============================================================
# This is the key fix: train on thousands of weak-labeled examples drawn
# from `remaining_pairs` (which excludes every golden-set row), not on the
# 200-row golden set. More, noisier data reliably beats less, "cleaner" data
# for this kind of text classification.

def build_training_pool(remaining_pairs):

    print("\nBuilding weak-labeled training pool...")

    size = min(TRAINING_POOL_SIZE, len(remaining_pairs))
    pool = remaining_pairs.sample(size, random_state=42).copy()

    pool["text"] = pool["customer_text"]
    pool["intent"] = pool["text"].apply(weak_label)

    print("Raw weak-label distribution:")
    print(pool["intent"].value_counts())

    # Cap the "Other" bucket so it doesn't drown out real intents.
    counts = pool["intent"].value_counts()
    non_other_counts = counts.drop("Other", errors="ignore")

    if "Other" in counts and len(non_other_counts) > 0:
        cap = int(non_other_counts.max() * MAX_OTHER_RATIO)
        other_rows = pool[pool["intent"] == "Other"]

        if len(other_rows) > cap:
            keep_other = other_rows.sample(cap, random_state=42)
            pool = pd.concat([pool[pool["intent"] != "Other"], keep_other])
            pool = pool.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nBalanced training distribution (Other capped):")
    print(pool["intent"].value_counts())

    path = PROCESSED / "training_pool.csv"
    pool[["text", "intent"]].to_csv(path, index=False)
    print(f"\u2713 Saved training pool: {path} ({len(pool)} rows)")

    return pool[["text", "intent"]]


# ============================================================
# STEP 9 - TRAIN INTENT MODEL (with model selection + baselines)
# ============================================================

def candidate_models():
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0),
        "linear_svm": CalibratedClassifierCV(LinearSVC(class_weight="balanced", C=1.0), cv=3),
        "naive_bayes": MultinomialNB(alpha=0.3),
    }


def select_best_model(X_train, y_train):

    print("\nCross-validating candidate models...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, clf in candidate_models().items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", clf),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
        results[name] = scores.mean()
        print(f"  {name:15s} CV macro-F1 = {scores.mean():.4f}")

    best_name = max(results, key=results.get)
    print(f"\n>> Best model: {best_name}")

    best_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", candidate_models()[best_name]),
    ])

    return best_name, best_pipe


def train_intent_model(training_pool):

    print("\n" + "=" * 60)
    print("TRAINING INTENT MODEL (on weak-labeled training pool)")
    print("=" * 60)

    X = training_pool["text"]
    y = training_pool["intent"]

    if y.nunique() < 2:
        print("Not enough intent classes.")
        return None, None

    best_name, best_pipe = select_best_model(X, y)
    best_pipe.fit(X, y)

    path = MODELS / "intent_model.joblib"
    joblib.dump(best_pipe, path)
    print(f"\u2713 Model saved: {path}")

    return best_pipe, best_name


# ============================================================
# STEP 10 - HISTORICAL RETRIEVAL
# ============================================================

def create_retrieval_index(pairs):

    print("\n" + "=" * 60)
    print("BUILDING HISTORICAL RETRIEVAL INDEX")
    print("=" * 60)

    data = pairs.copy()

    if len(data) > RETRIEVAL_SIZE:
        data = data.sample(RETRIEVAL_SIZE, random_state=42)

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = data["customer_text"].tolist()

    print("Creating embeddings...")
    embeddings = encoder.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    np.save(MODELS / "embeddings.npy", embeddings)
    data.to_csv(MODELS / "retrieval_data.csv", index=False)

    print("\u2713 Retrieval index created")

    return encoder, data, embeddings


# ============================================================
# STEP 11 - SEARCH SIMILAR CASES
# ============================================================

def retrieve(query, encoder, data, embeddings, top_k=5):

    query_embedding = encoder.encode([query], normalize_embeddings=True)
    scores = cosine_similarity(query_embedding, embeddings)[0]
    indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in indices:
        results.append({
            "score": float(scores[i]),
            "customer": data.iloc[i]["customer_text"],
            "response": data.iloc[i]["response_text"]
        })

    return results


# ============================================================
# STEP 12 - GENERATE REPLY
# ============================================================

def generate_reply(message, intent, evidence):

    if not evidence:
        return (
            "Thanks for contacting support. We need some additional "
            "information to investigate your request."
        )

    templates = {
        "Refund": "We understand that you are requesting a refund.",
        "Payment Issue": "We understand that you are having a payment issue.",
        "Account Problem": "We understand that you are having an account-related issue.",
        "Login Problem": "We understand that you are having trouble logging in.",
        "Delivery Problem": "We understand that you are having a delivery issue.",
        "Cancellation": "We understand that you want to cancel.",
        "Subscription": "We understand that you have a subscription-related question.",
        "Technical Issue": "We understand that you are facing a technical issue.",
        "Complaint": "We are sorry that you had this experience.",
        "Other": "Thanks for contacting support."
    }

    start = templates.get(intent, templates["Other"])

    reply = (
        start
        + " We found a similar previous support conversation that can help "
        "guide the response. Please provide the relevant order or account "
        "details through the approved support channel if additional "
        "information is required."
    )

    return reply


# ============================================================
# STEP 13 - AUTO HANDLE / ESCALATE
# ============================================================

SENSITIVE = {"Payment Issue", "Account Problem", "Complaint"}


def escalation_decision(intent, confidence, retrieval_score):

    if intent in SENSITIVE:
        return "ESCALATE", "Sensitive/risk-related intent"

    if confidence >= 0.80 and retrieval_score >= 0.60:
        return "AUTO-HANDLE", "High confidence and strong historical evidence"

    if confidence < 0.60:
        return "ESCALATE", "Low intent confidence"

    return "ESCALATE", "Insufficient historical evidence"


# ============================================================
# STEP 14 - COMPLETE AI AGENT
# ============================================================

def run_agent(message, model, encoder, data, embeddings):

    probabilities = model.predict_proba([message])[0]
    classes = model.classes_
    index = np.argmax(probabilities)

    intent = classes[index]
    confidence = float(probabilities[index])

    evidence = retrieve(message, encoder, data, embeddings, top_k=5)
    best_score = evidence[0]["score"] if evidence else 0.0

    reply = generate_reply(message, intent, evidence)
    decision, reason = escalation_decision(intent, confidence, best_score)

    return {
        "customer_message": message,
        "intent": intent,
        "confidence": round(confidence, 4),
        "similar_cases": evidence,
        "draft_reply": reply,
        "decision": decision,
        "reason": reason
    }


# ============================================================
# STEP 15 - EVALUATE (on held-out golden set ONLY - no leakage)
# ============================================================

def save_evaluation(model, training_pool, golden, golden_is_reviewed):

    print("\n" + "=" * 60)
    print("EVALUATION ON HELD-OUT GOLDEN SET")
    print("=" * 60)

    if not golden_is_reviewed:
        print(
            "\u26a0 Reminder: golden set has not been human-reviewed yet. "
            "Treat every number below as provisional.\n"
        )

    y_true = golden["true_intent"]
    texts = golden["text"]

    # Baseline 1: majority class (fit on TRAINING distribution, applied blind)
    majority_clf = DummyClassifier(strategy="most_frequent")
    majority_clf.fit(np.zeros((len(training_pool), 1)), training_pool["intent"])
    majority_preds = majority_clf.predict(np.zeros((len(golden), 1)))

    # Baseline 2: keyword rules (same function used to bootstrap training
    # labels, but now judged against INDEPENDENT ground truth - fair test)
    keyword_preds = texts.apply(weak_label)

    # Your model
    model_preds = model.predict(texts)

    def score(preds, name):
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

    results = {
        "golden_set_human_reviewed": golden_is_reviewed,
        "baseline_majority_class": score(majority_preds, "Baseline 1: Majority class"),
        "baseline_keyword_rules": score(keyword_preds, "Baseline 2: Keyword rules"),
        "your_model": score(model_preds, "Your model"),
    }

    print("\nClassification report (your model):")
    print(classification_report(y_true, model_preds, zero_division=0))

    pd.DataFrame([results["your_model"]]).to_csv(EVALUATION / "metrics.csv", index=False)

    with open(EVALUATION / "metrics_full.json", "w") as f:
        json.dump(results, f, indent=2)

    cm = confusion_matrix(y_true, model_preds, labels=sorted(y_true.unique()))
    np.savetxt(EVALUATION / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print("\n\u2713 Evaluation saved (metrics.csv, metrics_full.json, confusion_matrix.csv)")

    if results["your_model"]["accuracy"] < 0.90:
        print(
            "\nNOTE: below 90%. Once the golden set is genuinely human-reviewed, "
            "this is almost always fixed by: (1) merging intents that overlap "
            "in meaning, (2) adding more real (not just weak-labeled) examples "
            "for whichever intents show up worst in the confusion matrix, or "
            "(3) checking the golden labels themselves are consistent."
        )

    return results


# ============================================================
# STEP 16 - GENERATE REPORT
# ============================================================

def create_report(df, pairs, brand, best_model_name, eval_results):

    report = f"""
HIVER SDE INTERN ASSIGNMENT
===========================

Dataset
-------
Total tweets: {len(df)}
Conversation pairs: {len(pairs)}
Selected support account: {brand}

Intent Model
------------
Best model selected by cross-validated macro-F1: {best_model_name}
Trained on a weak-labeled pool (NOT the golden set) to avoid label leakage.

Evaluation (held-out golden set, human-reviewed: {eval_results['golden_set_human_reviewed']})
------------------------------------------------------------------------
Baseline 1 (majority class):  accuracy={eval_results['baseline_majority_class']['accuracy']:.4f}
Baseline 2 (keyword rules):   accuracy={eval_results['baseline_keyword_rules']['accuracy']:.4f}
Your model ({best_model_name}): accuracy={eval_results['your_model']['accuracy']:.4f}

System Flow
-----------
Customer Message -> Intent Classifier -> Historical Retrieval ->
Grounded Reply -> Confidence + Risk -> AUTO-HANDLE / ESCALATE

Safety Logic
------------
Sensitive intents (Payment Issue, Account Problem, Complaint) always escalate.
High confidence + strong historical evidence -> AUTO-HANDLE.
Everything else -> ESCALATE.

Important
---------
The Golden Set must be human reviewed before these numbers are trustworthy
for final submission. See data/golden/golden_set_TO_REVIEW.csv.
"""

    with open(EVALUATION / "REPORT.txt", "w", encoding="utf8") as f:
        f.write(report)

    print("\u2713 Report generated")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("        HIVER SDE AI CUSTOMER SUPPORT AGENT")
    print("=" * 70)

    create_folders()

    df = load_dataset()
    if df is None:
        return

    analyze_data(df)
    create_sample(df)

    brand = select_brand(df)
    pairs = create_pairs(df, brand)

    if len(pairs) < 20:
        print("\nNot enough conversation pairs.")
        return

    golden, remaining_pairs, golden_is_reviewed = create_golden_set(pairs)

    if len(remaining_pairs) < 50:
        print("\nNot enough remaining pairs to build a training pool.")
        return

    training_pool = build_training_pool(remaining_pairs)

    model, best_model_name = train_intent_model(training_pool)
    if model is None:
        return

    eval_results = save_evaluation(model, training_pool, golden, golden_is_reviewed)

    encoder, retrieval_data, embeddings = create_retrieval_index(pairs)

    create_report(df, pairs, brand, best_model_name, eval_results)

    print("\n" + "=" * 70)
    print("TESTING COMPLETE AI AGENT")
    print("=" * 70)

    message = "My payment was charged twice and I need help with the duplicate charge."
    result = run_agent(message, model, encoder, retrieval_data, embeddings)

    print("\nCUSTOMER:", result["customer_message"])
    print("\nINTENT:", result["intent"])
    print("\nCONFIDENCE:", result["confidence"])
    print("\nDECISION:", result["decision"])
    print("\nREASON:", result["reason"])
    print("\nDRAFT REPLY:", result["draft_reply"])

    print("\nTOP HISTORICAL CASES:")
    for i, case in enumerate(result["similar_cases"], 1):
        print(f"\n{i}. Similarity:", round(case["score"], 3))
        print("Customer:", case["customer"][:150])
        print("Response:", case["response"][:150])

    print("\n" + "=" * 70)
    print("PROJECT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
