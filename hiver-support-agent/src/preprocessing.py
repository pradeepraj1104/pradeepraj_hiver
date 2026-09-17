"""
Step 3-4 support: text cleaning, brand selection, and building
customer-message -> company-response conversation pairs.
"""

import re
import pandas as pd
from pathlib import Path


def clean_text(text) -> str:
    """Light cleaning: strip URLs and collapse whitespace.
    Deliberately keeps @handles/punctuation - they can carry intent signal
    (e.g. '!!!' or '?') and TF-IDF/embeddings handle noise reasonably well.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def select_brand(df: pd.DataFrame, min_replies: int = 1) -> str:
    """Step 4 (brand step): pick the support account with the most
    outbound replies - gives the richest, most consistent data to model.
    """

    print("\n" + "=" * 60)
    print("BRAND SELECTION")
    print("=" * 60)

    outbound = df[df["inbound"] == False]
    brands = outbound.groupby("author_id").size().sort_values(ascending=False)

    print("\nTop support accounts by reply volume:")
    print(brands.head(20))

    eligible = brands[brands >= min_replies]
    if len(eligible) == 0:
        raise ValueError("No brand has any outbound replies in this data.")

    selected = eligible.index[0]
    print(f"\nSelected brand: {selected}  ({eligible.iloc[0]} replies in this sample)")

    return selected


def create_pairs(df: pd.DataFrame, brand: str, out_path: str) -> pd.DataFrame:
    """Step 4: build (customer_text -> company response) pairs for one brand."""

    print(f"\nBuilding conversation pairs for brand: {brand}")

    customers = df[df["inbound"] == True].copy()
    responses = df[df["inbound"] == False].copy()

    pairs = customers.merge(
        responses,
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_customer", "_company"),
    )

    pairs = pairs[pairs["author_id_company"] == brand]

    pairs = pairs[[
        "tweet_id_customer", "text_customer",
        "tweet_id_company", "text_company",
    ]]
    pairs.columns = ["customer_id", "customer_text", "response_id", "response_text"]

    pairs["customer_text"] = pairs["customer_text"].apply(clean_text)
    pairs["response_text"] = pairs["response_text"].apply(clean_text)

    pairs = pairs.drop_duplicates().reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_path, index=False)

    print(f"\u2713 {len(pairs)} conversation pairs saved to {out_path}")
    return pairs
