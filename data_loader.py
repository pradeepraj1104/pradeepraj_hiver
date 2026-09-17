"""
Step 1-2: Load the raw Twitter customer-support dataset and take a
manageable subsample. Also prints the exploratory stats the assignment
asks for (Step 2 - Analyze Data).
"""

import pandas as pd
from pathlib import Path


def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw twcs.csv (or a sample of it)."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download 'thoughtvector/customer-support-on-twitter' "
            f"from Kaggle and place twcs.csv (or a sample of it) at this path."
        )

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def analyze_data(df: pd.DataFrame) -> pd.Series:
    """Step 2: Analyze the Data - prints the stats the assignment asks for."""

    print("\n" + "=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)

    print("\nNumber of tweets:", len(df))
    print("Unique authors:", df["author_id"].nunique())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nInbound (customer) vs outbound (company) messages:")
    print(df["inbound"].value_counts())

    avg_len = df["text"].fillna("").str.len().mean()
    print(f"\nAverage message length: {avg_len:.1f} characters")

    n_conversations = df["in_response_to_tweet_id"].notna().sum()
    print("Number of reply-linked messages:", n_conversations)

    print("\nTop company accounts by number of replies sent:")
    company_messages = df[df["inbound"] == False]
    companies = (
        company_messages.groupby("author_id").size()
        .sort_values(ascending=False).head(20)
    )
    print(companies)

    return companies


def create_sample(df: pd.DataFrame, sample_size: int, out_path: str, clean_fn, seed: int = 42) -> pd.DataFrame:
    """Step 3: take a manageable subsample and clean it.

    IMPORTANT (per assignment): never train/explore on the full ~3M-row
    dataset directly - always work off a subsample like this one.
    """

    size = min(sample_size, len(df))
    sample = df.sample(size, random_state=seed).copy()

    sample["text"] = sample["text"].apply(clean_fn)
    sample = sample[sample["text"].str.len() > 3]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)

    print(f"\u2713 Saved {len(sample)} cleaned rows to {out_path}")
    return sample
