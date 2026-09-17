"""
The two baselines the assignment requires you to compare your model
against (Step 5 / Step 10):

  1. Majority class (trivial floor)
  2. Keyword rules (src.intent_classifier.weak_label)

Both are only meaningful when scored against ground truth they did NOT
produce - i.e. the human-reviewed Golden Set, never their own labels.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.intent_classifier import weak_label


def majority_baseline_predict(train_labels: pd.Series, n_test: int) -> np.ndarray:
    """Predicts the single most common training-set label every time."""
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((len(train_labels), 1)), train_labels)
    return clf.predict(np.zeros((n_test, 1)))


def keyword_baseline_predict(texts: pd.Series) -> pd.Series:
    """Predicts using the hand-written keyword rules only."""
    return texts.apply(weak_label)
