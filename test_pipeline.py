"""
Fast unit tests that don't require the dataset - run with:
    pytest tests/
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing import clean_text
from src.intent_classifier import weak_label
from src.escalation import escalation_decision


def test_clean_text_strips_urls_and_whitespace():
    raw = "check this   out https://example.com/thing   now"
    cleaned = clean_text(raw)
    assert "https://" not in cleaned
    assert "  " not in cleaned


def test_clean_text_handles_nan():
    import pandas as pd
    assert clean_text(pd.NA) == ""
    assert clean_text(float("nan")) == ""


def test_weak_label_refund():
    assert weak_label("I want a refund for my order") == "Refund"


def test_weak_label_defaults_to_other():
    assert weak_label("just saying hi, love the new logo") == "Other"


def test_escalation_sensitive_intent_always_escalates():
    decision, reason = escalation_decision("Payment Issue", confidence=0.99, retrieval_score=0.99)
    assert decision == "ESCALATE"


def test_escalation_high_confidence_and_evidence_auto_handles():
    decision, reason = escalation_decision("Delivery Problem", confidence=0.9, retrieval_score=0.8)
    assert decision == "AUTO-HANDLE"


def test_escalation_low_confidence_escalates():
    decision, reason = escalation_decision("Other", confidence=0.3, retrieval_score=0.9)
    assert decision == "ESCALATE"


if __name__ == "__main__":
    import subprocess
    subprocess.run(["pytest", str(Path(__file__)), "-v"])
