"""
Step 8: Auto-handle or Escalate.

Simple, explainable rule stack - every decision comes with a plain-English
reason, which the assignment's report explicitly wants ("decisions ... with
reasons").
"""

# Intents where getting it wrong has real cost (money, account access,
# already-upset customer) - always escalate regardless of model confidence.
SENSITIVE_INTENTS = {"Payment Issue", "Account Problem", "Complaint"}

CONFIDENCE_THRESHOLD = 0.80
EVIDENCE_THRESHOLD = 0.60
LOW_CONFIDENCE_THRESHOLD = 0.60


def escalation_decision(intent: str, confidence: float, retrieval_score: float):
    """Returns (decision: 'AUTO-HANDLE' | 'ESCALATE', reason: str)."""

    if intent in SENSITIVE_INTENTS:
        return "ESCALATE", "Sensitive/risk-related intent"

    if confidence >= CONFIDENCE_THRESHOLD and retrieval_score >= EVIDENCE_THRESHOLD:
        return "AUTO-HANDLE", "High confidence and strong historical evidence"

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        return "ESCALATE", "Low intent confidence"

    return "ESCALATE", "Insufficient historical evidence"
