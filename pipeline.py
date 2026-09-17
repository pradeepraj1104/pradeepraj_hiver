"""
Ties intent classification + historical retrieval + reply generation +
escalation logic together into one call - this is "the agent" from the
assignment's Final System Flow diagram.
"""

import numpy as np


def run_agent(message: str, model, retriever, generate_reply_fn, escalation_fn, top_k: int = 5):
    """Run the full pipeline on a single customer message."""

    probabilities = model.predict_proba([message])[0]
    classes = model.classes_
    idx = int(np.argmax(probabilities))

    intent = classes[idx]
    confidence = float(probabilities[idx])

    evidence = retriever.retrieve(message, top_k=top_k)
    best_score = evidence[0]["score"] if evidence else 0.0

    reply = generate_reply_fn(message, intent, evidence)
    decision, reason = escalation_fn(intent, confidence, best_score)

    return {
        "customer_message": message,
        "intent": intent,
        "confidence": round(confidence, 4),
        "similar_cases": evidence,
        "draft_reply": reply,
        "decision": decision,
        "reason": reason,
    }
