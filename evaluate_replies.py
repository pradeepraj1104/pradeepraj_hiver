"""
Step 10 (reply portion): score draft replies for groundedness and
relevance.

Two tiers:
  1. Heuristic scoring (always available, no API key, deterministic) -
     good enough to sanity-check replies and catch obvious failures.
  2. Optional LLM-as-judge (only runs if ANTHROPIC_API_KEY is set) - closer
     to what the assignment calls "LLM-as-judge (relevance, helpfulness,
     grounding etc.)", for a stronger final report.
"""

import os
import re
from pathlib import Path

import pandas as pd


def _tokenize(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def heuristic_groundedness(reply: str, evidence_texts: list) -> float:
    """Rough proxy: how much of the reply's vocabulary overlaps with the
    retrieved historical evidence, vs. being generic boilerplate."""
    if not evidence_texts:
        return 0.0

    reply_tokens = _tokenize(reply)
    evidence_tokens = set()
    for t in evidence_texts:
        evidence_tokens |= _tokenize(t)

    if not reply_tokens:
        return 0.0

    overlap = reply_tokens & evidence_tokens
    return len(overlap) / len(reply_tokens)


def heuristic_relevance(message: str, reply: str) -> float:
    """Rough proxy: word overlap between the customer message and the reply."""
    msg_tokens = _tokenize(message)
    reply_tokens = _tokenize(reply)

    if not msg_tokens or not reply_tokens:
        return 0.0

    overlap = msg_tokens & reply_tokens
    return len(overlap) / len(msg_tokens)


def evaluate_replies_heuristic(records: list, out_path: str) -> pd.DataFrame:
    """`records` is a list of dicts each shaped like the output of
    src.pipeline.run_agent (needs customer_message, draft_reply,
    similar_cases)."""

    rows = []
    for r in records:
        evidence_texts = [c["response"] for c in r.get("similar_cases", [])]
        rows.append({
            "message": r["customer_message"],
            "intent": r["intent"],
            "reply": r["draft_reply"],
            "groundedness": round(heuristic_groundedness(r["draft_reply"], evidence_texts), 3),
            "relevance": round(heuristic_relevance(r["customer_message"], r["draft_reply"]), 3),
            "decision": r["decision"],
        })

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\u2713 Saved heuristic reply scores -> {out_path}")
    return df


def llm_judge_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_judge_reply(message: str, reply: str, evidence_texts: list) -> dict:
    """Optional: rate a single reply 1-5 on relevance/helpfulness/groundedness
    using Claude. Only runs if ANTHROPIC_API_KEY is set in the environment.
    Requires: pip install anthropic
    """
    if not llm_judge_available():
        raise RuntimeError("ANTHROPIC_API_KEY not set - skip or set it to use the LLM judge.")

    import anthropic
    client = anthropic.Anthropic()

    evidence_block = "\n".join(f"- {t}" for t in evidence_texts) or "(none)"

    prompt = f"""You are grading a customer support reply. Score 1-5 (5=best) on:
- relevance: does it address the customer's actual message?
- helpfulness: does it move the customer's problem forward?
- groundedness: is it consistent with the historical evidence, not making things up?

Customer message: {message}
Draft reply: {reply}
Historical evidence used:
{evidence_block}

Respond ONLY as JSON: {{"relevance": <1-5>, "helpfulness": <1-5>, "groundedness": <1-5>, "reason": "<one sentence>"}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    import json
    text = response.content[0].text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)
