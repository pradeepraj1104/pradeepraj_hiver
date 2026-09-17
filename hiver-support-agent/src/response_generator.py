"""
Step 7: Generate a grounded reply from the predicted intent + retrieved
historical evidence.

This is a template-based generator (fast, deterministic, no API key
needed, easy to grade/inspect). If you want LLM-drafted replies instead,
swap the body of `generate_reply` for a call to your LLM provider of
choice, keeping the same signature - everything else in the pipeline
(escalation logic, evaluation) stays unchanged.
"""

TEMPLATES = {
    "Refund": "We understand that you are requesting a refund.",
    "Payment Issue": "We understand that you are having a payment issue.",
    "Account Problem": "We understand that you are having an account-related issue.",
    "Login Problem": "We understand that you are having trouble logging in.",
    "Delivery Problem": "We understand that you are having a delivery issue.",
    "Cancellation": "We understand that you want to cancel.",
    "Subscription": "We understand that you have a subscription-related question.",
    "Technical Issue": "We understand that you are facing a technical issue.",
    "Complaint": "We are sorry that you had this experience.",
    "Other": "Thanks for contacting support.",
}


def generate_reply(message: str, intent: str, evidence: list) -> str:
    """Draft a grounded reply. `evidence` is the list returned by
    Retriever.retrieve() - used here to decide whether we have enough
    context to sound confident vs. asking for more information.
    """

    if not evidence:
        return (
            "Thanks for contacting support. We need some additional "
            "information to investigate your request."
        )

    start = TEMPLATES.get(intent, TEMPLATES["Other"])

    return (
        f"{start} We found a similar previous support conversation that can "
        "help guide the response. Please provide the relevant order or "
        "account details through the approved support channel if "
        "additional information is required."
    )
