"""Suggested auto-reply templates, one per support category."""
from __future__ import annotations

TEMPLATES: dict[str, str] = {
    "billing": (
        "Thanks for getting in touch about your billing. I've pulled up your account and "
        "I'm looking into the charge now. I'll confirm the amount and next steps within "
        "one business day — if anything needs refunding, I'll process it right away."
    ),
    "technical": (
        "Sorry you're hitting this issue. I've logged a technical ticket with the details "
        "you provided and our team is investigating. I'll update you as soon as we have a "
        "fix or a workaround — usually within 24 hours."
    ),
    "account": (
        "I can help with that account change. For security, I've sent a verification link to "
        "your registered email — once you confirm it, I'll complete the update and let you know."
    ),
    "delivery": (
        "I understand how frustrating a delayed or missing delivery is. I've traced your "
        "parcel and I'm chasing the courier now. I'll come back to you today with a status "
        "update and, if needed, a replacement or refund."
    ),
    "returns": (
        "No problem — I can sort out your return. I've started the returns process and sent "
        "you a prepaid label. Once the item arrives back with us, your refund will be "
        "processed within 3–5 business days."
    ),
    "product_info": (
        "Great question! Here's what I can tell you about that. If you'd like a more detailed "
        "comparison or a demo, just let me know and I'll arrange it."
    ),
    "complaint": (
        "I'm really sorry about your experience — that's not the standard we aim for. I've "
        "escalated your case to a senior team member who will review it personally and contact "
        "you within one business day."
    ),
    "feedback": (
        "Thank you so much for the feedback! I've shared it with the team — it's exactly this "
        "kind of input that helps us improve. We really appreciate you taking the time."
    ),
}


def get_template(category: str) -> str:
    return TEMPLATES.get(category, TEMPLATES["feedback"])
