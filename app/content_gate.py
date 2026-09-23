"""Local Services Content Gate: turn Laya's preset answers into governance
decisions, confidence-gated and routed.

Laya answers text-decision questions with calibrated confidence. This module
maps those answers onto the actions a local-services platform takes:

  listing description -> moderation_questions -> approve / reject
  user review         -> moderation_questions -> publish / hide (+ reason)
  merchant ticket     -> triage_questions     -> route to a team

Every decision is confidence-gated: confident -> auto, uncertain -> escalate
to a human. That is the whole System-1 value proposition -- cheap, fast,
calibrated decisions on the easy tail, humans only on the hard tail.
"""
from . import config

CONFIDENCE_AUTO = config.CONFIDENCE_AUTO

# triage intent -> owning team
TEAM_MAP = {
    "refund": "Billing",
    "billing_question": "Billing",
    "technical_help": "Technical Support",
    "information": "Sales",
    "cancellation": "Retention",
    "other": "Human review",
}

_HIDE_QUESTIONS = ("spam", "toxic", "harassment", "threat")


def _hide_signal(answers):
    """Best hide/publish signal across the moderation flags.

    The strongest *violation* signal (highest p(true)) drives the action --
    never the flag's confidence, which is maximised by clean text (a clean
    post has threat p=0.00 -> conf=1.00 and would otherwise mask a real
    spam p=0.99). ``p`` = max flag probability; ``confidence`` = max(p, 1-p),
    matching Laya's own noul convention.
    """
    best_q, best_p = None, 0.0
    for q in _HIDE_QUESTIONS:
        p = answers[q]["noul"]
        if p > best_p:
            best_q, best_p = q, p
    conf = max(best_p, 1.0 - best_p)
    if best_p >= 0.5:
        return "hide", best_q, round(conf, 4)
    return "publish", None, round(conf, 4)


def moderate_listing(answers):
    """Listing description -> approve/reject + confidence + per-flag signals."""
    action, flag, conf = _hide_signal(answers)
    verdict = "reject" if action == "hide" else "approve"
    signals = sorted(
        ({"key": q, "label": q, "value": round(answers[q]["noul"], 4),
          "weight": round(answers[q]["noul"], 4)} for q in _HIDE_QUESTIONS),
        key=lambda s: -s["weight"])
    return {"verdict": verdict, "reason": flag, "confidence": round(conf, 4),
            "route": gate(conf), "signals": signals}


def moderate_review(answers):
    """User review -> publish/hide + reason + confidence."""
    action, flag, conf = _hide_signal(answers)
    return {"verdict": action, "reason": flag, "confidence": round(conf, 4),
            "route": gate(conf)}


def triage_ticket(answers):
    """Merchant ticket -> intent + team + priority/retention flags."""
    intent = answers["intent"]["choice"]
    intent_conf = answers["intent"]["confidence"]
    urgent = answers["is_urgent"]["noul"] >= 0.5
    churn = answers["churn_risk"]["noul"] >= 0.5
    frustration = answers["frustration"]["score"]
    return {
        "intent": intent,
        "team": TEAM_MAP.get(intent, "Human review"),
        "confidence": round(intent_conf, 4),
        "route": gate(intent_conf),
        "urgent": urgent,
        "churn": churn,
        "frustration": round(frustration, 2),
    }


def gate(conf):
    """Confidence gate: auto-decide or escalate to a human."""
    return "auto" if conf >= CONFIDENCE_AUTO else "escalate"


# --- evaluation -----------------------------------------------------------

def summarize(traces):
    """Confusion counts + automation stats over the auto-decided subset."""
    s = {"listing": _blank(), "review": _blank(), "ticket": _blank()}
    for t in traces:
        k = t["kind"]
        s[k]["n"] += 1
        if t["decision"]["route"] == "escalate":
            s[k]["escalated"] += 1
            continue
        s[k]["auto"] += 1
        if t["correct"]:
            s[k]["auto_ok"] += 1
    return s


def _blank():
    return {"n": 0, "auto": 0, "auto_ok": 0, "escalated": 0}
