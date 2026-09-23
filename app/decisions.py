"""Prompt/state construction + routing + evaluation for the two pipelines.

Each pipeline asks Jev a single `score` question on a 5-level rubric. The
answer gives us a position (0..4) and a calibrated `confidence`; we use the
confidence to gate: high-confidence extremes are auto-decided, everything
else escalates to a human. This is the "confidence-gated cascade" that turns
a cheap decision model into a review-cost reducer.
"""
from . import config, features, models
from .jev_client import JevClient

client = JevClient()

DEDUP_LEVELS = ["definitely different", "probably different", "uncertain", "probably same", "definitely same"]
REVIEW_LEVELS = ["clearly reject", "probably reject", "needs review", "probably approve", "clearly approve"]


def _poi_text(p):
    return (
        f"name: {models.get(p.get('name'))}\n"
        f"address: {models.get(p.get('address'))}\n"
        f"category: {models.get(p.get('category'))}\n"
        f"phone: {models.get(p.get('phone'), '')}\n"
        f"lat: {models.get(p.get('lat'))}, lng: {models.get(p.get('lng'))}"
    )


# --- dedup ----------------------------------------------------------------
def build_dedup_state(a, b, dist_km):
    return (
        f"POI A:\n{_poi_text(a)}\n\n"
        f"POI B:\n{_poi_text(b)}\n\n"
        f"approximate distance between coordinates: {dist_km:.3f} km"
    )


def run_dedup(pair):
    state = build_dedup_state(pair["poi_a"], pair["poi_b"], pair["dist_km"])
    questions = {
        "same_or_not": {
            "type": "score",
            "instructions": "How likely are POI A and POI B to be the same physical place?",
            "criteria": DEDUP_LEVELS,
        }
    }
    res = client.decide(state, questions, hint={"score": pair["hint"], "seed": _seed(pair["id"])})
    ans = res["answers"]["same_or_not"]
    pos, conf = ans["score"], ans["confidence"]
    return {
        "auto_label": "same" if pos >= 2 else "different",
        "route": _route(pos, conf),
        "score": round(pos, 3),
        "probability": round(pos / 4, 3),
        "confidence": round(conf, 3),
        "level": DEDUP_LEVELS[min(4, max(0, round(pos)))],
        "raw": ans,
        "model": res["model"],
        "mock": res["mock"],
        "latency_ms": res["latency_ms"],
    }


# --- review ---------------------------------------------------------------
def build_review_state(m):
    return (
        f"Merchant POI submission:\n"
        f"name: {m['name']}\n"
        f"address: {m['address']}\n"
        f"category: {m['category']}\n"
        f"description: {m.get('description', '')}\n"
        f"phone: {m.get('phone', '')}"
    )


def run_review(sub):
    state = build_review_state(sub["merchant"])
    questions = {
        "verdict": {
            "type": "score",
            "instructions": "Rate this merchant POI submission from clearly-reject (spam/fraud/invalid) to clearly-approve (legitimate listing).",
            "criteria": REVIEW_LEVELS,
        }
    }
    res = client.decide(state, questions, hint={"score": sub["hint"], "seed": _seed(sub["id"])})
    ans = res["answers"]["verdict"]
    pos, conf = ans["score"], ans["confidence"]
    return {
        "auto_label": "approve" if pos >= 2 else "reject",
        "route": _route(pos, conf),
        "score": round(pos, 3),
        "probability": round(pos / 4, 3),
        "confidence": round(conf, 3),
        "level": REVIEW_LEVELS[min(4, max(0, round(pos)))],
        "raw": ans,
        "model": res["model"],
        "mock": res["mock"],
        "latency_ms": res["latency_ms"],
    }


def _route(pos, conf):
    if conf < config.CONFIDENCE_AUTO:
        return "escalate"
    if pos >= 3.0:
        return "auto_accept"
    if pos <= 1.0:
        return "auto_reject"
    return "escalate"


def _seed(pid):
    return sum(ord(c) for c in pid)


# --- feature-based dedup (candidate pipeline) -----------------------------
def build_dedup_state_with_features(pair, f):
    """State that feeds computed evidence to the model as facts, not asks it
    to recompute them."""
    a, b = pair["poi_a"], pair["poi_b"]

    def fmt(v):
        return "unknown" if v is None else v

    return "\n".join([
        "POI A:", _poi_text(a), "",
        "POI B:", _poi_text(b), "",
        "computed evidence:",
        f"  distance_km: {fmt(f['distance_km'])}",
        f"  name_similarity: {fmt(f['name_sim'])}",
        f"  brand_match: {fmt(f['brand_match'])}",
        f"  category_match: {fmt(f['category_match'])}",
        f"  phone_match: {fmt(f['phone_match'])}",
    ])


def run_candidate(pair):
    """Decide one blocking-generated candidate pair using computed features."""
    f = features.pair_features(pair["poi_a"], pair["poi_b"])
    state = build_dedup_state_with_features(pair, f)
    questions = {
        "same_or_not": {
            "type": "score",
            "instructions": "How likely are POI A and POI B to be the same physical place?",
            "criteria": DEDUP_LEVELS,
        }
    }
    hint = {"score": features.feature_hint(f), "seed": _seed(pair["id"])}
    res = client.decide(state, questions, hint=hint)
    ans = res["answers"]["same_or_not"]
    pos, conf = ans["score"], ans["confidence"]
    return {
        "id": pair["id"],
        "auto_label": "same" if pos >= 2 else "different",
        "route": _route(pos, conf),
        "score": round(pos, 3),
        "probability": round(pos / 4, 3),
        "confidence": round(conf, 3),
        "level": DEDUP_LEVELS[min(4, max(0, round(pos)))],
        "features": f,
        "model": res["model"],
        "mock": res["mock"],
        "latency_ms": res["latency_ms"],
    }
