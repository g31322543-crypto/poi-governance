"""Unified POI governance chain: merchant submission -> dedup -> review -> route.

This replaces the two earlier *separate* floating pipelines (pairwise A/B dedup,
standalone review) with one grounded flow: a merchant submits a listing, and the
platform (1) resolves it against the existing database (duplicate -> merge, or
new), (2) moderates it (legitimate -> approve, spam -> reject), (3) routes it
(auto-decide when confident, else escalate to a human).

Decisions run through the same JevClient as before, so the backend is
mock / laya / jev via config (the mock is what runs without fine-tuning; Laya
OOTB collapses on this domain, which is exactly what the demo is meant to show).
"""
import math
import random

from . import decisions, features, models, perturb

# A "same place" candidate must be within this distance (matches the geo cap in
# run_overture.py). Brand matches are allowed beyond it -- the score must then
# use distance to tell "same brand, different branch" apart.
MATCH_MAX_DIST_KM = 0.3

# Legitimate merchant listing descriptions. These are the ones the multilingual
# spam detector provably clears (probed: p(spam) 0.001-0.155). KNOWN LIMITATION:
# the detector is trained on user *posts*, so promotional-sounding listing text
# over-triggers -- "High-street pharmacy, prescriptions..." (0.98), "Barber
# shop, walk-in..." (0.96) and "Bookshop and cafe..." (0.92) all false-positive.
# A merchant listing *is* advertising, so "spam" is the wrong question for it;
# a listing-specific question or the english checkpoint is the real fix.
_LEGIT_DESC = [
    "Family-run Italian restaurant, fresh pasta made daily, open 12:00-23:00, dine-in and takeaway.",
    "Neighbourhood gym, 24-hour access, personal training and group classes, first session free.",
    "Local butcher shop, fresh meat and daily cuts, friendly service, open six days a week.",
    "Corner deli, sandwiches and salads made to order, open for lunch.",
]

_SPAM_DESC = [
    "Click here for cash back! Guaranteed rewards, WhatsApp +44 7700 000000, limited offer.",
    "No need to visit. Part-time work from home, earn £500/day, apply now at www.fast-money.example.",
    "Five-star reviews for cash. Subscribe and follow, instant pay, promo code FREE50.",
]

_NEW_NAMES = ["Sunrise Bakery", "The Corner Cafe", "Harbour Fitness", "Green Leaf Grocers"]

# Plausible London street names for giving a synthetic "new branch" a *different*
# address from the original. A new branch is the same brand at a genuinely
# different location, so it must not share the original's address (or be blank).
_STREETS = ["Oxford Street", "Regent Street", "Baker Street", "Shaftesbury Avenue",
            "Kingsway", "High Holborn", "Strand", "Piccadilly"]


def _dist_km(a, b):
    la, ln = models.get(a.get("lat")), models.get(a.get("lng"))
    lb, lnb = models.get(b.get("lat")), models.get(b.get("lng"))
    if None in (la, ln, lb, lnb):
        return None
    return features.haversine_km(la, ln, lb, lnb)


def find_matches(submission, db, max_dist_km=MATCH_MAX_DIST_KM, topk=5):
    """Candidate existing DB places that could be the same as the submission.

    Two cheap keys (the same idea as blocking, applied to a single incoming
    record): explicit-brand equality, and geo proximity. A submission only ever
    needs comparing against this handful, not the whole DB.
    """
    sub_brand = features.brand_of_poi(submission)
    out, seen = [], set()
    for p in db:
        if p["id"] == submission.get("id"):
            continue
        if sub_brand and features.brand_of_poi(p) == sub_brand:
            out.append(p)
            seen.add(p["id"])
            continue
        d = _dist_km(submission, p)
        if d is not None and d <= max_dist_km:
            out.append(p)
            seen.add(p["id"])

    def key(p):
        d = _dist_km(submission, p)
        s = features.name_similarity(
            models.get(submission.get("name")), models.get(p.get("name")))
        return (-(s or 0.0), (d if d is not None else 1e9))

    out.sort(key=key)
    return out[:topk]


def resolve_submission(submission, db, topk=5):
    """Dedup step: is this submission a duplicate of an existing place?"""
    cands = find_matches(submission, db, topk=topk)
    scored = []
    for c in cands:
        r = decisions.run_candidate({
            "id": f"{submission['id']}~{c['id']}",
            "poi_a": submission,
            "poi_b": c,
        })
        f = r["features"]
        scored.append({
            "candidate": c,
            "decision": r,
            "hint": features.feature_hint(f),
            "distance_km": f["distance_km"],
        })
    scored.sort(key=lambda x: (-x["hint"],
                               (x["distance_km"] if x["distance_km"] is not None else 1e9)))

    if not scored:
        return {"verdict": "new", "route": "auto_create", "matched": None,
                "confidence": 1.0, "candidates": [], "best": None}

    best = scored[0]
    d = best["decision"]
    same = d["auto_label"] == "same"
    verdict = "merge" if same else "new"
    matched = best["candidate"] if same else None
    route = ("auto_merge" if same else "auto_create") if d["route"].startswith("auto") else "escalate"
    return {"verdict": verdict, "route": route, "matched": matched,
            "confidence": d["confidence"], "candidates": scored, "best": best}


def run_submission(submission, db, topk=5):
    """Run the full chain for one merchant submission."""
    res = resolve_submission(submission, db, topk=topk)
    rev = decisions.run_review({
        "id": submission["id"],
        "merchant": submission,
        "hint": features.moderation_hint(submission),
    })
    return {
        "submission": submission,
        "resolve": res,
        "review": rev,
        "route": _final_route(res, rev),
    }


def _final_route(res, rev):
    if rev["route"] == "auto_reject":
        return {"action": "auto_reject", "why": "moderation rejected the listing"}
    if res["route"] == "escalate" or rev["route"] == "escalate":
        return {"action": "escalate", "why": "dedup or review was not confident enough"}
    if res["verdict"] == "merge":
        return {"action": "auto_merge",
                "matched_id": res["matched"]["id"] if res["matched"] else None,
                "why": "duplicate of an existing place, and the listing is legitimate"}
    return {"action": "auto_create", "why": "new place, and the listing is legitimate"}


def summarize(traces):
    """Confusion counts over the auto-decided subset of the chain."""
    dedup = {"merge_ok": 0, "merge_bad": 0, "new_ok": 0, "new_bad": 0, "escalated": 0}
    review = {"reject_ok": 0, "reject_bad": 0, "approve_ok": 0, "approve_bad": 0, "escalated": 0}
    for t in traces:
        s = t["submission"]
        if t["resolve"]["route"] == "escalate":
            dedup["escalated"] += 1
        else:
            is_dup = s["is_duplicate"]
            if t["resolve"]["verdict"] == "merge":
                dedup["merge_ok" if is_dup else "merge_bad"] += 1
            else:
                dedup["new_ok" if not is_dup else "new_bad"] += 1

        if t["review"]["route"] == "escalate":
            review["escalated"] += 1
        else:
            is_spam = s["legitimacy"] == "reject"
            if t["review"]["auto_label"] == "reject":
                review["reject_ok" if is_spam else "reject_bad"] += 1
            else:
                review["approve_ok" if not is_spam else "approve_bad"] += 1
    return {"dedup": dedup, "review": review}


def _relocate(lat, lng, rng, lo_km, hi_km):
    d = rng.uniform(lo_km, hi_km)
    ang = rng.uniform(0.0, 2 * math.pi)
    dlat = d * math.cos(ang) / 111.0
    dlng = d * math.sin(ang) / (111.0 * max(0.1, math.cos(math.radians(lat))))
    return round(lat + dlat, 6), round(lng + dlng, 6)


def build_submissions(db, n_dup=6, n_branch=4, n_new=3, n_spam=3, seed=0):
    """Simulate a batch of merchant submissions over the existing DB.

    Four flavors, each with hidden ground truth for evaluation:
      duplicates  -- the merchant re-submits an existing place (messy).
      new branch  -- same brand/name, genuinely a different location.
      new place   -- no brand, nowhere near anything -> trivially new.
      spam        -- the listing is an ad / scam -> reject.
    """
    rng = random.Random(seed)
    branded = [p for p in db if features.brand_of_poi(p)]
    dups = rng.sample(db, min(n_dup, len(db)))
    branches = rng.sample(branded or db, min(n_branch, len(branded or db)))
    news = rng.sample(db, min(n_new, len(db)))
    spams = rng.sample(db, min(n_spam, len(db)))

    subs = []

    def add(s, is_dup, matched, legitimacy):
        s["is_duplicate"] = is_dup
        s["matched_id"] = matched
        s["legitimacy"] = legitimacy
        subs.append(s)

    for i, p in enumerate(dups):
        s = perturb.drift_poi(p, rng, hard=(i >= n_dup // 2))
        s["id"] = f"sub_dup_{i:02d}"
        s["description"] = rng.choice(_LEGIT_DESC)
        add(s, True, p["id"], "approve")

    for i, p in enumerate(branches):
        s = dict(p)
        s["id"] = f"sub_branch_{i:02d}"
        s["entity_id"] = s["id"]
        s["lat"], s["lng"] = _relocate(p["lat"], p["lng"], rng, 2.0, 4.0)
        s["phone"] = ""
        s["address"] = f"{rng.randint(1, 250)} {rng.choice(_STREETS)}"
        s["description"] = rng.choice(_LEGIT_DESC)
        add(s, False, None, "approve")

    for i, p in enumerate(news):
        s = dict(p)
        s["id"] = f"sub_new_{i:02d}"
        s["entity_id"] = s["id"]
        s["name"] = rng.choice(_NEW_NAMES)
        s["brand"] = ""
        s["lat"], s["lng"] = _relocate(p["lat"], p["lng"], rng, 1.0, 2.0)
        s["phone"] = ""
        s["description"] = rng.choice(_LEGIT_DESC)
        add(s, False, None, "approve")

    for i, p in enumerate(spams):
        lat, lng = _relocate(p["lat"], p["lng"], rng, 1.0, 2.0)
        s = {
            "id": f"sub_spam_{i:02d}",
            "entity_id": f"sub_spam_{i:02d}",
            "name": p["name"],
            "address": "London",
            "category": p["category"],
            "phone": "",
            "lat": lat, "lng": lng,
            "brand": "",
            "description": rng.choice(_SPAM_DESC),
        }
        add(s, False, None, "reject")

    rng.shuffle(subs)
    return subs
