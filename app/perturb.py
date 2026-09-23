"""Labeled dedup pairs by perturbing REAL Overture places.

A single Overture release is conflated (one row per place), so it contains no
true intra-release duplicates. We build a labeled set two ways, both anchored on
real Overture records:

  positives  -- a real place vs a "dirty copy" (simulated second source): name
                drifted, coords jittered, phone reformatted. ~40% are *hard*
                (name stripped to its brand core, coords off 100-250 m, fields
                dropped) so the set isn't trivially separable.

  negatives  -- (a) easy: two nearby different-name places; (b) *hard*: two REAL
                same-brand different-branch places nearby (e.g. two Costa
                Coffee stores a street apart). These are the cases a name/
                brand/category feature layer is NOT designed to tell apart, and
                the honest reason F1 cannot be 1.0.
"""
import math
import random
import re

from . import features, models


def _drift_name(name, rng, hard=False):
    n = features.normalize_name(name)
    words = n.split()
    if not words:
        return name or ""
    if hard:
        keep = rng.choice([1, 1, 2])
        return " ".join(words[:keep])                 # brand core only
    r = rng.random()
    if len(words) >= 2 and r < 0.35:
        return " ".join(words[:-1])
    if r < 0.55:
        return n + " " + rng.choice(["london", "central london", "uk"])
    if len(words) >= 3 and r < 0.70:
        i = rng.randrange(1, len(words))
        return " ".join(words[:i] + words[i + 1:])
    return n


def _drift_coords(lat, lng, rng, hard=False):
    if lat is None or lng is None:
        return lat, lng
    d = rng.uniform(0.10, 0.25) if hard else rng.uniform(0.020, 0.060)  # km
    ang = rng.uniform(0.0, 2 * math.pi)
    dlat = d * math.cos(ang) / 111.0
    dlng = d * math.sin(ang) / (111.0 * max(0.1, math.cos(math.radians(lat))))
    return round(lat + dlat, 6), round(lng + dlng, 6)


def _drift_phone(phone, rng, hard=False):
    if hard and rng.random() < 0.5:
        return ""                                      # second source lacks phone
    if not phone:
        return phone
    digits = re.sub(r"\D", "", phone)
    if len(digits) >= 8:
        return "+44 " + digits[-8:-4] + " " + digits[-4:]
    return phone


def drift_poi(p, rng, hard=False):
    lat, lng = _drift_coords(models.get(p.get("lat")), models.get(p.get("lng")), rng, hard)
    return {
        "id": f"{p['id']}_dirty",
        "entity_id": p["entity_id"],
        "name": _drift_name(models.get(p.get("name")), rng, hard),
        "address": models.get(p.get("address")),
        "category": models.get(p.get("category")),
        "phone": _drift_phone(models.get(p.get("phone")), rng, hard),
        "lat": lat,
        "lng": lng,
        "brand": models.get(p.get("brand")),
        "confidence": models.get(p.get("confidence")),
        "sources": models.get(p.get("sources")) or [],
    }


def _dist(a, b):
    la, ln = models.get(a.get("lat")), models.get(a.get("lng"))
    lb, lbn = models.get(b.get("lat")), models.get(b.get("lng"))
    if None in (la, ln, lb, lbn):
        return float("inf")
    return features.haversine_km(la, ln, lb, lbn)


def _pick_negative(seeds, i, rng):
    a = seeds[i]
    cat = models.get(a.get("category"))
    others = [j for j in range(len(seeds)) if j != i]
    if cat is not None:
        same_cat_near = [j for j in others
                         if models.get(seeds[j].get("category")) == cat
                         and _dist(seeds[j], a) < 0.20]
        if same_cat_near:
            return rng.choice(same_cat_near)
    near = [j for j in others if _dist(seeds[j], a) < 0.50]
    if near:
        return rng.choice(near)
    return rng.choice(others)


def find_same_brand_pairs(pool, rng, n):
    """REAL hard negatives: same explicit brand, different id, 50 m-1 km apart.

    Two branches of one chain -- indistinguishable by name/brand/category; only
    distance (and address/phone) can tell them apart.
    """
    by_brand = {}
    for p in pool:
        b = models.get(p.get("brand"))
        if b:
            by_brand.setdefault(b, []).append(p)
    out = []
    for members in by_brand.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                d = _dist(members[i], members[j])
                if 0.05 < d < 1.0:
                    out.append((members[i], members[j]))
    rng.shuffle(out)
    return out[:n]


def build_labeled_pairs(pool, k=200, seed=0, hard_pos_ratio=0.4):
    """k positives (40% hard) + k negatives (half easy, half real same-brand)."""
    rng = random.Random(seed)
    seeds = rng.sample(pool, min(k, len(pool)))
    pairs = []

    for i, s in enumerate(seeds):
        d = drift_poi(s, rng, hard=(rng.random() < hard_pos_ratio))
        pairs.append({"id": f"pos_{i:03d}", "poi_a": s, "poi_b": d, "ground_truth": True})

    half = k // 2
    for i in range(half):
        j = _pick_negative(seeds, i, rng)
        d = drift_poi(seeds[j], rng, hard=False)
        pairs.append({"id": f"neg_easy_{i:03d}", "poi_a": seeds[i], "poi_b": d, "ground_truth": False})

    hard = find_same_brand_pairs(pool, rng, half)
    for i, (a, b) in enumerate(hard):
        pairs.append({"id": f"neg_hard_{i:03d}", "poi_a": a, "poi_b": b, "ground_truth": False})

    return pairs
