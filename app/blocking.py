"""Candidate pair generation via blocking.

Naive dedup compares every POI to every other POI -- O(n^2), infeasible at
scale. Blocking cuts the space first: two POIs are compared only if they
COULD plausibly be the same place, using cheap deterministic keys:

  - geo cell: same coarse lat/lng grid cell (~1km)
  - brand:     same leading brand token (catches coordinates that disagree)

Only pairs sharing a block become candidates. This is the difference between
a demo and something that can run over millions of records. (DESIGN.md §7)
"""
from . import features, models

# ~1.1km per cell at the equator, chosen so a duplicate that drifted by a few
# hundred meters still lands in the same (or an adjacent) cell.
GEO_STEP = 0.01


def geo_cell(lat, lng, step=GEO_STEP):
    return (round(lat / step), round(lng / step))


def _brand_key(poi):
    b = features.brand_of_poi(poi)
    return b or None


def generate_candidates(pool, geo_step=GEO_STEP, max_cell=500, max_dist_km=None):
    """Return candidate pairs that share a geo cell or a brand.

    Each pair carries ``ground_truth`` (same hidden entity or not) for
    evaluation. A pair emitted by both blockings is deduplicated.

    ``max_dist_km`` (optional) caps emitted pairs to within that great-circle
    distance -- chain brands otherwise pair up city-wide (O(n^2) on real data).
    """
    by_cell = {}
    by_brand = {}
    for p in pool:
        lat, lng = models.get(p.get("lat")), models.get(p.get("lng"))
        if lat is not None and lng is not None:
            by_cell.setdefault(geo_cell(lat, lng, geo_step), []).append(p)
        b = _brand_key(p)
        if b:
            by_brand.setdefault(b, []).append(p)

    seen = set()
    pairs = []

    def emit(a, b):
        if max_dist_km is not None:
            la, ln = models.get(a.get("lat")), models.get(a.get("lng"))
            lb, lnb = models.get(b.get("lat")), models.get(b.get("lng"))
            if None in (la, ln, lb, lnb):
                return
            if features.haversine_km(la, ln, lb, lnb) > max_dist_km:
                return
        key = tuple(sorted((a["id"], b["id"])))
        if key in seen:
            return
        seen.add(key)
        pairs.append({
            "id": f"cand_{len(pairs):03d}",
            "poi_a": a,
            "poi_b": b,
            "ground_truth": a["entity_id"] == b["entity_id"],
        })

    for members in by_cell.values():
        # Guard against a pathological cell blowing up to O(k^2).
        if len(members) > max_cell:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                emit(members[i], members[j])

    for members in by_brand.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                emit(members[i], members[j])

    return pairs


def blocking_quality(pool, candidates):
    """Recall/precision of blocking against the hidden ground truth.

    Recall = true-duplicate pairs found / all true-duplicate pairs.
    Precision = true-duplicate pairs / emitted candidate pairs.
    """
    true_pairs = set()
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if pool[i]["entity_id"] == pool[j]["entity_id"]:
                true_pairs.add(tuple(sorted((pool[i]["id"], pool[j]["id"]))))

    found = {tuple(sorted((c["poi_a"]["id"], c["poi_b"]["id"]))) for c in candidates}
    tp_found = len(true_pairs & found)
    recall = tp_found / len(true_pairs) if true_pairs else None
    precision = tp_found / len(found) if found else None
    return {
        "true_dup_pairs": len(true_pairs),
        "candidates": len(candidates),
        "true_dup_found": tp_found,
        "recall": round(recall, 4) if recall is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
    }
