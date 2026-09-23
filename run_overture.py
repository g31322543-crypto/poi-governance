"""End-to-end dedup pipeline on REAL Overture Maps data.

Loads the London GeoParquet, flattens it to the project schema, profiles field
coverage, blocks into distance-capped candidate pairs, and runs the decision
engine on a sample. Backend comes from JEV_BACKEND (mock | laya), default laya,
or the first CLI arg. Honest framing: a conflated source has no intra-release
duplicates, so the job is to confirm the pipeline separates nearby same-brand
*branches* and that Laya's known OOD behaviour shows up on real data too.

    python run_overture.py mock   # fast, feature-driven
    python run_overture.py laya   # real self-hosted model (slow, CPU)
"""
import os
import random
import sys
from collections import Counter

BACKEND = os.environ.get("JEV_BACKEND", "laya")
if len(sys.argv) > 1:
    BACKEND = sys.argv[1]
os.environ["JEV_BACKEND"] = BACKEND

from app import blocking, decisions, features, models, overture  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "overture_london.parquet")

# Central London (Soho / Covent Garden / West End) -- dense chains, the hard
# dedup case where two branches of one brand sit a street apart.
FOCAL = {"lng_lo": -0.145, "lng_hi": -0.115, "lat_lo": 51.505, "lat_hi": 51.518}

INTERESTING = {
    "restaurant", "cafe", "coffee_shop", "bar", "pub", "fast_food_restaurant",
    "hotel", "supermarket", "convenience_store", "pharmacy",
}

SAMPLE_PAIRS = 100


def main():
    pool = overture.load(DATA)
    n = len(pool)

    def cov(field):
        return sum(1 for p in pool if models.get(p.get(field))) / n

    print("=" * 72)
    print(f"OVERTURE London places -> project schema   (backend={BACKEND})")
    print("=" * 72)
    print(f"rows            : {n}")
    print(f"name coverage   : {cov('name'):.1%}")
    print(f"coords coverage : {sum(1 for p in pool if p['lat'] is not None) / n:.1%}")
    print(f"category cover  : {cov('category'):.1%}")
    print(f"brand coverage  : {cov('brand'):.1%}")
    print(f"phone coverage  : {cov('phone'):.1%}")
    lats = [p["lat"] for p in pool if p["lat"] is not None]
    lngs = [p["lng"] for p in pool if p["lng"] is not None]
    print(f"lat range       : {min(lats):.4f} .. {max(lats):.4f}")
    print(f"lng range       : {min(lngs):.4f} .. {max(lngs):.4f}")
    top = Counter(models.get(p.get("category")) for p in pool).most_common(12)
    print("top categories  : " + ", ".join(f"{k}={v}" for k, v in top if k))

    pool = [p for p in pool if models.get(p.get("name")) and p["lat"] is not None]
    print(f"\nfiltered (named + coords) : {len(pool)}")

    def in_focal(p):
        return (FOCAL["lng_lo"] <= p["lng"] <= FOCAL["lng_hi"] and
                FOCAL["lat_lo"] <= p["lat"] <= FOCAL["lat_hi"])

    focus = [p for p in pool if in_focal(p)]
    print(f"focal box (central London): {len(focus)}")
    interesting = [p for p in focus if models.get(p.get("category")) in INTERESTING]
    print(f"  + interesting categories: {len(interesting)}")
    pool = interesting or focus or pool
    print(f"pool for blocking         : {len(pool)}")

    cands = blocking.generate_candidates(pool, geo_step=0.002, max_dist_km=0.3)
    print(f"\nblocking (geo 220m, brand, <=300m): {len(cands)} candidates")

    if not cands:
        print("no candidates -- widen FOCAL or raise max_dist_km")
        return

    if len(cands) > SAMPLE_PAIRS:
        random.seed(7)
        sample = random.sample(cands, SAMPLE_PAIRS)
    else:
        sample = cands

    print(f"deciding {len(sample)} pairs ...")
    results = [(c, decisions.run_candidate(c)) for c in sample]

    routes = Counter(r["route"] for _, r in results)
    labels = Counter(r["auto_label"] for _, r in results)
    print("\n--- decision distribution ---")
    print("routes :", dict(routes))
    print("labels :", dict(labels))
    sc = [r["score"] for _, r in results]
    cf = [r["confidence"] for _, r in results]
    lt = [r["latency_ms"] for _, r in results]
    print(f"mean score {sum(sc)/len(sc):.2f} | mean conf {sum(cf)/len(cf):.2f} | "
          f"mean latency {sum(lt)/len(lt):.0f} ms/pair")

    print("\n--- spot-check (highest-score 6 + lowest-score 6) ---")
    ordered = sorted(results, key=lambda x: -x[1]["score"])
    for c, r in ordered[:6] + ordered[-6:]:
        a, b = c["poi_a"], c["poi_b"]
        f = r["features"]
        print(f"\n  {r['id']} [{r['route']:12s}] score={r['score']:.2f} conf={r['confidence']:.2f}")
        print(f"    A: {models.get(a['name'])} | {models.get(a.get('category')) or '?'} | {models.get(a.get('brand')) or '-'}")
        print(f"    B: {models.get(b['name'])} | {models.get(b.get('category')) or '?'} | {models.get(b.get('brand')) or '-'}")
        print(f"    dist={f['distance_km']:.3f}km  name_sim={f['name_sim']}  brand_match={f['brand_match']}  "
              f"cat_match={f['category_match']}  phone_match={f['phone_match']}")


if __name__ == "__main__":
    main()
