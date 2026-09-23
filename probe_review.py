"""Quick probe: is Laya OOTB actually in-distribution for POI review/routing?

Runs the review pipeline (approve / needs-review / reject) on real Overture
places -- known chains (high confidence, obviously legit) vs obscure low-
confidence listings -- and checks whether Laya's confidence is meaningful here
(unlike dedup, where it collapsed to ~0.11).
"""
import os

os.environ["JEV_BACKEND"] = "laya"

from app import decisions, models, overture  # noqa: E402

DATA = "overture_london.parquet"


def to_merchant(p):
    return {
        "name": models.get(p.get("name")),
        "address": models.get(p.get("address")),
        "category": models.get(p.get("category")),
        "phone": models.get(p.get("phone")),
        "description": "",
    }


def main():
    pool = overture.load(DATA)

    chains = ["starbucks", "pret a manger", "mcdonald", "tesco", "costa", "sainsbury"]
    legit = [p for p in pool
             if p["confidence"] is not None and p["confidence"] >= 0.9
             and any(c in models.get(p.get("name") or "").lower() for c in chains)]
    sketchy = [p for p in pool if p["confidence"] is not None and p["confidence"] < 0.3]

    print("=== LAYA OOTB on real POI review / routing ===\n")
    for label, group in [("LEGIT (high-confidence chains)", legit[:5]),
                         ("SKETCHY (low-confidence obscure)", sketchy[:5])]:
        print(f"-- {label} --")
        for p in group:
            r = decisions.run_review({"id": p["id"], "merchant": to_merchant(p), "hint": 0.5})
            print(f"  overture_conf={p['confidence']:.2f}  {models.get(p.get('name'))!r}")
            print(f"      -> level={r['level']!r}  score={r['score']:.2f}  "
                  f"model_conf={r['confidence']:.2f}  route={r['route']}")
        print()


if __name__ == "__main__":
    main()
