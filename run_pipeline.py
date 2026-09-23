"""Run the full governance chain (submission -> dedup -> review -> route) over
real Overture data.

    python run_pipeline.py            # mock backend (feature-driven)
    python run_pipeline.py laya       # self-hosted Laya (OOTB collapses; known)
    python run_pipeline.py jev        # TypeSafe System One API (needs key)
"""
import os
import sys

# Windows console defaults to GBK; the POI data is UTF-8 (accents, CJK).
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) > 1:
    os.environ["JEV_BACKEND"] = sys.argv[1]

from app import models, overture, pipeline  # noqa: E402

DATA = "overture_london.parquet"
FOCAL = {"lng_lo": -0.145, "lng_hi": -0.115, "lat_lo": 51.505, "lat_hi": 51.518}


def fmt(p):
    name = models.get(p.get("name")) or "?"
    addr = models.get(p.get("address")) or ""
    return f"{name} ({addr[:28]})" if addr else name


def main():
    db = overture.load(DATA)
    db = [p for p in db
          if models.get(p.get("name")) and p["lat"] is not None
          and FOCAL["lng_lo"] <= p["lng"] <= FOCAL["lng_hi"]
          and FOCAL["lat_lo"] <= p["lat"] <= FOCAL["lat_hi"]]
    print(f"existing POI DB: {len(db)} places (focal central London)")

    subs = pipeline.build_submissions(db, seed=0)
    print(f"merchant submissions: {len(subs)}")

    traces = [pipeline.run_submission(s, db) for s in subs]

    for t in traces:
        s = t["submission"]
        r = t["resolve"]
        v = t["review"]
        rt = t["route"]
        tag = "DUP" if s["is_duplicate"] else ("SPAM" if s["legitimacy"] == "reject" else "NEW")
        print(f"\n=== {s['id']}  [{tag}]  {fmt(s)} ===")
        if r["candidates"]:
            b = r["best"]
            d = f"{b['distance_km']:.3f}km" if b["distance_km"] is not None else "?"
            print(f"    dedup: top={fmt(b['candidate'])} ({d}, hint={b['hint']:.2f}, "
                  f"conf={b['decision']['confidence']:.2f}) -> {r['verdict']} ({r['route']})")
        else:
            print(f"    dedup: no match -> {r['verdict']} ({r['route']})")
        print(f"    review: {v['level']} (conf={v['confidence']:.2f}) -> {v['route']}")
        print(f"    => {rt['action']}: {rt['why']}")

    print("\n=== summary ===")
    sm = pipeline.summarize(traces)
    d = sm["dedup"]
    rv = sm["review"]
    print(f"dedup : merge_ok={d['merge_ok']}  merge_bad={d['merge_bad']}  "
          f"new_ok={d['new_ok']}  new_bad={d['new_bad']}  escalated={d['escalated']}")
    print(f"review: reject_ok={rv['reject_ok']}  reject_bad={rv['reject_bad']}  "
          f"approve_ok={rv['approve_ok']}  approve_bad={rv['approve_bad']}  "
          f"escalated={rv['escalated']}")


if __name__ == "__main__":
    main()
