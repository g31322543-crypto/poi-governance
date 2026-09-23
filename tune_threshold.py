"""Tune the auto-decision confidence threshold for F1, then validate held-out.

The user's method, exactly: split labeled pairs into train/holdout, sweep the
confidence threshold on train to maximize F1, then check the chosen threshold
holds on the held-out set. Labeled pairs come from app/perturb (real Overture
places + a simulated second source) because a conflated release has no true
intra-release duplicates. Negatives include REAL same-brand different-branch
pairs so the eval isn't trivially separable.

Decision rule under test: `same` when score position >= 2, but only finalized
when `confidence >= T`; otherwise escalate to human. We sweep T.

    python tune_threshold.py mock    # feature-driven backend (has signal)
"""
import os
import random
import sys

BACKEND = os.environ.get("JEV_BACKEND", "mock")
if len(sys.argv) > 1:
    BACKEND = sys.argv[1]
os.environ["JEV_BACKEND"] = BACKEND

from app import decisions, features, models, overture, perturb  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overture_london.parquet")
FOCAL = {"lng_lo": -0.145, "lng_hi": -0.115, "lat_lo": 51.505, "lat_hi": 51.518}
K = 200


def metrics(rows, T):
    tp = fp = fn = tn = esc = 0
    for p, r in rows:
        pred_same = r["auto_label"] == "same"
        if r["confidence"] < T:
            esc += 1
            continue
        gt = p["ground_truth"]
        if pred_same and gt:
            tp += 1
        elif pred_same and not gt:
            fp += 1
        elif not pred_same and gt:
            fn += 1
        else:
            tn += 1
    auto = tp + fp + fn + tn
    if auto == 0:
        return None
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"cov": auto / len(rows), "auto": auto, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "prec": prec, "rec": rec, "f1": f1, "esc": esc}


def show(name, rows, T):
    m = metrics(rows, T)
    if m is None:
        print(f"  {name} @T={T:.2f}: no auto-decisions")
        return None
    print(f"  {name} @T={T:.2f}: coverage={m['cov']*100:.0f}%  P={m['prec']:.3f}  "
          f"R={m['rec']:.3f}  F1={m['f1']:.3f}  (TP{m['tp']} FP{m['fp']} FN{m['fn']} "
          f"TN{m['tn']} esc{m['esc']})")
    return m


def main():
    pool = overture.load(DATA)
    pool = [p for p in pool
            if models.get(p.get("name")) and p["lat"] is not None
            and FOCAL["lng_lo"] <= p["lng"] <= FOCAL["lng_hi"]
            and FOCAL["lat_lo"] <= p["lat"] <= FOCAL["lat_hi"]]
    print(f"focal pool (central London): {len(pool)}")

    pairs = perturb.build_labeled_pairs(pool, k=K, seed=0)
    npos = sum(1 for p in pairs if p["ground_truth"])
    nneg = len(pairs) - npos
    nhard = sum(1 for p in pairs if p["id"].startswith("neg_hard"))
    print(f"labeled pairs: {len(pairs)}  ({npos} positive / {nneg} negative, "
          f"of which {nhard} hard same-brand negatives)")

    rows = [(p, decisions.run_candidate(p)) for p in pairs]

    ph = [features.feature_hint(r["features"]) for p, r in rows if p["ground_truth"]]
    nh = [features.feature_hint(r["features"]) for p, r in rows if not p["ground_truth"]]
    hh = [features.feature_hint(r["features"]) for p, r in rows if p["id"].startswith("neg_hard")]
    print(f"mean feature_hint: positive={sum(ph)/len(ph):.3f}  "
          f"negative={sum(nh)/len(nh):.3f}  (hard same-brand neg={sum(hh)/len(hh):.3f} if any)")

    # stratified split (50/50 per class)
    pos = [(p, r) for p, r in rows if p["ground_truth"]]
    neg = [(p, r) for p, r in rows if not p["ground_truth"]]
    rng = random.Random(1)
    rng.shuffle(pos)
    rng.shuffle(neg)
    train = pos[:len(pos) // 2] + neg[:len(neg) // 2]
    holdout = pos[len(pos) // 2:] + neg[len(neg) // 2:]
    rng.shuffle(train)
    rng.shuffle(holdout)
    print(f"train {len(train)} ({sum(1 for x in train if x[0]['ground_truth'])} pos) | "
          f"holdout {len(holdout)} ({sum(1 for x in holdout if x[0]['ground_truth'])} pos)")

    print("\n--- threshold sweep on TRAIN ---")
    sweep = [(T, metrics(train, T)) for T in [t / 20 for t in range(0, 21)]]

    best = None
    for T, m in sweep:
        if m is not None and m["cov"] >= 0.10 and (best is None or m["f1"] > best["f1"]):
            best = {"T": T, **m}

    print(f"{'T':>5} {'cov%':>5} {'auto':>5} {'P':>6} {'R':>6} {'F1':>6}")
    for T, m in sweep:
        if m is None:
            print(f"{T:5.2f}   (no auto-decisions)")
            continue
        flag = "  <-- best" if (best and abs(best["T"] - T) < 1e-9) else ""
        print(f"{T:5.2f} {m['cov']*100:5.0f} {m['auto']:5d} {m['prec']:6.3f} "
              f"{m['rec']:6.3f} {m['f1']:6.3f}{flag}")

    if best is None:
        print("\nno threshold auto-decides >=10% -- feature signal too weak to gate on")
        return

    print(f"\nchosen threshold T* = {best['T']:.2f}  "
          f"(max F1 among T with >=10% coverage on train)")
    print("--- validate on HOLDOUT ---")
    show("holdout @ T*", holdout, best["T"])
    show("holdout @ no-gate T=0", holdout, 0.0)

    # make the failure mode concrete
    print("\n--- where the mistakes come from: same-brand different-branch ---")
    hards = [(p, r) for p, r in rows if p["id"].startswith("neg_hard")][:6]
    for p, r in hards:
        f = r["features"]
        print(f"  {models.get(p['poi_a']['name'])}  vs  {models.get(p['poi_b']['name'])}")
        print(f"      dist={f['distance_km']:.3f}km  name_sim={f['name_sim']}  "
              f"brand_match={f['brand_match']}  -> hint={features.feature_hint(f):.2f}  "
              f"mock says '{r['auto_label']}'")


if __name__ == "__main__":
    main()
