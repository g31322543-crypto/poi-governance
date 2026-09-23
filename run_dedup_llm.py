"""Evaluate the LLM (System 2) POI dedup judge against the same labeled pairs
where Laya collapsed (conf ~0.1) and the deterministic feature layer sat at
~0.78 -- the bottleneck being same-brand different-branch.

    python run_dedup_llm.py [k]        # k positives + k negatives (default 40)

Prints a rules baseline first (no API needed), then the LLM judge (needs
DEEPSEEK_API_KEY), split by pair type so the hard same-brand negatives stand out.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from app import features, llm_judge, models, overture, perturb

DATA = "overture_london.parquet"
FOCAL = {"lng_lo": -0.145, "lng_hi": -0.115, "lat_lo": 51.505, "lat_hi": 51.518}


def focal_pool():
    pool = overture.load(DATA)
    return [p for p in pool if models.get(p.get("name")) and p.get("lat") is not None
            and FOCAL["lng_lo"] <= p["lng"] <= FOCAL["lng_hi"]
            and FOCAL["lat_lo"] <= p["lat"] <= FOCAL["lat_hi"]]


def metrics(preds, labels):
    tp = sum(1 for p, l in zip(preds, labels) if p and l)
    fp = sum(1 for p, l in zip(preds, labels) if p and not l)
    fn = sum(1 for p, l in zip(preds, labels) if not p and l)
    tn = sum(1 for p, l in zip(preds, labels) if not p and not l)
    n = len(labels)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "n": n}


def report(name, preds, pairs):
    labels = [p["ground_truth"] for p in pairs]
    idx = [i for i, p in enumerate(preds) if p is not None]   # drop failed calls
    n_fail = len(pairs) - len(idx)
    whole = metrics([preds[i] for i in idx], [labels[i] for i in idx])
    print(f"  {name:14s} overall acc={whole['acc']:.3f} F1={whole['f1']:.3f} "
          f"(P={whole['prec']:.3f} R={whole['rec']:.3f})"
          + (f"  [{n_fail} failed]" if n_fail else ""))
    for split, prefix in (("pos (same)", "pos_"), ("neg_easy", "neg_easy_"),
                          ("neg_hard (same-brand)", "neg_hard_")):
        sidx = [i for i in idx if pairs[i]["id"].startswith(prefix)]
        if not sidx:
            continue
        m = metrics([preds[i] for i in sidx], [labels[i] for i in sidx])
        print(f"      {split:24s} acc={m['acc']:.3f}  (n={m['n']})")


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    pool = focal_pool()
    pairs = perturb.build_labeled_pairs(pool, k=k, seed=0)
    labels = [p["ground_truth"] for p in pairs]
    npos = sum(labels)
    nneg = len(labels) - npos
    print(f"labeled pairs: {len(pairs)}  ({npos} same / {nneg} different)\n")

    # --- deterministic rules baseline (no API) ---
    rule_preds = [features.feature_hint(features.pair_features(p["poi_a"], p["poi_b"])) >= 0.5
                  for p in pairs]
    print("=== rules baseline (feature_hint >= 0.5 -> same) ===")
    report("rules", rule_preds, pairs)

    # --- LLM judge ---
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("\nDEEPSEEK_API_KEY not set -- skipping the LLM judge.\n"
              "Set it (export/set DEEPSEEK_API_KEY=sk-...) and re-run to score the LLM.")
        return

    print(f"\n=== LLM judge (deepseek, {llm_judge.DEEPSEEK_MODEL}) ===")
    llm_preds, reasons = [], {}
    for i, p in enumerate(pairs):
        try:
            j = llm_judge.judge_pair(p["poi_a"], p["poi_b"])
            llm_preds.append(j["same"])
            reasons[p["id"]] = j
        except Exception as e:
            llm_preds.append(None)
            reasons[p["id"]] = {"same": None, "reason": f"ERROR: {e}", "confidence": 0.0}
        if (i + 1) % 10 == 0 or i == len(pairs) - 1:
            print(f"  judged {i + 1}/{len(pairs)}", flush=True)
    report("llm", llm_preds, pairs)

    # --- show the hard same-brand cases (the whole point) ---
    print("\n--- hard same-brand different-branch (truth = DIFFERENT) ---")
    for p in pairs:
        if not p["id"].startswith("neg_hard_"):
            continue
        j = reasons.get(p["id"], {})
        f = features.pair_features(p["poi_a"], p["poi_b"])
        d = f"{f['distance_km']:.3f}km" if f["distance_km"] is not None else "?"
        a, b = models.get(p["poi_a"]["name"]), models.get(p["poi_b"]["name"])
        same = j.get("same")
        ok = "ok " if same is False else "BAD"
        print(f"  [{ok}] {a!r} vs {b!r} @ {d} -> same={same} (truth False)")
        print(f"         {j.get('reason', '')}")


if __name__ == "__main__":
    main()
