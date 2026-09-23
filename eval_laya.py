"""Empirical evaluation of the Laya backend on the synthetic dedup workload.

Runs every dedup pair through Laya and reports the numbers the demo cares
about: raw decision accuracy, confidence-gated auto-decision coverage, and
calibration (does confidence track correctness?). Run with:

    python eval_laya.py
"""
import os

os.environ["JEV_BACKEND"] = "laya"

from app import decisions, synthetic_data  # noqa: E402


def main():
    d = synthetic_data.generate_dataset()
    pairs = d["pairs"]

    rows = []
    for p in pairs:
        r = decisions.run_dedup(p)
        correct = (r["auto_label"] == "same") == p["ground_truth"]
        rows.append({"pair": p, "res": r, "correct": correct})

    total = len(rows)
    auto = [x for x in rows if x["res"]["route"].startswith("auto")]
    esc = [x for x in rows if x["res"]["route"] == "escalate"]
    auto_correct = sum(1 for x in auto if x["correct"])
    no_gate_correct = sum(1 for x in rows if x["correct"])

    print("=" * 64)
    print("LAYA (multilingual) on synthetic dedup workload")
    print("=" * 64)
    print(f"pairs            : {total}")
    print(f"auto-decided     : {len(auto)}  (accuracy {auto_correct}/{len(auto)} = "
          f"{auto_correct/len(auto):.2f})" if auto else "auto-decided: 0")
    print(f"escalated        : {len(esc)}")
    print(f"no-gate accuracy : {no_gate_correct}/{total} = {no_gate_correct/total:.3f}  "
          f"(model accuracy if we trusted every answer)")
    print(f"mean confidence  : {sum(x['res']['confidence'] for x in rows)/total:.3f}")
    print(f"mean score       : {sum(x['res']['score'] for x in rows)/total:.3f}")
    print(f"mean latency     : {sum(x['res']['latency_ms'] for x in rows)/total:.0f} ms/pair")

    # --- calibration: confidence bucket vs accuracy ---
    print("\ncalibration (confidence -> accuracy):")
    buckets = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    for lo, hi in buckets:
        b = [x for x in rows if lo <= x["res"]["confidence"] < hi]
        if not b:
            continue
        acc = sum(1 for x in b if x["correct"]) / len(b)
        print(f"  conf [{lo:.2f},{hi:.2f}): n={len(b):2d}  accuracy={acc:.2f}")

    # --- ECE (expected calibration error) ---
    ece = 0.0
    for x in rows:
        conf = x["res"]["confidence"]
        pred_p = x["res"]["probability"] if x["res"]["auto_label"] == "same" else 1 - x["res"]["probability"]
        ece += abs(conf - (1.0 if x["correct"] else 0.0)) / total
    print(f"  ECE (conf vs accuracy, L1): {ece:.3f}")

    # --- spot-check the cross-language + ambiguous cases ---
    print("\nspot-check (cross-language / ambiguous):")
    for x in rows:
        pid = x["pair"]["id"]
        if pid.startswith("dup_clr") or pid.startswith("dup_amb"):
            r = x["res"]
            mark = "OK " if x["correct"] else "ERR"
            print(f"  [{mark}] {pid}: gt={'same' if x['pair']['ground_truth'] else 'diff'}  "
                  f"label={r['auto_label']:9s} score={r['score']:.2f} conf={r['confidence']:.3f} "
                  f"route={r['route']}")


if __name__ == "__main__":
    main()
