"""English ROOT checkpoint (ModernBERT-large 421M) on English POI data.

This is the primary/full-size Laya checkpoint (repo root, no subfolder), the
one that matters for an overseas English POI use case. First run downloads the
~1GB encoder + head weights. Reuses the English pair builder from
eval_laya_en.py so the dataset is identical across the two probes.
"""
import os

os.environ["JEV_BACKEND"] = "laya"
os.environ["LAYA_SUBFOLDER"] = ""   # empty -> repo root = English ModernBERT-large

from eval_laya_en import build_pairs  # noqa: E402
from app import decisions             # noqa: E402


def main():
    pairs = build_pairs()
    rows = []
    for p in pairs:
        r = decisions.run_dedup(p)
        correct = (r["auto_label"] == "same") == p["ground_truth"]
        rows.append((p, r, correct))

    auto = [x for x in rows if x[1]["route"].startswith("auto")]
    no_gate = sum(1 for x in rows if x[2])
    print("=" * 64)
    print("LAYA English root (ModernBERT-large) on ENGLISH POI data")
    print("=" * 64)
    print(f"model            : {rows[0][1]['model']}")
    print(f"pairs            : {len(rows)}")
    print(f"auto-decided     : {len(auto)}")
    print(f"no-gate accuracy : {no_gate}/{len(rows)} = {no_gate/len(rows):.3f}")
    print(f"mean confidence  : {sum(x[1]['confidence'] for x in rows)/len(rows):.3f}")
    print()
    for p, r, c in rows:
        mark = "OK " if c else "ERR"
        print(f"  [{mark}] {p['id']:11s} gt={'same' if p['ground_truth'] else 'diff'}  "
              f"label={r['auto_label']:9s} score={r['score']:.2f} conf={r['confidence']:.3f}")


if __name__ == "__main__":
    main()
