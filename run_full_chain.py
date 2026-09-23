"""The complete governance chain, both decision engines in one flow (CLI).

    merchant submission
      -> 1. content gate (Laya, System 1): moderate the description text
      -> 2. dedup (blocking -> LLM judge, System 2): same place? merge or new
      -> 3. final route (confidence-gated): auto vs escalate

Ground truth is hidden on each submission (``is_duplicate``, ``matched_id``,
``legitimacy``) so every step is scored honestly. The chain itself lives in
``app/chain.py``, shared with the FastAPI server.

    DEEPSEEK_API_KEY=... python run_full_chain.py
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from app import chain as chain_mod, models, pipeline


def main():
    engine = chain_mod.Chain()
    db = engine.load_db()
    print(f"existing POI DB: {len(db)} places (focal central London)")

    subs = pipeline.build_submissions(db, seed=0)
    print(f"merchant submissions: {len(subs)}\n")

    traces = [engine.run_submission(s) for s in subs]

    for t in traces:
        s = t["submission"]
        m = t["moderation"]
        r = t["resolve"]
        tag = "DUP" if s["is_duplicate"] else ("SPAM" if s["legitimacy"] == "reject" else "NEW")
        name = models.get(s.get("name")) or "?"
        print(f"=== {s['id']} [{tag}] {name!r} ===")
        desc = (s.get("description") or "")[:60]
        print(f"    desc: {desc!r}")
        print(f"    [1] moderate: {m['verdict']:<8} conf={m['confidence']:.2f} [{m['route']}]")
        if r["matched"]:
            print(f"    [2] dedup: merge -> {models.get(r['matched'].get('name'))!r} "
                  f"conf={r['confidence']:.2f} [{r['route']}]")
        else:
            print(f"    [2] dedup: {r['verdict']} [{r['route']}]  {r['reason'][:60]}")
        print(f"    [3] => {t['route']['action']}: {t['route']['why'][:70]}")

    sm = chain_mod.summarize(traces)
    print("\n=== summary ===")
    md = sm["moderation"]
    dd = sm["dedup"]
    rt = sm["route"]
    print(f"moderation: ok={md['ok']} bad={md['bad']} escalated={md['esc']}")
    print(f"dedup     : merge_ok={dd['merge_ok']} merge_bad={dd['merge_bad']} "
          f"new_ok={dd['new_ok']} new_bad={dd['new_bad']} escalated={dd['esc']}")
    print(f"route     : auto_merge={rt['auto_merge']} auto_create={rt['auto_create']} "
          f"auto_reject={rt['auto_reject']} escalate={rt['escalate']} "
          f"(final correct {rt['route_ok']}/{len(traces)})")


if __name__ == "__main__":
    main()
