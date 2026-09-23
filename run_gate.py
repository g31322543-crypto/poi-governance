"""Local Services Content Gate: Laya OOTB over the text a local-services
platform moderates and triages.

    python run_gate.py               # multilingual checkpoint (cached, fast)
    python run_gate.py router        # Router: english + multilingual, script-routed

This is the redesign: Laya does what it was trained for (moderate / triage
text, not entity-resolve records). Each decision is confidence-gated -- auto
when confident, escalate to a human otherwise -- and the run reports accuracy,
automation rate, and cost vs an all-human baseline.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import laya

from app import config, content_data, content_gate, overture

DATA = "overture_london.parquet"
FOCAL = {"lng_lo": -0.145, "lng_hi": -0.115, "lat_lo": 51.505, "lat_hi": 51.518}


def make_decider(mode):
    if mode == "router":
        router = laya.Router(device="cpu")
        router.preload(["english", "multilingual"])

        def decide(state, questions):
            r = router.predict(state, questions)
            return r["answers"], r.get("routing")

        return decide, "router (english + multilingual, script-routed)"
    agent = laya.load("convaiinnovations/laya", device="cpu", subfolder="multilingual")

    def decide(state, questions):
        return agent.system_one(state, questions)["answers"], None

    return decide, "multilingual (mmBERT, single checkpoint)"


def run_item(it, decide):
    if it["kind"] == "listing":
        answers, routing = decide({"post": it["text"]}, laya.moderation_questions())
        d = content_gate.moderate_listing(answers)
        correct = d["verdict"] == it["truth"]
        return {"kind": "listing", "decision": d, "correct": correct,
                "truth": it["truth"], "routing": routing, "text": it["text"],
                "poi": it["poi"], "id": it["id"], "lang": it["lang"]}
    if it["kind"] == "review":
        answers, routing = decide({"post": it["text"]}, laya.moderation_questions())
        d = content_gate.moderate_review(answers)
        correct = d["verdict"] == it["truth"]
        return {"kind": "review", "decision": d, "correct": correct,
                "truth": it["truth"], "reason": it.get("reason"), "routing": routing,
                "text": it["text"], "poi": it["poi"], "id": it["id"], "lang": it["lang"]}
    answers, routing = decide({"message": it["text"]}, laya.triage_questions())
    d = content_gate.triage_ticket(answers)
    correct = d["intent"] == it["truth_intent"]
    return {"kind": "ticket", "decision": d, "correct": correct,
            "truth_intent": it["truth_intent"], "truth_urgent": it.get("truth_urgent"),
            "truth_churn": it.get("truth_churn"), "routing": routing,
            "text": it["text"], "poi": it["poi"], "id": it["id"], "lang": it["lang"]}


def fmt_routing(r):
    if not r:
        return ""
    return f"  [routed->{r.get('model')}: {r.get('reason')}]"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "multilingual"
    decide, label = make_decider(mode)
    print(f"decider: {label}\n")

    db = overture.load(DATA)
    db = [p for p in db if p.get("name") and p.get("lat") is not None
          and FOCAL["lng_lo"] <= p["lng"] <= FOCAL["lng_hi"]
          and FOCAL["lat_lo"] <= p["lat"] <= FOCAL["lat_hi"]]
    print(f"context: {len(db)} real Overture places (focal central London)\n")

    items = content_data.build_items()
    traces = [run_item(it, decide) for it in items]

    for t in traces:
        d = t["decision"]
        mark = "ok " if t["correct"] else "BAD"
        route = d["route"].upper()
        poi = f" @{t['poi']}" if t["poi"] else ""
        text = t["text"].replace("\n", " ")[:64]
        if t["kind"] == "ticket":
            extra = f" urgent={'Y' if d['urgent'] else 'n'} churn={'Y' if d['churn'] else 'n'}"
            print(f"[{mark}] {t['id']} ({t['lang']}) {text!r}{poi}")
            print(f"       -> intent={d['intent']:<16} team={d['team']:<16} "
                  f"conf={d['confidence']:.2f} [{route}]{extra}")
            if t["truth_intent"] != d["intent"]:
                print(f"       truth_intent={t['truth_intent']}")
        else:
            print(f"[{mark}] {t['id']} ({t['lang']}) {text!r}{poi}")
            print(f"       -> {d['verdict']:<8} ({d['reason'] or 'clean'}) conf={d['confidence']:.2f} "
                  f"[{route}]  truth={t['truth']}")
        r = fmt_routing(t["routing"])
        if r:
            print(r)

    sm = content_gate.summarize(traces)
    print("\n=== summary ===")
    for kind, label in (("listing", "listing moderation"),
                        ("review", "review moderation"),
                        ("ticket", "ticket triage")):
        m = sm[kind]
        acc = m["auto_ok"] / m["auto"] if m["auto"] else 0.0
        print(f"{label:20s} n={m['n']:2d}  auto={m['auto']:2d} (correct {m['auto_ok']})  "
              f"escalated={m['escalated']}  auto-acc={acc:.2f}")

    total = sum(m["n"] for m in sm.values())
    auto = sum(m["auto"] for m in sm.values())
    esc = sum(m["escalated"] for m in sm.values())
    cost = auto * config.COST_JEV + esc * config.COST_HUMAN
    baseline = total * config.COST_HUMAN
    print(f"\n{auto}/{total} auto-decided, {esc} escalated to a human")
    print(f"cost ${cost:.2f} vs all-human ${baseline:.2f} (saved ${baseline - cost:.2f})")


if __name__ == "__main__":
    main()
