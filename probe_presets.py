"""Verify the redesign hypothesis BEFORE building anything: does Laya OOTB make
correct, confident decisions on its own shipped presets, over the English text
content a local-services platform actually moderates/triages?

    python probe_presets.py            # multilingual checkpoint (cached, instant)
    python probe_presets.py english    # English checkpoint (downloads ~421M, better on English)

We are NOT forcing entity resolution this time. We feed Laya the text it was
trained for (a post / a message) and ask its own moderation / triage questions.
If confidence separates clean vs spam / routes tickets sensibly, the redesign
stands on solid ground; if it collapses to ~0.1 again, we learn something even
more fundamental.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import laya

WHICH = sys.argv[1] if len(sys.argv) > 1 else "multilingual"


def show(title, ans):
    a = ans["answers"]
    for qid, q in a.items():
        if q["type"] == "noul":
            print(f"    {qid:14s} -> p(true)={q['noul']:.2f}  conf={q['confidence']:.2f}")
        elif q["type"] == "score":
            print(f"    {qid:14s} -> score={q['score']:.2f}  conf={q['confidence']:.2f}")
        elif q["type"] == "choice":
            print(f"    {qid:14s} -> {q['choice']:20s} conf={q['confidence']:.2f}")
    print()


def main():
    agent = laya.load("convaiinnovations/laya", device="cpu",
                      subfolder=None if WHICH == "english" else "multilingual")
    print(f"checkpoint: {'english (ModernBERT-large)' if WHICH == 'english' else 'multilingual (mmBERT)'}\n")

    mod = laya.moderation_questions()
    print("=== moderation: listing descriptions & reviews (question 'post') ===")
    posts = {
        "spam_desc": "Click here for cash back! Guaranteed rewards, WhatsApp +44 7700 000000, limited offer.",
        "spam_review": "Get paid £500/day working from home, no experience needed, apply at www.fast-money.example.",
        "scam_review": "Buy followers and likes! Cheap and fast, message me on Telegram @faker for a promo code.",
        "legit_desc": "Family-run Italian restaurant, fresh pasta made daily, open 12:00-23:00, dine-in and takeaway.",
        "legit_review": "Great coffee and friendly staff, the croissants are excellent, will definitely come back.",
        "angry_review": "Worst place ever, the owner is a liar and a scammer, never go here.",
    }
    for name, post in posts.items():
        print(f"  [{name}] {post!r}")
        show(name, agent.system_one({"post": post}, mod))

    tri = laya.triage_questions()
    print("=== triage: merchant support tickets (question 'message') ===")
    tickets = {
        "refund": "I was charged twice for my listing subscription, please refund the duplicate charge now.",
        "wrong_hours": "Our opening hours are wrong on your app, we close at 23:00 not 22:00.",
        "cancel": "I would like to cancel my premium listing plan please.",
        "pricing": "How much does it cost to feature my business on the homepage?",
        "fake_review": "Someone posted a fake one-star review about my shop and it is hurting business, please remove it as soon as possible.",
    }
    for name, message in tickets.items():
        print(f"  [{name}] {message!r}")
        show(name, agent.system_one({"message": message}, tri))


if __name__ == "__main__":
    main()
