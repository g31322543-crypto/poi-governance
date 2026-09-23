"""Curated English content for the Local Services Content Gate demo.

The decision model (Laya) judges *text* -- a merchant listing description, a
user review, or a merchant support message. Each item carries hidden ground
truth so the gate can be evaluated honestly. Texts are hand-written to be
realistic and diverse (NOT template-matched against the model's spam lexicon):
the model has to generalise from its training, which is the point.

POI names are realistic central-London places. ``run_gate.py`` loads the real
Overture pool as extra grounding, but the decision input is always the text.
"""


def build_items():
    """Return the demo content set with hidden ground truth.

    kind / truth:
      listing -> truth in {"approve", "reject"}
      review  -> truth in {"publish", "hide"}, plus reason for hides
      ticket  -> truth_intent (a triage category), truth_urgent/churn where clear
    """
    items = []

    def add(it):
        it.setdefault("lang", "en")
        items.append(it)

    # --- merchant listing descriptions (moderation: spam or legit) ---------
    listings = [
        ("Family-run Italian restaurant, fresh pasta made daily, open 12:00-23:00, dine-in and takeaway.",
         "approve", "Trattoria Bella"),
        ("Neighbourhood gym, 24-hour access, personal training and group classes, first session free.",
         "approve", "Fitzrovia Fitness"),
        ("Independent coffee shop, specialty beans roasted on site, pastries baked every morning.",
         "approve", "Café Ciel"),
        ("Grocery and convenience store, open late, fresh produce and daily essentials.",
         "approve", "Green Leaf Grocers"),
        ("Click here for cash back! Guaranteed rewards, WhatsApp +44 7700 000000, limited offer.",
         "reject", "BrewDog Soho"),
        ("No need to visit. Part-time work from home, earn £500/day, apply now at www.fast-money.example.",
         "reject", "Pret A Manger"),
        ("Get rich quick with crypto! Join our Telegram group for free signals, 100x returns guaranteed.",
         "reject", "The Corner Shop"),
        ("Free phone recharge, just enter your number and share with 5 friends, promo code FREE50.",
         "reject", "Leicester Square News"),
    ]
    for i, (text, truth, poi) in enumerate(listings):
        add({"id": f"listing_{i:02d}", "kind": "listing", "poi": poi,
             "category": "business description", "text": text, "truth": truth})

    # --- user reviews (moderation: publish vs hide) ------------------------
    reviews = [
        ("Great coffee and friendly staff, the croissants are excellent, will definitely come back.",
         "publish", None, "Café Ciel"),
        ("Best ramen in town, quick service even at lunch, the broth is outstanding.",
         "publish", None, "Koya Soho"),
        ("Decent food but a bit pricey, service was fine and the room is cosy.",
         "publish", None, "Hawksmoor"),
        ("Food arrived cold and service was slow, expected better for the price.",
         "publish", None, "Barrafina"),          # critical but civil -> publish
        ("The owner is a liar and a scammer, they overcharge everyone, never go to this dump.",
         "hide", "abusive", "The Corner Shop"),
        ("Staff are thieves who steal from customers, stay away from this place.",
         "hide", "abusive", "BrewDog Soho"),
        ("Buy followers and likes! Cheap and fast, message me on Telegram @faker for a promo code.",
         "hide", "spam", "Pret A Manger"),
        ("Earn £500/day working from home, no experience needed, apply now at www.fast-money.example.",
         "hide", "spam", "Starbucks"),
        ("Get free gift cards, click the link below, limited time offer!!!",
         "hide", "spam", "Costa Coffee"),
    ]
    for i, (text, truth, reason, poi) in enumerate(reviews):
        add({"id": f"review_{i:02d}", "kind": "review", "poi": poi,
             "category": "customer review", "text": text, "truth": truth, "reason": reason})

    # --- merchant support tickets (triage: intent + urgency + churn) -------
    tickets = [
        ("I was charged twice for my listing subscription this month, please refund the duplicate charge now.",
         "refund", True, False, "Trattoria Bella"),
        ("My payout for last week never arrived in my bank account, can you check where it is?",
         "refund", False, False, "Fitzrovia Fitness"),
        ("Our opening hours are wrong on the app, we close at 23:00 not 22:00 and I cannot change them.",
         "technical_help", False, False, "The Corner Cafe"),
        ("My menu upload keeps failing with an error, the photos will not attach.",
         "technical_help", False, False, "Green Leaf Grocers"),
        ("How much does it cost to feature my business on the homepage for a week?",
         "billing_question", False, False, "Sunrise Bakery"),
        ("Can you explain the invoice I received? There is a line item I do not recognise.",
         "billing_question", False, False, "Harbour Fitness"),
        ("What are the requirements to verify my business as a registered merchant?",
         "information", False, False, "Café Ciel"),
        ("How do I reply to customer reviews from my merchant dashboard?",
         "information", False, False, "BrewDog Soho"),
        ("I would like to cancel my premium listing plan please.",
         "cancellation", False, True, "Pret A Manger"),
        ("Please close my merchant account, I am moving my business to another platform.",
         "cancellation", True, True, "The Ivy"),
    ]
    for i, (text, intent, urgent, churn, poi) in enumerate(tickets):
        add({"id": f"ticket_{i:02d}", "kind": "ticket", "poi": poi,
             "category": "merchant support", "text": text,
             "truth_intent": intent, "truth_urgent": urgent, "truth_churn": churn})

    # --- non-English content (Router demo: script detection -> checkpoint) -
    foreign = [
        ("Très bon café, personnel accueillant, je reviendrai avec plaisir.",
         "review", "publish", "fr", "Café Ciel"),
        ("フォロワーを買いませんか？今なら無料で現金がもらえます、詳しくはDMで。",
         "review", "hide", "ja", "Ramen Ichiban"),
        ("تم خصم المبلغ مرتين من حسابي، أرجو استرداد المبلغ في أقرب وقت.",
         "ticket", "refund", "ar", "Beirut Express"),
    ]
    for i, (text, kind, truth, lang, poi) in enumerate(foreign):
        add({"id": f"foreign_{i:02d}", "kind": kind, "poi": poi,
             "category": "customer review" if kind == "review" else "merchant support",
             "text": text, "lang": lang,
             "truth": truth if kind == "review" else None,
             "truth_intent": truth if kind == "ticket" else None})

    return items
