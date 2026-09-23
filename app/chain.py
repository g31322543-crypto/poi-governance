"""The full governance chain as a reusable service.

    merchant submission
      -> 1. content gate (Laya, System 1): moderate the description text
      -> 2. dedup (blocking -> LLM judge, System 2): same place? merge or new
      -> 3. final route (confidence-gated): auto vs escalate

This is the canonical implementation shared by the CLI (``run_full_chain.py``)
and the FastAPI server (``main.py``). Both engines are loaded lazily and cached
on first use: the Laya checkpoint (heavy, mmBERT) and the Overture focal DB
(fast). The System-2 dedup judge is DeepSeek when ``DEEPSEEK_API_KEY`` is set,
else a deterministic rules fallback so the app still runs offline.
"""
import os

from . import config, content_gate, features, llm_judge, models, overture, pipeline, store

DATA = "overture_london.parquet"
FOCAL = {"lng_lo": -0.145, "lng_hi": -0.115, "lat_lo": 51.505, "lat_hi": 51.518}


def dedup_backend():
    return "llm" if os.environ.get("DEEPSEEK_API_KEY", "").strip() else "rules"


class Chain:
    def __init__(self, data=DATA, focal=FOCAL):
        self.data = data
        self.focal = focal
        self._agent = None
        self._db = None

    # -- lazy resources -----------------------------------------------------
    def load_db(self):
        if self._db is None:
            db = overture.load(self.data)
            self._db = [p for p in db
                        if models.get(p.get("name")) and p.get("lat") is not None
                        and self.focal["lng_lo"] <= p["lng"] <= self.focal["lng_hi"]
                        and self.focal["lat_lo"] <= p["lat"] <= self.focal["lat_hi"]]
        return self._db

    def load_agent(self):
        if self._agent is None:
            import laya
            self._agent = laya.load(config.LAYA_MODEL_ID,
                                    device=config.LAYA_DEVICE or None,
                                    subfolder=config.LAYA_SUBFOLDER or None)
        return self._agent

    def status(self):
        return {
            "laya_model": config.LAYA_MODEL_ID,
            "laya_subfolder": config.LAYA_SUBFOLDER,
            "dedup_backend": dedup_backend(),
            "confidence_auto": config.CONFIDENCE_AUTO,
            "db_loaded": self._db is not None,
            "db_size": len(self._db) if self._db is not None else None,
            "agent_loaded": self._agent is not None,
        }

    # -- step 1: moderation (System 1) --------------------------------------
    def moderate(self, sub):
        import laya
        answers = self.load_agent().system_one(
            {"post": sub.get("description") or ""}, laya.moderation_questions())["answers"]
        return content_gate.moderate_listing(answers)

    # -- step 2: dedup (System 2) -------------------------------------------
    def _judge(self, sub, cand):
        if dedup_backend() == "llm":
            j = llm_judge.judge_pair(sub, cand)
            return {"same": j["same"], "confidence": j["confidence"],
                    "reason": j["reason"], "backend": "llm"}
        f = features.pair_features(sub, cand)
        hint = features.feature_hint(f)
        return {"same": hint >= 0.5, "confidence": round(max(hint, 1 - hint), 4),
                "reason": "rules fallback (no DEEPSEEK_API_KEY)", "backend": "rules"}

    def resolve(self, sub, topk=5):
        # Real-time index: read-only Overture seed + locally-created places.
        # New places enter the dedup scan the moment they are auto_created.
        db = self.load_db() + store.places()
        cands = pipeline.find_matches(sub, db, topk=topk)
        judged = []
        for c in cands:
            f = features.pair_features(sub, c)
            try:
                j = self._judge(sub, c)
            except Exception as e:  # network / API failure -> unjudgeable
                j = {"same": None, "confidence": 0.0, "reason": f"error: {e}",
                     "backend": dedup_backend()}
            judged.append({"poi": c, "distance_km": f.get("distance_km"),
                           "signals": features.explain(f), "judge": j})
        same_hits = [x for x in judged if x["judge"].get("same")]
        if not same_hits:
            return {"verdict": "new", "route": "auto_create", "matched": None,
                    "confidence": 1.0, "reason": "no same-place candidate",
                    "backend": dedup_backend(), "candidates": judged, "signals": []}
        best = max(same_hits, key=lambda x: x["judge"].get("confidence", 0.0))
        c, j = best["poi"], best["judge"]
        if j["confidence"] >= config.CONFIDENCE_AUTO:
            return {"verdict": "merge", "route": "auto_merge", "matched": c,
                    "confidence": j["confidence"], "reason": j["reason"],
                    "backend": j["backend"], "candidates": judged,
                    "signals": best["signals"]}
        return {"verdict": "uncertain", "route": "escalate", "matched": None,
                "confidence": j["confidence"], "reason": j["reason"],
                "backend": j["backend"], "candidates": judged,
                "signals": best["signals"]}

    # -- step 3: route ------------------------------------------------------
    def final_route(self, mod, res):
        if mod["verdict"] == "reject":
            return {"action": "auto_reject", "why": "moderation rejected the listing"}
        if mod["route"] == "escalate":
            return {"action": "escalate", "why": "moderation was not confident"}
        if res["route"] == "escalate":
            return {"action": "escalate", "why": "dedup was not confident"}
        if res["verdict"] == "merge":
            return {"action": "auto_merge",
                    "matched_id": res["matched"]["id"] if res["matched"] else None,
                    "why": res["reason"]}
        return {"action": "auto_create", "why": "new place, listing is legitimate"}

    # -- run one submission end-to-end --------------------------------------
    def run_submission(self, sub, topk=5, persist=True):
        mod = self.moderate(sub)
        res = self.resolve(sub, topk=topk)
        route = self.final_route(mod, res)
        if persist:
            try:
                store.record(sub, mod, res, route)
            except Exception as e:
                print(f"[store] persist failed: {e}")
        return {"submission": sub, "moderation": mod, "resolve": res, "route": route}

    # -- example submissions for the UI -------------------------------------
    def examples(self):
        db = self.load_db()
        out = []
        seen = set()
        for p in db:
            b = features.brand_of_poi(p)
            if not b or b in seen:
                continue
            seen.add(b)
            out.append({
                "id": f"dup_{len(out)}",
                "label": f"Duplicate of {models.get(p.get('name'))}",
                "kind": "dup",
                "submission": self._example_sub(
                    models.get(p.get("name")), models.get(p.get("address")) or "",
                    "Family-run Italian restaurant, fresh pasta made daily, open 12:00-23:00, dine-in and takeaway.",
                    models.get(p.get("category")) or "", "", models.get(p.get("brand")) or "",
                    p.get("lat"), p.get("lng")),
            })
            if len(out) >= 3:
                break
        out.append({
            "id": "spam_0", "label": "Spam / scam listing", "kind": "spam",
            "submission": self._example_sub(
                "Fast Cash Rewards", "London",
                "Click here for cash back! Guaranteed rewards, WhatsApp +44 7700 000000, limited offer.",
                "other", "", "", 51.5116, -0.1270),
        })
        out.append({
            "id": "new_0", "label": "New place (no existing match)", "kind": "new",
            "submission": self._example_sub(
                "Sunrise Bakery", "14 Monmouth Street, London",
                "Corner deli, sandwiches and salads made to order, open for lunch.",
                "bakery", "", "", 51.5132, -0.1278),
        })
        return out

    @staticmethod
    def _example_sub(name, address, description, category, phone, brand, lat, lng):
        return {"id": "example", "entity_id": "example", "name": name, "address": address,
                "description": description, "category": category, "phone": phone,
                "brand": brand, "lat": lat, "lng": lng}
