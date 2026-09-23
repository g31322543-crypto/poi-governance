"""Thin wrapper over the TypeSafe System One (Jev) API.

Jev is a "System 1" decision model: it takes a shared `state` plus a set of
typed `questions` and returns structured, calibrated answers -- `noul`
(0..1 probability), `choice` (option + distribution + confidence) and
`score` (position on an ordered rubric + distribution + confidence).

We only use `score` questions here because they return an explicit
`confidence`, which drives the confidence-gated routing.

When TYPESAFE_API_KEY is absent we use a deterministic mock, driven by a
per-sample "hint score", so the whole app runs offline.
"""
import math
import random
import time

import requests

from . import config


class JevClient:
    def __init__(self):
        self.backend = config.BACKEND
        self.mock = self.backend == "mock"
        self.model = config.JEV_MODEL
        self.base_url = config.JEV_BASE_URL
        self.api_key = config.TYPESAFE_API_KEY
        self._laya_agent = None

    # -- public -------------------------------------------------------------
    def decide(self, state, questions, hint=None):
        """Run `questions` against `state`.

        `hint` is only used by the mock; it is a dict with a scalar
        ``score`` in [0, 1] (how strongly "positive" the case is) plus a
        stable ``seed`` so a given sample always produces the same answer.
        """
        start = time.perf_counter()
        if self.backend == "mock":
            answers = self._mock_answers(questions, hint or {})
        elif self.backend == "laya":
            answers = self._laya_answers(state, questions)
        else:
            answers = self._http_answers(state, questions)
        model_label = {
            "mock": "mock (deterministic)",
            "jev": self.model,
            "laya": f"{config.LAYA_MODEL_ID} ({config.LAYA_SUBFOLDER})",
        }[self.backend]
        return {
            "answers": answers,
            "model": model_label,
            "mock": self.mock,
            "backend": self.backend,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    # -- real API -----------------------------------------------------------
    def _http_answers(self, state, questions):
        payload = {"model": self.model, "state": state, "questions": questions}
        resp = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["answers"]

    # -- Laya (self-hosted System One) --------------------------------------
    def _get_laya_agent(self):
        if self._laya_agent is None:
            import laya
            self._laya_agent = laya.load(
                config.LAYA_MODEL_ID,
                device=config.LAYA_DEVICE or None,
                subfolder=config.LAYA_SUBFOLDER or None,
            )
        return self._laya_agent

    def _laya_answers(self, state, questions):
        agent = self._get_laya_agent()
        return agent.system_one(state, questions)["answers"]

    # -- mock ---------------------------------------------------------------
    def _mock_answers(self, questions, hint):
        base = float(hint.get("score", 0.5))
        seed = int(hint.get("seed", 0))
        out = {}
        for idx, (qid, q) in enumerate(questions.items()):
            rng = random.Random(seed * 1000 + idx)
            if q["type"] == "score":
                out[qid] = self._mock_score(base, q, rng)
            else:  # noul
                out[qid] = self._mock_noul(base, qid, rng)
        return out

    def _mock_score(self, base, q, rng):
        levels = q["criteria"]
        n = len(levels)
        noisy = max(0.0, min(1.0, base + rng.gauss(0, 0.08)))
        pos = noisy * (n - 1)
        # Confidence is high at the extremes and low in the middle --
        # the "calibrated" behaviour that makes gating meaningful.
        confidence = max(0.03, min(1.0, 2 * abs(noisy - 0.5)))
        probs = {}
        for i in range(n):
            d = (i - pos) / max(n - 1, 1)
            probs[levels[i]] = round(math.exp(-(d * d) / (2 * 0.18 ** 2)), 4)
        tot = sum(probs.values()) or 1.0
        probs = {k: round(v / tot, 4) for k, v in probs.items()}
        return {
            "type": "score",
            "score": round(pos, 3),
            "legend": {str(i): lvl for i, lvl in enumerate(levels)},
            "probabilities": probs,
            "confidence": round(confidence, 3),
        }

    def _mock_noul(self, base, qid, rng):
        polarity = -1 if any(k in qid.lower() for k in ("risk", "spam", "flag", "fake")) else 1
        p = base if polarity > 0 else 1 - base
        return {"type": "noul", "noul": round(max(0.0, min(1.0, p + rng.gauss(0, 0.08))), 3)}
