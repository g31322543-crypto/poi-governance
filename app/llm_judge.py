"""POI dedup judge backed by an LLM (System 2) -- the hard reasoning tail.

Laya (System 1) does the high-volume text decisions; this LLM judge does the
one task that needs world-knowledge reasoning: deciding whether two POI records
are the same physical place, especially the same-brand-different-branch case
that a deterministic feature layer cannot tell apart. It runs only on a
blocking shortlist, never against the whole DB.

Default backend is DeepSeek (OpenAI-compatible). Set ``DEEPSEEK_API_KEY``.
"""
import json
import os
import re

from openai import OpenAI

from . import features, models

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

_SYSTEM = (
    "You are a POI (point-of-interest) entity-resolution judge for a local-services "
    "platform. Given two records A and B plus precomputed evidence, decide whether "
    "they are the SAME physical place.\n"
    "Rules:\n"
    "- If the distance is small and the name matches (allowing for noise), they are "
    "the SAME place (a duplicate).\n"
    "- If the brand/name matches but the distance is clearly a different location, "
    "they are DIFFERENT branches of the same chain, NOT the same place.\n"
    "- Weigh distance first, then name, then phone.\n"
    "Respond with a single JSON object and no other text:\n"
    '{"same": true, "confidence": 0.9, "reason": "one short sentence"}'
)

_client = None


def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Set it in your environment, e.g.\n"
                "  export DEEPSEEK_API_KEY=sk-...   (bash) or\n"
                "  set DEEPSEEK_API_KEY=sk-...       (cmd/PowerShell)\n"
                "then re-run.")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
    return _client


def _field(p, key):
    return str(models.get(p.get(key)) or "").strip()


def _render(a, b, f):
    def rec(p):
        return (f"  name: {_field(p, 'name') or '(missing)'}\n"
                f"  address: {_field(p, 'address') or '(missing)'}\n"
                f"  category: {_field(p, 'category') or '(missing)'}\n"
                f"  phone: {_field(p, 'phone') or '(missing)'}\n"
                f"  brand: {_field(p, 'brand') or '(none)'}")

    def yn(v):
        return "true" if v is True else ("false" if v is False else "unknown")

    d = f["distance_km"]
    d = f"{d:.3f} km" if d is not None else "unknown"
    ns = f["name_sim"]
    ns = f"{ns:.2f}" if ns is not None else "unknown"
    return (f"Record A:\n{rec(a)}\n\nRecord B:\n{rec(b)}\n\n"
            f"Computed evidence:\n"
            f"  distance_km: {d}\n"
            f"  name_similarity: {ns}\n"
            f"  brand_match: {yn(f['brand_match'])}\n"
            f"  category_match: {yn(f['category_match'])}\n"
            f"  phone_match: {yn(f['phone_match'])}\n")


def _parse(text):
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in response: {text!r}")
    return json.loads(m.group(0))


def judge_pair(a, b):
    """Return {same, confidence, reason} for one POI pair (needs a live API)."""
    f = features.pair_features(a, b)
    prompt = _render(a, b, f)
    client = _get_client()
    kwargs = dict(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
    )
    try:
        resp = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
    except Exception:
        # some gateways reject response_format; retry without it
        resp = client.chat.completions.create(**kwargs)
    d = _parse(resp.choices[0].message.content)
    return {
        "same": bool(d.get("same")),
        "confidence": float(d.get("confidence", 0.5)),
        "reason": str(d.get("reason", "")).strip(),
        "raw": d,
    }
