"""Feature engineering: turn raw POI fields into deterministic evidence.

Jev (System One) is a decision model, not a calculator. Anything we can
compute deterministically we compute HERE and hand to the model as evidence --
distance, name similarity, brand/category/phone match. The model's job is to
weigh that evidence and give a calibrated answer, not to do arithmetic or
string matching. (DESIGN.md §2 原则3: 证据与判断分离)
"""
import re
from difflib import SequenceMatcher
from math import asin, cos, radians, sin, sqrt

from . import models


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km between two coordinates."""
    r = 6371.0
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lng2 - lng1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * r * asin(sqrt(a))


def normalize_name(name):
    """Lowercase, collapse punctuation/whitespace, keep CJK + alnum."""
    if not name:
        return ""
    s = re.sub(r"[^\w一-鿿]+", " ", str(name).lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


# Brand/chain gazetteer: canonical brand -> aliases (CJK + latin). Production
# POI systems keep exactly such a gazetteer, so a transliterated map-vendor
# name ("Starbucks (China World Mall)") still matches the merchant CJK name.
# Used to raise the brand-match signal and to bridge the multilingual gap.
BRANDS = {
    "星巴克": ["星巴克", "starbucks"],
    "海底捞": ["海底捞", "haidilao"],
    "必胜客": ["必胜客", "pizza hut", "pizzahut"],
    "7天连锁酒店": ["7天", "7天连锁", "7 days", "7days"],
    "乐刻健身": ["乐刻", "乐刻健身", "lefit"],
    "全家便利店": ["全家", "全家便利店", "familymart", "family mart"],
    "木屋烧烤": ["木屋烧烤", "wooden house"],
    "同仁堂": ["同仁堂", "tongrentang"],
    "瑞幸咖啡": ["瑞幸", "瑞幸咖啡", "luckin"],
    "喜茶": ["喜茶", "heytea"],
    "麦当劳": ["麦当劳", "mcdonald"],
    "茶百道": ["茶百道", "chabaidao"],
    "优衣库": ["优衣库", "uniqlo"],
    "全聚德": ["全聚德", "quanjude"],
}


def brand_of(name):
    """Return the canonical brand for a name, or "" if none is known.

    '海底捞火锅(三里屯店)' -> '海底捞'
    'Starbucks (China World Mall)' -> '星巴克'
    """
    n = normalize_name(name)
    if not n:
        return ""
    for canon, aliases in BRANDS.items():
        for alias in aliases:
            if alias in n:
                return canon
    # Chinese names are brand-first, so a leading CJK run is a usable brand
    # signal; English names are not ("The ...", "Leicester Square ..."), and
    # the authoritative English brand lives in the POI's `brand` field anyway.
    m = re.search(r"[一-鿿]+", n)
    if m:
        return m.group(0)
    return ""


def brand_of_poi(p):
    """Authoritative brand when the source carries one, else name-based guess.

    Real English sources (Overture) expose an explicit ``brand`` field; that is
    far more reliable than inferring a brand from the display name, so we prefer
    it. Synthetic records have no ``brand`` field, so they fall back unchanged.
    """
    explicit = models.get(p.get("brand"))
    if explicit:
        return normalize_name(explicit)
    return brand_of(models.get(p.get("name")))


def name_similarity(a, b):
    """0..1 similarity: normalized string ratio + brand-match boost.

    Messy resubmissions truncate names ("D4 Productions" -> "d4"), and a raw
    SequenceMatcher ratio scores a short name against its full form absurdly low
    (it loses to unrelated short names). Token containment fixes that: one
    name's tokens being a subset of the other's is a strong same-entity signal.
    """
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    if ta and (ta <= tb or tb <= ta):
        return 0.85
    ratio = SequenceMatcher(None, na, nb).ratio()
    ba, bb = brand_of(a), brand_of(b)
    boost = 0.2 if (ba and ba == bb) else 0.0
    return round(min(1.0, ratio + boost), 4)


def category_match(a, b):
    """Exact match after normalization; None if either side missing."""
    va, vb = normalize_name(a), normalize_name(b)
    if not va or not vb:
        return None
    return va == vb


def phone_match(a, b):
    """Exact phone match; None if either side missing."""
    if models.is_blank(a) or models.is_blank(b):
        return None
    return str(a).strip() == str(b).strip()


def pair_features(a, b):
    """Compute all deterministic evidence for a candidate dedup pair.

    Every feature is either a number/boolean or None (= missing, don't use).
    """
    name_a = models.get(a.get("name"))
    name_b = models.get(b.get("name"))
    lat_a, lng_a = models.get(a.get("lat")), models.get(a.get("lng"))
    lat_b, lng_b = models.get(b.get("lat")), models.get(b.get("lng"))

    dist = None
    if None not in (lat_a, lng_a, lat_b, lng_b):
        dist = round(haversine_km(lat_a, lng_a, lat_b, lng_b), 4)

    ba, bb = brand_of_poi(a), brand_of_poi(b)
    return {
        "name_sim": name_similarity(name_a, name_b) if (name_a and name_b) else None,
        "brand": {"a": ba, "b": bb},
        "brand_match": (ba == bb) if (ba and bb) else None,
        "distance_km": dist,
        "category_match": category_match(models.get(a.get("category")), models.get(b.get("category"))),
        "phone_match": phone_match(models.get(a.get("phone")), models.get(b.get("phone"))),
        "has_coords": None not in (lat_a, lng_a, lat_b, lng_b),
    }


def feature_hint(f):
    """Turn evidence into a 0..1 'same place' prior for the mock model.

    The real Jev would learn this mapping from data; the mock uses a hand-set
    weighting so its answer tracks the evidence we actually computed. This
    makes the feature layer meaningfully drive the demo's accuracy.
    """
    score = 0.0
    if f.get("name_sim") is not None:
        score += 0.45 * f["name_sim"]
    if f.get("distance_km") is not None:
        score += 0.30 * max(0.0, 1.0 - f["distance_km"] / 1.5)
    if f.get("brand_match"):
        score += 0.10
    if f.get("category_match"):
        score += 0.05
    if f.get("phone_match"):
        score += 0.10
    return round(min(1.0, max(0.0, score)), 4)


def explain(f):
    """Weight-ordered attribution for a dedup pair: which evidence moved the
    same-place hint, and by how much. Weights mirror ``feature_hint`` so the
    bars sum to the model's prior (证据与判断分离 -> the judgement is legible).
    """
    out = []
    if f.get("name_sim") is not None:
        out.append({"key": "name", "label": "name", "value": f["name_sim"],
                    "weight": round(0.45 * f["name_sim"], 4)})
    if f.get("distance_km") is not None:
        out.append({"key": "distance", "label": "distance",
                    "value": f["distance_km"], "unit": "km",
                    "weight": round(0.30 * max(0.0, 1.0 - f["distance_km"] / 1.5), 4)})
    if f.get("brand_match") is not None:
        out.append({"key": "brand", "label": "brand", "value": f["brand_match"],
                    "weight": 0.10 if f["brand_match"] else 0.0})
    if f.get("category_match") is not None:
        out.append({"key": "category", "label": "category", "value": f["category_match"],
                    "weight": 0.05 if f["category_match"] else 0.0})
    if f.get("phone_match") is not None:
        out.append({"key": "phone", "label": "phone", "value": f["phone_match"],
                    "weight": 0.10 if f["phone_match"] else 0.0})
    return sorted(out, key=lambda s: -s["weight"])


# --- moderation (review) evidence -----------------------------------------
SPAM_SIGNALS = [
    "cash back", "cashback", "whatsapp", "telegram", "wechat", "weixin", "vx", "qq ",
    "click here", "rewards", "bitcoin", "crypto", "part-time", "part time",
    "work from home", "earn £", "earn $", "apply now", "subscribe", "follow",
    "instant pay", "promo code", "discount code", "buy now", "recruit", "hiring",
    "guaranteed", "limited offer", "no need to",
]


def moderation_hint(sub):
    """0..1 'legitimate' prior for the moderation step, from cheap signals.

    Placeholder evidence, parallel to feature_hint: the real decision model
    (fine-tuned Laya) learns this mapping from data; until then we compute
    obvious spam + completeness signals deterministically. Low = likely
    spam/fraud; high = likely legitimate.
    """
    name = models.get(sub.get("name")) or ""
    desc = models.get(sub.get("description")) or ""
    addr = models.get(sub.get("address")) or ""
    phone = models.get(sub.get("phone")) or ""
    text = f"{name} {desc}".lower()

    spam = any(s in text for s in SPAM_SIGNALS) or any(
        t in text for t in ("http", "www", "@"))
    if spam:
        return 0.06

    score = 0.72
    if name.strip():
        score += 0.06
    if desc.strip():
        score += 0.06
    if len(addr.strip()) >= 8:
        score += 0.06
    if phone.strip():
        score += 0.05
    return round(min(1.0, score), 4)
