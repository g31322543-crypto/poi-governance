"""POI data model with field-level source + quality.

Real POI data arrives from many sources -- merchant self-report, map vendors,
user-generated content -- and they conflict. We attach a source + quality to
every field so downstream layers can (a) resolve conflicts and (b) prefer
high-quality fields when computing evidence.

`source` rank is the primary tie-breaker; `quality` (0..1) breaks ties within
a source. This is the "字段级来源可信度" from DESIGN.md §6.
"""

# Lower rank = more trusted.
SOURCE_RANK = {
    "merchant": 0,     # 商家自报 -- accurate on name/phone, coords self-reported
    "map_vendor": 1,   # 地图供应商 -- accurate on coordinates
    "ugc": 2,          # 用户贡献 -- least trusted
}


def field(value, source="merchant", quality=1.0):
    """Wrap a raw field value with its provenance."""
    return {"value": value, "source": source, "quality": quality}


def get(f, default=None):
    """Unwrap a field value (accepts wrapped fields or bare values)."""
    if isinstance(f, dict) and "value" in f:
        return f["value"]
    return default if f is None else f


def is_blank(v):
    return v is None or (isinstance(v, str) and not v.strip())


def resolve(*fields):
    """Resolve conflicting values for one logical field across sources.

    Returns the highest-trust non-blank value, preferring source rank then
    quality. Used when merging duplicate POIs (§6 conflict arbitration).
    """
    scored = []
    for f in fields:
        v = get(f)
        if is_blank(v):
            continue
        if isinstance(f, dict):
            src = f.get("source", "ugc")
            q = f.get("quality", 0.0)
        else:
            src, q = "ugc", 0.0
        scored.append((v, src, q))
    if not scored:
        return None
    scored.sort(key=lambda t: (SOURCE_RANK.get(t[1], 99), -t[2]))
    return scored[0][0]
