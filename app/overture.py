"""Real-data ingestion: Overture Maps GeoParquet -> project POI schema.

Overture's place theme is rich (struct/list columns). We flatten the fields the
dedup pipeline consumes into the same bare shape the synthetic data uses --
name/address/category/phone/lat/lng -- plus provenance (brand/confidence/
sources) so the demo can show real-data quality signals.

A single Overture release is already conflated (one row per place, ``sources``
lists every contributor), so there are no true intra-release duplicates. That
is exactly the point: the pipeline still has to tell apart nearby same-brand
*branches* (the hard case), and its "same" calls should be sparse. We set
``entity_id = id`` (each row its own entity) so the blocking layer treats every
row as a distinct entity without crashing on missing ground truth.
"""
import pyarrow.parquet as pq

try:
    from shapely import wkb as _wkb
except Exception:  # shapely optional; bbox centroid is the fallback
    _wkb = None


def _coord(geometry, bbox):
    """(lng, lat) from WKB geometry, falling back to bbox centroid."""
    if _wkb is not None and geometry is not None:
        try:
            g = _wkb.loads(geometry)
            c = g.centroid
            return round(c.x, 6), round(c.y, 6)
        except Exception:
            pass
    if bbox is not None:
        return (
            round((bbox["xmin"] + bbox["xmax"]) / 2, 6),
            round((bbox["ymin"] + bbox["ymax"]) / 2, 6),
        )
    return None, None


def load(path):
    """Load a GeoParquet file into a list of project-schema POI dicts."""
    table = pq.read_table(path)
    pool = []
    for row in table.to_pylist():
        names = row.get("names") or {}
        cats = row.get("categories") or {}
        brand = row.get("brand") or {}
        brand_names = brand.get("names") or {}
        addrs = row.get("addresses") or []
        addr = ""
        if addrs:
            a = addrs[0]
            addr = a.get("freeform") or " ".join(
                x for x in (a.get("locality"), a.get("postcode"), a.get("region")) if x
            )
        phones = row.get("phones") or []
        lng, lat = _coord(row.get("geometry"), row.get("bbox"))
        pool.append({
            "id": row.get("id"),
            "entity_id": row.get("id"),  # one row = one place in a conflated release
            "name": names.get("primary"),
            "address": addr,
            "category": cats.get("primary"),
            "phone": phones[0] if phones else "",
            "lat": lat,
            "lng": lng,
            "brand": brand_names.get("primary"),
            "confidence": row.get("confidence"),
            "sources": [s.get("dataset") for s in (row.get("sources") or [])],
        })
    return pool
