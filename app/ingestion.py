"""Simulated multi-source POI ingestion.

Generates a pool of POI records as they'd actually arrive in production: the
same real-world shop submitted by several sources (merchant self-report, map
vendor, user UGC), each with its own noise profile -- name drift, coordinate
offset (incl. un-normalized coordinate systems), missing fields, and
category/address conflicts.

Every record carries a hidden ``entity_id`` used only to evaluate downstream
layers (blocking recall, dedup accuracy). Field-level source + quality comes
from models.field(). (DESIGN.md §1.3 痛点 + §6 来源可信度)
"""
import random

from . import models

# True-world entities (hidden ground truth). "real" fields are what a perfect
# record would hold. A few carry ``en`` (transliteration) to exercise the
# multilingual gap that scenario 6 must later close.
ENTITIES = [
    {"entity": "starbucks_guomao", "name": "星巴克(国贸店)", "en": "Starbucks (China World Mall)",
     "cat": "咖啡", "addr": "北京市朝阳区建国门外大街1号国贸商城B1层", "lat": 39.9087, "lng": 116.4583, "phone": "010-65051234"},
    {"entity": "haidilao_sanlitun", "name": "海底捞火锅(三里屯店)", "en": "Haidilao Hot Pot (Sanlitun)",
     "cat": "火锅", "addr": "北京市朝阳区工体北路8号三里屯SOHO 1层", "lat": 39.9360, "lng": 116.4554, "phone": "010-85987654"},
    {"entity": "pizzahut_xidan", "name": "必胜客(西单店)", "en": "Pizza Hut (Xidan)",
     "cat": "西餐", "addr": "北京市西城区西单北大街120号", "lat": 39.9093, "lng": 116.3733, "phone": "010-66012345"},
    {"entity": "7days_wangfujing", "name": "7天连锁酒店(王府井店)", "en": "7 Days Inn (Wangfujing)",
     "cat": "酒店", "addr": "北京市东城区东安门大街53号", "lat": 39.9160, "lng": 116.4109, "phone": "010-65123456"},
    {"entity": "lefit_wangjing", "name": "乐刻健身(望京SOHO店)", "en": "Lefit Gym (Wangjing SOHO)",
     "cat": "健身房", "addr": "北京市朝阳区望京街10号望京SOHO T1", "lat": 39.9966, "lng": 116.4817, "phone": "010-64777777"},
    {"entity": "familymart_zhongguancun", "name": "全家便利店(中关村店)", "en": "FamilyMart (Zhongguancun)",
     "cat": "便利店", "addr": "北京市海淀区海淀中街6号", "lat": 39.9830, "lng": 116.3160, "phone": "010-82668888"},
    {"entity": "mumu_wudaokou", "name": "木屋烧烤(五道口店)", "en": "Wooden House BBQ (Wudaokou)",
     "cat": "烧烤", "addr": "北京市海淀区成府路35号", "lat": 39.9910, "lng": 116.3350, "phone": "010-62345678"},
    {"entity": "tongrentang_qianmen", "name": "同仁堂药店(前门店)", "en": "Tongrentang Pharmacy (Qianmen)",
     "cat": "药房", "addr": "北京市东城区前门大街42号", "lat": 39.8990, "lng": 116.4010, "phone": "010-67081111"},
    {"entity": "luckin_jinrong", "name": "瑞幸咖啡(金融街店)", "en": "Luckin Coffee (Financial Street)",
     "cat": "咖啡", "addr": "北京市西城区金融大街35号", "lat": 39.9150, "lng": 116.3560, "phone": "400-100-0000"},
    {"entity": "heytea_joycity", "name": "喜茶(朝阳大悦城店)", "en": "Heytea (Chaoyang Joy City)",
     "cat": "奶茶", "addr": "北京市朝阳区朝阳北路101号朝阳大悦城5层", "lat": 39.9240, "lng": 116.5060, "phone": "010-85555555"},
    # --- extra entities forming tight clusters (hard cross-entity negatives) ---
    {"entity": "mcdonalds_sanlitun", "name": "麦当劳(三里屯店)", "en": "McDonald's (Sanlitun)",
     "cat": "快餐", "addr": "北京市朝阳区三里屯路19号", "lat": 39.9355, "lng": 116.4560, "phone": "010-85981111"},
    {"entity": "chabaidao_sanlitun", "name": "茶百道(三里屯店)", "en": "Chabaidao (Sanlitun)",
     "cat": "奶茶", "addr": "北京市朝阳区工体北路13号", "lat": 39.9365, "lng": 116.4548, "phone": "010-85982222"},
    {"entity": "uniqlo_joycity", "name": "优衣库(朝阳大悦城店)", "en": "Uniqlo (Chaoyang Joy City)",
     "cat": "服装", "addr": "北京市朝阳区朝阳北路101号", "lat": 39.9238, "lng": 116.5058, "phone": "010-85556666"},
    {"entity": "quanjude_qianmen", "name": "全聚德(前门店)", "en": "Quanjude (Qianmen)",
     "cat": "烤鸭", "addr": "北京市东城区前门大街30号", "lat": 39.8988, "lng": 116.4006, "phone": "010-63081234"},
]

_CATS = ["咖啡", "火锅", "西餐", "酒店", "健身房", "便利店", "烧烤", "药房", "奶茶", "快餐", "服装", "烤鸭", "美发", "KTV"]


def _wrong_cat(cat, rng):
    return rng.choice([c for c in _CATS if c != cat])


def _colloquial_name(name, rng):
    base = name.split("(")[0]
    return base + rng.choice([" 那家店", "（老店）", " 这家", ""])


def _merchant(e, rng):
    # self-reported: accurate name/phone, noisy self-GPS coords, occasional
    # wrong category / missing phone.
    lat = e["lat"] + rng.uniform(-0.004, 0.004)
    lng = e["lng"] + rng.uniform(-0.004, 0.004)
    rec = {
        "id": f"{e['entity']}::merchant",
        "entity_id": e["entity"],
        "name": models.field(e["name"], "merchant", 0.95),
        "address": models.field(e["addr"], "merchant", 0.90),
        "category": models.field(e["cat"], "merchant", 0.85),
        "phone": models.field(e["phone"], "merchant", 0.90),
        "lat": models.field(round(lat, 6), "merchant", 0.80),
        "lng": models.field(round(lng, 6), "merchant", 0.80),
    }
    if rng.random() < 0.25:
        rec["phone"] = models.field("", "merchant", 0.0)
    if rng.random() < 0.15:
        rec["category"] = models.field(_wrong_cat(e["cat"], rng), "merchant", 0.30)
    return rec


def _map_vendor(e, rng):
    # map vendor: accurate coords, name may be transliterated (en), phone often
    # missing. Uses the CJK name by default; ``en`` entities create a real
    # cross-language duplicate that the feature layer scores lower.
    name = e.get("en") if (e.get("en") and rng.random() < 0.5) else e["name"]
    return {
        "id": f"{e['entity']}::map",
        "entity_id": e["entity"],
        "name": models.field(name, "map_vendor", 0.90),
        "address": models.field(e["addr"], "map_vendor", 0.95),
        "category": models.field(e["cat"], "map_vendor", 0.90),
        "phone": models.field("", "map_vendor", 0.0),
        "lat": models.field(round(e["lat"] + rng.uniform(-0.0002, 0.0002), 6), "map_vendor", 0.98),
        "lng": models.field(round(e["lng"] + rng.uniform(-0.0002, 0.0002), 6), "map_vendor", 0.98),
    }


def _ugc(e, rng):
    # user contribution: colloquial name, noisy coords, missing contact/category.
    return {
        "id": f"{e['entity']}::ugc",
        "entity_id": e["entity"],
        "name": models.field(_colloquial_name(e["name"], rng), "ugc", 0.50),
        "address": models.field("", "ugc", 0.0),
        "category": models.field("", "ugc", 0.0),
        "phone": models.field("", "ugc", 0.0),
        "lat": models.field(round(e["lat"] + rng.uniform(-0.01, 0.01), 6), "ugc", 0.40),
        "lng": models.field(round(e["lng"] + rng.uniform(-0.01, 0.01), 6), "ugc", 0.40),
    }


def generate_pool(seed=42, entities=None):
    """Generate a multi-source POI pool with hidden ground truth."""
    rng = random.Random(seed)
    ents = entities or ENTITIES
    pool = []
    for e in ents:
        pool.append(_merchant(e, rng))
        if rng.random() < 0.6:
            pool.append(_map_vendor(e, rng))
        if rng.random() < 0.3:
            pool.append(_ugc(e, rng))
    return pool
