"""Synthetic POI data for the demo.

We generate two workloads that mirror the TikTok Local Services POI-governance
pain points:

1. ``pairs``       -- candidate POI dedup pairs (same place? different place?).
2. ``submissions`` -- merchant-submitted POI listings that need review triage.

Every sample carries a hidden ``ground_truth`` label (used only for evaluation)
and a ``hint`` score in [0, 1] (used only by the mock Jev client to fake a
calibrated answer). In real-Jev mode neither is ever sent to the model -- only
the readable ``state`` text is.
"""
import random

# --- base POI pool (Beijing coords) --------------------------------------
BASE = [
    {"name": "星巴克(国贸店)", "en": "Starbucks (China World Mall)", "cat": "咖啡", "addr": "北京市朝阳区建国门外大街1号国贸商城B1层", "lat": 39.9087, "lng": 116.4583, "phone": "010-65051234"},
    {"name": "海底捞火锅(三里屯店)", "en": "Haidilao Hot Pot (Sanlitun)", "cat": "火锅", "addr": "北京市朝阳区工体北路8号三里屯SOHO 1层", "lat": 39.9360, "lng": 116.4554, "phone": "010-85987654"},
    {"name": "必胜客(西单店)", "en": "Pizza Hut (Xidan)", "cat": "西餐", "addr": "北京市西城区西单北大街120号", "lat": 39.9093, "lng": 116.3733, "phone": "010-66012345"},
    {"name": "7天连锁酒店(王府井店)", "en": "7 Days Inn (Wangfujing)", "cat": "酒店", "addr": "北京市东城区东安门大街53号", "lat": 39.9160, "lng": 116.4109, "phone": "010-65123456"},
    {"name": "乐刻健身(望京SOHO店)", "en": "Lefit Gym (Wangjing SOHO)", "cat": "健身房", "addr": "北京市朝阳区望京街10号望京SOHO T1", "lat": 39.9966, "lng": 116.4817, "phone": "010-64777777"},
    {"name": "全家便利店(中关村店)", "en": "FamilyMart (Zhongguancun)", "cat": "便利店", "addr": "北京市海淀区海淀中街6号", "lat": 39.9830, "lng": 116.3160, "phone": "010-82668888"},
    {"name": "木屋烧烤(五道口店)", "en": "Wooden House BBQ (Wudaokou)", "cat": "烧烤", "addr": "北京市海淀区成府路35号", "lat": 39.9910, "lng": 116.3350, "phone": "010-62345678"},
    {"name": "同仁堂药店(前门店)", "en": "Tongrentang Pharmacy (Qianmen)", "cat": "药房", "addr": "北京市东城区前门大街42号", "lat": 39.8990, "lng": 116.4010, "phone": "010-67081111"},
    {"name": "瑞幸咖啡(金融街店)", "en": "Luckin Coffee (Financial Street)", "cat": "咖啡", "addr": "北京市西城区金融大街35号", "lat": 39.9150, "lng": 116.3560, "phone": "400-100-0000"},
    {"name": "喜茶(朝阳大悦城店)", "en": "Heytea (Chaoyang Joy City)", "cat": "奶茶", "addr": "北京市朝阳区朝阳北路101号朝阳大悦城5层", "lat": 39.9240, "lng": 116.5060, "phone": "010-85555555"},
]


def _poi(base, **overrides):
    d = {"name": base["name"], "address": base["addr"], "category": base["cat"],
         "lat": base["lat"], "lng": base["lng"], "phone": base["phone"]}
    d.update(overrides)
    return d


def _haversine_km(lat1, lng1, lat2, lng2):
    import math
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _make_pair(pid, a, b, label, hint):
    return {
        "id": pid,
        "poi_a": a,
        "poi_b": b,
        "dist_km": round(_haversine_km(a["lat"], a["lng"], b["lat"], b["lng"]), 3),
        "ground_truth": label,       # True == same physical place
        "hint": hint,
        "decision": None,
        "human": None,               # "merge" | "keep" once a human reviews
    }


def _ambiguous_pairs():
    """The genuinely hard cases: same brand / tiny name drift, ambiguous truth."""
    rows = []
    def add(pid, a, b, label, hint):
        rows.append(_make_pair(pid, a, b, label, hint))

    h = BASE[1]; add("dup_amb_01", _poi(h), _poi(h, name="海底捞火锅(工体店)", addr="北京市朝阳区工体北路4号院", lat=39.9310, lng=116.4560, phone="010-85981234"), False, 0.48)
    h = BASE[0]; add("dup_amb_02", _poi(h), _poi(h, name="星巴克(国贸商城店)", addr="北京市朝阳区建国门外大街1号国贸商城1层", lat=39.9085, lng=116.4586, phone="010-65050001"), True, 0.52)
    h = BASE[8]; add("dup_amb_03", _poi(h), _poi(h, name="Luckin Coffee Financial Street", addr="Financial Street 35, Xicheng, Beijing", lat=39.9152, lng=116.3562), True, 0.53)
    h = BASE[6]; add("dup_amb_04", _poi(h), _poi(h, name="木屋烧烤(成府路店)", addr="北京市海淀区成府路45号", lat=39.9890, lng=116.3370, phone="010-62349999"), False, 0.47)
    h = BASE[5]; add("dup_amb_05", _poi(h), _poi(h, name="全家便利店(海淀中街店)", addr="北京市海淀区海淀中街6号院", lat=39.9832, lng=116.3161), True, 0.51)
    h = BASE[4]; add("dup_amb_06", _poi(h), _poi(h, name="乐刻健身(望京SOHO T1店)", addr="北京市朝阳区望京街10号望京SOHO", lat=39.9968, lng=116.4820), True, 0.52)
    h = BASE[7]; add("dup_amb_07", _poi(h), _poi(h, name="同仁堂药店(大栅栏店)", addr="北京市西城区大栅栏街34号", lat=39.8980, lng=116.3960, phone="010-63081234"), False, 0.46)
    h = BASE[3]; add("dup_amb_08", _poi(h), _poi(h, name="7天连锁酒店(东单店)", addr="北京市东城区东单北大街66号", lat=39.9140, lng=116.4210, phone="010-65121111"), False, 0.45)
    return rows


def _submissions(rng):
    """Merchant-submitted POI listings: approve / reject / ambiguous review."""
    legit_desc = [
        "本店提供现磨咖啡与简餐，营业时间 7:00-22:00，支持堂食与外带。",
        "正宗川味火锅，食材每日新鲜到店，可预订包间，营业至凌晨2点。",
        "连锁健身房，24小时营业，提供私教与团课，新客可免费体验一次。",
        "品牌奶茶，主打鲜果茶与轻乳茶，支持小程序下单免排队。",
    ]
    spam_desc = [
        "加微信 xx12345 领大额优惠，全网最低价，扫码进群，刷单返现，QQ 987654。",
        "无需到店，线上兼职日结，联系 VX: zhaopin888，非诚勿扰。",
        "新店冲量，五星好评返现，添加客服领取，不买也送。",
    ]
    invalid_desc = [
        "这家店很好吃，环境不错，值得推荐。",  # no usable listing info
        "待定",  # meaningless
    ]

    subs = []
    seq = {"i": 0}

    def make(merchant, truth, hint):
        seq["i"] += 1
        return {
            "id": f"sub_{seq['i']:03d}",
            "merchant": merchant,
            "ground_truth": truth,   # "approve" | "reject" | "review"
            "hint": hint,
            "decision": None,
            "human": None,           # "approve" | "reject" once a human reviews
        }

    # legit submissions
    for i, b in enumerate(BASE):
        subs.append(make(
            {"name": b["name"], "address": b["addr"], "category": b["cat"],
             "description": legit_desc[i % len(legit_desc)], "phone": b["phone"]},
            "approve", 0.90 + rng.random() * 0.08,
        ))

    # spam / invalid submissions (reject)
    for i in range(6):
        base = BASE[rng.randrange(len(BASE))]
        subs.append(make(
            {"name": base["name"], "address": "北京市", "category": base["cat"],
             "description": spam_desc[i % len(spam_desc)], "phone": ""},
            "reject", 0.02 + rng.random() * 0.08,
        ))
    for i in range(4):
        base = BASE[rng.randrange(len(BASE))]
        subs.append(make(
            {"name": base["name"], "address": base["addr"],
             "category": "火锅" if base["cat"] != "火锅" else "美发",
             "description": invalid_desc[i % len(invalid_desc)], "phone": ""},
            "reject", 0.05 + rng.random() * 0.10,
        ))

    # ambiguous submissions (needs human review)
    amb = [
        {"name": "无名烧烤(新开业)", "address": "北京市朝阳区百子湾路12号", "category": "烧烤",
         "description": "新店开业，试营业期间全场8折，欢迎品尝。", "phone": ""},
        {"name": "老王美发工作室", "address": "北京市海淀区学院路30号院内", "category": "美发",
         "description": "资深发型师，剪发烫染均可，需提前预约。", "phone": "13800000000"},
        {"name": "小众咖啡馆", "address": "北京市东城区鼓楼东大街88号", "category": "咖啡",
         "description": "手冲咖啡，安静，适合办公。", "phone": ""},
    ]
    for m in amb:
        subs.append(make(m, "review", 0.5))

    rng.shuffle(subs)
    return subs


def generate_dataset(seed=42):
    rng = random.Random(seed)
    pairs = []
    n = len(BASE)

    # clear duplicates (same place, name transliterated / minor wording drift)
    for i, b in enumerate(BASE):
        jx = 0.0001 * rng.uniform(-1, 1)
        a = _poi(b)
        bb = _poi(b, name=b["en"], lat=b["lat"] + jx, lng=b["lng"] + jx * 0.7)
        pairs.append(_make_pair(f"dup_clr_{i:02d}", a, bb, True, 0.90 + rng.random() * 0.08))

    # clear non-duplicates (two unrelated POIs)
    for i in range(n):
        j = (i + n // 2) % n
        pairs.append(_make_pair(f"dup_neg_{i:02d}", _poi(BASE[i]), _poi(BASE[j]), False, 0.02 + rng.random() * 0.08))

    # ambiguous
    pairs += _ambiguous_pairs()
    rng.shuffle(pairs)

    return {"pairs": pairs, "submissions": _submissions(rng)}
