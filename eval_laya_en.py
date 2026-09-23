"""English-data probe: does Laya improve when the POI records are English?

Runs the *already-downloaded* multilingual checkpoint (no new download) on a
small English POI dedup dataset mirroring the Chinese one, to isolate the
"data language" variable. Compare the accuracy against eval_laya.py (Chinese
data = 50%).
"""
import os

os.environ.setdefault("JEV_BACKEND", "laya")  # allow JEV_BACKEND=mock override

from app import decisions, features  # noqa: E402


def poi(name, address, category, lat, lng, phone=""):
    return {"name": name, "address": address, "category": category,
            "lat": lat, "lng": lng, "phone": phone}


def pair(pid, a, b, gt):
    dist = features.haversine_km(a["lat"], a["lng"], b["lat"], b["lng"])
    return {"id": pid, "poi_a": a, "poi_b": b, "dist_km": round(dist, 3),
            "ground_truth": gt, "hint": 0.5}


ENT = {
    "starbucks": poi("Starbucks (China World Mall)", "B1, China World Mall, 1 Jianguomenwai Ave, Chaoyang, Beijing", "Coffee", 39.9087, 116.4583, "010-65051234"),
    "haidilao": poi("Haidilao Hot Pot (Sanlitun)", "1F, Sanlitun SOHO, 8 Gongti North Rd, Chaoyang, Beijing", "Hot Pot", 39.9360, 116.4554, "010-85987654"),
    "pizzahut": poi("Pizza Hut (Xidan)", "120 Xidan North St, Xicheng, Beijing", "Western", 39.9093, 116.3733, "010-66012345"),
    "7days": poi("7 Days Inn (Wangfujing)", "53 Dong'anmen St, Dongcheng, Beijing", "Hotel", 39.9160, 116.4109, "010-65123456"),
    "lefit": poi("Lefit Gym (Wangjing SOHO)", "T1, Wangjing SOHO, 10 Wangjing St, Chaoyang, Beijing", "Gym", 39.9966, 116.4817, "010-64777777"),
    "familymart": poi("FamilyMart (Zhongguancun)", "6 Haidian Middle St, Haidian, Beijing", "Convenience", 39.9830, 116.3160, "010-82668888"),
    "mumu": poi("Wooden House BBQ (Wudaokou)", "35 Chengfu Rd, Haidian, Beijing", "BBQ", 39.9910, 116.3350, "010-62345678"),
    "tongrentang": poi("Tongrentang Pharmacy (Qianmen)", "42 Qianmen St, Dongcheng, Beijing", "Pharmacy", 39.8990, 116.4010, "010-67081111"),
    "luckin": poi("Luckin Coffee (Financial Street)", "35 Financial St, Xicheng, Beijing", "Coffee", 39.9150, 116.3560, "400-100-0000"),
    "heytea": poi("Heytea (Chaoyang Joy City)", "5F, Joy City, 101 Chaoyang North Rd, Beijing", "Milk Tea", 39.9240, 116.5060, "010-85555555"),
    "mcdonalds": poi("McDonald's (Sanlitun)", "19 Sanlitun Rd, Chaoyang, Beijing", "Fast Food", 39.9355, 116.4560, "010-85981111"),
    "chabaidao": poi("Chabaidao (Sanlitun)", "13 Gongti North Rd, Chaoyang, Beijing", "Milk Tea", 39.9365, 116.4548, "010-85982222"),
    "uniqlo": poi("Uniqlo (Chaoyang Joy City)", "1F, Joy City, 101 Chaoyang North Rd, Beijing", "Clothing", 39.9238, 116.5058, "010-85556666"),
    "quanjude": poi("Quanjude (Qianmen)", "30 Qianmen St, Dongcheng, Beijing", "Roast Duck", 39.8988, 116.4006, "010-63081234"),
}


def build_pairs():
    keys = list(ENT)
    rows = []

    # clear duplicates: same place, minor name/address drift, tiny coord offset
    for i, k in enumerate(keys[:10]):
        a = ENT[k]
        b = poi(a["name"], a["address"], a["category"],
                a["lat"] + 0.0002, a["lng"] + 0.0001, a["phone"])
        rows.append(pair(f"dup_clr_{i:02d}", a, b, True))

    # clear non-duplicates: unrelated entities
    for i in range(10):
        a, b = ENT[keys[i]], ENT[keys[(i + 5) % 10]]
        rows.append(pair(f"dup_neg_{i:02d}", a, b, False))

    # ambiguous / hard cases (same brand different branch, cross-brand same area)
    amb = [
        ("dup_amb_01", ENT["starbucks"], poi("Starbucks (Guomao Store)", "China World Mall, Chaoyang, Beijing", "Coffee", 39.9090, 116.4590), True),
        ("dup_amb_02", ENT["starbucks"], poi("Starbucks (Sanlitun)", "Sanlitun, Chaoyang, Beijing", "Coffee", 39.9355, 116.4560), False),
        ("dup_amb_03", ENT["mcdonalds"], ENT["haidilao"], False),
        ("dup_amb_04", ENT["luckin"], poi("Luckin Coffee Financial Street Store", "Financial Street, Xicheng, Beijing", "Coffee", 39.9152, 116.3562), True),
        ("dup_amb_05", ENT["heytea"], ENT["chabaidao"], False),
        ("dup_amb_06", ENT["7days"], poi("7 Days Inn (Dongdan)", "66 Dongdan North St, Dongcheng, Beijing", "Hotel", 39.9140, 116.4210), False),
    ]
    for pid, a, b, gt in amb:
        rows.append(pair(pid, a, b, gt))
    return rows


def main():
    pairs = build_pairs()
    rows = []
    for p in pairs:
        r = decisions.run_dedup(p)
        correct = (r["auto_label"] == "same") == p["ground_truth"]
        rows.append((p, r, correct))

    auto = [x for x in rows if x[1]["route"].startswith("auto")]
    no_gate = sum(1 for x in rows if x[2])
    print("=" * 64)
    print("LAYA multilingual checkpoint on ENGLISH POI data")
    print("=" * 64)
    print(f"pairs            : {len(rows)}")
    print(f"auto-decided     : {len(auto)}")
    print(f"no-gate accuracy : {no_gate}/{len(rows)} = {no_gate/len(rows):.3f}")
    print(f"mean confidence  : {sum(x[1]['confidence'] for x in rows)/len(rows):.3f}")
    print()
    for p, r, c in rows:
        mark = "OK " if c else "ERR"
        print(f"  [{mark}] {p['id']:11s} gt={'same' if p['ground_truth'] else 'diff'}  "
              f"label={r['auto_label']:9s} score={r['score']:.2f} conf={r['confidence']:.3f}")


if __name__ == "__main__":
    main()
