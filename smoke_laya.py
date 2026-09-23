"""Smoke-test the Laya backend on a real Chinese dedup pair.

Exercises the full path config -> JevClient -> laya.load -> system_one ->
routing, so we verify the integration end to end. First run downloads the
model weights (~600MB) from Hugging Face, so it can take a few minutes.
"""
import os

os.environ["JEV_BACKEND"] = "laya"  # force this backend for the smoke test

from app import decisions  # noqa: E402  (must import after the env override)


def main():
    pair = {
        "id": "smoke_01",
        "poi_a": {"name": "星巴克(国贸店)", "address": "北京市朝阳区建国门外大街1号国贸商城B1层",
                  "category": "咖啡", "phone": "010-65051234", "lat": 39.9087, "lng": 116.4583},
        "poi_b": {"name": "星巴克国贸商城店", "address": "北京市朝阳区建国门外大街1号国贸商城1层",
                  "category": "咖啡", "phone": "010-65050001", "lat": 39.9085, "lng": 116.4586},
    }
    res = decisions.run_candidate(pair)
    print("backend: ", decisions.client.backend)
    print("model:   ", res["model"])
    print("score:   ", res["score"], "| confidence:", res["confidence"],
          "| route:", res["route"], "| label:", res["auto_label"])
    print("level:   ", res["level"])
    print("features:", {k: res["features"][k] for k in ("name_sim", "distance_km", "brand_match", "category_match")})


if __name__ == "__main__":
    main()
