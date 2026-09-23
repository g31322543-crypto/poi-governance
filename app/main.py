"""FastAPI app for the POI governance full chain.

Serves the HTML/JS frontend (``app/static``) and a small JSON API:
  GET  /api/status     app + model + dedup-backend config
  GET  /api/examples   pre-filled example submissions for the UI
  POST /api/submit     run the full chain on one merchant submission

The chain (Laya moderation -> blocking + LLM dedup -> route) lives in
``app/chain.py`` and is loaded lazily. The Laya checkpoint is heavy, so the
server warms it in a background thread at startup; the first request before
that finishes waits, every later one is instant.
"""
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import chain, store

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app):
    def _warm():
        try:
            ENGINE.load_db()
            ENGINE.load_agent()
        except Exception as e:  # model warm is best-effort; requests trigger retry
            print(f"[warmup] engine warm failed: {e}")

    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="POI Governance Chain (Laya + LLM)", lifespan=lifespan)

ENGINE = chain.Chain()


class Submission(BaseModel):
    name: str
    address: str = ""
    description: str = ""
    category: str = ""
    phone: str = ""
    brand: str = ""
    lat: float | None = None
    lng: float | None = None


def _to_sub(d: Submission):
    return {"id": "live", "entity_id": "live", "name": d.name, "address": d.address,
            "description": d.description, "category": d.category, "phone": d.phone,
            "brand": d.brand, "lat": d.lat, "lng": d.lng}


@app.get("/api/status")
def status():
    s = ENGINE.status()
    try:
        s["store"] = store.stats()
    except Exception:
        s["store"] = None
    return s


@app.get("/api/audit")
def audit(limit: int = 50):
    return store.recent(limit)


@app.get("/api/examples")
def examples():
    return ENGINE.examples()


@app.post("/api/submit")
def submit(d: Submission):
    return ENGINE.run_submission(_to_sub(d))


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
