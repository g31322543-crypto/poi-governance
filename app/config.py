"""Runtime configuration. Everything is overridable via environment variables."""
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path=_PROJECT_ROOT / ".env"):
    """Load KEY=VALUE pairs from ``.env`` into the environment (existing env wins)."""
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if k:
                os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

# torch.compile / Dynamo crashes on Windows GBK locales (it reads inductor
# .jinja templates with the locale codec instead of UTF-8) and is a pure speed
# optimisation we don't need on CPU. Disable it before torch/transformers get
# imported by the Laya backend. Set TORCHDYNAMO_DISABLE=0 to opt back in.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# --- Backend selection ----------------------------------------------------
# Which decision backend to use: "mock" | "jev" | "laya".
# Leave empty to auto-pick: "jev" when TYPESAFE_API_KEY is set, else "mock".
JEV_BACKEND = os.environ.get("JEV_BACKEND", "").strip().lower()

# --- Jev / TypeSafe System One -------------------------------------------
TYPESAFE_API_KEY = os.environ.get("TYPESAFE_API_KEY", "").strip()
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")
JEV_BASE_URL = os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai/v1/systemone")

# --- Laya (open-source System One, self-hosted, Jev-compatible) ----------
LAYA_MODEL_ID = os.environ.get("LAYA_MODEL_ID", "convaiinnovations/laya")
# Repo root is the English checkpoint; "multilingual" (mmBERT) is for Chinese.
LAYA_SUBFOLDER = os.environ.get("LAYA_SUBFOLDER", "multilingual")
LAYA_DEVICE = os.environ.get("LAYA_DEVICE", "").strip()  # '' = auto-detect


def resolve_backend():
    if JEV_BACKEND in ("mock", "jev", "laya"):
        return JEV_BACKEND
    return "jev" if TYPESAFE_API_KEY else "mock"


BACKEND = resolve_backend()
MOCK_MODE = BACKEND == "mock"

# --- Decision routing -----------------------------------------------------
# A decision with confidence below this threshold escalates to human review.
CONFIDENCE_AUTO = float(os.environ.get("CONFIDENCE_AUTO", "0.65"))

# --- Cost model (USD per decision) ----------------------------------------
# Only used to tell the "human-effort / cost saved" story on the dashboard.
COST_HUMAN = 0.50                      # one human review
COST_LLM = 0.004                       # one general LLM call (~2k tokens total)
COST_JEV = 0.042 / 1e6 * 800           # ~800 input tokens, output tokens free
