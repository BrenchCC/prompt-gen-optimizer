# -*- coding: utf-8 -*-
import os

from dotenv import load_dotenv

load_dotenv()

def _parse_optional_float(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)

# ====== Worker model config ======
WORKER_API_KEY = os.environ.get("WORKER_API_KEY", "")
WORKER_BASE_URL = os.environ.get("WORKER_BASE_URL", "https://api.openai.com/v1")
WORKER_MODEL_NAME = os.environ.get("WORKER_MODEL_NAME", "gpt-4o-mini")
WORKER_THINKING = os.environ.get("WORKER_THINKING", "enabled")
WORKER_TEMPERATURE = _parse_optional_float(os.environ.get("WORKER_TEMPERATURE"))
WORKER_TOP_P = _parse_optional_float(os.environ.get("WORKER_TOP_P"))

# ====== Master model config ======
MASTER_API_KEY = os.environ.get("MASTER_API_KEY", "")
MASTER_BASE_URL = os.environ.get("MASTER_BASE_URL", "https://api.openai.com/v1")
MASTER_MODEL_NAME = os.environ.get("MASTER_MODEL_NAME", "gpt-4o")
MASTER_THINKING = os.environ.get("MASTER_THINKING", "enabled")
MASTER_TEMPERATURE = _parse_optional_float(os.environ.get("MASTER_TEMPERATURE"))
MASTER_TOP_P = _parse_optional_float(os.environ.get("MASTER_TOP_P"))

# ====== Default optimizer settings ======
DEFAULT_ROUNDS = int(os.environ.get("DEFAULT_ROUNDS", "5"))
DEFAULT_PATIENCE = int(os.environ.get("DEFAULT_PATIENCE", "2"))
DEFAULT_CONCURRENCY = int(os.environ.get("DEFAULT_CONCURRENCY", "8"))
DEFAULT_SEED = int(os.environ.get("DEFAULT_SEED", "42"))
DEFAULT_TRAIN_SIZE = float(os.environ.get("DEFAULT_TRAIN_SIZE", "0.7"))
DEFAULT_VAL_SIZE = float(os.environ.get("DEFAULT_VAL_SIZE", "0.3"))
DEFAULT_PRIMARY_METRIC = os.environ.get("DEFAULT_PRIMARY_METRIC", "f1")

# ====== Output defaults ======
RESULTS_JSON_NAME = "results.json"
BEST_PROMPT_NAME = "best_prompt.md"
BEST_SCORE_NAME = "best_score.json"
