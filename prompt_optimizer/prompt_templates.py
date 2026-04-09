# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompts_template"


@lru_cache(maxsize=None)
def load_prompt_template(name: str) -> str:
    path = _TEMPLATE_DIR / name
    return path.read_text(encoding="utf-8").strip()


def render_prompt_template(name: str, **kwargs: object) -> str:
    return load_prompt_template(name).format(**kwargs)
