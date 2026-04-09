# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any


def strip_code_fence(text: str | None) -> str:
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    lines = cleaned.splitlines()
    if not lines:
        return ""

    body = lines[1:]
    if body and body[-1].strip() == "```":
        body = body[:-1]
    return "\n".join(body).strip()


def parse_json_dict(text: str | None) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    if not cleaned:
        return {}

    try:
        obj = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}

    return obj if isinstance(obj, dict) else {}


def parse_string_dict(text: str | None, fallback_key: str = "raw_output") -> dict[str, str]:
    obj = parse_json_dict(text)
    if obj:
        return {str(key): str(value) for key, value in obj.items()}

    cleaned = strip_code_fence(text)
    if not cleaned:
        return {}
    return {fallback_key: cleaned}
