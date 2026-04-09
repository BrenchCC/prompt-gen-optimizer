# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Any, Callable

from loguru import logger

import config as cfg
from utils.llm_server import LLM_Client


def build_llm_client(
    *,
    mode: str,
    api_key: str,
    base_url: str,
    model_name: str,
) -> LLM_Client:
    return LLM_Client(
        mode=mode,
        api_key=api_key,
        base_url=base_url,
        default_model=model_name,
        print_stream=False,
    )


def resolve_model_config(
    *,
    role: str,
    task_config: Any,
) -> dict[str, Any]:
    role_name = role.strip().lower()
    if role_name not in {"worker", "master"}:
        raise ValueError(f"Unsupported role: {role}")

    prefix = role_name.upper()
    return {
        "mode": getattr(task_config, f"{role_name}_mode", "ark") or "ark",
        "api_key": getattr(task_config, f"{role_name}_api_key", "") or getattr(cfg, f"{prefix}_API_KEY"),
        "base_url": getattr(task_config, f"{role_name}_base_url", "") or getattr(cfg, f"{prefix}_BASE_URL"),
        "model_name": getattr(task_config, f"{role_name}_model_name", "") or getattr(cfg, f"{prefix}_MODEL_NAME"),
        "model_identifier": (
            getattr(task_config, f"{role_name}_model_identifier", "")
            or getattr(cfg, f"{prefix}_MODEL_IDENTIFIER")
        ),
        "temperature": getattr(task_config, f"{role_name}_temperature", None) or getattr(cfg, f"{prefix}_TEMPERATURE"),
        "top_p": getattr(task_config, f"{role_name}_top_p", None) or getattr(cfg, f"{prefix}_TOP_P"),
        "reasoning_option": getattr(task_config, f"{role_name}_reasoning_option", "") or getattr(cfg, f"{prefix}_THINKING"),
    }


def call_llm_with_retries(
    *,
    client: LLM_Client,
    label: str,
    max_retries: int,
    request_fn: Callable[[], tuple[str, str]],
    fallback: tuple[str, str] = ("", ""),
) -> tuple[str, str]:
    for attempt in range(max_retries):
        try:
            return request_fn()
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"{label} retry {attempt + 1}/{max_retries} after {wait}s: {exc}")
                time.sleep(wait)
                continue
            logger.error(f"{label} API Error (all retries failed): {exc}")
    return fallback
