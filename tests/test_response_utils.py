# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prompt_optimizer.response_utils import parse_json_dict, parse_string_dict, strip_code_fence


def test_strip_code_fence_plain_text() -> None:
    assert strip_code_fence("hello") == "hello"


def test_strip_code_fence_markdown_block() -> None:
    raw = "```markdown\nline1\nline2\n```"
    assert strip_code_fence(raw) == "line1\nline2"


def test_parse_json_dict_from_code_block() -> None:
    raw = '```json\n{"label": "是"}\n```'
    assert parse_json_dict(raw) == {"label": "是"}


def test_parse_json_dict_invalid_json() -> None:
    assert parse_json_dict("not-json") == {}


def test_parse_string_dict_with_json_values() -> None:
    raw = '```json\n{"a": 1, "b": true}\n```'
    assert parse_string_dict(raw) == {"a": "1", "b": "True"}


def test_parse_string_dict_fallback_to_raw_output() -> None:
    assert parse_string_dict("plain output") == {"raw_output": "plain output"}
