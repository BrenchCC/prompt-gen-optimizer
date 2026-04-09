# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prompt_optimizer.prompt_templates import load_prompt_template, render_prompt_template


def test_load_prompt_template() -> None:
    content = load_prompt_template("master_system_prompt.md")
    assert "长期系统级优化建议" in content


def test_render_prompt_template() -> None:
    rendered = render_prompt_template(
        "suggestion_relation_prompt.md",
        current_suggestion="A",
        candidate_suggestion="B",
    )
    assert "建议A：A" in rendered
    assert "建议B：B" in rendered
