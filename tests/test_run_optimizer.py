# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import run_optimizer
from prompt_optimizer.task_config import TaskConfig


def test_cli_accepts_precision_pos(monkeypatch) -> None:
    cfg = TaskConfig()
    cfg.master_reasoning_option = "enabled"

    monkeypatch.setattr(
        run_optimizer.TaskConfig,
        "from_yaml",
        staticmethod(lambda path: cfg),
    )

    captured: dict[str, object] = {}

    class DummyOptimizer:
        def __init__(self, task_config: TaskConfig) -> None:
            captured["metric"] = task_config.primary_metric

        def optimize(self) -> dict[str, str]:
            return {"status": "ok"}

    monkeypatch.setattr(run_optimizer, "PromptOptimizer", DummyOptimizer)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_optimizer.py", "--config", "demo.yaml", "--metric", "precision_pos"],
    )

    run_optimizer.main()

    assert cfg.primary_metric == "precision_pos"
    assert captured["metric"] == "precision_pos"
