# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prompt_optimizer.evaluator import Evaluator
from prompt_optimizer.optimizer import PromptOptimizer
from prompt_optimizer.task_config import TaskConfig


def build_config(tmp_path) -> TaskConfig:
    cfg = TaskConfig()
    cfg.task_name = "demo"
    cfg.task_version = "v1"
    cfg.task_type = "classify"
    cfg.data_file = str(tmp_path / "data.csv")
    cfg.data_format = "csv"
    cfg.text_columns = ["text"]
    cfg.label_column = "label"
    cfg.label_map = {"yes": "是", "no": "不是"}
    cfg.positive_label = "是"
    cfg.prompt_file = str(tmp_path / "prompt.md")
    cfg.output_dir = str(tmp_path / "output")
    cfg.suggestion_pool_dir = str(tmp_path / "suggestions")
    cfg.master_reasoning_option = "enabled"
    cfg.primary_metric = "precision_pos"
    cfg.iterations = 1
    cfg.patience = 1
    cfg.train_sample_size = 2
    cfg.val_sample_size = 2
    cfg.prompt_candidate_count = 2
    return cfg


def test_ensure_eval_samples_reuses_same_split(tmp_path, monkeypatch) -> None:
    cfg = build_config(tmp_path)
    monkeypatch.setattr(Evaluator, "_init_worker_client", lambda self: object())
    monkeypatch.setattr(PromptOptimizer, "_init_master_client", lambda self: object())
    optimizer = PromptOptimizer(cfg)

    data = [
        {"fields": {"text": f"text-{idx}"}, "label": "是" if idx % 2 else "不是"}
        for idx in range(8)
    ]
    train_1, valid_1 = optimizer._ensure_eval_samples(data)
    train_2, valid_2 = optimizer._ensure_eval_samples(list(reversed(data)))

    assert train_1 == train_2
    assert valid_1 == valid_2


def test_optimize_only_trains_winning_candidate(tmp_path, monkeypatch) -> None:
    cfg = build_config(tmp_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(cfg.prompt_file, "w", encoding="utf-8") as f:
        f.write("baseline prompt")

    monkeypatch.setattr(Evaluator, "_init_worker_client", lambda self: object())
    monkeypatch.setattr(PromptOptimizer, "_init_master_client", lambda self: object())
    optimizer = PromptOptimizer(cfg)

    data = [
        {"fields": {"text": "a"}, "label": "是"},
        {"fields": {"text": "b"}, "label": "不是"},
        {"fields": {"text": "c"}, "label": "是"},
        {"fields": {"text": "d"}, "label": "不是"},
    ]
    cfg.load_data = lambda: data

    optimizer._get_or_generate_task_description = lambda prompt: "demo task"
    optimizer._save_config_snapshot = lambda: None
    optimizer._init_worker_prompt_log = lambda: None
    optimizer._append_results = lambda path, record: None
    optimizer._append_log = lambda path, step, prompt, metrics, analysis: None
    optimizer._save_best = lambda prompt, score, metrics: None
    optimizer._save_comparison = lambda original, best: None
    optimizer._run_final_evaluation = lambda data, original, best: {"status": "ok"}
    optimizer.evaluator.get_cache_stats = lambda: {"hits": 0, "misses": 0, "size": 0}

    eval_calls: list[tuple[str, bool, str]] = []
    train_only_calls: list[str] = []

    def fake_eval_prompt(prompt: str, system_prompt: str, step_label: str, run_train: bool = True):
        eval_calls.append((prompt, run_train, step_label))
        if step_label == "Step0":
            return (
                {"precision_pos": 0.50},
                {"precision_pos": 0.50},
                [{"id": "train-base"}],
                [{"id": "valid-base"}],
            )
        if prompt == "candidate-a":
            return {}, {"precision_pos": 0.58}, [], [{"id": "valid-a"}]
        return {}, {"precision_pos": 0.62}, [], [{"id": "valid-b"}]

    def fake_evaluate_train_only(prompt: str, system_prompt: str, step_label: str):
        train_only_calls.append(prompt)
        return {"precision_pos": 0.61}, [{"id": "train-winner"}]

    optimizer._eval_prompt = fake_eval_prompt
    optimizer._evaluate_train_only = fake_evaluate_train_only
    optimizer._improve_prompt = lambda *args, **kwargs: (
        [
            {"strategy_name": "候选A", "patch_focus": "边界澄清", "candidate_prompt": "candidate-a"},
            {"strategy_name": "候选B", "patch_focus": "删减冗余", "candidate_prompt": "candidate-b"},
        ],
        ["SUG-0001"],
        {"added_ids": [], "updated_ids": [], "experience_summary": "ok"},
    )

    class DummySuggestionPool:
        def apply_feedback(self, suggestion_ids, metric_delta):
            return suggestion_ids

        def snapshot(self, step, selected_ids, metric_delta=None):
            return None

    optimizer.suggestion_pool = DummySuggestionPool()

    result = optimizer.optimize()

    assert result["best_score"] == 0.62
    assert train_only_calls == ["candidate-b"]
    candidate_eval_calls = [call for call in eval_calls if call[2].startswith("Step1-Cand")]
    assert candidate_eval_calls == [
        ("candidate-a", False, "Step1-Cand1"),
        ("candidate-b", False, "Step1-Cand2"),
    ]
