# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

from prompt_optimizer.task_config import TaskConfig
from prompt_optimizer.master_prompt import build_master_prompt, format_errors

SHENPING_YAML = os.path.join(ROOT, "tasks", "shenping", "v1", "config", "test_v1.yaml")


@pytest.fixture
def task_config() -> TaskConfig:
    return TaskConfig.from_yaml(SHENPING_YAML, project_root=ROOT)


class TestFormatErrors:
    def test_empty_errors(self, task_config: TaskConfig) -> None:
        text, total = format_errors([], task_config)
        assert text == "无错误样本。"
        assert total == 0

    def test_single_error(self, task_config: TaskConfig) -> None:
        errors = [{
            "fields": {"文章": "test article", "评论": "test comment"},
            "display_text": "test article",
            "expected": "是",
            "predicted": "不是",
        }]
        text, total = format_errors(errors, task_config)
        assert total == 1
        assert "Expected=是" in text
        assert "Predicted=不是" in text
        assert "[GT]: 是（是神评）" in text

    def test_respects_max_samples(self, task_config: TaskConfig) -> None:
        errors = [
            {"fields": {"文章": f"a{i}", "评论": f"b{i}"}, "display_text": f"a{i}",
             "expected": "是", "predicted": "不是"}
            for i in range(20)
        ]
        text, total = format_errors(errors, task_config, max_samples=5)
        assert total == 20
        assert text.count("[错误样本 #") == 5

    def test_preserves_full_fields(self, task_config: TaskConfig) -> None:
        long_text = "x" * 500
        errors = [{
            "fields": {"文章": long_text, "评论": "short"},
            "expected": "是", "predicted": "不是",
        }]
        text, _ = format_errors(errors, task_config)
        assert long_text in text

    def test_includes_worker_io(self, task_config: TaskConfig) -> None:
        errors = [{
            "fields": {"文章": "article", "评论": "comment"},
            "expected": "是", "predicted": "不是",
            "user_input": "原文内容：article\n评论内容：comment",
            "worker_output": '{"reasoning": "blah", "is_shenping": false}',
        }]
        text, _ = format_errors(errors, task_config)
        assert "[Worker 输入]" in text
        assert "[Worker 输出]" in text
        assert "article_content:" not in text
        assert "comment_content:" not in text
        assert "原文内容：article" in text
        assert '"is_shenping": false' in text

    def test_reconstructs_worker_input_from_fields(self, task_config: TaskConfig) -> None:
        errors = [{
            "fields": {"文章": "article", "评论": "comment"},
            "expected": "是", "predicted": "不是",
        }]
        text, _ = format_errors(errors, task_config)
        assert "[GT]: 是（是神评）" in text
        assert "[Worker 输入]" in text
        assert "原文内容：article" in text
        assert "评论内容：comment" in text
        assert "article_content:" not in text


class TestBuildMasterPrompt:
    def test_contains_key_sections(self, task_config: TaskConfig) -> None:
        prompt = build_master_prompt(
            current_prompt="test prompt",
            metrics_train={"f1": 0.8, "accuracy": 0.85},
            metrics_val={"f1": 0.75, "accuracy": 0.8},
            errors_str="[错误样本 #1] Expected=是 Predicted=不是",
            total_errors=5,
            history=[],
            task_config=task_config,
        )
        assert "Prompt 工程师" in prompt
        assert "test prompt" in prompt
        assert "增量改进" in prompt
        assert "错误样本剖析" in prompt
        assert "改进指南与约束" in prompt

    def test_includes_history(self, task_config: TaskConfig) -> None:
        history = [
            {"step": 0, "train_metrics": {"f1": 0.7}, "val_metrics": {"f1": 0.65}, "total_errors": 10},
            {"step": 1, "train_metrics": {"f1": 0.8}, "val_metrics": {"f1": 0.75}, "total_errors": 7},
        ]
        prompt = build_master_prompt(
            current_prompt="test",
            metrics_train={"f1": 0.85},
            metrics_val={"f1": 0.8},
            errors_str="none",
            total_errors=3,
            history=history,
            task_config=task_config,
        )
        assert "Step 0" in prompt
        assert "Step 1" in prompt

    def test_mentions_worker_input_in_instructions(self, task_config: TaskConfig) -> None:
        prompt = build_master_prompt(
            current_prompt="test",
            metrics_train={"f1": 0.85},
            metrics_val={"f1": 0.8},
            errors_str="none",
            total_errors=3,
            history=[],
            task_config=task_config,
        )
        assert "[GT]" in prompt
        assert "[Worker 输入]" in prompt
