# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import tempfile

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.insert(0, ROOT)

from prompt_optimizer.task_config import TaskConfig


SHENPING_YAML = os.path.join(ROOT, "tasks", "shenping", "v1", "config", "test_v1.yaml")


@pytest.fixture
def shenping_config() -> TaskConfig:
    return TaskConfig.from_yaml(SHENPING_YAML, project_root=ROOT)


def test_from_yaml_loads_task_section(shenping_config: TaskConfig) -> None:
    assert shenping_config.task_name == "shenping"
    assert shenping_config.task_version == "v1"
    assert shenping_config.task_type == "classify"


def test_from_yaml_loads_data_section(shenping_config: TaskConfig) -> None:
    assert shenping_config.data_format == "xlsx"
    assert "文章" in shenping_config.text_columns
    assert "评论" in shenping_config.text_columns
    assert shenping_config.label_column == "人工打标是否神评"
    assert "是" in shenping_config.label_map
    assert "不是" in shenping_config.label_map


def test_from_yaml_loads_optimizer_section(shenping_config: TaskConfig) -> None:
    assert isinstance(shenping_config.iterations, int) and shenping_config.iterations > 0
    assert isinstance(shenping_config.patience, int) and shenping_config.patience > 0
    assert shenping_config.primary_metric in ("accuracy", "f1", "precision", "recall")
    assert shenping_config.concurrency == 8
    assert shenping_config.seed == 42


def test_from_yaml_resolves_paths(shenping_config: TaskConfig) -> None:
    assert os.path.isabs(shenping_config.data_file)
    assert os.path.isabs(shenping_config.prompt_file)
    assert os.path.isabs(shenping_config.output_dir)


def test_all_labels(shenping_config: TaskConfig) -> None:
    labels = shenping_config.all_labels
    assert isinstance(labels, list)
    assert len(labels) == 3
    assert "是" in labels
    assert "不是" in labels
    assert "不确定" in labels


def test_output_paths(shenping_config: TaskConfig) -> None:
    assert shenping_config.output_results_path.endswith("results.json")
    assert shenping_config.output_best_prompt_path.endswith("best_prompt.md")
    assert shenping_config.output_best_score_path.endswith("best_score.json")
    assert shenping_config.output_log_path.endswith("optimization_log.md")
    assert shenping_config.output_worker_prompt_log_path.endswith("worker_prompt_log.md")


def test_load_data(shenping_config: TaskConfig) -> None:
    data = shenping_config.load_data()
    assert len(data) > 0
    item = data[0]
    assert "fields" in item
    assert "label" in item
    assert "文章" in item["fields"] or "评论" in item["fields"]


def test_read_prompt(shenping_config: TaskConfig) -> None:
    content = shenping_config.read_prompt()
    assert len(content) > 0
    assert "神评" in content


def test_write_prompt(shenping_config: TaskConfig) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        tmp_path = f.name

    try:
        shenping_config.prompt_file = tmp_path
        shenping_config.write_prompt("test prompt content")
        with open(tmp_path, "r") as f:
            assert f.read() == "test prompt content"
    finally:
        os.unlink(tmp_path)


def test_user_prompt_template_loaded(shenping_config: TaskConfig) -> None:
    assert shenping_config.user_prompt_template != ""
    assert "{{article_content}}" in shenping_config.user_prompt_template
    assert "{{comment_content}}" in shenping_config.user_prompt_template


def test_build_user_prompt_with_template(shenping_config: TaskConfig) -> None:
    fields = {"文章": "测试文章内容", "评论": "测试评论内容", "text": "测试文章内容"}
    result = shenping_config.build_user_prompt(fields)
    assert "原文内容：测试文章内容" in result
    assert "评论内容：测试评论内容" in result


def test_build_user_prompt_without_template() -> None:
    cfg = TaskConfig()
    cfg.task_type = "classify"
    cfg.text_columns = ["text"]
    cfg.prompt_variables = {}
    cfg.user_prompt_template = ""
    fields = {"text": "hello world"}
    assert cfg.build_user_prompt(fields) == "hello world"


def test_build_user_prompt_judge_fallback() -> None:
    cfg = TaskConfig()
    cfg.task_type = "judge"
    cfg.text_columns = ["question"]
    cfg.prompt_variables = {"question": "q_col", "answer": "a_col"}
    cfg.user_prompt_template = ""
    fields = {"q_col": "what?", "a_col": "yes"}
    result = cfg.build_user_prompt(fields)
    assert "question: what?" in result
    assert "answer: yes" in result


def test_validate_bad_task_type() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({
            "task": {"type": "invalid"},
            "data": {
                "file": "fake.csv", "text_columns": ["text"],
                "label_column": "label", "label_map": {"a": "a"},
            },
        }, f)
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="不支持的任务类型"):
            TaskConfig.from_yaml(tmp_path, project_root=ROOT)
    finally:
        os.unlink(tmp_path)


def test_validate_missing_label_map() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({
            "task": {"type": "classify"},
            "data": {
                "file": "fake.csv", "text_columns": ["text"],
                "label_column": "label",
            },
        }, f)
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="未指定 label_map"):
            TaskConfig.from_yaml(tmp_path, project_root=ROOT)
    finally:
        os.unlink(tmp_path)
