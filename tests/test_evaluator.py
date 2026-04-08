# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest

from prompt_optimizer.task_config import TaskConfig
from prompt_optimizer.evaluator import Evaluator

SHENPING_YAML = os.path.join(ROOT, "tasks", "shenping", "v1", "config", "test_v1.yaml")


@pytest.fixture
def evaluator() -> Evaluator:
    cfg = TaskConfig.from_yaml(SHENPING_YAML, project_root=ROOT)
    return Evaluator(cfg)


class TestParsePrediction:
    def test_output_field_true(self, evaluator: Evaluator) -> None:
        output = '{"reasoning": "test", "is_shenping": true, "category": "抖机灵调侃型"}'
        assert evaluator._parse_prediction(output) == "是"

    def test_output_field_false(self, evaluator: Evaluator) -> None:
        output = '{"reasoning": "test", "is_shenping": false, "category": null}'
        assert evaluator._parse_prediction(output) == "不是"

    def test_output_field_null(self, evaluator: Evaluator) -> None:
        output = '{"reasoning": "test", "is_shenping": null, "category": null}'
        assert evaluator._parse_prediction(output) == "不确定"

    def test_json_in_code_block(self, evaluator: Evaluator) -> None:
        output = '```json\n{"reasoning": "test", "is_shenping": true, "category": "精准评价型"}\n```'
        assert evaluator._parse_prediction(output) == "是"

    def test_empty_output(self, evaluator: Evaluator) -> None:
        assert evaluator._parse_prediction("") == "PARSE_FAIL"
        assert evaluator._parse_prediction(None) == "PARSE_FAIL"

    def test_plain_label_fallback(self, evaluator: Evaluator) -> None:
        assert evaluator._parse_prediction("是") == "是"
        assert evaluator._parse_prediction("不是") == "不是"
        assert evaluator._parse_prediction("不确定") == "不确定"

    def test_custom_parser(self) -> None:
        cfg = TaskConfig.from_yaml(SHENPING_YAML, project_root=ROOT)
        cfg.custom_parser = "tasks.shenping.v1.parser.parse_output"
        cfg.output_field = ""
        e = Evaluator(cfg)
        assert e._parse_prediction('{"is_shenping": true}') == "是"
        assert e._parse_prediction('{"is_shenping": false}') == "不是"

    def test_generic_json_fallback(self) -> None:
        cfg = TaskConfig.from_yaml(SHENPING_YAML, project_root=ROOT)
        cfg.output_field = ""
        cfg.output_map = {}
        e = Evaluator(cfg)
        assert e._parse_prediction('{"label": "是"}') == "是"
        assert e._parse_prediction('{"result": "不是"}') == "不是"


class TestBuildQuery:
    def test_shenping_query(self, evaluator: Evaluator) -> None:
        item = {
            "fields": {"文章": "测试文章", "评论": "测试评论", "text": "测试文章"},
            "label": "是",
        }
        query = evaluator._build_query("dummy_prompt", item)
        assert "原文内容" in query
        assert "评论内容" in query
        assert "测试文章" in query
        assert "测试评论" in query


class TestEvaluate:
    def test_all_correct(self, evaluator: Evaluator) -> None:
        dataset = [
            {"fields": {"文章": "a", "评论": "b"}, "label": "是"},
            {"fields": {"文章": "c", "评论": "d"}, "label": "不是"},
        ]
        preds = ["是", "不是"]
        metrics, errors = evaluator.evaluate(preds, dataset)
        assert metrics["accuracy"] == 1.0
        assert len(errors) == 0

    def test_all_wrong(self, evaluator: Evaluator) -> None:
        dataset = [
            {"fields": {"文章": "a", "评论": "b"}, "label": "是"},
            {"fields": {"文章": "c", "评论": "d"}, "label": "不是"},
        ]
        preds = ["不是", "是"]
        metrics, errors = evaluator.evaluate(preds, dataset)
        assert metrics["accuracy"] == 0.0
        assert len(errors) == 2

    def test_empty_dataset(self, evaluator: Evaluator) -> None:
        metrics, errors = evaluator.evaluate([], [])
        assert metrics == {}
        assert errors == []
