# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from prompt_optimizer.suggestion_pool import SuggestionPool
from prompt_optimizer.task_config import TaskConfig

def build_config(tmp_path) -> TaskConfig:
    cfg = copy.deepcopy(TaskConfig())
    cfg.task_name = "test"
    cfg.task_version = "v1"
    cfg.suggestion_pool_dir = str(tmp_path)
    cfg.suggestion_similarity_threshold = 0.82
    return cfg


def test_ingest_new_suggestion_creates_pool_files(tmp_path) -> None:
    cfg = build_config(tmp_path)
    pool = SuggestionPool(cfg)
    result = pool.ingest([
        {
            "title": "补充边界规则",
            "category": "规则缺失",
            "root_cause": "缺少边界说明",
            "suggestion": "补充正反边界与冲突消解规则",
            "keywords": ["边界", "冲突"],
            "risk": "可能增加保守性",
            "confidence": 0.8,
            "source_samples": ["sample-1"],
            "source_payload": {
                "query": "q1",
                "reasoning_content": "r1",
                "raw_output": "{\"suggestion\":\"x\"}",
            },
        }
    ], step=1)

    assert result["added_ids"] == ["SUG-0001"]
    assert result["updated_ids"] == []
    assert os.path.exists(cfg.suggestion_pool_path)
    assert os.path.exists(cfg.suggestion_index_path)
    assert pool.active_suggestions()[0]["keywords"] == ["边界", "冲突"]
    assert pool.active_suggestions()[0]["generation_traces"][0]["query"] == "q1"


def test_similar_suggestions_are_merged_by_threshold(tmp_path) -> None:
    cfg = build_config(tmp_path)
    cfg.suggestion_similarity_threshold = 0.6
    pool = SuggestionPool(cfg)
    base = {
        "title": "补充边界规则",
        "category": "规则缺失",
        "root_cause": "缺少边界说明",
        "suggestion": "补充正反边界与冲突消解规则",
        "keywords": ["边界", "冲突"],
        "source_samples": ["sample-1"],
    }
    pool.ingest([base], step=1)
    result = pool.ingest([{
        "title": "补充边界规则说明",
        "category": "规则缺失",
        "root_cause": "边界不清晰",
        "suggestion": "补充正反边界与冲突消解规则，并明确适用边界",
        "keywords": ["边界", "冲突"],
        "source_samples": ["sample-2"],
        "source_payload": {
            "query": "q2",
            "reasoning_content": "r2",
            "raw_output": "{\"suggestion\":\"y\"}",
        },
    }], step=2)

    active = pool.active_suggestions()
    assert result["added_ids"] == []
    assert result["updated_ids"] == ["SUG-0001"]
    assert len(active) == 1
    assert active[0]["id"] == "SUG-0001"
    assert active[0]["version"] == 2
    assert active[0]["generation_traces"][-1]["query"] == "q2"


def test_suggestions_are_preserved_when_below_threshold(tmp_path) -> None:
    cfg = build_config(tmp_path)
    cfg.suggestion_similarity_threshold = 0.95
    pool = SuggestionPool(cfg)
    pool.ingest([{
        "title": "补充边界规则",
        "category": "规则缺失",
        "root_cause": "缺少边界说明",
        "suggestion": "补充正反边界与冲突消解规则",
        "keywords": ["边界", "冲突"],
        "source_samples": ["sample-1"],
    }], step=1)
    pool.ingest([{
        "title": "输出格式强约束",
        "category": "输出格式",
        "root_cause": "输出不稳定",
        "suggestion": "强制只输出标签且禁止解释，避免跑题",
        "keywords": ["格式", "约束"],
        "source_samples": ["sample-2"],
    }], step=2)
    assert len(pool.active_suggestions()) == 2


def test_index_groups_by_category_and_keyword(tmp_path) -> None:
    cfg = build_config(tmp_path)
    cfg.suggestion_similarity_threshold = 0.99
    pool = SuggestionPool(cfg)
    pool.ingest([{
        "title": "澄清神评边界",
        "category": "边界模糊",
        "root_cause": "定义模糊",
        "suggestion": "补充神评与非神评的边界说明",
        "keywords": ["边界", "定义"],
        "source_samples": ["sample-1"],
    }], step=1)
    pool.ingest([{
        "title": "澄清神评判定边界",
        "category": "边界模糊",
        "root_cause": "边界不清",
        "suggestion": "补充神评判定时的边界说明",
        "keywords": ["边界", "定义"],
        "source_samples": ["sample-2"],
    }], step=2)

    assert len(pool.active_suggestions()) == 2
    assert "边界模糊" in pool.index["by_category"]
    assert "边界" in pool.index["by_keyword"]


def test_apply_feedback_and_snapshot(tmp_path) -> None:
    cfg = build_config(tmp_path)
    pool = SuggestionPool(cfg)
    result = pool.ingest([{
        "title": "补充边界规则",
        "category": "规则缺失",
        "root_cause": "缺少边界说明",
        "suggestion": "补充正反边界与冲突消解规则",
        "keywords": ["边界", "冲突"],
        "source_samples": ["sample-1"],
    }], step=1)

    updated = pool.apply_feedback(result["selected_ids"], metric_delta=0.02)
    pool.snapshot(step=1, selected_ids=result["selected_ids"], metric_delta=0.02)
    active = pool.active_suggestions()

    assert updated == ["SUG-0001"]
    assert active[0]["positive_hits"] == 1
    assert os.path.exists(cfg.suggestion_snapshots_path)


def test_review_fn_can_keep_grey_zone_suggestions_independent(tmp_path) -> None:
    cfg = build_config(tmp_path)
    cfg.suggestion_similarity_threshold = 0.6
    pool = SuggestionPool(cfg, review_fn=lambda current, candidate: "independent")
    pool.ingest([{
        "title": "补充边界规则",
        "category": "规则缺失",
        "root_cause": "缺少边界说明",
        "suggestion": "补充正反边界与冲突消解规则",
        "keywords": ["边界", "冲突"],
        "source_samples": ["sample-1"],
    }], step=1)
    result = pool.ingest([{
        "title": "补充边界规则说明",
        "category": "规则缺失",
        "root_cause": "边界不清晰",
        "suggestion": "补充正反边界与冲突消解规则，并明确适用边界",
        "keywords": ["边界", "冲突"],
        "source_samples": ["sample-2"],
    }], step=2)

    assert result["added_ids"] == ["SUG-0002"]
    assert len(pool.active_suggestions()) == 2
