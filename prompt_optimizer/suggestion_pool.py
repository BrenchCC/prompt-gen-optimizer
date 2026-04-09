# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import difflib
import json
import os
from datetime import datetime
from typing import Any, Callable

from prompt_optimizer.task_config import TaskConfig


def _utcnow() -> str:
    return datetime.now().isoformat()

def unique_keywords(keywords: list[Any] | None) -> list[str]:
    values: list[str] = []
    for item in keywords or []:
        value = str(item).strip().lower()
        if value and value not in values:
            values.append(value)
    return values

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _jaccard(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 0.0
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

class SuggestionPool:
    def __init__(
        self,
        task_config: TaskConfig,
        review_fn: Callable[[dict[str, Any], dict[str, Any]], str] | None = None,
    ) -> None:
        self.cfg = task_config
        self.review_fn = review_fn
        self.pool = self._load_json(
            self.cfg.suggestion_pool_path,
            {
                "meta": {
                    "task_name": self.cfg.task_name,
                    "task_version": self.cfg.task_version,
                    "next_id": 1,
                    "updated_at": "",
                },
                "active_suggestions": [],
                "suggestion_history": [],
            },
        )
        self.snapshots = self._load_json(self.cfg.suggestion_snapshots_path, [])
        self.index = self._build_index()

    @staticmethod
    def _load_json(path: str, default: Any) -> Any:
        if not os.path.exists(path):
            return copy.deepcopy(default)
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return copy.deepcopy(default)

    def _save_json(self, path: str, content: Any) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)

    def _build_index(self) -> dict[str, Any]:
        by_keyword: dict[str, list[str]] = {}
        by_category: dict[str, list[str]] = {}
        for record in self.pool.get("active_suggestions", []):
            record_id = record["id"]
            category = str(record.get("category", "")).strip()
            if category:
                by_category.setdefault(category, []).append(record_id)
            for keyword in unique_keywords(record.get("keywords")):
                by_keyword.setdefault(keyword, []).append(record_id)
        return {
            "updated_at": _utcnow(),
            "by_keyword": by_keyword,
            "by_category": by_category,
        }

    def save(self) -> None:
        self.pool["meta"]["updated_at"] = _utcnow()
        self.index = self._build_index()
        self._save_json(self.cfg.suggestion_pool_path, self.pool)
        self._save_json(self.cfg.suggestion_index_path, self.index)

    def active_suggestions(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.pool.get("active_suggestions", []))

    def build_system_prompt_text(self, limit: int = 12) -> str:
        active = sorted(
            self.pool.get("active_suggestions", []),
            key=lambda item: (
                item.get("effectiveness_score", 0),
                item.get("updated_at", ""),
            ),
            reverse=True,
        )
        if not active:
            return "暂无历史系统建议，可完全基于当前轮错误样本做增量归纳。"
        lines: list[str] = []
        for item in active[:limit]:
            lines.append(
                f"- [{item['id']}] v{item['version']} | score={item['effectiveness_score']} | "
                f"{item['category']} | {item['suggestion']}"
            )
        return "\n".join(lines)

    def snapshot(
        self,
        step: int,
        selected_ids: list[str],
        metric_delta: float | None = None,
    ) -> None:
        snapshot = {
            "step": step,
            "timestamp": _utcnow(),
            "metric_delta": metric_delta,
            "selected_ids": selected_ids,
            "active_suggestions": self.active_suggestions(),
            "index_summary": {
                "keywords": len(self.index.get("by_keyword", {})),
                    "categories": len(self.index.get("by_category", {})),
            },
        }
        self.snapshots.append(snapshot)
        self._save_json(self.cfg.suggestion_snapshots_path, self.snapshots)

    def apply_feedback(self, suggestion_ids: list[str], metric_delta: float) -> list[str]:
        updated_ids: list[str] = []
        for suggestion_id in suggestion_ids:
            current = self._find_active(suggestion_id)
            if current is None:
                continue
            updated = copy.deepcopy(current)
            updated["status"] = "active"
            updated["version"] = current["version"] + 1
            updated["updated_at"] = _utcnow()
            updated["last_metric_delta"] = metric_delta
            updated["supersedes"] = current["version"]
            if metric_delta > 0:
                updated["positive_hits"] = current.get("positive_hits", 0) + 1
                updated["negative_hits"] = current.get("negative_hits", 0)
            else:
                updated["positive_hits"] = current.get("positive_hits", 0)
                updated["negative_hits"] = current.get("negative_hits", 0) + 1
            updated["effectiveness_score"] = (
                updated["positive_hits"] - updated["negative_hits"]
            )
            self._replace_active(updated)
            self.pool["suggestion_history"].append(copy.deepcopy(updated))
            updated_ids.append(suggestion_id)
        if updated_ids:
            self.save()
        return updated_ids

    def ingest(
        self,
        suggestions: list[dict[str, Any]],
        step: int,
    ) -> dict[str, Any]:
        decisions: list[dict[str, Any]] = []
        selected_ids: list[str] = []
        added_ids: list[str] = []
        updated_ids: list[str] = []

        threshold = float(getattr(self.cfg, "suggestion_similarity_threshold", 0.82) or 0.82)

        for suggestion in suggestions:
            prepared = self._prepare_record(suggestion, step)

            matched, similarity = self._find_best_match(prepared)
            decision = None
            if matched is not None and similarity >= threshold:
                if self.review_fn is not None:
                    decision = self.review_fn(matched, prepared)
                else:
                    decision = "duplicate" if similarity >= max(threshold, 0.95) else "merge"

            if decision in {"duplicate", "merge"} and matched is not None:
                merged = self._merge_records(matched, prepared, step=step, decision=decision, similarity=similarity)
                self._replace_active(merged)
                selected_ids.append(merged["id"])
                if decision == "merge":
                    updated_ids.append(merged["id"])
                    self.pool["suggestion_history"].append(copy.deepcopy(merged))
                decisions.append({
                    "decision": decision,
                    "id": merged["id"],
                    "title": merged.get("title", ""),
                    "similarity": round(similarity, 4),
                })
                continue

            prepared["id"] = self._allocate_id()
            self.pool["active_suggestions"].append(prepared)
            self.pool["suggestion_history"].append(copy.deepcopy(prepared))
            selected_ids.append(prepared["id"])
            added_ids.append(prepared["id"])
            decisions.append({
                "decision": "new",
                "id": prepared["id"],
                "title": prepared["title"],
            })

        self.save()
        return {
            "selected_ids": selected_ids,
            "added_ids": added_ids,
            "updated_ids": updated_ids,
            "decisions": decisions,
            "system_prompt_text": self.build_system_prompt_text(),
        }

    def _similarity(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        a_kw = unique_keywords(a.get("keywords"))
        b_kw = unique_keywords(b.get("keywords"))
        kw_sim = _jaccard(a_kw, b_kw)

        a_text = f"{a.get('title', '')}|{a.get('category', '')}|{a.get('suggestion', '')}".strip()
        b_text = f"{b.get('title', '')}|{b.get('category', '')}|{b.get('suggestion', '')}".strip()
        seq_sim = difflib.SequenceMatcher(None, a_text, b_text).ratio() if a_text and b_text else 0.0

        if a_kw or b_kw:
            return max(0.0, min(1.0, 0.7 * seq_sim + 0.3 * kw_sim))
        return max(0.0, min(1.0, seq_sim))

    def _find_best_match(self, prepared: dict[str, Any]) -> tuple[dict[str, Any] | None, float]:
        best: dict[str, Any] | None = None
        best_sim = 0.0
        for record in self.pool.get("active_suggestions", []):
            sim = self._similarity(record, prepared)
            if sim > best_sim:
                best_sim = sim
                best = record
        return best, best_sim

    def _merge_records(
        self,
        current: dict[str, Any],
        candidate: dict[str, Any],
        step: int,
        decision: str,
        similarity: float,
    ) -> dict[str, Any]:
        merged = copy.deepcopy(current)
        merged["updated_at"] = _utcnow()

        if decision == "merge":
            merged["version"] = int(merged.get("version", 1)) + 1
            merged["supersedes"] = current.get("version", 1)

        merged_keywords = unique_keywords((merged.get("keywords") or []) + (candidate.get("keywords") or []))
        merged["keywords"] = merged_keywords

        merged_samples = []
        for item in (merged.get("source_samples") or []) + (candidate.get("source_samples") or []):
            val = str(item).strip()
            if val and val not in merged_samples:
                merged_samples.append(val)
        merged["source_samples"] = merged_samples

        merged_traces = list(merged.get("generation_traces") or [])
        for trace in candidate.get("generation_traces") or []:
            if isinstance(trace, dict):
                merged_traces.append(copy.deepcopy(trace))
        merged["generation_traces"] = merged_traces

        source_suggestions = list(merged.get("source_suggestions") or [])
        for src in candidate.get("source_suggestions") or []:
            if isinstance(src, dict):
                source_suggestions.append(copy.deepcopy(src))
        merged["source_suggestions"] = source_suggestions

        merged_from = list(merged.get("merged_from") or [])
        merged_from.append({
            "step": step,
            "timestamp": _utcnow(),
            "decision": decision,
            "similarity": round(float(similarity), 4),
            "title": str(candidate.get("title", "")).strip(),
            "category": str(candidate.get("category", "")).strip(),
        })
        merged["merged_from"] = merged_from
        return merged

    def _allocate_id(self) -> str:
        next_id = int(self.pool["meta"].get("next_id", 1))
        self.pool["meta"]["next_id"] = next_id + 1
        return f"SUG-{next_id:04d}"

    def _prepare_record(self, suggestion: dict[str, Any], step: int) -> dict[str, Any]:
        title = str(suggestion.get("title", "")).strip() or "未命名建议"
        text = str(suggestion.get("suggestion", "")).strip() or str(
            suggestion.get("root_cause", "")
        ).strip()
        category = str(suggestion.get("category", "")).strip() or "未分类"
        source_samples = suggestion.get("source_samples")
        if not isinstance(source_samples, list):
            source_samples = [source_samples] if source_samples else []
        source_payload = suggestion.get("source_payload")
        generation_traces: list[dict[str, Any]] = []
        if isinstance(source_payload, dict) and source_payload:
            generation_traces.append({
                "step": step,
                "timestamp": _utcnow(),
                "query": str(source_payload.get("query", "")),
                "reasoning_content": str(source_payload.get("reasoning_content", "")),
                "raw_output": str(source_payload.get("raw_output", "")),
            })
        return {
            "id": "",
            "version": 1,
            "status": "active",
            "title": title,
            "category": category,
            "root_cause": str(suggestion.get("root_cause", "")).strip(),
            "suggestion": text,
            "keywords": unique_keywords(suggestion.get("keywords")),
            "risk": str(suggestion.get("risk", "")).strip(),
            "confidence": max(0.0, min(1.0, _safe_float(suggestion.get("confidence"), 0.5))),
            "positive_hits": 0,
            "negative_hits": 0,
            "effectiveness_score": 0,
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "step_created": step,
            "supersedes": None,
            "merged_from": [],
            "source_samples": source_samples,
            "generation_traces": generation_traces,
            "source_suggestions": [
                {
                    "title": title,
                    "suggestion": text,
                    "category": category,
                }
            ],
        }

    def _replace_active(self, record: dict[str, Any]) -> None:
        active = self.pool.get("active_suggestions", [])
        for idx, item in enumerate(active):
            if item["id"] == record["id"]:
                active[idx] = record
                return
        active.append(record)

    def _find_active(self, suggestion_id: str) -> dict[str, Any] | None:
        for item in self.pool.get("active_suggestions", []):
            if item["id"] == suggestion_id:
                return item
        return None


def build_round_suggestions_text(suggestions: list[dict[str, Any]]) -> str:
    if not suggestions:
        return "本轮暂无新增建议。"
    lines: list[str] = []
    for idx, item in enumerate(suggestions, 1):
        lines.append(
            f"- #{idx} [{item.get('category', '未分类')}] {item.get('title', '未命名建议')}："
            f"{item.get('suggestion', '')}"
        )
    return "\n".join(lines)


def build_experience_feedback_text(feedback: dict[str, Any] | None) -> str:
    if not feedback:
        return "本轮暂无结构化错误经验反馈，请结合历史趋势与新增建议自行归纳。"

    summary = str(feedback.get("summary", "")).strip() or "暂无整体总结。"
    effective = feedback.get("effective_strategies", [])
    ineffective = feedback.get("ineffective_or_risky_strategies", [])
    priorities = feedback.get("optimization_priorities", [])

    def _format_list(title: str, values: Any) -> str:
        items = [str(item).strip() for item in values if str(item).strip()] if isinstance(values, list) else []
        if not items:
            return f"### {title}\n- 暂无\n"
        return "### " + title + "\n" + "\n".join(f"- {item}" for item in items) + "\n"

    return (
        f"### 整体总结\n{summary}\n\n"
        f"{_format_list('有效策略', effective)}\n"
        f"{_format_list('无效或高风险方向', ineffective)}\n"
        f"{_format_list('本轮优化优先级', priorities)}"
    ).strip()
