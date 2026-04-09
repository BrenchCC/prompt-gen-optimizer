# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from prompt_optimizer.task_config import TaskConfig
from prompt_optimizer.evaluator import Evaluator
from prompt_optimizer.llm_utils import build_llm_client, call_llm_with_retries, resolve_model_config
from prompt_optimizer.master_prompt import (
    build_master_prompt,
    build_master_system_prompt,
    build_suggestion_prompt,
    format_error,
    sample_errors,
)
from prompt_optimizer.prompt_templates import render_prompt_template
from prompt_optimizer.response_utils import parse_json_dict, parse_string_dict, strip_code_fence
from prompt_optimizer.suggestion_pool import (
    SuggestionPool,
    build_experience_feedback_text,
    build_round_suggestions_text,
)


class PromptOptimizer:
    def __init__(self, task_config: TaskConfig) -> None:
        self.cfg = task_config
        self.evaluator = Evaluator(task_config)
        self.master_options = resolve_model_config(role="master", task_config=task_config)
        self.master_client = self._init_master_client()
        self.history: list[dict[str, Any]] = []
        self.suggestion_pool = SuggestionPool(task_config)

    def _init_master_client(self):
        return build_llm_client(
            mode=self.master_options["mode"],
            api_key=self.master_options["api_key"],
            base_url=self.master_options["base_url"],
            model_name=self.master_options["model_name"],
        )

    def _call_master(self, prompt: str, system_prompt: str | None = None) -> tuple[str, str]:
        temp = self.master_options["temperature"]
        top_p = self.master_options["top_p"]
        reasoning = self.master_options["reasoning_option"]

        def request() -> tuple[str, str]:
            reasoning_content, result, _, _ = self.master_client.chat(
                input_query=prompt,
                system_prompt=system_prompt,
                reasoning_option=reasoning,
                temperature=temp,
                top_p=top_p,
            )
            return (
                reasoning_content if isinstance(reasoning_content, str) else "",
                result if isinstance(result, str) else "",
            )

        return call_llm_with_retries(
            client=self.master_client,
            label="Master LLM",
            max_retries=self.cfg.max_retries,
            request_fn=request,
        )

    def _get_or_generate_task_description(self, original_prompt: str) -> str:
        desc_path = self.cfg.output_task_description_path
        if os.path.exists(desc_path):
            with open(desc_path, "r", encoding="utf-8") as f:
                cached = f.read().strip()
            if cached:
                logger.info(f"复用已有任务背景描述: {desc_path}")
                return cached

        logger.info("生成任务背景描述 (LLM)...")
        labels_str = ", ".join(f"{k}: {v}" for k, v in self.cfg.label_descriptions.items()) \
            if self.cfg.label_descriptions else ", ".join(self.cfg.label_map.values())
        vars_str = ", ".join(f"{k} → {v}" for k, v in self.cfg.prompt_variables.items()) \
            if self.cfg.prompt_variables else "无"

        query = render_prompt_template(
            "task_description_prompt.md",
            task_type=self.cfg.task_type,
            labels_str=labels_str,
            vars_str=vars_str,
            original_prompt=original_prompt,
        )

        _, desc = self._call_master(query)
        if not desc:
            desc = f"任务类型: {self.cfg.task_type}, 标签: {labels_str}"
            logger.warning("LLM 生成任务描述失败，使用默认描述")

        self.cfg.ensure_output_dir()
        with open(desc_path, "w", encoding="utf-8") as f:
            f.write(desc)
        logger.info(f"任务背景描述已保存: {desc_path}")
        return desc

    def _sample_train_val(self, data: list[dict]) -> tuple[list[dict], list[dict]]:
        train_size = self.cfg.train_sample_size
        val_size = self.cfg.val_sample_size
        total_needed = train_size + val_size

        if total_needed > len(data):
            logger.warning(f"所需样本数({total_needed})超过数据总量({len(data)})，按比例缩减")
            ratio = len(data) / total_needed
            train_size = max(1, int(train_size * ratio))
            val_size = max(1, len(data) - train_size)
            total_needed = train_size + val_size

        sampled = random.sample(data, total_needed)
        train_set = sampled[:train_size]
        val_set = sampled[train_size:]
        logger.info(f"抽样完成: train={len(train_set)}, valid={len(val_set)}, 总数据={len(data)}")
        return train_set, val_set

    def _init_worker_prompt_log(self) -> None:
        log_path = self.cfg.output_worker_prompt_log_path
        self.cfg.ensure_output_dir()
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Worker System Prompt Log\n\n")

    def _append_worker_prompt_log(
        self,
        stage: str,
        system_prompt: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        log_path = self.cfg.output_worker_prompt_log_path
        self.cfg.ensure_output_dir()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"## {stage}\n\n")
            if extra:
                for key, value in extra.items():
                    f.write(f"- {key}: {value}\n")
                f.write("\n")
            prompt_text = system_prompt if isinstance(system_prompt, str) else ""
            f.write(f"```markdown\n{prompt_text}\n```\n\n")
            f.write("---\n\n")

    def _eval_prompt(
        self, prompt: str, system_prompt: str, data: list[dict], step_label: str
    ) -> tuple[dict, dict, list, list]:
        self._append_worker_prompt_log(
            stage=step_label,
            system_prompt=system_prompt,
            extra={
                "mode": "train+valid",
                "prompt_lines": len(system_prompt.splitlines()) if system_prompt else 0,
            },
        )
        train_sample, val_sample = self._sample_train_val(data)

        t_preds, t_raws, t_queries = self.evaluator.run_prompt(
            prompt, train_sample, desc=f"{step_label} Train", system_prompt=system_prompt
        )
        m_train, t_errors = self.evaluator.evaluate(t_preds, train_sample, t_raws, t_queries)

        v_preds, v_raws, v_queries = self.evaluator.run_prompt(
            prompt, val_sample, desc=f"{step_label} Valid", system_prompt=system_prompt
        )
        m_val, v_errors = self.evaluator.evaluate(v_preds, val_sample, v_raws, v_queries)

        return m_train, m_val, t_errors, v_errors

    def _clean_prompt(self, raw: str) -> str:
        return strip_code_fence(raw)

    def _parse_json_object(self, raw: str) -> dict[str, Any]:
        return parse_json_dict(raw)

    def _review_suggestion_pair(
        self,
        current: dict[str, Any],
        candidate: dict[str, Any],
    ) -> str:
        query = render_prompt_template(
            "suggestion_relation_prompt.md",
            current_suggestion=json.dumps(current, ensure_ascii=False),
            candidate_suggestion=json.dumps(candidate, ensure_ascii=False),
        )
        _, raw = self._call_master(query)
        decision = raw.strip().lower()
        if decision in {"duplicate", "merge", "independent"}:
            return decision
        return "merge"

    def _collect_sampled_errors(
        self,
        val_errors: list[dict[str, Any]],
        train_errors: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        all_errors = val_errors + train_errors
        total_errors = len(all_errors)
        if not all_errors:
            return [], 0

        def infer_positive_label() -> str:
            pos_label = getattr(self.cfg, "positive_label", "") or ""
            if pos_label:
                return pos_label
            labels = set(self.cfg.label_map.values())
            if "是" in labels:
                return "是"
            if len(labels) == 2:
                return sorted(labels)[0]
            return ""

        def diverse_sample(errors: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
            if len(errors) <= max_samples:
                return list(errors)
            buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for e in errors:
                key = (str(e.get("expected", "")).strip(), str(e.get("predicted", "")).strip())
                buckets.setdefault(key, []).append(e)
            keys = sorted(buckets.keys(), key=lambda k: len(buckets[k]), reverse=True)
            for k in keys:
                random.shuffle(buckets[k])
            picked: list[dict[str, Any]] = []
            while len(picked) < max_samples:
                progressed = False
                for k in keys:
                    bucket = buckets.get(k) or []
                    if not bucket:
                        continue
                    picked.append(bucket.pop())
                    progressed = True
                    if len(picked) >= max_samples:
                        break
                if not progressed:
                    break
            if len(picked) < max_samples:
                remaining: list[dict[str, Any]] = []
                for k in keys:
                    remaining.extend(buckets.get(k) or [])
                if remaining:
                    picked.extend(random.sample(remaining, min(len(remaining), max_samples - len(picked))))
            return picked

        max_samples = min(total_errors, self.cfg.max_error_samples)
        primary = self.cfg.primary_metric

        if primary in {"precision_pos", "recall_pos"}:
            pos_label = infer_positive_label()
            if pos_label:
                fp = [e for e in all_errors if e.get("predicted") == pos_label and e.get("expected") != pos_label]
                fn = [e for e in all_errors if e.get("predicted") != pos_label and e.get("expected") == pos_label]
                other = [e for e in all_errors if e not in fp and e not in fn]
                preferred = fp if primary == "precision_pos" else fn
                preferred_pick = diverse_sample(preferred, min(len(preferred), max_samples))
                rest = max_samples - len(preferred_pick)
                if rest <= 0:
                    return preferred_pick, total_errors
                rest_pool = (fn if primary == "precision_pos" else fp) + other
                return preferred_pick + diverse_sample(rest_pool, rest), total_errors

        return diverse_sample(all_errors, max_samples), total_errors

    def _build_master_messages(
        self,
        current_prompt: str,
        metrics_train: dict[str, float],
        metrics_val: dict[str, float],
        total_errors: int,
        round_suggestions: list[dict[str, Any]],
        experience_feedback: dict[str, Any] | None,
    ) -> tuple[str, str]:
        system_prompt = build_master_system_prompt(
            self.cfg,
            self.suggestion_pool.build_system_prompt_text(),
        )
        user_prompt = build_master_prompt(
            current_prompt=current_prompt,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            total_errors=total_errors,
            history=self.history,
            task_config=self.cfg,
            task_description=self.task_description,
            round_suggestions=build_round_suggestions_text(round_suggestions),
            experience_feedback=build_experience_feedback_text(experience_feedback),
        )
        return system_prompt, user_prompt

    def _parse_suggestion_response(
        self,
        raw: str,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        payload = self._parse_json_object(raw)
        if not payload:
            return {}, []
        raw_suggestions = payload.get("suggestions", [])
        if isinstance(raw_suggestions, dict):
            suggestions = [raw_suggestions]
        else:
            suggestions = [item for item in raw_suggestions if isinstance(item, dict)] if isinstance(raw_suggestions, list) else []
        feedback = {
            "summary": payload.get("summary", ""),
            "effective_strategies": payload.get("effective_strategies", []),
            "ineffective_or_risky_strategies": payload.get("ineffective_or_risky_strategies", []),
            "optimization_priorities": payload.get("optimization_priorities", []),
        }
        return feedback, suggestions

    def _generate_batch_suggestions(
        self,
        sampled_errors: list[dict[str, Any]],
        current_prompt: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        formatted_errors = [
            format_error(error, self.cfg, index=index)
            for index, error in enumerate(sampled_errors, 1)
        ]
        query = build_suggestion_prompt(
            errors=sampled_errors,
            current_prompt=current_prompt,
            task_config=self.cfg,
            task_description=self.task_description,
        )
        reasoning_content, raw = self._call_master(query)
        if not sampled_errors:
            return [], {}, {
                "query": "",
                "reasoning_content": "",
                "raw_output": "",
                "sample_count": 0,
                "sample_errors": [],
            }
        experience_feedback, parsed = self._parse_suggestion_response(raw)
        suggestions: list[dict[str, Any]] = []
        for item in parsed:
            suggestions.append({
                "title": str(item.get("title", "")).strip() or "未命名建议",
                "category": str(item.get("category", "")).strip() or "未分类",
                "root_cause": str(item.get("root_cause", "")).strip(),
                "suggestion": str(item.get("suggestion", "")).strip(),
                "keywords": item.get("keywords", []),
                "risk": str(item.get("risk", "")).strip(),
                "confidence": item.get("confidence", 0.5),
                "source_samples": formatted_errors,
                "source_payload": {
                    "query": query,
                    "reasoning_content": reasoning_content,
                    "raw_output": raw,
                },
            })
        if len(suggestions) > 1:
            merged_keywords: list[str] = []
            merged_root_causes: list[str] = []
            merged_suggestions: list[str] = []
            merged_risks: list[str] = []
            confidences: list[float] = []
            categories: list[str] = []
            for s in suggestions:
                category = str(s.get("category", "")).strip()
                if category and category not in categories:
                    categories.append(category)
                for kw in s.get("keywords", []) or []:
                    kw_s = str(kw).strip()
                    if kw_s and kw_s not in merged_keywords:
                        merged_keywords.append(kw_s)
                rc = str(s.get("root_cause", "")).strip()
                if rc and rc not in merged_root_causes:
                    merged_root_causes.append(rc)
                sug = str(s.get("suggestion", "")).strip()
                if sug and sug not in merged_suggestions:
                    merged_suggestions.append(sug)
                risk = str(s.get("risk", "")).strip()
                if risk and risk not in merged_risks:
                    merged_risks.append(risk)
                try:
                    confidences.append(float(s.get("confidence", 0.5)))
                except (TypeError, ValueError):
                    pass
            merged = {
                "title": "本轮总结建议",
                "category": " / ".join(categories) if categories else "未分类",
                "root_cause": "；".join(merged_root_causes),
                "suggestion": "\n".join(f"{i}. {text}" for i, text in enumerate(merged_suggestions, 1)),
                "keywords": merged_keywords[:5],
                "risk": "；".join(merged_risks),
                "confidence": sum(confidences) / len(confidences) if confidences else 0.5,
                "source_samples": formatted_errors,
                "source_payload": {
                    "query": query,
                    "reasoning_content": reasoning_content,
                    "raw_output": raw,
                },
            }
            suggestions = [merged]
        if not suggestions:
            fallback_text = "\n\n---\n\n".join(formatted_errors) if formatted_errors else "无错误样本。"
            suggestions.append({
                "title": "批量建议生成失败回退",
                "category": "未分类",
                "root_cause": "建议生成输出未能解析为 JSON 数组",
                "suggestion": fallback_text,
                "keywords": [],
                "risk": "需要人工复核",
                "confidence": 0.0,
                "source_samples": formatted_errors,
                "source_payload": {
                    "query": query,
                    "reasoning_content": reasoning_content,
                    "raw_output": raw,
                },
            })
        if not experience_feedback:
            experience_feedback = {
                "summary": "本轮建议生成结果未返回结构化错误经验反馈，需要结合新增建议人工归纳。",
                "effective_strategies": [],
                "ineffective_or_risky_strategies": [],
                "optimization_priorities": [],
            }
        trace = {
            "query": query,
            "reasoning_content": reasoning_content,
            "raw_output": raw,
            "sample_count": len(sampled_errors),
            "sample_errors": formatted_errors,
        }
        return suggestions, experience_feedback, trace

    def _merge_and_store_suggestions(
        self,
        step: int,
        suggestions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not suggestions:
            return {
                "selected_ids": [],
                "added_ids": [],
                "updated_ids": [],
                "decisions": [],
                "system_prompt_text": self.suggestion_pool.build_system_prompt_text(),
                "experience_summary": "本轮暂无经验生成。",
                "experience_feedback": {},
            }
        result = self.suggestion_pool.ingest(suggestions, step=step)
        result["experience_summary"] = build_round_suggestions_text(suggestions)
        result["experience_feedback"] = {}
        return result

    def _log_master_step(
        self,
        step: int,
        suggestions: list[dict[str, Any]],
        suggestion_trace: dict[str, Any],
        pool_result: dict[str, Any],
        master_system_prompt: str,
        master_user_prompt: str,
        reasoning_content: str,
        raw_output: str,
    ) -> None:
        log_path = self.cfg.output_master_log_path
        self.cfg.ensure_output_dir()
        summary = [
            {
                "title": suggestion.get("title", ""),
                "category": suggestion.get("category", ""),
                "suggestion": suggestion.get("suggestion", ""),
                "keywords": suggestion.get("keywords", []),
                "risk": suggestion.get("risk", ""),
                "confidence": suggestion.get("confidence", 0.0),
            }
            for suggestion in suggestions
        ]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"# Step {step}\n\n")
            f.write("## Error Experience Feedback\n")
            f.write(
                f"```text\n{build_experience_feedback_text(pool_result.get('experience_feedback'))}\n```\n\n"
            )
            f.write("## Experience Generation Summary\n")
            f.write(f"```json\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n```\n\n")
            f.write("## Experience Generation Call\n")
            f.write(f"**sample_count:** {suggestion_trace.get('sample_count', 0)}\n\n")
            f.write("### sampled_errors\n\n")
            f.write(
                f"```text\n{chr(10).join(suggestion_trace.get('sample_errors', []))}\n```\n\n"
            )
            f.write("### query\n\n")
            f.write(f"```text\n{suggestion_trace.get('query', '')}\n```\n\n")
            f.write("### reasoning_content\n\n")
            f.write(f"```text\n{suggestion_trace.get('reasoning_content', '')}\n```\n\n")
            f.write("### raw_output\n\n")
            f.write(f"```text\n{suggestion_trace.get('raw_output', '')}\n```\n\n")
            f.write("## Dedupe Decisions\n")
            f.write(f"```json\n{json.dumps(pool_result.get('decisions', []), ensure_ascii=False, indent=2)}\n```\n\n")
            f.write("## Merged Suggestions\n")
            f.write(f"```text\n{pool_result.get('system_prompt_text', '')}\n```\n\n")
            f.write("## Master Full Prompt\n")
            f.write("### system_prompt\n\n")
            f.write(f"```text\n{master_system_prompt}\n```\n\n")
            f.write("### user_prompt\n\n")
            f.write(f"```text\n{master_user_prompt}\n```\n\n")
            f.write("## Master System Prompt\n")
            f.write(f"```text\n{master_system_prompt}\n```\n\n")
            f.write("## Master Input Query\n")
            f.write(f"```text\n{master_user_prompt}\n```\n\n")
            f.write("## Master Reasoning Content\n")
            f.write(f"```text\n{reasoning_content}\n```\n\n")
            f.write("## Master Raw Output\n")
            f.write(f"```text\n{raw_output}\n```\n\n")
            f.write("---\n\n")

    def _improve_prompt(
        self,
        step: int,
        current_prompt: str,
        metrics_train: dict,
        metrics_val: dict,
        val_errors: list,
        train_errors: list,
    ) -> tuple[str, list[str], dict[str, Any]]:
        sampled_errors, total_errors = self._collect_sampled_errors(
            val_errors,
            train_errors,
        )
        suggestions, experience_feedback, suggestion_trace = self._generate_batch_suggestions(sampled_errors, current_prompt)
        pool_result = self._merge_and_store_suggestions(step, suggestions)
        pool_result["experience_feedback"] = experience_feedback
        master_system_prompt, master_query = self._build_master_messages(
            current_prompt=current_prompt,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            total_errors=total_errors,
            round_suggestions=suggestions,
            experience_feedback=experience_feedback,
        )

        reasoning_content, raw = self._call_master(master_query, system_prompt=master_system_prompt)
        new_prompt = self._clean_prompt(raw)
        self._log_master_step(
            step=step,
            suggestions=suggestions,
            suggestion_trace=suggestion_trace,
            pool_result=pool_result,
            master_system_prompt=master_system_prompt,
            master_user_prompt=master_query,
            reasoning_content=reasoning_content,
            raw_output=raw,
        )

        if not new_prompt:
            logger.warning("Master 返回空 prompt，保持原 prompt 不变")
            return current_prompt, pool_result.get("selected_ids", []), pool_result

        return new_prompt, pool_result.get("selected_ids", []), pool_result

    def _append_results(self, path: str, record: dict) -> None:
        records = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except (json.JSONDecodeError, ValueError):
                records = []
        records.append(record)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def _append_log(self, path: str, step: int, prompt: str, metrics: dict, analysis: str) -> None:
        with open(path, "a", encoding="utf-8") as f:
            metrics_str = " | ".join([f"**{k}:** {v:.4f}" for k, v in metrics.items()])
            f.write(f"## Step {step}\n")
            f.write(f"{metrics_str}\n\n")
            f.write(f"**Prompt ({len(prompt.splitlines())} lines):**\n```markdown\n{prompt}\n```\n\n")
            f.write(f"**Error Analysis:**\n{analysis}\n\n")
            f.write("---\n\n")

    def _save_config_snapshot(self) -> None:
        config_data = asdict(self.cfg)
        for secret_key in ("worker_api_key", "master_api_key"):
            if secret_key in config_data and config_data[secret_key]:
                config_data[secret_key] = "***"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "task_config": config_data,
            "derived_paths": {
                "output_results_path": self.cfg.output_results_path,
                "output_best_prompt_path": self.cfg.output_best_prompt_path,
                "output_best_score_path": self.cfg.output_best_score_path,
                "output_log_path": self.cfg.output_log_path,
                "output_master_log_path": self.cfg.output_master_log_path,
                "output_worker_prompt_log_path": self.cfg.output_worker_prompt_log_path,
                "output_config_path": self.cfg.output_config_path,
                "suggestion_pool_path": self.cfg.suggestion_pool_path,
                "suggestion_index_path": self.cfg.suggestion_index_path,
                "suggestion_snapshots_path": self.cfg.suggestion_snapshots_path,
            },
        }
        self.cfg.ensure_output_dir()
        with open(self.cfg.output_config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _save_best(self, prompt: str, score: float, metrics: dict) -> None:
        self.cfg.ensure_output_dir()
        with open(self.cfg.output_best_prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(self.cfg.output_best_score_path, "w", encoding="utf-8") as f:
            json.dump(
                {"best_score": score, "metric": self.cfg.primary_metric, "metrics": metrics,
                 "timestamp": datetime.now().isoformat()},
                f, ensure_ascii=False, indent=2,
            )

    def _save_comparison(self, original: str, best: str) -> None:
        self.cfg.ensure_output_dir()
        path = self.cfg.output_comparison_path
        changed = original.strip() != best.strip()
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Prompt Comparison: Original vs Best\n\n")
            f.write(f"**状态:** {'Prompt 已优化' if changed else '未发生变更（原始即最优）'}\n\n")
            f.write("---\n\n")
            f.write("## Original Prompt\n\n")
            f.write(f"```markdown\n{original}\n```\n\n")
            f.write("---\n\n")
            f.write("## Best Prompt\n\n")
            f.write(f"```markdown\n{best}\n```\n\n")

    @staticmethod
    def _parse_raw_json(raw: str) -> dict[str, str]:
        return parse_string_dict(raw)

    def _run_final_evaluation(
        self, data: list[dict], original_prompt: str, best_prompt: str
    ) -> dict[str, Any]:
        logger.info(f"全量数据推理: {len(data)} 条样本")

        prompt_changed = original_prompt.strip() != best_prompt.strip()

        logger.info("--- 原始 Prompt 全量推理 ---")
        self._append_worker_prompt_log(
            stage="Final-Original",
            system_prompt=original_prompt,
            extra={
                "mode": "full-dataset",
                "prompt_lines": len(original_prompt.splitlines()) if original_prompt else 0,
            },
        )
        orig_preds, orig_raws, _ = self.evaluator.run_prompt(
            original_prompt, data, desc="Final-Original", system_prompt=original_prompt
        )
        orig_metrics, orig_errors = self.evaluator.evaluate(orig_preds, data)
        logger.info(f"原始 Prompt 全量指标: {orig_metrics}")

        if prompt_changed:
            logger.info("--- 最佳 Prompt 全量推理 ---")
            self._append_worker_prompt_log(
                stage="Final-Best",
                system_prompt=best_prompt,
                extra={
                    "mode": "full-dataset",
                    "prompt_lines": len(best_prompt.splitlines()) if best_prompt else 0,
                },
            )
            best_preds, best_raws, _ = self.evaluator.run_prompt(
                best_prompt, data, desc="Final-Best", system_prompt=best_prompt
            )
            best_metrics, best_errors = self.evaluator.evaluate(best_preds, data)
            logger.info(f"最佳 Prompt 全量指标: {best_metrics}")
        else:
            best_preds, best_raws = orig_preds, orig_raws
            best_metrics = orig_metrics
            best_errors = orig_errors
            logger.info("原始 Prompt 即最优，跳过重复推理")

        rows: list[dict[str, str]] = []
        for i, item in enumerate(data):
            row: dict[str, str] = {"index": str(i)}
            for col_name, col_val in item["fields"].items():
                row[col_name] = str(col_val) if col_val is not None else ""
            row["label"] = str(item["label"])

            o_pred = str(orig_preds[i]) if i < len(orig_preds) else ""
            row["original_pred"] = o_pred
            row["original_correct"] = str(o_pred == row["label"])
            orig_parsed = self._parse_raw_json(orig_raws[i] if i < len(orig_raws) else "")
            for k, v in orig_parsed.items():
                row[f"original_{k}"] = str(v)

            if prompt_changed:
                b_pred = str(best_preds[i]) if i < len(best_preds) else ""
                row["best_pred"] = b_pred
                row["best_correct"] = str(b_pred == row["label"])
                best_parsed = self._parse_raw_json(best_raws[i] if i < len(best_raws) else "")
                for k, v in best_parsed.items():
                    row[f"best_{k}"] = str(v)
            else:
                row["best_pred"] = row["original_pred"]
                row["best_correct"] = row["original_correct"]

            rows.append(row)

        primary = self.cfg.primary_metric
        result = {
            "total_samples": len(data),
            "timestamp": datetime.now().isoformat(),
            "primary_metric": primary,
            "original_prompt": {
                "metrics": orig_metrics,
                "total_errors": len(orig_errors),
            },
            "best_prompt": {
                "metrics": best_metrics,
                "total_errors": len(best_errors),
            },
            "improvement": {
                k: round(best_metrics.get(k, 0) - orig_metrics.get(k, 0), 4)
                for k in orig_metrics
            },
        }

        self.cfg.ensure_output_dir()

        with open(self.cfg.output_final_eval_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        xlsx_path = self.cfg.output_final_eval_path.replace(".json", ".xlsx")
        df = pd.DataFrame(rows, dtype=str)
        df.to_excel(xlsx_path, index=False, engine="openpyxl")
        logger.info(f"全量推理 xlsx 已保存: {xlsx_path} ({len(df)} 行, {len(df.columns)} 列)")

        logger.info(
            f"全量评测完成: 原始 {primary}={orig_metrics.get(primary, 0):.4f} → "
            f"最佳 {primary}={best_metrics.get(primary, 0):.4f} "
            f"(Δ={best_metrics.get(primary, 0) - orig_metrics.get(primary, 0):+.4f})"
        )
        return result

    def optimize(self) -> dict[str, Any]:
        random.seed(self.cfg.seed)
        data = self.cfg.load_data()
        if not data:
            logger.error("没有可用数据，优化终止。")
            return {"status": "error", "message": "no data"}

        self.cfg.ensure_output_dir()
        results_path = self.cfg.output_results_path
        log_path = self.cfg.output_log_path
        primary = self.cfg.primary_metric
        self._save_config_snapshot()

        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            backup = log_path.replace(".md", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
            os.rename(log_path, backup)
            logger.info(f"旧日志已备份: {backup}")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Prompt Optimization Log\n\n")
        self._init_worker_prompt_log()

        current_prompt = self.cfg.read_prompt()
        if not current_prompt:
            logger.error("Prompt 文件为空，优化终止。")
            return {"status": "error", "message": "empty prompt"}

        original_prompt = current_prompt
        system_prompt = current_prompt
        best_prompt = current_prompt

        self.task_description = self._get_or_generate_task_description(original_prompt)
        best_val_score = 0.0
        no_improve_count = 0

        logger.info(
            f"配置: task={self.cfg.task_name}/{self.cfg.task_version}, "
            f"type={self.cfg.task_type}, iterations={self.cfg.iterations}, "
            f"patience={self.cfg.patience}, metric={primary}, "
            f"train_sample={self.cfg.train_sample_size}, val_sample={self.cfg.val_sample_size}"
        )

        logger.info("=== Step 0: 评测初始 Prompt (Baseline) ===")
        metrics_train, metrics_val, train_errors, val_errors = self._eval_prompt(
            current_prompt, system_prompt, data, "Step0"
        )

        val_primary = metrics_val.get(primary, 0)
        train_primary = metrics_train.get(primary, 0)
        metrics_str = " | ".join(
            [f"{k}: Train={metrics_train[k]:.4f}/Valid={metrics_val[k]:.4f}" for k in metrics_val]
        )
        logger.info(f"Baseline: {metrics_str} | Prompt Lines: {len(current_prompt.splitlines())}")

        best_val_score = val_primary
        best_prompt = current_prompt
        best_metrics = metrics_val.copy()

        self._append_results(results_path, {
            "step": 0, "status": "baseline",
            "train_metrics": metrics_train, "val_metrics": metrics_val,
            "prompt_lines": len(current_prompt.splitlines()),
            "timestamp": datetime.now().isoformat(),
        })

        if val_primary == 1.0 and train_primary == 1.0:
            logger.info("初始 prompt 已达满分，无需优化。")
            self._append_log(log_path, 0, current_prompt, metrics_val, "Perfect score on baseline.")
            self._save_best(best_prompt, best_val_score, best_metrics)
            self._save_comparison(original_prompt, best_prompt)
            logger.info("=== 全量数据最终评测 ===")
            final_result = self._run_final_evaluation(data, original_prompt, best_prompt)
            return {
                "status": "perfect", "best_score": best_val_score,
                "best_metrics": best_metrics, "steps": 0,
                "final_evaluation": final_result,
            }

        for step in range(1, self.cfg.iterations + 1):
            logger.info(f"--- Iteration {step}/{self.cfg.iterations}: 生成新 prompt ---")

            total_errors = len(train_errors) + len(val_errors)
            logger.info(f"错误样本: train={len(train_errors)}, valid={len(val_errors)}, total={total_errors}")

            self._append_log(
                log_path, step - 1, current_prompt, metrics_val,
                f"errors: train={len(train_errors)}, valid={len(val_errors)}"
            )
            self.history.append({
                "step": step - 1,
                "prompt": current_prompt,
                "train_metrics": metrics_train,
                "val_metrics": metrics_val,
                "total_errors": total_errors,
            })

            logger.info("Generating new prompt (Master)...")
            new_prompt, selected_suggestion_ids, pool_result = self._improve_prompt(
                step, current_prompt, metrics_train, metrics_val, val_errors, train_errors
            )

            system_prompt = new_prompt
            current_prompt = new_prompt
            logger.info(f"New prompt written ({len(new_prompt.splitlines())} lines)")

            logger.info(f"--- Iteration {step}/{self.cfg.iterations}: 评测新 prompt ---")
            metrics_train, metrics_val, train_errors, val_errors = self._eval_prompt(
                current_prompt, system_prompt, data, f"Step{step}"
            )

            val_primary = metrics_val.get(primary, 0)
            train_primary = metrics_train.get(primary, 0)
            metrics_str = " | ".join(
                [f"{k}: Train={metrics_train[k]:.4f}/Valid={metrics_val[k]:.4f}" for k in metrics_val]
            )
            logger.info(f"{metrics_str} | Prompt Lines: {len(current_prompt.splitlines())}")
            metric_delta = val_primary - best_val_score

            if val_primary > best_val_score:
                best_val_score = val_primary
                best_prompt = current_prompt
                best_metrics = metrics_val.copy()
                no_improve_count = 0
                status = "keep"
                logger.info(f"Valid {primary} improved! Best: {best_val_score:.4f}")
            else:
                no_improve_count += 1
                status = "discard"
                logger.warning(
                    f"Valid {primary} not improved ({no_improve_count}/{self.cfg.patience}). "
                    f"Best: {best_val_score:.4f}"
                )
                current_prompt = best_prompt
                system_prompt = best_prompt
                logger.info("Rolled back to best prompt.")

            feedback_updated_ids = self.suggestion_pool.apply_feedback(
                selected_suggestion_ids,
                metric_delta=metric_delta,
            )
            self.suggestion_pool.snapshot(
                step=step,
                selected_ids=selected_suggestion_ids,
                metric_delta=metric_delta,
            )

            self._append_results(results_path, {
                "step": step, "status": status,
                "train_metrics": metrics_train, "val_metrics": metrics_val,
                "prompt_lines": len(current_prompt.splitlines()),
                "selected_suggestion_ids": selected_suggestion_ids,
                "suggestion_added_ids": pool_result.get("added_ids", []),
                "suggestion_updated_ids": pool_result.get("updated_ids", []),
                "feedback_updated_ids": feedback_updated_ids,
                "timestamp": datetime.now().isoformat(),
            })
            self._append_log(
                log_path,
                step,
                current_prompt,
                metrics_val,
                (
                    f"errors: train={len(train_errors)}, valid={len(val_errors)}\n"
                    f"experience generation:\n{pool_result.get('experience_summary', '本轮暂无经验生成。')}\n"
                    f"suggestions selected: {selected_suggestion_ids}\n"
                    f"suggestions added: {pool_result.get('added_ids', [])}\n"
                    f"suggestions updated: {pool_result.get('updated_ids', [])}\n"
                    f"feedback updated: {feedback_updated_ids}\n"
                    f"metric delta vs best-before-step: {metric_delta:+.4f}"
                ),
            )

            if val_primary == 1.0 and train_primary == 1.0:
                logger.info("Perfect score. Stopping.")
                break

            if no_improve_count >= self.cfg.patience:
                logger.warning(f"Early stopping: valid {primary} 连续 {self.cfg.patience} 轮未提升。")
                self._append_results(results_path, {
                    "step": step, "status": "early_stop",
                    "train_metrics": metrics_train, "val_metrics": metrics_val,
                    "prompt_lines": len(best_prompt.splitlines()),
                    "timestamp": datetime.now().isoformat(),
                    "description": "rolled back to best",
                })
                break

        logger.info("=== OPTIMIZATION COMPLETE ===")
        logger.info(f"BEST VALID {primary.upper()}: {best_val_score:.4f}")

        self._save_best(best_prompt, best_val_score, best_metrics)
        logger.info(f"Best prompt saved to {self.cfg.output_best_prompt_path}")

        self._save_comparison(original_prompt, best_prompt)
        logger.info(f"Prompt comparison saved to {self.cfg.output_comparison_path}")

        logger.info("=== 全量数据最终评测 ===")
        final_result = self._run_final_evaluation(data, original_prompt, best_prompt)
        logger.info(f"Final evaluation saved to {self.cfg.output_final_eval_path}")

        logger.info(f"Results: {results_path} | Log: {log_path}")

        return {
            "status": "completed",
            "best_score": best_val_score,
            "best_metrics": best_metrics,
            "steps": len(self.history),
            "final_evaluation": final_result,
        }
