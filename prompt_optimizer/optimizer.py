# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from utils.llm_server import LLM_Client
import config as cfg
from prompt_optimizer.task_config import TaskConfig
from prompt_optimizer.evaluator import Evaluator
from prompt_optimizer.master_prompt import build_master_prompt, format_errors


class PromptOptimizer:
    def __init__(self, task_config: TaskConfig) -> None:
        self.cfg = task_config
        self.evaluator = Evaluator(task_config)
        self.master_client = self._init_master_client()
        self.history: list[dict[str, Any]] = []

    def _init_master_client(self) -> LLM_Client:
        return LLM_Client(
            mode=self.cfg.master_mode,
            api_key=cfg.MASTER_API_KEY,
            base_url=cfg.MASTER_BASE_URL,
            default_model=cfg.MASTER_MODEL_NAME,
            print_stream=False,
        )

    def _call_master(self, prompt: str) -> tuple[str, str]:
        temp = self.cfg.master_temperature or cfg.MASTER_TEMPERATURE
        top_p = self.cfg.master_top_p or cfg.MASTER_TOP_P
        reasoning = self.cfg.master_reasoning_option or cfg.MASTER_THINKING

        for attempt in range(self.cfg.max_retries):
            try:
                reasoning_content, result, _, _ = self.master_client.chat(
                    input_query=prompt,
                    reasoning_option=reasoning,
                    temperature=temp,
                    top_p=top_p,
                )
                return (
                    reasoning_content if isinstance(reasoning_content, str) else "",
                    result if isinstance(result, str) else "",
                )
            except Exception as e:
                if attempt < self.cfg.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Master LLM retry {attempt + 1}/{self.cfg.max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Master LLM API Error (all retries failed): {e}")
                    return "", ""

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

        query = (
            "请基于以下 system prompt 内容，撰写一段简洁的任务背景描述（3-5句话）。\n"
            "描述需包含：任务目标、输入数据格式、输出分类标签及含义、关键判断逻辑。\n"
            "不要复述 prompt 原文，用你自己的话概括。\n\n"
            f"## 任务类型\n{self.cfg.task_type}\n\n"
            f"## 标签定义\n{labels_str}\n\n"
            f"## 输入变量\n{vars_str}\n\n"
            f"## System Prompt 原文\n{original_prompt}"
        )

        _, desc = self._call_master(query)
        if not desc:
            desc = f"任务类型: {self.cfg.task_type}, 标签: {labels_str}"
            logger.warning("LLM 生成任务描述失败，使用默认描述")

        os.makedirs(self.cfg.output_dir, exist_ok=True)
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
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Worker System Prompt Log\n\n")

    def _append_worker_prompt_log(
        self,
        stage: str,
        system_prompt: str | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        log_path = self.cfg.output_worker_prompt_log_path
        os.makedirs(self.cfg.output_dir, exist_ok=True)
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
        new_prompt = raw.strip()
        if new_prompt.startswith("```") and new_prompt.endswith("```"):
            new_prompt = "\n".join(new_prompt.split("\n")[1:-1]).strip()
        if new_prompt.startswith("```markdown"):
            new_prompt = "\n".join(new_prompt.split("\n")[1:]).strip()
            if new_prompt.endswith("```"):
                new_prompt = new_prompt[:-3].strip()
        if new_prompt.startswith("```"):
            new_prompt = "\n".join(new_prompt.split("\n")[1:]).strip()
            if new_prompt.endswith("```"):
                new_prompt = new_prompt[:-3].strip()
        return new_prompt

    def _improve_prompt(
        self,
        step: int,
        current_prompt: str,
        metrics_train: dict,
        metrics_val: dict,
        val_errors: list,
        train_errors: list,
    ) -> str:
        all_errors = val_errors + train_errors
        errors_str, total_errors = format_errors(
            all_errors, self.cfg,
            max_samples=self.cfg.max_error_samples,
        )

        master_query = build_master_prompt(
            current_prompt=current_prompt,
            metrics_train=metrics_train,
            metrics_val=metrics_val,
            errors_str=errors_str,
            total_errors=total_errors,
            history=self.history,
            task_config=self.cfg,
            task_description=self.task_description,
        )

        reasoning_content, raw = self._call_master(master_query)
        new_prompt = self._clean_prompt(raw)

        # 记录 master log
        log_path = self.cfg.output_master_log_path
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"# Step {step}\n\n")
            f.write("## Master Input Query\n")
            f.write(f"```text\n{master_query}\n```\n\n")
            f.write("## Master Reasoning Content\n")
            f.write(f"```text\n{reasoning_content}\n```\n\n")
            f.write("## Master Raw Output\n")
            f.write(f"```text\n{raw}\n```\n\n")
            f.write("---\n\n")

        if not new_prompt:
            logger.warning("Master 返回空 prompt，保持原 prompt 不变")
            return current_prompt

        return new_prompt

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

    def _save_best(self, prompt: str, score: float, metrics: dict) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        with open(self.cfg.output_best_prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(self.cfg.output_best_score_path, "w", encoding="utf-8") as f:
            json.dump(
                {"best_score": score, "metric": self.cfg.primary_metric, "metrics": metrics,
                 "timestamp": datetime.now().isoformat()},
                f, ensure_ascii=False, indent=2,
            )

    def _save_comparison(self, original: str, best: str) -> None:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
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
        if not raw:
            return {}
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return {k: str(v) for k, v in obj.items()}
        except (json.JSONDecodeError, ValueError):
            pass
        return {"raw_output": raw}

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

        os.makedirs(self.cfg.output_dir, exist_ok=True)

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

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        results_path = self.cfg.output_results_path
        log_path = self.cfg.output_log_path
        primary = self.cfg.primary_metric

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
            new_prompt = self._improve_prompt(
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

            self._append_results(results_path, {
                "step": step, "status": status,
                "train_metrics": metrics_train, "val_metrics": metrics_val,
                "prompt_lines": len(current_prompt.splitlines()),
                "timestamp": datetime.now().isoformat(),
            })

            if val_primary == 1.0 and train_primary == 1.0:
                logger.info("Perfect score. Stopping.")
                self._append_log(log_path, step, current_prompt, metrics_val, "Perfect score reached.")
                break

            if no_improve_count >= self.cfg.patience:
                logger.warning(f"Early stopping: valid {primary} 连续 {self.cfg.patience} 轮未提升。")
                self._append_log(
                    log_path, step, current_prompt, metrics_val,
                    f"Early stop. Best valid {primary}={best_val_score:.4f}."
                )
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
