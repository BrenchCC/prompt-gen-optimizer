# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from loguru import logger
from prompt_optimizer.prompt_templates import render_prompt_template

if TYPE_CHECKING:
    from prompt_optimizer.task_config import TaskConfig


def sample_errors(errors: list[dict[str, Any]], max_samples: int = 10) -> tuple[list[dict[str, Any]], int]:
    total = len(errors)
    if not errors:
        return [], 0
    return random.sample(errors, min(total, max_samples)), total


def format_error(error: dict[str, Any], task_config: "TaskConfig", index: int = 1) -> str:
    parts = [f"[错误样本 #{index}] Expected={error['expected']} | Predicted={error['predicted']}"]

    user_input = error.get("user_input")
    if not user_input and error.get("fields"):
        user_input = task_config.build_user_prompt(error["fields"])
    gt_desc = task_config.label_descriptions.get(error["expected"], "")
    pred_desc = task_config.label_descriptions.get(error["predicted"], "")

    gt_tag = f"{error['expected']}（{gt_desc}）" if gt_desc else error["expected"]
    pred_tag = f"{error['predicted']}（{pred_desc}）" if pred_desc else error["predicted"]
    parts.append(f"  [GT]: {gt_tag}")
    parts.append(f"  [Pred]: {pred_tag}")

    if user_input:
        parts.append(f"  [Worker 输入]:\n{user_input}")
    elif error.get("fields"):
        for var_name, col_name in task_config.prompt_variables.items():
            val = error["fields"].get(col_name, "")
            parts.append(f"  {var_name}: {val}")

    if error.get("worker_output"):
        parts.append(f"  [Worker 输出]:\n{error['worker_output']}")

    if error.get("feedback"):
        parts.append(f"  ⚠ [人工备注]: {error['feedback']}")

    return "\n".join(parts)


def format_errors(
    errors: list[dict[str, Any]],
    task_config: "TaskConfig",
    max_samples: int = 10,
) -> tuple[str, int]:
    sampled, total = sample_errors(errors, max_samples=max_samples)
    if not sampled:
        return "无错误样本。", 0
    return "\n\n---\n\n".join(
        format_error(error, task_config, index=i)
        for i, error in enumerate(sampled, 1)
    ), total


def _build_placeholder_note(task_config: "TaskConfig") -> str:
    if task_config.task_type == "judge":
        var_list = ", ".join(f"{{{{{k}}}}}" for k in task_config.prompt_variables)
        return f"prompt 中必须保留以下变量占位符：{var_list}"
    return "保留 prompt 中已有的结构和关键标记"


def _format_history(history: list[dict], primary: str, last_n: int = 3) -> str:
    if not history:
        return "首轮优化，暂无历史记录。\n"
    recent = history[-last_n:]
    lines: list[str] = []
    for idx, h in enumerate(recent):
        delta = ""
        if idx > 0:
            prev_val = recent[idx - 1]["val_metrics"].get(primary, 0)
            curr_val = h["val_metrics"].get(primary, 0)
            diff = curr_val - prev_val
            arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
            delta = f" (Valid Δ={diff:+.4f} {arrow})"

        lines.append(
            f"- Step {h['step']}: "
            f"Train={h['train_metrics'].get(primary, 0):.4f}, "
            f"Valid={h['val_metrics'].get(primary, 0):.4f}, "
            f"errors={h['total_errors']}{delta}"
        )
    return "\n".join(lines) + "\n"


def build_suggestion_prompt(
    errors: list[dict[str, Any]],
    current_prompt: str,
    task_config: "TaskConfig",
    task_description: str = "",
) -> str:
    task_desc = task_description if task_description else f"任务类型: {task_config.task_type}"
    error_text = (
        "\n\n---\n\n".join(
            format_error(error, task_config, index=index)
            for index, error in enumerate(errors, 1)
        )
        if errors
        else "无错误样本。"
    )
    return render_prompt_template(
        "suggestion_prompt.md",
        task_desc=task_desc,
        current_prompt=current_prompt,
        error_text=error_text,
        error_count=len(errors),
    )


def build_master_system_prompt(
    task_config: "TaskConfig",
    suggestions_text: str,
) -> str:
    return render_prompt_template(
        "master_system_prompt.md",
        primary=task_config.primary_metric,
        suggestions_text=suggestions_text or "暂无历史系统建议，可完全基于当前轮错误样本做增量归纳。",
        placeholder_note=_build_placeholder_note(task_config),
    )


def build_master_user_prompt(
    current_prompt: str,
    metrics_train: dict,
    metrics_val: dict,
    total_errors: int,
    history: list[dict],
    task_config: "TaskConfig",
    task_description: str = "",
    round_suggestions: str = "本轮暂无新增建议。",
    experience_feedback: str = "本轮暂无结构化错误经验反馈，请结合历史趋势与新增建议自行归纳。",
    candidate_count: int = 3,
) -> str:
    primary = task_config.primary_metric
    max_samples = task_config.max_error_samples

    task_desc = task_description if task_description else f"任务类型: {task_config.task_type}"
    history_str = _format_history(history, primary)

    logger.debug(
        "Building master prompt | total_errors={} | history_rounds={}",
        total_errors,
        len(history),
    )

    return render_prompt_template(
        "master_user_prompt.md",
        task_desc=task_desc,
        history_str=history_str,
        current_prompt=current_prompt,
        primary=primary,
        train_score=metrics_train.get(primary, 0),
        val_score=metrics_val.get(primary, 0),
        total_errors=total_errors,
        shown_errors=min(total_errors, max_samples),
        round_suggestions=round_suggestions,
        experience_feedback=experience_feedback,
        candidate_count=max(1, candidate_count),
    )


def build_master_prompt(
    current_prompt: str,
    metrics_train: dict,
    metrics_val: dict,
    total_errors: int,
    history: list[dict],
    task_config: "TaskConfig",
    task_description: str = "",
    round_suggestions: str = "本轮暂无新增建议。",
    experience_feedback: str = "本轮暂无结构化错误经验反馈，请结合历史趋势与新增建议自行归纳。",
    candidate_count: int = 3,
) -> str:
    return build_master_user_prompt(
        current_prompt=current_prompt,
        metrics_train=metrics_train,
        metrics_val=metrics_val,
        total_errors=total_errors,
        history=history,
        task_config=task_config,
        task_description=task_description,
        round_suggestions=round_suggestions,
        experience_feedback=experience_feedback,
        candidate_count=candidate_count,
    )
