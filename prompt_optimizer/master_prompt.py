# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from prompt_optimizer.task_config import TaskConfig


def format_errors(
    errors: list[dict],
    task_config: "TaskConfig",
    max_samples: int = 10,
) -> tuple[str, int]:
    total = len(errors)
    if not errors:
        return "无错误样本。", 0

    sampled = random.sample(errors, min(total, max_samples))
    lines: list[str] = []

    for i, e in enumerate(sampled, 1):
        parts = [f"[错误样本 #{i}] Expected={e['expected']} | Predicted={e['predicted']}"]

        # --- GT 描述 ---
        user_input = e.get("user_input")
        if not user_input and e.get("fields"):
            user_input = task_config.build_user_prompt(e["fields"])
        gt_desc = task_config.label_descriptions.get(e["expected"], "")
        pred_desc = task_config.label_descriptions.get(e["predicted"], "")

        gt_tag = f"{e['expected']}（{gt_desc}）" if gt_desc else e["expected"]
        pred_tag = f"{e['predicted']}（{pred_desc}）" if pred_desc else e["predicted"]
        parts.append(f"  [GT]: {gt_tag}")
        parts.append(f"  [Pred]: {pred_tag}")

        # --- Worker 输入 ---
        if user_input:
            parts.append(f"  [Worker 输入]:\n{user_input}")
        elif e.get("fields"):
            for var_name, col_name in task_config.prompt_variables.items():
                val = e["fields"].get(col_name, "")
                parts.append(f"  {var_name}: {val}")

        # --- Worker 输出 ---
        if e.get("worker_output"):
            parts.append(f"  [Worker 输出]:\n{e['worker_output']}")

        # --- 人工备注（提升视觉权重） ---
        if e.get("feedback"):
            parts.append(f"  ⚠ [人工备注]: {e['feedback']}")

        lines.append("\n".join(parts))

    return "\n\n---\n\n".join(lines), total


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


_MASTER_TEMPLATE = """\
你是一位顶尖的 Prompt 工程师和算法专家。你的唯一目标：通过对当前 prompt 进行精准的增量改进，修复模型在评测中犯的错误，提升 {primary} 指标。

## 任务背景
{task_desc}

## 近几轮实验历史
{history_str}
请从历史趋势中识别：哪些类型的改动带来了提升？哪些改动未见效或导致回退？本轮改进应延续有效策略、规避已知无效方向。

## 当前 Prompt（待改进）
<current_prompt>
{current_prompt}
</current_prompt>

## 当前表现
- Train {primary} = {train_score:.4f}
- Valid {primary} = {val_score:.4f}
- 错误样本概览：总计 {total_errors} 条，以下随机展示 {shown_errors} 条。

## 错误样本剖析
每条样本包含：
- Expected（人工标注）/ Predicted（当前模型预测）
- [GT]：人工标注结论，必要时附带标签含义
- [Pred]：模型预测结论，附带标签含义，便于理解模型的错误方向
- [Worker 输入]：发送给模型的实际输入
- [Worker 输出]：模型的错误推理过程——重点关注"它在哪一步想错了"
- [人工备注]（若有）：人工标注者的真实判定依据，**最高可信度参考**

{errors_str}

## 改进方法论

### Step 1：错误模式聚类
先将上述错误样本按根因归类（如：定义边界模糊、规则缺失、规则冲突、上下文遗漏等），识别出 2-4 个主要错误模式。

### Step 2：逐一归因定位
对每个错误模式，精确定位当前 prompt 中的哪些条款（或缺失的条款）导致了该类错误。尤其注意：
- [Worker 输出] 中模型的推理链在哪一步偏离了正确方向
- [人工备注] 指出的判定依据，在当前 prompt 中是否有对应规则

### Step 3：增量打补丁
- **保留**当前 prompt 中已验证有效的主体结构和规则
- **仅针对** Step 2 定位的漏洞，进行条款增补、边界澄清或逻辑修正
- 改动幅度应与错误严重程度成正比——小问题微调措辞，大问题补充子条款
- 切忌推翻重写

### Step 4：反向验证（心理模拟）
在输出前，心理模拟新 prompt 对以下场景的表现：
- 本轮展示的错误样本是否能被修复
- 之前已正确判定的典型样本是否仍然正确（避免误伤）

## 硬性约束
1. **严禁过拟合**：将错误样本中的具体问题抽象为通用判定原则，禁止将具体案例的原文、特定实体名、特定数值写入 prompt 作为规则。
2. **格式约束**：{placeholder_note}
3. **输出格式**：直接输出修改后的完整 prompt 文本。禁止包含任何开场白、解释说明、markdown 代码块包裹（如 ```）、或 "以下是优化后的 prompt" 等前缀。第一个字符就是 prompt 正文的第一个字符。"""


def build_master_prompt(
    current_prompt: str,
    metrics_train: dict,
    metrics_val: dict,
    errors_str: str,
    total_errors: int,
    history: list[dict],
    task_config: "TaskConfig",
    task_description: str = "",
) -> str:
    primary = task_config.primary_metric
    max_samples = task_config.max_error_samples

    task_desc = task_description if task_description else f"任务类型: {task_config.task_type}"
    placeholder_note = _build_placeholder_note(task_config)
    history_str = _format_history(history, primary)

    logger.debug(
        "Building master prompt | total_errors={} | history_rounds={}",
        total_errors,
        len(history),
    )

    return _MASTER_TEMPLATE.format(
        task_desc=task_desc,
        history_str=history_str,
        current_prompt=current_prompt,
        primary=primary,
        train_score=metrics_train.get(primary, 0),
        val_score=metrics_val.get(primary, 0),
        total_errors=total_errors,
        shown_errors=min(total_errors, max_samples),
        errors_str=errors_str,
        placeholder_note=placeholder_note,
    )