# -*- coding: utf-8 -*-
"""
自定义解析函数示例 — shenping 任务

在 YAML 中注册：
  data:
    custom_parser: tasks.shenping.v1.parser.parse_output

函数签名要求：
    def parse_output(raw_output: str) -> str
    - 入参: LLM 原始输出字符串
    - 返回: 标准标签字符串（与 label_map 的 value 对齐）
    - 返回 None 表示解析失败
"""
import json


def parse_output(raw_output: str) -> str:
    text = raw_output.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        obj = json.loads(text)
        val = obj.get("is_shenping")
        if val is True:
            return "是"
        elif val is False:
            return "不是"
        return "不确定"
    except (json.JSONDecodeError, KeyError, TypeError):
        if "是" in text and "不是" not in text:
            return "是"
        if "不是" in text:
            return "不是"
        return "不确定"
