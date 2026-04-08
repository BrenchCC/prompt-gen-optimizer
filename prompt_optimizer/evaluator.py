# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from utils.llm_server import LLM_Client
import config as cfg

if TYPE_CHECKING:
    from prompt_optimizer.task_config import TaskConfig


class Evaluator:
    def __init__(self, task_config: "TaskConfig") -> None:
        self.task_config = task_config
        self.task_type: str = task_config.task_type
        self.text_columns: list[str] = task_config.text_columns
        self.label_map: dict[str, str] = task_config.label_map
        self.label_descriptions: dict[str, str] = task_config.label_descriptions
        self.prompt_variables: dict[str, str] = task_config.prompt_variables
        self.concurrency: int = task_config.concurrency
        self.vote_count: int = task_config.vote_count
        self.max_retries: int = task_config.max_retries
        self.output_field = task_config.output_field
        self.output_map = task_config.output_map
        self.custom_parser_fn = task_config.load_custom_parser()
        self.worker_client = self._init_worker_client()

    def _init_worker_client(self) -> LLM_Client:
        tc = self.task_config
        api_key = cfg.WORKER_API_KEY
        base_url = cfg.WORKER_BASE_URL
        model_name = cfg.WORKER_MODEL_NAME
        mode = getattr(tc, "worker_mode", "ark") or "ark"
        return LLM_Client(
            mode = mode,
            api_key = api_key,
            base_url = base_url,
            default_model = model_name,
            print_stream = False,
        )

    def _build_query(self, prompt: str, item: dict[str, Any]) -> str:
        return self.task_config.build_user_prompt(item["fields"])

    def _parse_prediction(self, output: str) -> str:
        if not output:
            return "PARSE_FAIL"

        if self.custom_parser_fn:
            try:
                result = self.custom_parser_fn(output)
                return str(result) if result is not None else "PARSE_FAIL"
            except Exception as e:
                logger.debug(f"自定义解析函数异常: {e}")
                return "PARSE_FAIL"

        text = output.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        if self.output_field:
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and self.output_field in obj:
                    raw_val = obj[self.output_field]
                    if raw_val is None:
                        str_val = "null"
                    elif isinstance(raw_val, bool):
                        str_val = "true" if raw_val else "false"
                    else:
                        str_val = str(raw_val).strip().lower()
                    if self.output_map:
                        for src, dst in self.output_map.items():
                            if str_val == str(src).lower():
                                return dst
                    mapped = self.label_map.get(str(raw_val), None)
                    if mapped:
                        return mapped
                    if str(raw_val) in self.label_map.values():
                        return str(raw_val)
                    return str(raw_val)
            except (json.JSONDecodeError, ValueError):
                pass

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                for k in ("label", "result", "prediction", "answer", "category", "output"):
                    if k in obj:
                        v = str(obj[k]).strip()
                        if v in self.label_map:
                            return self.label_map[v]
                        if v in self.label_map.values():
                            return v
        except (json.JSONDecodeError, ValueError):
            pass

        for lbl in sorted(self.label_map.values(), key=len, reverse=True):
            if lbl in text:
                return lbl

        return "PARSE_FAIL"

    def _call_worker(
        self,
        input_query: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        tc = self.task_config
        temp = temperature if temperature is not None else getattr(tc, "worker_temperature", None) or cfg.WORKER_TEMPERATURE
        top_p = getattr(tc, "worker_top_p", None) or cfg.WORKER_TOP_P
        reasoning = getattr(tc, "worker_reasoning_option", cfg.WORKER_THINKING)

        for attempt in range(self.max_retries):
            try:
                _, result, _, _ = self.worker_client.chat(
                    input_query=input_query,
                    system_prompt=system_prompt,
                    reasoning_option=reasoning,
                    temperature=temp,
                    top_p=top_p,
                )
                return result if isinstance(result, str) else ""
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Worker LLM retry {attempt + 1}/{self.max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Worker LLM API Error (all retries failed): {e}")
                    return ""

    def _evaluate_single(
        self,
        idx: int,
        item: dict[str, Any],
        prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[int, str, str, str, str, str]:
        query = self._build_query(prompt, item)

        if self.vote_count <= 1:
            out = self._call_worker(query, system_prompt=system_prompt, temperature=0.0)
            pred = self._parse_prediction(out)
        else:
            votes: list[str] = []
            for _ in range(self.vote_count):
                out = self._call_worker(query, system_prompt=system_prompt, temperature=0.3)
                votes.append(self._parse_prediction(out))
            pred = Counter(votes).most_common(1)[0][0]

        return idx, pred, item["label"], out, query, item["fields"].get(self.text_columns[0], "")[:30]

    def run_prompt(
        self,
        prompt: str,
        dataset: list[dict[str, Any]],
        desc: str = "Eval",
        system_prompt: str | None = None,
    ) -> tuple[list[str], list[str], list[str]]:
        preds: list[str | None] = [None] * len(dataset)
        raw_outputs: list[str | None] = [None] * len(dataset)
        queries: list[str | None] = [None] * len(dataset)

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            future_to_idx = {
                executor.submit(
                    self._evaluate_single, idx, item, prompt, system_prompt
                ): idx
                for idx, item in enumerate(dataset)
            }

            completed_count = 0
            for future in tqdm(as_completed(future_to_idx), total=len(dataset), desc=desc):
                idx, pred, gt, raw, query, display = future.result()
                preds[idx] = pred
                raw_outputs[idx] = raw
                queries[idx] = query
                completed_count += 1

                if completed_count % 5 == 0:
                    logger.info(
                        f"[{desc}] {completed_count}/{len(dataset)} | "
                        f"{display}... | Pred: {pred} | GT: {gt}"
                    )

        return preds, raw_outputs, queries  # type: ignore[return-value]

    def evaluate(
        self,
        preds: list[str],
        dataset: list[dict[str, Any]],
        raw_outputs: list[str] | None = None,
        queries: list[str] | None = None,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        if not dataset:
            return {}, []

        gt = [x["label"] for x in dataset]
        all_labels = sorted(set(self.label_map.values()))

        metrics: dict[str, float] = {
            "accuracy": accuracy_score(gt, preds),
            "f1": f1_score(gt, preds, labels=all_labels, average="macro", zero_division=0),
            "precision": precision_score(gt, preds, labels=all_labels, average="macro", zero_division=0),
            "recall": recall_score(gt, preds, labels=all_labels, average="macro", zero_division=0),
        }

        errors: list[dict[str, Any]] = []
        for i, (p, g) in enumerate(zip(preds, gt)):
            if p != g:
                err: dict[str, Any] = {
                    "fields": dataset[i]["fields"],
                    "expected": g,
                    "predicted": p,
                }
                if queries and i < len(queries):
                    err["user_input"] = queries[i]
                if raw_outputs and i < len(raw_outputs):
                    err["worker_output"] = raw_outputs[i]
                if dataset[i].get("feedback"):
                    err["feedback"] = dataset[i]["feedback"]
                errors.append(err)

        return metrics, errors
