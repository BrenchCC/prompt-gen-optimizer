# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from threading import Lock
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from prompt_optimizer.llm_utils import build_llm_client, call_llm_with_retries, resolve_model_config
from prompt_optimizer.response_utils import parse_json_dict, strip_code_fence

if TYPE_CHECKING:
    from prompt_optimizer.task_config import TaskConfig


class Evaluator:
    def __init__(self, task_config: "TaskConfig") -> None:
        self.task_config = task_config
        self.custom_parser_fn = task_config.load_custom_parser()
        self.worker_options = resolve_model_config(role="worker", task_config=task_config)
        self.worker_client = self._init_worker_client()
        self._inference_cache: dict[str, tuple[str, str]] = {}
        self._cache_lock = Lock()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def concurrency(self) -> int:
        return self.task_config.concurrency

    @property
    def vote_count(self) -> int:
        return self.task_config.vote_count

    @property
    def max_retries(self) -> int:
        return self.task_config.max_retries

    @property
    def label_map(self) -> dict[str, str]:
        return self.task_config.label_map

    @property
    def output_field(self) -> str:
        return self.task_config.output_field

    @property
    def output_map(self) -> dict[str, str]:
        return self.task_config.output_map

    def get_cache_stats(self) -> dict[str, int]:
        with self._cache_lock:
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "size": len(self._inference_cache),
            }

    def _init_worker_client(self):
        return build_llm_client(
            mode=self.worker_options["mode"],
            api_key=self.worker_options["api_key"],
            base_url=self.worker_options["base_url"],
            model_name=self.worker_options["model_name"],
        )

    def _build_query(self, prompt: str | None, item: dict[str, Any]) -> str:
        return self.task_config.build_user_prompt(item["fields"])

    @staticmethod
    def _stringify_prediction_value(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip()

    def _map_prediction_value(self, value: Any) -> str:
        str_val = self._stringify_prediction_value(value)
        normalized = str_val.lower()

        if self.output_map:
            for src, dst in self.output_map.items():
                if normalized == str(src).strip().lower():
                    return dst

        mapped = self.label_map.get(str_val)
        if mapped:
            return mapped

        if str_val in self.label_map.values():
            return str_val

        return str_val

    def _parse_prediction(self, output: str) -> str:
        if not output:
            return "PARSE_FAIL"

        if self.custom_parser_fn:
            try:
                result = self.custom_parser_fn(output)
                if result is None:
                    return "PARSE_FAIL"
                return self._map_prediction_value(result)
            except Exception as e:
                logger.debug(f"自定义解析函数异常: {e}")
                return "PARSE_FAIL"

        text = strip_code_fence(output)
        obj = parse_json_dict(output)

        if self.output_field:
            if self.output_field in obj:
                return self._map_prediction_value(obj[self.output_field])

        for key in ("label", "result", "prediction", "answer", "category", "output"):
            if key in obj:
                return self._map_prediction_value(obj[key])

        if self.output_map:
            for src, dst in self.output_map.items():
                if str(src) in text:
                    return dst

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
        temp = temperature if temperature is not None else self.worker_options["temperature"]
        top_p = self.worker_options["top_p"]
        reasoning = self.worker_options["reasoning_option"]

        def request() -> tuple[str, str]:
            reasoning_content, result, _, _ = self.worker_client.chat(
                input_query=input_query,
                system_prompt=system_prompt,
                reasoning_option=reasoning,
                temperature=temp,
                top_p=top_p,
            )
            return (
                reasoning_content if isinstance(reasoning_content, str) else "",
                result if isinstance(result, str) else "",
            )

        _, result = call_llm_with_retries(
            client=self.worker_client,
            label="Worker LLM",
            max_retries=self.max_retries,
            request_fn=request,
        )
        return result

    def _build_cache_key(
        self,
        item: dict[str, Any],
        prompt: str,
        query: str,
        system_prompt: str | None,
    ) -> str:
        sample_signature = {
            "fields": item.get("fields", {}),
            "label": item.get("label", ""),
        }
        payload = {
            "prompt": prompt,
            "query": query,
            "system_prompt": system_prompt or "",
            "sample": sample_signature,
            "model": self.worker_options.get("model_name", ""),
            "mode": self.worker_options.get("mode", ""),
            "reasoning_option": self.worker_options.get("reasoning_option", ""),
            "temperature": self.worker_options.get("temperature", ""),
            "top_p": self.worker_options.get("top_p", ""),
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(encoded.encode("utf-8")).hexdigest()

    def _evaluate_single(
        self,
        idx: int,
        item: dict[str, Any],
        prompt: str,
        system_prompt: str | None = None,
    ) -> tuple[int, str, str, str, str, str]:
        query = self._build_query(prompt, item)
        cache_key = self._build_cache_key(item, prompt, query, system_prompt)

        if self.vote_count <= 1:
            with self._cache_lock:
                cached = self._inference_cache.get(cache_key)
                if cached is not None:
                    self._cache_hits += 1
                else:
                    self._cache_misses += 1
            if cached is not None:
                pred, out = cached
            else:
                out = self._call_worker(query, system_prompt=system_prompt, temperature=0.0)
                pred = self._parse_prediction(out)
                with self._cache_lock:
                    self._inference_cache[cache_key] = (pred, out)
        else:
            votes: list[str] = []
            for _ in range(self.vote_count):
                out = self._call_worker(query, system_prompt=system_prompt, temperature=0.3)
                votes.append(self._parse_prediction(out))
            pred = Counter(votes).most_common(1)[0][0]

        return idx, pred, item["label"], out, query, item["fields"].get(self.task_config.text_columns[0], "")[:30]

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

        precision_macro = precision_score(gt, preds, labels=all_labels, average="macro", zero_division=0)
        metrics: dict[str, float] = {
            "accuracy": accuracy_score(gt, preds),
            "f1": f1_score(gt, preds, labels=all_labels, average="macro", zero_division=0),
            "precision": precision_macro,
            "precision_macro": precision_macro,
            "recall": recall_score(gt, preds, labels=all_labels, average="macro", zero_division=0),
        }

        pos_label = getattr(self.task_config, "positive_label", "") or ""
        if not pos_label:
            if "是" in all_labels:
                pos_label = "是"
            elif len(all_labels) == 2:
                pos_label = all_labels[0]
        if pos_label:
            tp = 0
            fp = 0
            fn = 0
            for p, g in zip(preds, gt):
                if p == pos_label and g == pos_label:
                    tp += 1
                elif p == pos_label and g != pos_label:
                    fp += 1
                elif p != pos_label and g == pos_label:
                    fn += 1
            metrics["precision_pos"] = (tp / (tp + fp)) if (tp + fp) else 0.0
            metrics["recall_pos"] = (tp / (tp + fn)) if (tp + fn) else 0.0

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
