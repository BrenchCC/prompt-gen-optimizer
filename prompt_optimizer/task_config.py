# -*- coding: utf-8 -*-
import csv
import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml
from loguru import logger

import config as default_cfg


@dataclass
class TaskConfig:
    task_name: str = ""
    task_version: str = ""
    task_type: str = "classify"

    data_file: str = ""
    data_format: str = "csv"
    text_columns: List[str] = field(default_factory=list)
    label_column: str = ""
    label_map: Dict[str, str] = field(default_factory=dict)
    label_descriptions: Dict[str, str] = field(default_factory=dict)
    positive_label: str = ""
    output_field: str = ""
    output_map: Dict[str, str] = field(default_factory=dict)
    custom_parser: str = ""
    feedback_column: str = ""

    prompt_file: str = ""
    prompt_variables: Dict[str, str] = field(default_factory=dict)
    user_prompt_template: str = ""

    iterations: int = default_cfg.DEFAULT_ROUNDS
    patience: int = default_cfg.DEFAULT_PATIENCE
    primary_metric: str = default_cfg.DEFAULT_PRIMARY_METRIC
    train_sample_size: int = 0
    val_sample_size: int = 0
    concurrency: int = default_cfg.DEFAULT_CONCURRENCY
    vote_count: int = 1
    max_retries: int = 3
    max_error_samples: int = 10
    prompt_candidate_count: int = 3
    suggestion_similarity_threshold: float = 0.82
    suggestion_concurrency: int = 4
    suggestion_pool_dir: str = ""
    seed: int = default_cfg.DEFAULT_SEED

    output_dir: str = ""
    results_file: str = default_cfg.RESULTS_JSON_NAME
    best_prompt_file: str = default_cfg.BEST_PROMPT_NAME
    best_score_file: str = default_cfg.BEST_SCORE_NAME
    log_file: str = "optimization_log.md"

    worker_mode: str = "ark"
    worker_api_key: str = ""
    worker_base_url: str = ""
    worker_model_name: str = ""
    worker_model_identifier: str = ""
    worker_temperature: Optional[float] = None
    worker_top_p: Optional[float] = None
    worker_reasoning_option: str = "disabled"

    master_mode: str = "ark"
    master_api_key: str = ""
    master_base_url: str = ""
    master_model_name: str = ""
    master_model_identifier: str = ""
    master_temperature: Optional[float] = None
    master_top_p: Optional[float] = None
    master_reasoning_option: str = "disabled"

    project_root: str = ""

    @classmethod
    def from_yaml(cls, yaml_path: str, project_root: str = None) -> "TaskConfig":
        yaml_path = os.path.abspath(yaml_path)
        if project_root is None:
            project_root = str(Path(__file__).resolve().parent.parent)
        project_root = os.path.abspath(project_root)

        logger.info(f"加载任务配置: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        cfg = cls()
        cfg.project_root = project_root

        task_sec = raw.get("task", {})
        cfg.task_name = task_sec.get("name", cfg.task_name)
        cfg.task_version = task_sec.get("version", cfg.task_version)
        cfg.task_type = task_sec.get("type", cfg.task_type)

        data_sec = raw.get("data", {})
        cfg.data_file = cls._resolve(project_root, data_sec.get("file", ""))
        cfg.data_format = data_sec.get("format", cfg.data_format)
        cfg.text_columns = data_sec.get("text_columns", cfg.text_columns)
        cfg.label_column = data_sec.get("label_column", cfg.label_column)
        cfg.label_map = data_sec.get("label_map", cfg.label_map)
        cfg.label_descriptions = data_sec.get("label_descriptions", cfg.label_descriptions)
        cfg.positive_label = str(data_sec.get("positive_label", cfg.positive_label) or "").strip()
        cfg.output_field = data_sec.get("output_field", cfg.output_field)
        cfg.output_map = data_sec.get("output_map", cfg.output_map)
        cfg.custom_parser = data_sec.get("custom_parser", cfg.custom_parser)
        cfg.feedback_column = data_sec.get("feedback_column", cfg.feedback_column)

        prompt_sec = raw.get("prompt", {})
        cfg.prompt_file = cls._resolve(project_root, prompt_sec.get("file", ""))
        cfg.prompt_variables = prompt_sec.get("variables", cfg.prompt_variables)
        cfg.user_prompt_template = prompt_sec.get("user_prompt_template", "").strip()

        opt_sec = raw.get("optimizer", {})
        cfg.iterations = opt_sec.get("iterations", cfg.iterations)
        cfg.patience = opt_sec.get("patience", cfg.patience)
        cfg.primary_metric = opt_sec.get("primary_metric", cfg.primary_metric)
        cfg.train_sample_size = opt_sec.get("train_sample_size", cfg.train_sample_size)
        cfg.val_sample_size = opt_sec.get("val_sample_size", cfg.val_sample_size)
        cfg.concurrency = opt_sec.get("concurrency", cfg.concurrency)
        cfg.vote_count = opt_sec.get("vote_count", cfg.vote_count)
        cfg.max_retries = opt_sec.get("max_retries", cfg.max_retries)
        cfg.max_error_samples = opt_sec.get("max_error_samples", cfg.max_error_samples)
        cfg.prompt_candidate_count = opt_sec.get("prompt_candidate_count", cfg.prompt_candidate_count)
        cfg.suggestion_similarity_threshold = opt_sec.get(
            "suggestion_similarity_threshold",
            cfg.suggestion_similarity_threshold,
        )
        cfg.suggestion_concurrency = opt_sec.get(
            "suggestion_concurrency",
            cfg.suggestion_concurrency,
        )
        cfg.suggestion_pool_dir = cls._resolve(project_root, opt_sec.get("suggestion_pool_dir", ""))
        cfg.seed = opt_sec.get("seed", cfg.seed)

        out_sec = raw.get("output", {})
        cfg.output_dir = cls._resolve(project_root, out_sec.get("dir", ""))
        cfg.results_file = out_sec.get("results_file", cfg.results_file)
        cfg.best_prompt_file = out_sec.get("best_prompt_file", cfg.best_prompt_file)
        cfg.best_score_file = out_sec.get("best_score_file", cfg.best_score_file)
        cfg.log_file = out_sec.get("log_file", cfg.log_file)

        worker_sec = raw.get("worker", {})
        cfg.worker_mode = worker_sec.get("mode", cfg.worker_mode)
        cfg.worker_api_key = worker_sec.get("api_key", "") or default_cfg.WORKER_API_KEY
        cfg.worker_base_url = worker_sec.get("base_url", "") or default_cfg.WORKER_BASE_URL
        cfg.worker_model_name = (
            worker_sec.get("name")
            or worker_sec.get("model_name")
            or default_cfg.WORKER_MODEL_NAME
        )
        cfg.worker_model_identifier = (
            worker_sec.get("identifier")
            or worker_sec.get("model_identifier")
            or default_cfg.WORKER_MODEL_IDENTIFIER
        )
        cfg.worker_temperature = worker_sec.get("temperature", default_cfg.WORKER_TEMPERATURE)
        cfg.worker_top_p = worker_sec.get("top_p", default_cfg.WORKER_TOP_P)
        cfg.worker_reasoning_option = worker_sec.get("reasoning_option", default_cfg.WORKER_THINKING)

        master_sec = raw.get("master", {})
        cfg.master_mode = master_sec.get("mode", cfg.master_mode)
        cfg.master_api_key = master_sec.get("api_key", "") or default_cfg.MASTER_API_KEY
        cfg.master_base_url = master_sec.get("base_url", "") or default_cfg.MASTER_BASE_URL
        cfg.master_model_name = (
            master_sec.get("name")
            or master_sec.get("model_name")
            or default_cfg.MASTER_MODEL_NAME
        )
        cfg.master_model_identifier = (
            master_sec.get("identifier")
            or master_sec.get("model_identifier")
            or default_cfg.MASTER_MODEL_IDENTIFIER
        )
        cfg.master_temperature = master_sec.get("temperature", default_cfg.MASTER_TEMPERATURE)
        cfg.master_top_p = master_sec.get("top_p", default_cfg.MASTER_TOP_P)
        cfg.master_reasoning_option = master_sec.get("reasoning_option", default_cfg.MASTER_THINKING)

        cfg._validate()
        logger.info(f"配置加载完成: task={cfg.task_name}/{cfg.task_version}, type={cfg.task_type}")
        return cfg

    @staticmethod
    def _resolve(project_root: str, rel_path: str) -> str:
        if not rel_path:
            return ""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.normpath(os.path.join(project_root, rel_path))

    @staticmethod
    def _is_reasoning_enabled(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        normalized = str(value or "").strip().lower()
        if not normalized:
            return False
        return normalized not in {"disabled", "off", "false", "none", "0"}

    def _validate(self) -> None:
        if self.task_type not in ("classify", "judge"):
            raise ValueError(f"不支持的任务类型: {self.task_type}")

        if self.primary_metric not in ("accuracy", "f1", "precision", "precision_pos", "recall"):
            raise ValueError("primary_metric 必须是 accuracy/f1/precision/precision_pos/recall 之一")

        if self.data_format not in ("csv", "xlsx"):
            raise ValueError(f"不支持的数据格式: {self.data_format}")

        if not self.data_file:
            raise ValueError("未指定数据文件路径")

        if not self.text_columns:
            raise ValueError("未指定 text_columns")

        if not self.label_column:
            raise ValueError("未指定 label_column")

        if not self.label_map:
            raise ValueError("未指定 label_map")

        if self.positive_label and self.positive_label not in set(self.label_map.values()):
            raise ValueError("positive_label 必须是 label_map 的值之一")

        if self.primary_metric == "precision_pos":
            labels = set(self.label_map.values())
            if not self.positive_label and "是" not in labels and len(labels) != 2:
                raise ValueError("primary_metric=precision_pos 时需配置 data.positive_label（或标签中包含“是”，或为二分类）")

        if not 1 <= self.max_error_samples <= 20:
            raise ValueError("max_error_samples 必须在 1 到 20 之间")

        if self.prompt_candidate_count < 1:
            raise ValueError("prompt_candidate_count 必须大于等于 1")

        if not 0 < self.suggestion_similarity_threshold < 1:
            raise ValueError("suggestion_similarity_threshold 必须在 0 到 1 之间")

        if self.suggestion_concurrency < 1:
            raise ValueError("suggestion_concurrency 必须大于等于 1")

        if not self.suggestion_pool_dir:
            raise ValueError("未指定 suggestion_pool_dir")

        if not self._is_reasoning_enabled(self.master_reasoning_option):
            raise ValueError("master.reasoning_option 必须开启思考")

    def ensure_output_dir(self) -> str:
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir

    def ensure_prompt_dir(self) -> str:
        prompt_dir = os.path.dirname(self.prompt_file)
        if prompt_dir:
            os.makedirs(prompt_dir, exist_ok=True)
        return prompt_dir

    @property
    def output_results_path(self) -> str:
        return os.path.join(self.output_dir, self.results_file)

    @property
    def output_best_prompt_path(self) -> str:
        return os.path.join(self.output_dir, self.best_prompt_file)

    @property
    def output_best_score_path(self) -> str:
        return os.path.join(self.output_dir, self.best_score_file)

    @property
    def output_log_path(self) -> str:
        return os.path.join(self.output_dir, self.log_file)

    @property
    def output_comparison_path(self) -> str:
        return os.path.join(self.output_dir, "prompt_comparison.md")

    @property
    def output_task_description_path(self) -> str:
        return os.path.join(self.output_dir, "task_description.txt")

    @property
    def output_final_eval_path(self) -> str:
        return os.path.join(self.output_dir, "final_evaluation.json")

    @property
    def output_config_path(self) -> str:
        return os.path.join(self.output_dir, "config.json")

    @property
    def output_master_log_path(self) -> str:
        return os.path.join(self.output_dir, "master_log.md")

    @property
    def output_worker_prompt_log_path(self) -> str:
        return os.path.join(self.output_dir, "worker_prompt_log.md")

    @property
    def suggestion_store_dir(self) -> str:
        return self.suggestion_pool_dir

    @property
    def suggestion_pool_path(self) -> str:
        return os.path.join(self.suggestion_store_dir, "suggestion_pool.json")

    @property
    def suggestion_snapshots_path(self) -> str:
        return os.path.join(self.suggestion_store_dir, "suggestion_snapshots.json")

    @property
    def suggestion_index_path(self) -> str:
        return os.path.join(self.suggestion_store_dir, "suggestion_index.json")

    @property
    def all_labels(self) -> List[str]:
        return sorted(set(self.label_map.values()))

    def load_data(self) -> List[Dict[str, Any]]:
        path = self.data_file
        if not os.path.exists(path):
            logger.warning(f"数据文件不存在: {path}")
            return []

        if self.data_format == "xlsx":
            return self._load_xlsx(path)
        return self._load_csv(path)

    def _load_csv(self, path: str) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = self._parse_row(row)
                if item is not None:
                    data.append(item)
        logger.info(f"CSV 数据加载完成: {len(data)} 条, 文件={path}")
        return data

    def _load_xlsx(self, path: str) -> List[Dict[str, Any]]:
        import pandas as pd

        df = pd.read_excel(path, engine="openpyxl")
        data: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            row_dict = {str(k): str(v) if pd.notna(v) else "" for k, v in row.items()}
            item = self._parse_row(row_dict)
            if item is not None:
                data.append(item)
        logger.info(f"XLSX 数据加载完成: {len(data)} 条, 文件={path}")
        return data

    def _parse_row(self, row: Dict[str, str]) -> Optional[Dict[str, Any]]:
        fields: Dict[str, str] = {}
        for col in self.text_columns:
            val = row.get(col, "").strip()
            if not val and col == self.text_columns[0]:
                return None
            fields[col] = val

        label_str = row.get(self.label_column, "").strip()
        label = self.label_map.get(label_str, label_str)

        if self.task_type == "classify":
            fields["text"] = fields.get(self.text_columns[0], "")

        item: Dict[str, Any] = {"fields": fields, "label": label}

        if self.feedback_column:
            fb = row.get(self.feedback_column, "").strip()
            if fb:
                item["feedback"] = fb

        return item

    def load_custom_parser(self) -> Optional[Callable[[str], str]]:
        if not self.custom_parser:
            return None
        try:
            module_path, func_name = self.custom_parser.rsplit(".", 1)
            mod = importlib.import_module(module_path)
            func = getattr(mod, func_name)
            logger.info(f"已加载自定义解析函数: {self.custom_parser}")
            return func
        except Exception as e:
            logger.error(f"加载自定义解析函数失败 ({self.custom_parser}): {e}")
            return None

    def build_user_prompt(self, fields: Dict[str, str]) -> str:
        if self.user_prompt_template:
            result = self.user_prompt_template
            for var_name, col_name in self.prompt_variables.items():
                placeholder = "{{" + var_name + "}}"
                result = result.replace(placeholder, fields.get(col_name, ""))
            return result.strip()

        if self.task_type == "judge":
            parts: List[str] = []
            for var_name, col_name in self.prompt_variables.items():
                parts.append(f"{var_name}: {fields.get(col_name, '')}")
            return "\n".join(parts).strip()

        return fields.get("text", fields.get(self.text_columns[0], "")).strip()

    def read_prompt(self) -> str:
        if not os.path.exists(self.prompt_file):
            logger.warning(f"Prompt 文件不存在: {self.prompt_file}")
            return ""
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
        logger.info(f"Prompt 已读取: {self.prompt_file} ({len(content)} 字符)")
        return content

    def write_prompt(self, content: str) -> None:
        self.ensure_prompt_dir()
        with open(self.prompt_file, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Prompt 已写入: {self.prompt_file} ({len(content)} 字符)")
