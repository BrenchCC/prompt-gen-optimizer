# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys

from loguru import logger

from prompt_optimizer.task_config import TaskConfig
from prompt_optimizer.optimizer import PromptOptimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt Optimizer CLI")
    parser.add_argument("--config", type=str, required=True, help="任务 YAML 配置文件路径")
    parser.add_argument("--iterations", type=int, default=None, help="覆盖优化迭代轮数")
    parser.add_argument("--patience", type=int, default=None, help="覆盖 early stop 耐心值")
    parser.add_argument("--metric", type=str, default=None,
                        choices=["accuracy", "f1", "precision", "precision_pos", "recall"], help="覆盖主评估指标")
    parser.add_argument("--concurrency", type=int, default=None, help="覆盖 LLM 并发数")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    task_config = TaskConfig.from_yaml(args.config)

    if args.iterations is not None:
        task_config.iterations = args.iterations
    if args.patience is not None:
        task_config.patience = args.patience
    if args.metric is not None:
        task_config.primary_metric = args.metric
    if args.concurrency is not None:
        task_config.concurrency = args.concurrency
    if args.seed is not None:
        task_config.seed = args.seed

    optimizer = PromptOptimizer(task_config)
    result = optimizer.optimize()

    logger.info(f"优化结果: {result}")


if __name__ == "__main__":
    main()
