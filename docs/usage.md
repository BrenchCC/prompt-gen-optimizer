# Prompt Optimizer 使用文档

## 系统概览

Prompt Optimizer 是一个基于 Agent 的 Prompt 自动优化框架，通过 Worker 模型执行任务评测、Master 模型分析错误并增量改写 Prompt，实现自动化的 Prompt 迭代优化。

### 核心流程

```
标注数据 → Train/Val 抽样 → Worker 并发预测 → 收集错误 → Master 增量改写 → 评估新 Prompt → 保留/回滚
```

## 快速开始

### 1. 环境准备

```bash
# 使用 conda 环境
conda activate prompt_optimizer

# 配置 .env 文件（参考 .env.example）
cp .env.example .env
# 编辑 .env，填入真实的 API Key 和模型配置
```

### 2. 运行优化（shenping 任务示例）

```bash
python run_optimizer.py --config tasks/shenping/v1/config/test_v1.yaml
```

### 3. 命令行参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config` | **必填** YAML 配置文件路径 | `tasks/shenping/v1/config/test_v1.yaml` |
| `--iterations` | 覆盖迭代轮数 | `--iterations 10` |
| `--patience` | 覆盖 early stop 耐心值 | `--patience 3` |
| `--metric` | 覆盖主评估指标 | `--metric accuracy` |
| `--concurrency` | 覆盖 LLM 并发数 | `--concurrency 16` |
| `--seed` | 覆盖随机种子 | `--seed 123` |
| `--log-level` | 日志级别 | `--log-level DEBUG` |

## YAML 配置文件说明

每个任务通过独立的 YAML 文件配置。完整字段说明：

```yaml
task:
  name: shenping          # 任务名称
  version: v1             # 版本号
  type: classify          # 任务类型: classify | judge

data:
  file: path/to/data.xlsx # 标注数据文件（相对项目根目录）
  format: xlsx            # 数据格式: csv | xlsx
  text_columns:           # 输入文本列名（按顺序）
    - 文章
    - 评论
  label_column: 标签列    # 标签列名
  label_map:              # 标签值映射: 原始值 → 统一值
    是: "是"
    不是: "不是"
  label_descriptions:     # 标签含义（用于 Master 分析）
    是: 是神评
    不是: 不是神评

prompt:
  file: path/to/prompt.md # Prompt 文件路径
  variables:              # 模板变量映射: 占位符名 → 数据列名
    article_content: 文章
    comment_content: 评论

optimizer:
  iterations: 5           # 优化轮数
  patience: 2             # early stop 耐心值
  primary_metric: f1      # 主指标: accuracy | f1 | precision | recall
  train_sample_size: 60   # 每轮 train 抽样数
  val_sample_size: 30     # 每轮 valid 抽样数
  concurrency: 8          # LLM 并发数
  vote_count: 1           # 多数投票次数（1=不投票）
  max_retries: 3          # LLM 调用最大重试次数
  max_error_samples: 10   # 提供给 Master 的最大错误样本数
  seed: 42                # 随机种子

output:
  dir: path/to/output     # 输出目录
  results_file: results.json
  best_prompt_file: best_prompt.md
  best_score_file: best_score.json
  log_file: optimization_log.md

worker:                   # Worker 模型配置（覆盖 .env 默认值）
  mode: ark               # ark | openai
  temperature: 0.3
  top_p: 0.7
  reasoning_option: disabled

master:                   # Master 模型配置（覆盖 .env 默认值）
  mode: ark
  temperature: 0.7
  top_p: 0.95
  reasoning_option: disabled
```

## Python API

### 基本用法

```python
from prompt_optimizer import TaskConfig, PromptOptimizer

# 加载配置
config = TaskConfig.from_yaml("tasks/shenping/v1/config/test_v1.yaml")

# 创建优化器并运行
optimizer = PromptOptimizer(config)
result = optimizer.optimize()

# result 示例:
# {
#     "status": "completed",     # completed | perfect | error
#     "best_score": 0.85,
#     "best_metrics": {"accuracy": 0.9, "f1": 0.85, "precision": 0.88, "recall": 0.83},
#     "steps": 5
# }
```

### 单独使用 Evaluator

```python
from prompt_optimizer import TaskConfig, Evaluator

config = TaskConfig.from_yaml("tasks/shenping/v1/config/test_v1.yaml")
evaluator = Evaluator(config)

# 加载数据
data = config.load_data()

# 读取 prompt
system_prompt = config.read_prompt()

# 运行评测
preds = evaluator.run_prompt(
    prompt="dummy",           # 构建 query 用（shenping 任务中实际通过 variables 构建）
    dataset=data[:10],        # 子集
    desc="Test",
    system_prompt=system_prompt
)

# 计算指标
metrics, errors = evaluator.evaluate(preds, data[:10])
```

### 单独使用 TaskConfig

```python
from prompt_optimizer import TaskConfig

config = TaskConfig.from_yaml("tasks/shenping/v1/config/test_v1.yaml")

# 访问配置
print(config.task_name)        # "shenping"
print(config.all_labels)       # ["不是", "不确定", "是"]
print(config.primary_metric)   # "f1"

# 加载数据
data = config.load_data()      # [{"fields": {...}, "label": "是"}, ...]

# 读写 prompt
prompt = config.read_prompt()
config.write_prompt("new prompt content")
```

## 扩展新任务

### 1. 创建任务目录结构

```
tasks/
└── your_task/
    └── v1/
        ├── config/
        │   └── config.yaml    # 任务配置
        ├── data/
        │   └── data.xlsx      # 标注数据
        └── prompt/
            └── 001.md         # 初始 prompt
```

### 2. 编写 YAML 配置

复制 `tasks/shenping/v1/config/test_v1.yaml` 为模板，修改以下关键字段：
- `task.name` / `task.version`
- `data.*` — 数据文件、列名、标签映射
- `prompt.*` — prompt 文件和变量映射

### 3. 运行

```bash
python run_optimizer.py --config tasks/your_task/v1/config/config.yaml
```

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `results.json` | 每轮迭代的详细指标记录（JSON 数组） |
| `best_prompt.md` | 最优 prompt 文本 |
| `best_score.json` | 最优分数和对应指标 |
| `optimization_log.md` | 可读的实验日志（含每轮 prompt 和错误分析） |
| `prompt_comparison.md` | 原始 prompt 与最佳 prompt 的对比文件 |
| `final_evaluation.json` | 最终全量数据推理结果（原始/最佳 prompt 的指标对比 + 逐条预测详情） |

> **注意:** 优化过程不会覆盖原始 prompt 文件，最优 prompt 仅保存至 `output/best_prompt.md`。

## 项目结构

```
prompt-optimizer/
├── config.py                   # 全局配置（从 .env 读取）
├── run_optimizer.py            # CLI 入口
├── prompt_optimizer/           # 核心包
│   ├── __init__.py
│   ├── task_config.py          # 任务配置加载与验证
│   ├── evaluator.py            # LLM 评测与指标计算
│   ├── optimizer.py            # 主优化循环
│   └── master_prompt.py        # Master 模型 prompt 模板
├── utils/
│   ├── __init__.py
│   └── llm_server.py           # LLM 客户端封装
├── tasks/
│   └── shenping/v1/            # shenping 任务
├── tests/                      # 单元测试
└── docs/                       # 文档
```
