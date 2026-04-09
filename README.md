# Prompt Optimizer

基于 Agent 的 Prompt 自动优化框架。当前版本采用“固定评测集 → Worker 评测 → 批量错误经验提炼 → 建议池沉淀 → Master 多候选改写 → 两阶段选优”的多阶段优化架构，实现 Prompt 的自动化迭代优化。

## 核心流程

```text
标注数据
  -> 固定 Train/Valid 抽样（单次抽样，整轮复用）
  -> Baseline 评测（Train + Valid）
  -> 采样错误样本
  -> Master 批量提炼 2-4 条系统级建议
  -> 建议池召回 / 去重 / 灰区复核
  -> Master 生成多个候选 Prompt（不同 patch_focus）
  -> 候选 Valid 评测
  -> 赢家补跑 Train
  -> Valid 提升则保留，否则回滚到 Best Prompt
  -> Early Stop / 满分停止
  -> 全量数据最终评测（原始 vs 最佳，输出 json + xlsx）
```

### 关键设计

- **固定验证集** — 优化开始时一次性抽取 Train/Valid，整轮复用，避免指标波动误导 keep/discard
- **两阶段评测** — 每轮候选先只跑 Valid，只有赢家补跑 Train，减少无效 Worker 调用
- **多候选搜索** — Master 每轮生成多个候选 Prompt，在同一验证集上对比选优，降低局部最优风险
- **错误驱动优化** — Master 模型基于 Worker 的真实错误样本分析共性问题，而非凭空改写
- **建议池沉淀** — 系统级建议沉淀到任务级建议池，支持召回、去重、版本化与历史快照
- **缓存复用** — 同一轮内对相同 Prompt/样本的 Worker 推理走缓存，减少重复请求
- **自动回滚** — 验证集指标未提升则回滚到历史最优 Prompt
- **Early Stop** — 连续 N 轮无提升自动终止，避免无效计算
- **不覆盖原始 Prompt** — 原始 Prompt 文件始终不变，最优结果保存到 output 目录

## 项目结构

```
prompt-optimizer/
├── config.py                       # 全局配置（从 .env 读取模型密钥和默认参数）
├── run_optimizer.py                # CLI 入口脚本
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量模板
│
├── prompt_optimizer/               # 核心优化器包
│   ├── __init__.py
│   ├── task_config.py              # YAML 任务配置加载与验证
│   ├── evaluator.py                # LLM 评测：并发推理、结果解析、指标计算
│   ├── optimizer.py                # 主优化循环：baseline → 迭代改写 → early stop → 全量评测
│   ├── master_prompt.py            # Master 模型 Prompt 模板构建
│   └── suggestion_pool.py          # 建议池：去重、版本控制、索引、快照
│
├── utils/
│   ├── __init__.py
│   └── llm_server.py               # LLM 客户端封装（支持 Ark / OpenAI 双模式）
│
├── tasks/                          # 任务目录（按 task/version 组织）
│   └── shenping/v1/                # 示例：神评判断任务
│       ├── config/test_v1.yaml     # 任务配置
│       ├── data/v1-100case.xlsx    # 标注数据
│       ├── prompt/001.md           # 初始 Prompt
│       ├── output-v1-100/          # 优化结果输出
│       └── artifacts/suggestions/  # 任务级建议池
│
├── tests/                          # 单元测试
│   ├── test_task_config.py
│   ├── test_evaluator.py
│   ├── test_master_prompt.py
│   ├── test_optimizer.py
│   └── test_run_optimizer.py
│
└── docs/
    └── usage.md                    # 详细 API 使用文档
```

## 快速开始

### 1. 环境配置

```bash
# 激活 conda 环境
conda activate prompt_optimizer

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入真实的 API Key 和模型端点
```

### 2. 环境变量说明

`.env` 文件配置两组模型：

```bash
# Worker 模型 — 执行任务推理（建议用快速/低成本模型）
WORKER_API_KEY=your_worker_api_key
WORKER_BASE_URL=https://ark-cn-beijing.bytedance.net/api/v3
WORKER_MODEL_NAME=your-worker-endpoint
WORKER_THINKING=disabled

# Master 模型 — 错误分析 + Prompt 改写（建议用强推理模型）
MASTER_API_KEY=your_master_api_key
MASTER_BASE_URL=https://ark-cn-beijing.bytedance.net/api/v3
MASTER_MODEL_NAME=your-master-endpoint
MASTER_THINKING=enabled
```

### 3. 运行优化

```bash
# 使用默认配置运行 shenping 任务
python run_optimizer.py --config tasks/shenping/v1/config/test_v1.yaml

# 覆盖部分参数
python run_optimizer.py \
    --config tasks/shenping/v1/config/test_v1.yaml \
    --iterations 10 \
    --patience 3 \
    --metric precision_pos \
    --concurrency 16

# 调整日志级别
python run_optimizer.py --config tasks/shenping/v1/config/test_v1.yaml --log-level DEBUG
```

### 4. CLI 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--config` | str | **必填**，YAML 配置文件路径 |
| `--iterations` | int | 覆盖迭代轮数 |
| `--patience` | int | 覆盖 early stop 耐心值 |
| `--metric` | str | 覆盖主指标（`accuracy` / `f1` / `precision` / `precision_pos` / `recall`） |
| `--concurrency` | int | 覆盖 LLM 并发数 |
| `--seed` | int | 覆盖随机种子 |
| `--log-level` | str | 日志级别（`DEBUG` / `INFO` / `WARNING` / `ERROR`） |

## YAML 配置文件

每个任务通过独立的 YAML 文件进行完整配置，示例：

```yaml
task:
  name: shenping                        # 任务名称
  version: v1                           # 版本号
  type: classify                        # 任务类型: classify | judge

data:
  file: tasks/shenping/v1/data/v1-100case.xlsx
  format: xlsx                          # csv | xlsx
  text_columns:                         # 输入文本列名
    - 文章
    - 评论
  label_column: 人工打标是否神评          # 标签列名
  label_map:                            # 原始标签 → 统一标签
    是: "是"
    不是: "不是"
    不确定: "是"
  label_descriptions:                   # 标签语义（用于 Master 分析）
    是: 是神评
    不是: 不是神评
  positive_label: "是"                  # 正类标签，用于 precision_pos 等指标
  feedback_column: 人工备注               # （可选）提供给 Master 的错误原因/反馈列
  output_field: is_shenping             # LLM 输出 JSON 中的分类结果字段名
  output_map:                           # 输出字段值 → 标准标签
    "true": "是"
    "false": "不是"

prompt:
  file: tasks/shenping/v1/prompt/001.md # 初始 Prompt 文件
  variables:                            # 变量映射: 占位符名 → 数据列名
    article_content: 文章
    comment_content: 评论
  user_prompt_template: |               # 发送给 Worker 的 user_prompt 模板
    原文内容：{{article_content}}
    评论内容：{{comment_content}}

optimizer:
  iterations: 10                        # 优化轮数（不含 baseline）
  patience: 3                           # early stop 耐心值
  primary_metric: precision_pos         # 主指标
  train_sample_size: 65                 # 每轮 train 抽样数
  val_sample_size: 35                   # 每轮 valid 抽样数
  prompt_candidate_count: 3             # 每轮生成的候选 Prompt 数
  concurrency: 8                        # LLM 并发数
  vote_count: 1                         # 多数投票次数（1=不投票）
  max_retries: 3                        # LLM 调用最大重试次数
  max_error_samples: 10                 # 提供给 Master 的最大错误样本数
  suggestion_similarity_threshold: 0.82 # 建议池语义去重阈值
  suggestion_concurrency: 4             # 单样本建议分析并发数
  suggestion_pool_dir: tasks/shenping/v1/artifacts/suggestions
  seed: 42                              # 随机种子

output:
  dir: tasks/shenping/v1/output
  results_file: results.json
  best_prompt_file: best_prompt.md
  best_score_file: best_score.json
  log_file: optimization_log.md

worker:                                 # Worker 模型参数（覆盖 .env）
  mode: ark                             # ark | openai
  temperature: 0.3
  top_p: 0.7
  reasoning_option: disabled

master:                                 # Master 模型参数（覆盖 .env）
  mode: ark
  temperature: 0.7
  top_p: 0.95
  reasoning_option: enabled             # 必须开启思考
```

### user_prompt_template 说明

`user_prompt_template` 定义了发送给 Worker 模型的用户输入格式，使用 `{{变量名}}` 引用 `variables` 中的映射关系。这使得不同任务可以完全通过配置定义输入格式，无需修改代码。

## 输出文件

优化完成后，所有结果保存到配置的 `output.dir` 目录：

| 文件 | 说明 |
|------|------|
| `results.json` | 每轮迭代的详细指标记录（含候选评分、选中候选、缓存统计） |
| `best_prompt.md` | 最优 Prompt 文本 |
| `best_score.json` | 最优分数及对应指标 |
| `optimization_log.md` | 可读的实验日志（每轮 Prompt + 指标 + 错误分析） |
| `prompt_comparison.md` | 原始 Prompt 与最佳 Prompt 的对比文件 |
| `worker_prompt_log.md` | 每次 Worker system prompt 的评测记录 |
| `master_log.md` | 每次调用 Master 模型的 system/user 输入、单样本建议调用、去重决策与原始输出 |
| `final_evaluation.json` | 全量数据最终评测的汇总指标（原始 vs 最佳） |
| `final_evaluation.xlsx` | 全量数据逐条推理详情（含 JSON 字段拆分） |

任务级建议池会额外保存在 `tasks/{task}/{version}/artifacts/suggestions/` 下：

| 文件 | 说明 |
|------|------|
| `suggestion_pool.json` | 当前活跃建议池与建议级历史 |
| `suggestion_index.json` | 关键词索引与文本指纹索引 |
| `suggestion_snapshots.json` | 每轮优化后的池级快照 |

### final_evaluation.xlsx 列说明

| 列 | 说明 |
|----|------|
| `index` | 数据序号 |
| *数据字段列* | 原始数据的所有字段（完整内容，不截断） |
| `label` | 人工标注标签 |
| `original_pred` | 原始 Prompt 的预测结果 |
| `original_correct` | 原始 Prompt 是否预测正确 |
| `original_*` | 原始 Prompt LLM 输出 JSON 的拆分字段 |
| `best_pred` | 最佳 Prompt 的预测结果 |
| `best_correct` | 最佳 Prompt 是否预测正确 |
| `best_*` | 最佳 Prompt LLM 输出 JSON 的拆分字段 |

> 所有列均以字符串形式存储，防止 pandas 类型推断导致数据截断或溢出。
> LLM 输出的 JSON 字段会自动拆分为独立列，适用于任何任务的输出格式。

## Python API

### 基本用法

```python
from prompt_optimizer import TaskConfig, PromptOptimizer

config = TaskConfig.from_yaml("tasks/shenping/v1/config/test_v1.yaml")
optimizer = PromptOptimizer(config)
result = optimizer.optimize()

# result:
# {
#     "status": "completed",       # completed | perfect | error
#     "best_score": 0.85,
#     "best_metrics": {"accuracy": 0.9, "f1": 0.85, "precision": 0.88, "recall": 0.83},
#     "steps": 5,
#     "final_evaluation": { ... }  # 全量评测结果
# }
```

### 单独使用 Evaluator

```python
from prompt_optimizer import TaskConfig, Evaluator

config = TaskConfig.from_yaml("tasks/shenping/v1/config/test_v1.yaml")
evaluator = Evaluator(config)

data = config.load_data()
system_prompt = config.read_prompt()

preds = evaluator.run_prompt(
    prompt="dummy", dataset=data[:10],
    desc="Test", system_prompt=system_prompt
)
metrics, errors = evaluator.evaluate(preds, data[:10])
```

### 单独使用 TaskConfig

```python
from prompt_optimizer import TaskConfig

config = TaskConfig.from_yaml("tasks/shenping/v1/config/test_v1.yaml")

# 访问配置
print(config.task_name)        # "shenping"
print(config.all_labels)       # ["不是", "是"]
print(config.primary_metric)   # "precision_pos"

# 加载数据
data = config.load_data()      # [{"fields": {...}, "label": "是"}, ...]

# 构建 user_prompt
fields = {"文章": "...", "评论": "..."}
user_prompt = config.build_user_prompt(fields)

# 读取 Prompt
prompt = config.read_prompt()
```

## 扩展新任务

### 1. 创建任务目录

```
tasks/
└── your_task/
    └── v1/
        ├── config/
        │   └── config.yaml     # 任务配置
        ├── data/
        │   └── data.xlsx       # 标注数据
        └── prompt/
            └── 001.md          # 初始 Prompt
```

### 2. 编写 YAML 配置

以 `tasks/shenping/v1/config/test_v1.yaml` 为模板，修改：

- `task.*` — 任务名称和类型
- `data.*` — 数据文件路径、列名、标签映射
- `prompt.*` — Prompt 文件路径、变量映射、user_prompt_template
- `optimizer.*` — 优化参数

### 3. 运行

```bash
python run_optimizer.py --config tasks/your_task/v1/config/config.yaml
```

## 运行测试

```bash
conda activate prompt_optimizer
python -m pytest tests/ -v
```

## 双模型架构

| 角色 | 用途 | 建议配置 |
|------|------|----------|
| **Worker** | 执行任务推理（分类/评测） | 快速低成本模型，低温度，关闭思考模式 |
| **Master** | 批量错误分析、建议去重复核、候选 Prompt 生成 | 强推理模型，较高温度，必须开启思考模式 |

两组模型通过 `.env` 文件配置 API 密钥和端点，YAML 中可覆盖 `temperature`、`top_p`、`reasoning_option` 等推理参数。LLM 调用通过 `utils/llm_server.py` 封装，支持 Ark（火山引擎）和 OpenAI 两种后端。

## 优化架构

### 分层设计

| 层级 | 核心职责 | 关键产物 |
|------|----------|----------|
| **评测层** | Worker 对固定 train/valid 集做并发推理，候选阶段先 Valid、赢家再 Train | 指标、错误样本、缓存统计 |
| **建议生成层** | 针对一批错误样本调用 Master，生成 2-4 条结构化 suggestion | 批量建议、错误经验总结 |
| **建议池层** | 对 suggestion 做召回、去重、灰区复核、版本控制、快照维护 | `suggestion_pool.json` / `suggestion_index.json` / `suggestion_snapshots.json` |
| **Prompt 改写层** | 将长期 merged suggestions 注入 Master system_prompt，把本轮错误样本和新增建议注入 user prompt | 候选 Prompt 列表 |
| **反馈闭环层** | 根据本轮 valid 指标变化回写建议有效性，更新 `positive_hits / negative_hits / effectiveness_score` | 建议效果历史 |

### 调用链路

1. Worker 在固定 train/valid 集下完成 baseline 推理，并收集错误样本。
2. 从错误样本中按 `max_error_samples` 采样，调用 Master 生成 2-4 条结构化 suggestion 和错误经验总结。
3. 新 suggestion 写入任务级建议池，先做关键词/类别召回，再对灰区候选做 LLM 复核。
4. 建议池输出去重后的系统级建议摘要，作为 Master 的长期 system_prompt。
5. 当前轮错误样本、实验历史、当前 prompt、本轮新增建议摘要作为 Master 的 user prompt。
6. Master 输出多个增量优化候选 Prompt；系统先在固定 valid 集上比较，选出赢家后再补跑 train。
7. 若赢家 valid 指标未提升，则回滚到历史最佳 prompt；否则更新 best prompt。
8. 本轮被采纳的建议根据指标变化写回效果分数，并生成池级快照。

### 关键约束

- **固定对比基准**：同一轮实验中的候选 Prompt 必须在同一 valid 集上比较。
- **系统级建议沉淀**：原始错误样本不直接进入长期 system_prompt，进入的是合并后的系统级建议。
- **可追溯性**：建议池保留 `source_samples`、`source_suggestions`、`merged_from`、版本号和快照。
- **配置化去重**：语义去重阈值通过 `suggestion_similarity_threshold` 控制。
- **Master 强约束**：`master.reasoning_option` 必须开启，保证错误分析与候选生成稳定性。
