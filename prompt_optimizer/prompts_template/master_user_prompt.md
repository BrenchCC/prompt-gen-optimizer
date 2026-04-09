## 目标
对 <current_prompt> 做小幅增量修改，修复本轮错误并提升 {primary}。只输出修改后的完整 prompt，不要输出分析过程。

## 任务背景
{task_desc}

## 近几轮实验历史
{history_str}

## 当前 Prompt（待改进）
<current_prompt>
{current_prompt}
</current_prompt>

## 当前表现
- Train {primary} = {train_score:.4f}
- Valid {primary} = {val_score:.4f}
- 错误样本概览：总计 {total_errors} 条，本轮已独立分析 {shown_errors} 条并提炼为优化建议。

## 本轮错误经验与建议
### 错误经验反馈（已抽象）
{experience_feedback}

### 建议摘要（本轮新增）
{round_suggestions}

## 改进方法论（不要输出）
### Step 1：错误模式聚类
基于错误经验反馈 + 建议摘要 + 历史趋势，选出 2-4 个最高优先级的错误模式。

### Step 2：逐一归因定位
在当前 prompt 中定位对应条款或缺口：缺失/冲突/边界模糊/覆盖不足。

### Step 3：增量打补丁
只做小幅增补与边界澄清，保持主体结构与占位符不变。

### Step 4：反向验证
心理模拟：是否修复本轮问题；是否可能误伤之前正确的样本。
