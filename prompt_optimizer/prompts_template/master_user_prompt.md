## 目标
对 <current_prompt> 做小幅增量修改，修复本轮错误并提升 {primary}。输出 {candidate_count} 个候选 Prompt，供同一验证集比较。

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

## 候选生成要求
1. 输出严格遵守 system prompt 里的 JSON 结构。
2. `candidates` 长度必须等于 {candidate_count}。
3. 每个候选都只做增量补丁，不可推倒重写。
4. 每个候选必须有不同的 `patch_focus`，示例方向可包括：
   - 边界澄清
   - 冲突优先级
   - 输出约束收紧
   - 低价值规则删减
5. 优先把规则写短、写直接、写成高优先级列表，不要保留重复表述。
6. 只保留对当前任务判定真正有帮助的规则；重复、空泛、互相打架的表述要删掉或合并。
7. 心理模拟后再输出：既要修复本轮问题，也要尽量降低对历史有效样本的误伤。
