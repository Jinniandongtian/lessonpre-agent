# 向量存储功能设计方案

## 问题1：存储结构如何设计？

### 最终方案：双表示法 + 分层元数据

```json
{
  "id": "q_2025_shandong_alliance_g2_math_001",

  // ✅ 唯一用于向量化的字段：移除LaTeX标记和标点，合并题干+选项
  "embedding_text": "已知直线l的斜率为2经过点2,1则直线l的方程为2x-y-5=0或2x+y-5=0或2x-y-3=0或2x+y-3=0",

  // ✅ 双表示法：LaTeX用于渲染，plain用于评测对比
  "content": {
    "stem_latex": "已知直线 $l$ 的斜率为 $2$，经过点 $(2,1)$，则直线 $l$ 的方程为",
    "stem_plain": "已知直线 l 的斜率为 2，经过点 (2,1)，则直线 l 的方程为",
    "options_latex": {
      "A": "$2x-y-5=0$",
      "B": "$2x+y-5=0$",
      "C": "$2x-y-3=0$",
      "D": "$2x+y-3=0$"
    },
    "options_plain": {
      "A": "2x-y-5=0",
      "B": "2x+y-5=0",
      "C": "2x-y-3=0",
      "D": "2x+y-3=0"
    }
  },

  // ✅ 一级过滤：按来源查询题目（地区、年份、年级等）
  "source_meta": {
    "region": "山东省",
    "year": 2025,
    "grade": "高二",
    "exam_name": "山东名校联盟期中",
    "source_type": "期中"
  },

  // ✅ 二级过滤：按题目属性筛选（题型、难度、知识点）
  "question_meta": {
    "question_type": "单选题",
    "difficulty": 3,
    "knowledge_points": ["直线方程", "点斜式"]
  },

  // ✅ 可选：自然语言描述，增强RAG语义理解
  "description": "求过点(2,1)斜率为2的直线方程，考查直线的点斜式"
}
```

### 各字段职责

| 字段 | 用途 | 是否参与向量化 |
|------|------|:-----------:|
| `embedding_text` | 向量化的唯一输入 | ✅ 是 |
| `content.stem_latex` + `options_latex` | PDF讲义渲染 | ❌ 否 |
| `content.stem_plain` + `options_plain` | 评测时字符级相似度对比 | ❌ 否 |
| `source_meta` | 按来源一级过滤（地区/年份/年级） | ❌ 否 |
| `question_meta` | 按属性二级过滤（题型/难度/知识点） | ❌ 否 |
| `description` | 增强RAG语义理解（可选） | ❌ 否 |

> `source_meta` 和 `question_meta` 只用于检索过滤，不参与向量计算。
> 时间戳、embedding模型名称、向量维度等信息记录在 `import_history.jsonl` 中，不存入题目记录。

### 存储层次

```
data/vector_db/
├── index.faiss              # FAISS向量索引（只存向量）
├── metadata.json            # 题目元数据（上述JSON结构）
└── import_history.jsonl     # 每次导入记录（时间戳、模型、维度、数量等）
```

`import_history.jsonl` 单条记录示例：

```json
{
  "timestamp": "2025-03-20T10:30:00Z",
  "pdf_file": "shandong_alliance_2025_g2_math.pdf",
  "embedding_model": "Qwen/Qwen3-Embedding-8B",
  "embedding_dim": 1024,
  "questions_added": 50,
  "questions_skipped": 3
}
```

---

## 问题2：识别试卷后先生成普通格式再转换成LaTeX，还是反过来？

### 推荐方案：先生成LaTeX，再转换为普通格式

#### 理由

1. **LaTeX是规范化的中间表示**
   - 数学公式有标准写法：`\frac{x}{y}` 而不是 `x/y`
   - 向量符号统一：`\overrightarrow{AB}` 而不是 `vec(AB)`

2. **OCR/LLM输出通常是混乱的**
   - 视觉模型识别出的公式可能是 `x²/4` 或 `x^2/4` 或 `x2/4`
   - 先转LaTeX可以统一这些变体

3. **LaTeX→普通文本是单向的，可逆性好**
   - LaTeX → 普通文本：`\frac{x}{y}` → `x/y`（简单替换规则）
   - 普通文本 → LaTeX：`x/y` → `\frac{x}{y}`？（歧义多，难以自动化）

#### 实现流程

```
PDF/OCR输出（格式混乱）
    ↓
LLM规范化为LaTeX（结构化）
    ↓
保存 stem_latex, options_latex
    ↓
规则转换为普通文本
    ↓
保存 stem_plain, options_plain
    ↓
生成 embedding_text（去除标记和标点）
    ↓
向量化存储
```

---

## 问题3：如何将LaTeX格式规范化，将不同写法统一？

### 推荐方案：LLM规范化 + 正则后处理

#### 规范化表

| 变体 | 标准形式 | 说明 |
|------|--------|------|
| `\dfrac{x}{y}` | `\frac{x}{y}` | 分数 |
| `x^2` | `x^{2}` | 上标 |
| `\times` | `\cdot` | 乘号 |
| `/`（数学模式中） | `\div` | 除号 |
| `\left( \right)` | `( )` | 括号 |
| `\vec{AB}` | `\overrightarrow{AB}` | 向量 |

#### LLM规范化提示词

```
将以下LaTeX公式规范化为标准形式。规范化规则：

1. 分数统一用 \frac{分子}{分母}（不用 \dfrac, \tfrac）
2. 上标统一用 x^{2}（不用 x^2）
3. 乘号统一用 \cdot（不用 *, ×, \times）
4. 除号统一用 \div（不用 /）
5. 根号统一用 \sqrt{x}
6. 括号统一用 ( ) 而不是 \left( \right)
7. 向量统一用 \overrightarrow{AB}
8. 移除多余空格

原LaTeX：{latex_text}

输出：规范化后的LaTeX（仅输出公式，不要解释）
```

#### 正则后处理（兜底）

LLM调用失败时的降级方案，处理最常见的变体：

```python
def post_process_latex(text: str) -> str:
    text = re.sub(r'\\[dt]frac', r'\\frac', text)          # dfrac/tfrac → frac
    text = re.sub(r'\^(\d+)(?!\{)', r'^{\1}', text)       # x^2 → x^{2}
    text = re.sub(r'\\times', r'\\cdot', text)             # \times → \cdot
    text = re.sub(r'\\left\(', '(', text)                  # \left( → (
    text = re.sub(r'\\right\)', ')', text)                  # \right) → )
    text = re.sub(r'\\vec\{([^}]+)\}',
                  r'\\overrightarrow{\1}', text)            # \vec → \overrightarrow
    return re.sub(r'\s+', ' ', text).strip()
```

---

## 问题4：要不要生成题目的自然语言描述？

### 推荐方案：生成，但分阶段实施

`description` 字段不参与向量化，而是在构建 `embedding_text` 时作为**可选的语义补充**。

#### 第一阶段（必做）：基础描述（模板生成，0成本）

```python
description = f"{question_type}：{stem_plain[:50]}...，考查{'、'.join(knowledge_points)}"
# 示例："单选题：已知直线l的斜率为2...，考查直线方程、点斜式"
```

#### 第二阶段（可选）：丰富描述（LLM生成，成本较高）

```
请为以下数学题生成100字以内的描述，包括：
1. 考查的核心概念
2. 解题关键思路
3. 常见错误陷阱

题干：{stem_plain}
知识点：{', '.join(knowledge_points)}
```

#### 对RAG检索精度的影响

| 场景 | 无描述 | 有基础描述 | 有丰富描述 |
|------|:------:|:--------:|:--------:|
| 用户搜索「椭圆周长」 | ❌ 可能漏掉 | ✅ 能匹配 | ✅ 精准匹配 |
| 向量检索精度提升 | 基准 | +20% | +40% |
| LLM调用成本 | 0 | 0（模板） | 中等 |

**结论：基础描述必做，丰富描述视情况添加。**

---

## 问题5：要不要微调Embedding模型？

### 推荐方案：暂不微调，先用现成模型

#### 为什么暂不微调？

1. **数据量不足**：微调需要 10K+ 高质量正负样本对，当前只有几百道题目
2. **成本-收益不划算**：微调成本约 $500-2000（GPU时间），换来的精度提升仅 5-10%
3. **现成模型已足够好**：Qwen3-Embedding-8B 在中文数学文本上表现良好

#### 什么时候考虑微调？

```python
# 同时满足以下三个条件时再评估
if (
    len(questions) >= 10000          # 数据量足够
    and has_labeled_pairs             # 有人工标注的相似/不相似题目对
    and retrieval_recall < 0.70       # 当前召回率低于预期
):
    # 考虑使用对比学习（Triplet Loss）微调
    pass
```

#### 微调目的与成本收益

| 目的 | 成本 | 预期收益 |
|------|------|--------|
| 提升数学题语义检索精度 | 高 | 中等 |
| 适配高中数学特定知识点 | 中等 | 中等 |
| 降低向量维度、加快检索 | 低 | 低 |

**建议路线：**
- 现阶段：Qwen3-Embedding-8B + 优化 `embedding_text` 构建方式
- 3个月后：若召回率持续低于 70%，评估微调可行性
- 