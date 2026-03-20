# 向量存储功能 - 设计与实现方案

## 📌 快速导航

| 需要 | 打开 |
|------|------|
| 5分钟快速了解 | `SUMMARY.md` |
| 完整设计方案 | `VECTOR_STORAGE_DESIGN.md` |
| 实现代码（数据模型、规范化器、生成器） | `IMPLEMENTATION_CODE_PART1.md` |
| 实现代码（集成、示例、数据流图） | `IMPLEMENTATION_CODE_PART2.md` |
| 速查表、测试清单、常见问题 | `QUICK_REFERENCE.md` |

---

## 🎯 核心设计：5个问题的答案

### 1. 存储结构

**双表示法 + 分层元数据**，每道题目存储为：

```json
{
  "id": "q_001",
  "embedding_text": "已知直线l的斜率为2经过点2,1则直线l的方程为2x-y-5=0或...",
  "content": {
    "stem_latex":    "已知直线 $l$ 的斜率为 $2$，经过点 $(2,1)$，则直线 $l$ 的方程为",
    "stem_plain":    "已知直线 l 的斜率为 2，经过点 (2,1)，则直线 l 的方程为",
    "options_latex": {"A": "$2x-y-5=0$", "B": "$2x+y-5=0$", "C": "$2x-y-3=0$", "D": "$2x+y-3=0$"},
    "options_plain": {"A": "2x-y-5=0",  "B": "2x+y-5=0",  "C": "2x-y-3=0",  "D": "2x+y-3=0"}
  },
  "source_meta": {
    "region": "山东省", "year": 2025, "grade": "高二",
    "exam_name": "山东名校联盟期中", "source_type": "期中"
  },
  "question_meta": {
    "question_type": "单选题", "difficulty": 3,
    "knowledge_points": ["直线方程", "点斜式"]
  },
  "description": "求过点(2,1)斜率为2的直线方程，考查直线的点斜式"
}
```

**字段职责速查：**

| 字段 | 用途 | 参与向量化 |
|------|------|:--------:|
| `embedding_text` | 向量化的唯一输入 | ✅ |
| `content.stem_latex` / `options_latex` | PDF讲义渲染 | ❌ |
| `content.stem_plain` / `options_plain` | 评测字符级对比 | ❌ |
| `source_meta` | 一级过滤（地区/年份/年级） | ❌ |
| `question_meta` | 二级过滤（题型/难度/知识点） | ❌ |
| `description` | 增强RAG语义理解（可选） | ❌ |

> 时间戳、embedding模型、向量维度记录在 `import_history.jsonl`，不存入题目记录。

---

### 2. 生成顺序：先LaTeX后普通文本

```
OCR输出（混乱）→ LLM规范化为LaTeX → 转换为plain → build_embedding_text → 向量化
```

---

### 3. LaTeX规范化：LLM + 正则兜底

统一写法：`\dfrac`→`\frac`，`x^2`→`x^{2}`，`\times`→`\cdot`，`\vec`→`\overrightarrow` 等。

---

### 4. 自然语言描述：生成基础描述（必做）

- `description` **不参与向量化**，只增强RAG语义理解
- 基础描述用模板生成（0成本，+20% RAG精度）
- 丰富描述用LLM生成（按需启用，+40% RAG精度）

---

### 5. 微调Embedding：暂不微调

数据量不足（当前 < 1K，需要 10K+），未来满足条件后再评估。

---

## 📁 新增/修改的源码文件

### 新增
```
src/utils/latex_normalizer.py       # LaTeX规范化器
src/utils/description_generator.py  # 描述生成器
```

### 修改
```
src/data_models/question.py          # Question模型（新字段结构）
src/data_processing/pdf_processor.py # 集成 enrich_question_with_representations()
src/vector_store/embedding.py        # 新增 encode_question()，只用 embedding_text
src/api/teacher_api.py               # 集成规范化器和描述生成器
```

---

## 🗂️ 存储层次

```
data/vector_db/
├── index.faiss             # FAISS向量索引
├── metadata.json           # 题目元数据（Question.to_dict()）
└── import_history.jsonl    # 每次导入记录
```

`import_history.jsonl` 单条示例：
```json
{
  "timestamp": "2025-03-20T10:30:00Z",
  "pdf_file": "exam.pdf",
  "embedding_model": "Qwen/Qwen3-Embedding-8B",
  "embedding_dim": 1024,
  "questions_added": 50,
  "questions_skipped": 3
}
```

---

## 🚀 实现步骤（约4小时）

1. **Question模型**（30分钟）→ `src/data_models/question.py`
2. **LaTeX规范化器**（1小时）→ `src/utils/latex_normalizer.py`
3. **描述生成器**（30分钟）→ `src/utils/description_generator.py`
4. **PDF处理集成**（1小时）→ `src/data_processing/pdf_processor.py`
5. **向量化和API更新**（1小时）→ `embedding.py` + `teacher_api.py`
