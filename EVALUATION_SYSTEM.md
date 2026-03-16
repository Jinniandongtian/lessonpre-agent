# 评测系统架构梳理

## 系统概览

评测系统分为 **4 层指标** 和 **3 种使用方式**：

```
┌─────────────────────────────────────────────────────────────┐
│                    评测系统架构                              │
├─────────────────────────────────────────────────────────────┤
│ 指标层                                                       │
│  ├─ A 类（格式与可用性，不需金标）                          │
│  │  ├─ A1: JSON 解析成功率                                  │
│  │  ├─ A2: Schema 合格率                                    │
│  │  ├─ A3: 结构完整率（按题型）                             │
│  │  └─ A4: 题号一致性                                       │
│  ├─ B 类（覆盖与切分，需金标）                              │
│  │  ├─ B1: Precision / Recall / F1                          │
│  │  └─ B2: 疑似合并 / 拆分检测                              │
│  └─ C 类（内容正确性，需金标含题干/选项）                   │
│     ├─ C1: 选项文本准确率                                   │
│     └─ C2: 题干相似度                                       │
├─────────────────────────────────────────────────────────────┤
│ 使用方式                                                     │
│  ├─ CLI 命令行（仅 A 类）                                   │
│  ├─ HTTP API（A/B/C 类）                                    │
│  └─ Python 直接调用（A/B/C 类）                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 文件职责

### 1. `quality_metrics.py` — A 类指标评测
**职责**：评测题目格式与可用性（不需要金标）

**核心类/函数**：
- `MetricResult` — 单个指标结果
- `EvaluationReport` — 完整评测报告
- `QualityEvaluator` — 评测器
  - `evaluate(questions)` — 运行 A1-A4 评测
  - `_evaluate_a1_json_parse()` — JSON 解析成功率
  - `_evaluate_a2_schema_compliance()` — Schema 合格率
  - `_evaluate_a3_structure_completeness()` — 结构完整率
  - `_evaluate_a4_number_consistency()` — 题号一致性
- `evaluate_questions(questions)` — 便捷函数
- `print_evaluation_report(report)` — 打印报告

**输入**：题目列表 `[{"content": "...", "question_type": "...", ...}]`

**输出**：
```json
{
  "total_questions": 19,
  "quality_metrics": {
    "a2_schema_compliance": {"score": 1.0, "passed": 19, "total": 19, ...},
    "a3_structure_completeness": {"score": 0.947, "passed": 18, "total": 19, ...},
    "a4_number_consistency": {"score": 1.0, "passed": 19, "total": 19, ...}
  },
  "quality_summary": {
    "a2_schema_compliance": 1.0,
    "a3_structure_completeness": 0.947,
    "a4_number_consistency": 1.0,
    "overall": 0.982
  }
}
```

---

### 2. `coverage_metrics.py` — B 类指标评测
**职责**：评测题目覆盖率与切分准确性（需要金标）

**核心类/函数**：
- `CoverageResult` — 覆盖评测结果
- `SegmentationResult` — 切分评测结果
- `GoldStandardEvaluator` — 评测器
  - `evaluate(gold_data, pred_questions, strategy)` — 运行 B1-B2 评测
  - `_evaluate_coverage()` — 计算 Precision/Recall/F1
  - `_evaluate_segmentation()` — 检测疑似合并/拆分

**输入**：
- 金标数据：`{"questions": [{"num": 1, "stem": "...", "options": {...}}, ...]}`
- 预测题目：`[{"content": "...", "question_type": "...", ...}]`

**输出**：
```json
{
  "coverage_metrics": {
    "gold_count": 19,
    "pred_count": 19,
    "tp": 19,
    "fp": 0,
    "fn": 0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "unparsable_rate": 0.0,
    "duplicate_rate": 0.0,
    "missing_nums": [],
    "extra_nums": []
  },
  "segmentation_metrics": {
    "merge_count": 0,
    "split_count": 0,
    "merge_suspects": [],
    "split_suspects": []
  }
}
```

---

### 3. `content_metrics.py` — C 类指标评测
**职责**：评测题干与选项的文本准确性（需要金标含题干/选项）

**核心类/函数**：
- `OptionMatchResult` — 单个选项匹配结果
- `QuestionContentResult` — 单题内容评测结果
- `ContentEvaluator` — 评测器
  - `evaluate(gold_data, pred_questions)` — 运行 C1-C2 评测
  - `_evaluate_options()` — 选项文本匹配（严格匹配 + 字符相似度）
  - `_evaluate_stem()` — 题干相似度

**输入**：
- 金标数据：`{"questions": [{"num": 1, "stem": "$\\vec{a}=...$", "options": {"A": "$...$", ...}}, ...]}`
- 预测题目：`[{"content": "...", ...}]`

**输出**：
```json
{
  "choice_count": 9,
  "c1_option_exact_match_rate": 0.95,
  "c1_option_avg_similarity": 0.98,
  "total_matched": 19,
  "total_gold": 19,
  "c2_stem_exact_match_rate": 0.85,
  "c2_stem_avg_similarity": 0.92,
  "unmatched_gold_nums": [],
  "unmatched_pred_nums": []
}
```

---

### 4. `full_evaluator.py` — 统一评测器
**职责**：整合 A/B/C 类指标，生成完整报告

**核心类/函数**：
- `FullEvaluator` — 统一评测器
  - `evaluate_all(questions, gold_data, strategy, source_name)` — 运行所有指标
  - `generate_markdown_report(report, output_dir)` — 生成 Markdown 报告
- `run_full_evaluation()` — 便捷函数

**输入**：
- 题目列表
- 金标数据（可选）
- 策略标识（如 `"vision_llm"`）
- 评测对象名称

**输出**：完整报告（JSON + Markdown 文件）

---

### 5. `cli.py` — 命令行工具
**职责**：提供命令行接口，仅支持 A 类指标

**支持的数据源**：
- `vector_db` — 从向量库加载题目
- `file` — 从 JSON 文件加载题目
- `pdf` — 从 PDF 提取题目

---

## 使用方式

### 方式一：CLI 命令行（仅 A 类指标）

```bash
# 1. 评测向量库中的题目
python -m src.evaluation.cli --source vector_db

# 2. 评测 JSON 文件
python -m src.evaluation.cli --source file --input questions.json

# 3. 评测 PDF（自动提取题目）
python -m src.evaluation.cli --source pdf --input data/pdfs/exam.pdf --ocr

# 4. 限制评测题目数量
python -m src.evaluation.cli --source pdf --input exam.pdf --ocr --limit 10

# 5. 保存结果到 JSON
python -m src.evaluation.cli --source pdf --input exam.pdf --ocr --output results.json
```

**输出**：终端打印 A 类指标汇总

---

### 方式二：HTTP API（A/B/C 类指标）

需要先启动 API 服务：
```bash
python main.py
```

#### 2.1 仅评测 PDF（A 类）
```bash
curl -X POST http://localhost:8000/evaluation/pdf \
  -F "pdf_file=@data/pdfs/exam.pdf" \
  -F "ocr_enabled=true"
```

#### 2.2 完整评测 PDF（A/B/C 类，需金标）
```bash
curl -X POST http://localhost:8000/evaluation/full/pdf \
  -F "pdf_file=@data/pdfs/exam.pdf" \
  -F "gold_file=@data/gold_standards/gold.json" \
  -F "ocr_enabled=true" \
  -F "strategy=vision_llm" \
  -F "save_report=true"
```

#### 2.3 评测向量库（A 类）
```bash
curl -X GET "http://localhost:8000/evaluation/vector_db?limit=0&use_llm=false"
```

#### 2.4 评测向量库（A/B/C 类，需金标）
```bash
curl -X POST http://localhost:8000/evaluation/full/vector_db \
  -H "Content-Type: application/json" \
  -d '{
    "gold_file_path": "data/gold_standards/gold.json",
    "strategy": "vision_llm",
    "save_report": true
  }'
```

#### 2.5 评测传入的题目列表（A 类）
```bash
curl -X POST http://localhost:8000/evaluation/questions \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      {"content": "已知...", "question_type": "选择题", ...},
      ...
    ]
  }'
```

---



---

## 指标说明

| 指标 | 需金标 | 说明 |
|------|--------|------|
| **A1** | ❌ | LLM 输出中能被解析为合法 JSON 的比例 |
| **A2** | ❌ | 题目通过 Schema 校验的比例（必填字段存在、类型正确） |
| **A3** | ❌ | 按题型分别评测：选择题有 ABCD、填空题有空标、解答题有内容 |
| **A4** | ❌ | 检测重复题号、跳号、非法题号 |
| **B1** | ✅ | Precision（提取的题里有多少是真的）、Recall（真实题覆盖了多少）、F1 |
| **B2** | ✅ | 疑似合并（两题粘一起）、疑似拆分（一题拆成两题）的数量 |
| **C1** | ✅ | 选择题 A-D 选项与金标的匹配程度（严格匹配率 + 字符相似度） |
| **C2** | ✅ | 题干文本与金标的匹配程度（字符级相似度/编辑距离） |

---

## 快速参考

### 最常用的三条命令

**1. 仅评测 A 类（快速检查格式）**
```bash
python -m src.evaluation.cli --source pdf --input exam.pdf --ocr
```

**2. 完整评测（A/B/C 类，需金标）**
```bash
curl -X POST http://localhost:8000/evaluation/full/pdf \
  -F "pdf_file=@exam.pdf" \
  -F "gold_file=@gold.json" \
  -F "ocr_enabled=true" \
  -F "save_report=true"
```

**3. 清空题库 + 入库 + 评测**
```bash
# 清空
curl -X POST http://localhost:8000/vector_db/reset -d '{}'

# 入库（使用视觉模型）
curl -X POST http://localhost:8000/pdf/upload \
  -F "pdf_file=@exam.pdf" \
  -F "ocr_enabled=true"

# 评测
curl -X GET "http://localhost:8000/evaluation/vector_db"
```
