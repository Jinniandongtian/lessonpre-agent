# 独立老师备课助手 - PDF处理与RAG讲义生成

基于 RAG 技术的智能备课系统，自动处理本地 PDF 试卷，提取题目并生成个性化讲义。

---

## 核心功能

### 1. PDF 试卷智能处理
- **自动类型识别**：智能判断扫描版 PDF（OCR）或原生 PDF（文本提取）
- **视觉大模型 OCR**：使用 Qwen2.5-VL-72B 视觉模型进行图文识别，完整支持数学公式、向量符号
- **Tesseract OCR（备用）**：Otsu 自动阈值二值化、灰度预处理、OCR 纠错提示词
- **题目提取**：LLM 智能提取（分块处理、完整性验证、缺题补救）
- **元数据识别**：自动识别地区、年份、年级、考试名称、考试类型（含置信度评分）
- **智能标注**：自动识别题型、知识点、难度等级
- **三轮去重**：同 PDF 去重 + 向量库 ID 去重 + 语义相似度去重
- **实时预览**：上传前预览提取效果，查看质量指标和潜在问题
- **大题说明过滤**：自动识别并过滤「一、选择题：本题共…」等题型说明，不混入题目

### 2. 向量化存储与检索
- **向量化**：支持 OpenAI Embeddings / Sentence-BERT / Qwen3-Embedding
- **FAISS 存储**：高性能向量检索
- **向量库管理**：按元数据查询、删除、重建、导入历史追踪

### 3. RAG 讲义生成
- **语义检索**：基于向量相似度检索相关题目
- **个性化生成**：根据知识点、地区、班级水平定制讲义
- **本地化**：使用本地试卷中的真实题目
- **PDF 导出**：一键导出精美 PDF 讲义

### 4. 质量评测系统

评测分为三大类，支持带/不带金标两种模式：

| 类别 | 需要金标 | 指标 |
|------|---------|------|
| **A 类（格式可用性）** | 否 | JSON 解析率、Schema 合格率、结构完整率、题号一致性 |
| **B 类（覆盖与切分）** | 是 | Precision / Recall / F1、疑似合并数、疑似拆分数 |
| **C 类（内容正确性）** | 是（含 stem/options）| 选项文本准确率、题干相似度 |

**金标数据格式说明**：金标文件采用 LaTeX 格式存储题干和选项（便于后续讲义渲染），提取结果为 Unicode 普通文字格式。C 类指标目前为字符级相似度对比，存在格式差异导致的偏低问题，后续计划通过 LLM 规范化后再对比。

**当前评测基准（山东名校联盟2025高二期中，原卷版，视觉模型提取）：**

| 指标 | 得分 | 备注 |
|------|------|------|
| A2 Schema 合格率 | 100% | |
| A3 结构完整率 | 100% | |
| A4 题号一致性 | 100% | |
| B1 Precision | 100% | |
| B1 Recall | 100% | |
| B1 F1 | 100% | |
| B2 疑似合并 | 0 处 | |
| B2 疑似拆分 | 0 处 | |
| C2 题干平均相似度 | 73.11% | 格式差异（LaTeX vs 普通文字），内容实际正确 |
| C1 选项平均相似度 | 59.81% | 同上 |

> C 类严格匹配率 0% 是预期结果：金标为 LaTeX（`$\frac{x^2}{4}$`），提取结果为 Unicode（`x²/4`），内容一致但字符不同。

---

## 技术架构

```
备课助手系统
├── PDF处理模块（src/data_processing/）
│   ├── PDFProcessor：类型判断、文本提取、视觉OCR/Tesseract OCR
│   ├── QuestionExtractor：LLM提取、缺题补救、完整性验证、大题说明过滤
│   └── ExamMetaExtractor：元数据识别（正则+LLM兜底）
├── 去重模块（src/api/teacher_api.py）
│   ├── 第一轮：同PDF内部去重（pdf_hash + 题号）
│   ├── 第二轮：向量库ID去重
│   └── 第三轮：语义相似度去重（文本+向量双阈值）
├── 向量存储模块（src/vector_store/）
│   ├── EmbeddingModel：文本向量化
│   └── VectorDatabase：FAISS存储与检索
├── RAG讲义生成（src/agents/）
│   └── RAGHandoutAgent：检索+生成+导出
├── 质量评测模块（src/evaluation/）
│   ├── quality_metrics：A类指标
│   ├── coverage_metrics：B类指标（兼容旧格式 list/dict 金标）
│   ├── content_metrics：C类指标
│   └── full_evaluator：统一评测入口，生成 Markdown 报告
└── API层（src/api/teacher_api.py）
    └── FastAPI：所有对外接口
```

---

## PDF 处理完整流程

```
1. PDF上传
   ↓
2. 类型判断（detect_ocr_need）
   - 检查前3页文本质量
   - 文本长度 < 100字符 → 走OCR
   - 有意义字符占比 < 20% → 走OCR
   - 检测到加密/占位文本 → 走OCR
   ↓
3. 文本提取
   - 视觉模型模式（推荐）：PDF转图片 → Qwen2.5-VL-72B 逐页识别 → 合并全文
   - Tesseract OCR模式（备用）：DPI=300, 灰度化 → Otsu阈值二值化 → Tesseract (chi_sim+eng)
   - 原生模式：PyMuPDF直接提取文本层
   ↓
4. 元数据识别（前2页）
   - 正则提取：年份(置信度0.95)、年级(0.9)、考试类型(0.85)、地区(0.75)
   - LLM兜底：任意字段置信度 < 0.6 时调用LLM补全
   - 合并策略：用户手动传入 > 自动识别
   ↓
5. 题目提取（全文，QuestionExtractor）
   - LLM分块提取（按页分块，相邻页重叠）
   - 完整性验证：必须含题号+题干+选项（选择题）
   - 批量过滤试卷说明（LLM判断 + 关键词备用）
   - 大题说明过滤：「一、选择题：本题共…」等中文序号开头的说明自动过滤
   ↓
6. 缺题补救
   - 推断预期题号范围（排除年份1900-2100，排除>200的数字）
   - LLM二次提取缺失题号
   - 正则备用补救（仅在LLM补救无结果时）
   ↓
7. 三轮去重
   - 第一轮：pdf_hash+题号（同PDF内部精确去重）
   - 第二轮：向量库现有ID查询（跳过已入库题目）
   - 第三轮：文本相似度 ≥ 0.92 OR 向量余弦相似度 ≥ 0.80 → 判定重复
   ↓
8. 题目丰富化（可选，ENRICH_WITH_LLM=1 时启用）
   - 批量标注知识点（2-6个）
   - 难度评级（1-5分）
   ↓
9. 向量化存储
   - EmbeddingModel.encode() 生成向量
   - 存入FAISS索引（data/vector_db/index.faiss）
   - 元数据保存（data/vector_db/metadata.json）
   - 写入导入历史（data/import_history.jsonl）
```

---

## 快速开始

### 1. 安装依赖

```bash
# 创建conda环境
conda create -n lessonpre-agent python=3.10 -y
conda activate lessonpre-agent

# 安装Python依赖
pip install -r requirements.txt

# macOS 系统依赖
brew install tesseract tesseract-lang poppler

# Ubuntu / Debian 系统依赖
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim poppler-utils
sudo apt-get install -y libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

> 说明：服务启动时会做一次系统依赖自检，若缺少 tesseract/poppler 或 PDF 导出依赖，会在控制台打印提示与安装命令。

### 2. 配置 API Key

编辑 `.env` 文件：

```bash
# SiliconFlow API（推荐）
SILICONFLOW_API_KEY=your-api-key-here
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_MODEL=deepseek-ai/DeepSeek-V3.2

# 视觉大模型（用于OCR识别，支持数学公式）
VISION_MODEL=Qwen/Qwen2.5-VL-72B-Instruct

# Embedding 模型
# 同时配置 EMBEDDING_MODEL 和 SILICONFLOW_API_KEY 时，优先使用 SiliconFlow 远端 embedding
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
# 如果未配置远端 embedding，则回退到本地 sentence-transformers
# 本地模型名在代码中通过 EmbeddingModel(model_name=...) 传入
# 不传时默认使用 sentence-transformers/all-MiniLM-L6-v2

# 可选：启用题目丰富化（会增加LLM调用次数）
# ENRICH_WITH_LLM=1

# 可选：去重阈值调整
# DEDUPE_TEXT_SIM_THRESHOLD=0.92
# DEDUPE_EMB_SIM_THRESHOLD=0.80
```

### 3. 启动服务

```bash
python main.py
```

访问 `http://localhost:8000/docs` 查看完整 API 文档。

---

## 使用流程

### 推荐工作流：先预览，再入库

#### 步骤1：预览PDF提取效果（推荐）

**方式A：Web界面（最直观）**

启动服务后，浏览器直接打开项目根目录下的 `preview_ui.html`：
- 拖拽或点击上传 PDF 文件
- 实时查看提取的题目列表、题型、知识点
- 查看质量指标（Schema 合格率、结构完整率）
- 系统自动检测潜在问题（缺题、题目过短、选项缺失、OCR 错误）
- 确认无误后一键「正式入库」

**方式B：API调用**

```bash
curl -X POST "http://localhost:8000/pdf/preview" \
  -F "pdf_file=@exam.pdf" \
  -F "ocr_enabled=true" \
  -F "auto_meta=true"
```

#### 步骤2：正式入库

```bash
curl -X POST "http://localhost:8000/pdf/upload" \
  -F "pdf_file=@exam.pdf" \
  -F "ocr_enabled=true" \
  -F "auto_meta=true"
```

也可以手动指定元数据（优先级高于自动识别）：

```bash
curl -X POST "http://localhost:8000/pdf/upload" \
  -F "pdf_file=@exam.pdf" \
  -F "region=山东省" \
  -F "year=2025" \
  -F "grade=高一" \
  -F "source_type=期中" \
  -F "ocr_enabled=true"
```

#### 步骤3：完整评测（含金标对比）

```bash
curl -X POST "http://localhost:8000/evaluation/full/pdf" \
  -F "pdf_file=@exam.pdf" \
  -F "gold_file=@gold_standard.json" \
  -F "strategy=vision_llm" \
  -F "save_report=true"
```

金标文件为 JSON 格式，支持新格式（`{"paper_id": ..., "questions": [...]}`）和旧格式（直接 `[...]` 数组）。

#### 步骤4：生成讲义

```bash
curl -X POST "http://localhost:8000/lesson/handout" \
  -F "topic=高二 椭圆的标准方程" \
  -F "region=山东省" \
  -F "grade=高二" \
  -F "class_level=中等" \
  -F "num_examples=3" \
  -F "num_practice=5"
```

---

## API 接口一览

### PDF 处理
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/pdf/preview` | 预览提取效果（不入库），返回题目摘要、质量指标、问题检测 |
| POST | `/pdf/upload` | 正式上传入库，支持自动元数据识别 |

### 讲义生成
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/lesson/handout` | RAG 生成讲义，导出 PDF |
| GET | `/exports/{filename}` | 下载导出文件 |

### 向量库管理
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/vector_db/stats` | 总量与后端类型 |
| GET | `/vector_db/query` | 按元数据字段查询题目 |
| GET | `/vector_db/groups` | 按字段分组统计 |
| POST | `/vector_db/delete` | 按条件删除（支持 dry_run） |
| POST | `/vector_db/reset` | 清空重建 |
| GET | `/vector_db/import_history` | 查看历次导入记录 |

### 质量评测
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/evaluation/vector_db` | 评测当前向量库题目质量 |
| POST | `/evaluation/pdf` | 上传 PDF 评测（不入库） |
| POST | `/evaluation/questions` | 评测题目列表 |
| POST | `/evaluation/gold_standard` | 与金标数据对比 |
| POST | `/evaluation/full/pdf` | 完整评测（A+B+C 类指标，生成 MD 报告） |

---

## 金标数据格式

金标文件存放于 `data/gold_standards/`，当前已有：

| 文件 | 试卷 |
|------|------|
| `shandong_alliance_2025_first_mid_g2_math.json` | 山东名校联盟 2025 高二上期中数学 |
| `shandong_shiyan_2025_g2_upper_mid_math_gold.json` | 山东实验中学 2025 高二上期中数学 |
| `jinan_province-shiyan_2025_first_mid_g1_math.json` | 济南省实验中学 2025 高一上期中数学 |

**格式规范（新格式）**：
```json
{
  "paper_id": "唯一标识",
  "description": "试卷描述",
  "questions": [
    {
      "num": 1,
      "type": "单选题",
      "stem": "题干（LaTeX格式，如 $\\frac{x^2}{4}+\\frac{y^2}{3}=1$）",
      "options": {
        "A": "选项A（LaTeX格式）",
        "B": "选项B",
        "C": "选项C",
        "D": "选项D"
      }
    }
  ]
}
```

**兼容旧格式**：直接传题目数组 `[{"num": 1, ...}]` 也可正常使用。

**关于格式选择**：金标采用 LaTeX 格式，便于后续讲义/试卷渲染。C 类评测指标当前使用字符级相似度，LaTeX 与普通文字的格式差异会导致相似度偏低（约 60-73%），但内容实际正确。后续计划通过 LLM 规范化后再对比，以获得准确的内容相似度评分。

---

## 技术栈

| 模块 | 技术选型 |
|------|----------|
| PDF 文本提取 | PyMuPDF (fitz) |
| 视觉 OCR | Qwen2.5-VL-72B-Instruct (via SiliconFlow) |
| Tesseract OCR（备用）| pytesseract + pdf2image |
| 向量存储 | FAISS |
| Embedding | Qwen3-Embedding-8B (via SiliconFlow) |
| LLM | DeepSeek-V3.2 (via SiliconFlow) |
| 后端框架 | FastAPI + uvicorn |
| PDF 导出 | WeasyPrint（降级为 HTML 当依赖未安装时）|

---

## 项目结构

```
lessonpre-agent/
├── main.py                    # 启动入口
├── requirements.txt
├── .env                       # API Key 配置
├── preview_ui.html            # PDF预览Web界面
├── src/
│   ├── api/teacher_api.py     # FastAPI 接口层
│   ├── data_processing/
│   │   ├── pdf_processor.py   # PDF处理、OCR、题目提取
│   │   └── meta_extractor.py  # 元数据识别
│   ├── vector_store/
│   │   ├── embedding.py       # Embedding生成
│   │   └── vector_db.py       # FAISS向量库
│   ├── agents/
│   │   └── rag_handout_agent.py  # RAG讲义生成
│   ├── evaluation/
│   │   ├── quality_metrics.py    # A类指标
│   │   ├── coverage_metrics.py   # B类指标
│   │   ├── content_metrics.py    # C类指标
│   │   ├── full_evaluator.py     # 统一评测入口
│   │   ├── cli.py                # 命令行评测工具
│   │   └── results/              # 评测报告（Markdown）
│   ├── export/pdf_exporter.py    # PDF导出
│   └── utils/
│       ├── config.py             # 配置管理
│       └── llm_client.py         # LLM客户端
└── data/
    ├── pdfs/                  # 上传的PDF文件
    ├── vector_db/             # FAISS索引与元数据
    ├── gold_standards/        # 金标数据（LaTeX格式）
    └── import_history.jsonl   # 导入历史
```

---

## 开发进度

### 已完成
- ✅ PDF 类型自动判断（原生/扫描）
- ✅ 视觉大模型 OCR（Qwen2.5-VL-72B，支持数学公式、向量符号）
- ✅ Tesseract OCR（Otsu 阈值、灰度预处理、后处理，备用）
- ✅ LLM 智能题目提取（分块、OCR 纠错提示词、完整性验证）
- ✅ 大题说明自动过滤（「一、选择题：本题共…」不再误入题库）
- ✅ 元数据自动识别（正则+LLM 兜底，含置信度）
- ✅ 三轮去重机制
- ✅ 缺题补救（LLM + 正则备用）
- ✅ FAISS 向量化存储
- ✅ 向量库管理接口（查询、删除、重建、历史）
- ✅ 质量评测系统（A+B+C 类指标，金标对比，生成 Markdown 报告）
- ✅ 金标数据兼容旧格式（list/dict 均可）
- ✅ 实时预览功能（`preview_ui.html` + `/pdf/preview` API）
- ✅ 金标数据集（3 份山东高中数学期中试卷）

### 已知问题 / 待优化
- ⚠️ C 类指标偏低：金标 LaTeX 格式 vs 提取结果普通文字，字符相似度失真；计划引入 LLM 规范化后对比
- ⚠️ PDF 导出依赖（WeasyPrint/wkhtmltopdf）未安装时降级为 HTML
- ⚠️ Chroma 向量库不可用（`No module named 'chromadb'`），当前使用 FAISS

### 待开发
- ⏳ C 类指标：LLM 语义规范化后再对比（解决 LaTeX vs 普通文字格式差异）
- ⏳ PaddleOCR 集成（双引擎策略，针对低质量扫描件）
- ⏳ 图像去噪与倾斜校正（opencv）
- ⏳ RAG 讲义生成优化
- ⏳ 批量 PDF 处理
- ⏳ Web 前端界面
- ⏳ 题目图片提取与理解

---

## 许可证

MIT License
