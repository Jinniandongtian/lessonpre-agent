# 系统架构设计文档

## 概述

本系统专注于PDF试卷处理和基于RAG的讲义生成，采用模块化设计，便于扩展和维护。

## 核心模块

### 1. PDF处理模块 (`src/data_processing/pdf_processor.py`)

#### PDFProcessor
- **功能**：PDF文本提取
- **支持**：
  - 扫描版PDF：使用OCR（Tesseract）
  - 原生PDF：直接提取文本（PyMuPDF）
- **方法**：
  - `is_scanned_pdf()`: 判断PDF类型
  - `extract_text_with_ocr()`: OCR提取
  - `extract_text_native()`: 原生提取
  - `extract_text()`: 自动判断并提取
  - `clean_text()`: 文本清洗

#### QuestionExtractor
- **功能**：从文本中提取题目
- **方法**：
  - `identify_question_type()`: 识别题型
  - `extract_questions_from_text()`: 提取题目
  - `enrich_question_with_llm()`: LLM丰富题目信息

### 2. 向量存储模块 (`src/vector_store/`)

#### EmbeddingModel (`embedding.py`)
- **功能**：文本向量化
- **支持模型**：
  - OpenAI Embeddings（推荐）
  - Sentence-BERT（本地模型）
- **方法**：
  - `encode()`: 批量编码
  - `encode_single()`: 单文本编码

#### VectorDatabase (`vector_db.py`)
- **功能**：向量存储和检索
- **支持后端**：
  - FAISS（推荐，高性能）
  - Chroma（可选）
  - Simple（备用，内存存储）
- **方法**：
  - `add_questions()`: 添加题目和向量
  - `search()`: 向量相似度搜索
  - `count()`: 获取题目数量

### 3. RAG讲义生成模块 (`src/agents/rag_handout_agent.py`)

#### RAGHandoutAgent
- **功能**：基于RAG生成讲义
- **流程**：
  1. 解析主题，提取知识点
  2. 使用向量检索相关题目
  3. 筛选和排序题目
  4. 生成讲义内容
- **方法**：
  - `generate_lesson_topic()`: 生成课程主题
  - `retrieve_questions()`: RAG检索题目
  - `select_example_and_practice_questions()`: 选择例题和练习题
  - `build_handout_markdown()`: 生成讲义Markdown
  - `generate_handout()`: 完整流程

## 数据流

### PDF上传流程

```
PDF文件上传
    ↓
PDFProcessor判断类型（扫描版/原生）
    ↓
提取文本（OCR或直接提取）
    ↓
QuestionExtractor提取题目
    ↓
LLM标注知识点和难度
    ↓
EmbeddingModel生成向量
    ↓
VectorDatabase存储
```

### 讲义生成流程

```
用户输入主题
    ↓
RAGHandoutAgent解析知识点
    ↓
生成查询向量
    ↓
VectorDatabase检索相似题目
    ↓
筛选和排序（按知识点、难度）
    ↓
LLM生成讲义（使用检索到的题目）
    ↓
导出PDF
```

## API接口

### POST `/pdf/upload`
上传PDF试卷，自动处理并存储到向量库。

**请求参数**：
- `pdf_file`: PDF文件
- `region`: 地区（如"北京市"）
- `year`: 年份
- `grade`: 年级（高一/高二/高三）
- `exam_name`: 考试名称（可选）
- `source_type`: 来源类型（期中/期末/高考/模拟）
- `ocr_enabled`: 是否启用OCR（默认true）

**响应**：
```json
{
  "success": true,
  "message": "成功提取并存储 N 道题目",
  "questions_extracted": 10,
  "vector_db_total": 100
}
```

### POST `/lesson/handout`
使用RAG生成讲义。

**请求参数**：
- `topic`: 主题（如"高一 平面向量的概念和运算"）
- `region`: 地区
- `grade`: 年级
- `class_level`: 班级水平（可选）
- `template_style`: 模板样式（可选）
- `num_examples`: 例题数量（默认3）
- `num_practice`: 练习题数量（默认5）

**响应**：
```json
{
  "success": true,
  "lesson_topic": {...},
  "handout_content": "...",
  "pdf_url": "/exports/handout_xxx.pdf",
  "examples_count": 3,
  "practice_count": 5
}
```

### GET `/vector_db/stats`
获取向量库统计信息。

## 技术选型

### PDF处理
- **PyMuPDF (fitz)**: 原生PDF文本提取
- **pytesseract**: OCR识别
- **pdf2image**: PDF转图片

### 向量存储
- **FAISS**: Facebook AI Similarity Search，高性能向量检索
- **Chroma**: 可选，更易用的向量数据库

### Embedding
- **OpenAI Embeddings**: 高质量，需要API
- **Sentence-BERT**: 本地模型，免费但质量略低

### LLM
- **SiliconFlow DeepSeek**: 推荐，性价比高
- **OpenAI GPT**: 备选

## 性能优化

1. **批量处理**：PDF题目批量生成向量
2. **索引优化**：使用FAISS的索引结构加速检索
3. **缓存机制**：缓存常用查询的向量
4. **异步处理**：PDF处理可以异步进行

## 扩展方向

1. **多模态支持**：处理图片题目
2. **题目去重**：使用向量相似度去重
3. **题目分类**：更精细的题型分类
4. **质量评估**：评估题目质量并排序
5. **增量更新**：支持向量库增量更新

