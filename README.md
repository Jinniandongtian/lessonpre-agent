# 独立老师备课助手 - PDF处理与RAG讲义生成

基于RAG技术的智能备课系统，自动处理本地PDF试卷，提取题目并生成个性化讲义。

## 核心功能

### 1. PDF试卷处理
- **自动识别**：扫描版PDF（OCR）或原生PDF（文本提取）
- **题目提取**：自动从PDF中提取题目
- **智能标注**：自动识别题型、知识点、难度
- **向量化存储**：题目自动存储到向量库

### 2. RAG讲义生成
- **语义检索**：基于向量相似度检索相关题目
- **个性化生成**：根据知识点和地区生成定制化讲义
- **本地化**：使用本地试卷中的真实题目
- **PDF导出**：一键导出精美PDF讲义

## 技术架构

```
备课助手系统
├── PDF处理模块
│   ├── OCR处理（扫描版PDF）
│   ├── 文本提取（原生PDF）
│   ├── 题目提取与分类
│   └── 知识点自动标注
├── 向量存储模块
│   ├── Embedding生成（OpenAI/Sentence-BERT）
│   └── 向量数据库（FAISS/Chroma）
├── RAG检索模块
│   ├── 向量相似度检索
│   └── 题目筛选与排序
└── 讲义生成模块
    ├── RAG增强生成
    └── PDF导出
```

## 快速开始

### 1. 安装依赖

```bash
# 创建conda环境
conda create -n lessonpre-agent python=3.10 -y
conda activate lessonpre-agent

# 安装依赖
pip install -r requirements.txt

# 安装Tesseract OCR（macOS）
brew install tesseract tesseract-lang
# Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim
```

### 2. 配置API Key

编辑 `.env` 文件：
```bash
# SiliconFlow API（推荐）
SILICONFLOW_API_KEY=your-api-key-here

# 或使用OpenAI
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. 启动服务

```bash
python main.py
```

访问：http://localhost:8000/docs

## 使用流程

### 步骤1：上传PDF试卷

```bash
curl -X POST "http://localhost:8000/pdf/upload" \
  -F "pdf_file=@exam.pdf" \
  -F "region=北京市" \
  -F "year=2023" \
  -F "grade=高一" \
  -F "exam_name=2023年北京市期中考试" \
  -F "source_type=期中" \
  -F "ocr_enabled=true"
```

系统会：
1. 自动判断PDF类型（扫描版/原生）
2. 提取题目文本
3. 识别题型和知识点
4. 生成向量并存储

### 步骤2：生成讲义

```bash
curl -X POST "http://localhost:8000/lesson/handout" \
  -F "topic=高一 平面向量的概念和运算" \
  -F "region=北京市" \
  -F "grade=高一" \
  -F "class_level=中等" \
  -F "num_examples=3" \
  -F "num_practice=5"
```

系统会：
1. 解析主题，提取知识点
2. 使用RAG检索相关题目
3. 生成个性化讲义
4. 返回PDF下载链接

## API接口

### 上传PDF试卷
- **POST** `/pdf/upload`
- 参数：pdf_file, region, year, grade, exam_name, source_type, ocr_enabled

### 生成讲义
- **POST** `/lesson/handout`
- 参数：topic, region, grade, class_level, num_examples, num_practice

### 向量库统计
- **GET** `/vector_db/stats`
- 返回：题目总数、后端类型等

## 技术栈

- **PDF处理**：PyMuPDF (fitz), pytesseract, pdf2image
- **向量存储**：FAISS（推荐）或 Chroma
- **Embedding**：OpenAI Embeddings 或 Sentence-BERT
- **LLM**：SiliconFlow DeepSeek（推荐）或 OpenAI
- **后端**：FastAPI
- **PDF导出**：WeasyPrint

## 项目结构

```
src/
  data_processing/    # PDF处理（OCR、文本提取、题目提取）
  vector_store/       # 向量存储（Embedding、向量数据库）
  agents/             # RAG讲义生成Agent
  api/                # FastAPI接口
  export/             # PDF导出
  utils/              # 工具类（LLM客户端、配置）
  data_models/        # 数据模型
```

## 注意事项

1. **OCR依赖**：扫描版PDF需要安装Tesseract OCR
2. **向量库**：首次使用需要先上传PDF试卷建立向量库
3. **API Key**：需要配置LLM和Embedding的API Key
4. **性能**：大量PDF处理可能需要较长时间

## 开发计划

- [x] PDF处理和OCR
- [x] 题目提取和标注
- [x] 向量化存储
- [x] RAG检索
- [x] 讲义生成
- [ ] Web前端界面
- [ ] 批量PDF处理
- [ ] 题目去重和合并

## 许可证

MIT License
