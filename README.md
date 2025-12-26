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

# 系统依赖（OCR / PDF处理 / PDF导出）
# 1) OCR：tesseract
# 2) pdf2image：poppler（提供 pdftoppm/pdfinfo）
# 3) PDF 导出：优先 weasyprint（需要 cairo/pango 等系统库）；或可选 pdfkit + wkhtmltopdf

# macOS（Homebrew）
brew install tesseract tesseract-lang poppler

# Ubuntu / Debian
# OCR + poppler
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim poppler-utils

# WeasyPrint 依赖（Ubuntu / Debian，若你使用 weasyprint 导出 PDF）
sudo apt-get install -y libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info

# 可选：pdfkit + wkhtmltopdf（另一条 PDF 导出链路）
# sudo apt-get install -y wkhtmltopdf
```

说明：服务启动时会做一次系统依赖自检，若缺少 tesseract/poppler 或 PDF 导出依赖，会在控制台打印提示与安装命令。

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

- [√] PDF处理和OCR
- [√] 题目提取和标注
- [√] 向量化存储
- [ RAG检索
- [] 讲义生成
- [] 重复题目检测
- [] Web前端界面
- [] 批量PDF处理
- [] 题目去重和合并

## 开发进度 
12月24日
1、对现有离线 OCR（Tesseract）做了明显增强（DPI/参数/预处理/后处理/保留符号），保留更强PaddleOCR的可能性
2、补救题目优先使用llm，再使用正则
3、对于试卷中题目的原始题号，不打算存储，因为后续功能大概率不涉及找到哪张试卷的哪道题

12月25日
1、对导入题目进行三轮去重：同pdf去重+库id去重+语义相似度去重
2、去掉聚合题目的逻辑，让llm判断不要把两个题目聚合在一起
3、对导入题库的质量和数量进行评测(待完成)

12月26日
1、增加向量库管理接口：按 exam_name/region 查询、删除/重建、导入进度与去重统计，方便维护题库。
2、添加纠错提示词，使v3模型获得与v3.2类似的效果

## 许可证

MIT License
