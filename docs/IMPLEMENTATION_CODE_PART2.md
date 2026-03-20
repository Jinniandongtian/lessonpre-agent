# 向量存储实现代码 - 第2部分

## 4. 集成到PDF处理流程

```python
# src/data_processing/pdf_processor.py 中添加

from ..utils.latex_normalizer import LaTeXNormalizer
from ..utils.description_generator import DescriptionGenerator
from ..data_models.question import Question


def enrich_question_with_representations(
    question: Dict[str, Any],
    llm_client,
    normalizer: LaTeXNormalizer,
    desc_generator: DescriptionGenerator,
) -> Question:
    """
    为原始题目添加双表示法、embedding_text 和 description。

    流程：
    1. LLM规范化为LaTeX
    2. 转换为普通文本（plain）
    3. 生成 embedding_text（向量化的唯一输入）
    4. 生成 description（可选，增强RAG）
    5. 组装 Question 对象
    """
    # 1. 规范化为LaTeX
    stem_latex = normalizer.normalize_latex(question.get('stem_raw', ''))
    options_raw = question.get('options_raw', {})
    options_latex = {k: normalizer.normalize_latex(v) for k, v in options_raw.items()}

    # 2. 转换为普通文本
    stem_plain = normalizer.latex_to_plain(stem_latex)
    options_plain = {k: normalizer.latex_to_plain(v) for k, v in options_latex.items()}

    # 3. 生成 embedding_text
    embedding_text = normalizer.build_embedding_text(stem_latex, options_latex)

    # 4. 生成 description
    question_dict = {
        'content': {
            'stem_plain': stem_plain,
            'stem_latex': stem_latex,
            'options_plain': options_plain,
            'options_latex': options_latex,
        },
        'question_meta': {
            'question_type': question.get('question_type', ''),
            'difficulty': question.get('difficulty', 3),
            'knowledge_points': question.get('knowledge_points', []),
        },
    }
    description = desc_generator.generate_basic_description(question_dict)

    # 5. 组装 Question 对象
    return Question(
        id=question.get('id', ''),
        embedding_text=embedding_text,
        content={
            'stem_latex': stem_latex,
            'stem_plain': stem_plain,
            'options_latex': options_latex,
            'options_plain': options_plain,
        },
        source_meta=question.get('source_meta', {}),
        question_meta={
            'question_type': question.get('question_type', ''),
            'difficulty': question.get('difficulty', 3),
            'knowledge_points': question.get('knowledge_points', []),
        },
        description=description,
    )
```

---

## 5. 更新向量化逻辑

```python
# src/vector_store/embedding.py 中添加

def encode_question(self, question: Dict[str, Any]) -> List[float]:
    """
    用 embedding_text 生成向量。
    embedding_text 是唯一用于向量化的字段，其他字段（source_meta、
    question_meta、content 等）只用于过滤和渲染，不参与向量计算。
    """
    text = question.get('embedding_text', '')

    # 兜底：旧数据可能没有 embedding_text 字段
    if not text:
        text = question.get('content', {}).get('stem_plain', '')
    if not text:
        text = question.get('content', {}).get('stem_latex', '')

    return self.encode_single(text)
```

---

## 6. 在API中集成

```python
# src/api/teacher_api.py

from ..utils.latex_normalizer import LaTeXNormalizer
from ..utils.description_generator import DescriptionGenerator
from ..data_processing.pdf_processor import enrich_question_with_representations

# 全局变量
normalizer: LaTeXNormalizer = None # 默认值为None
desc_generator: DescriptionGenerator = None


def init_agents():
    global llm_client, vision_llm_client, vector_db, embedding_model, rag_agent
    global normalizer, desc_generator
    try:
        llm_client = get_default_llm_client()
        vision_llm_client = get_vision_llm_client()
        vector_db = VectorDatabase(str(Config.VECTOR_DB_PATH))
        embedding_model = EmbeddingModel()
        rag_agent = RAGHandoutAgent(
            llm_client=llm_client,
            vector_db=vector_db,
            embedding_model=embedding_model,
        )
        normalizer = LaTeXNormalizer(llm_client)
        desc_generator = DescriptionGenerator(llm_client)
        print("✓ 所有Agent初始化成功")
        print(f"✓ 向量库中有 {vector_db.count()} 道题目")
    except Exception as e:
        print(f"⚠ 警告：Agent初始化失败: {e}")


@app.post("/pdf/upload")
async def upload_pdf_exam(...):
    """上传PDF试卷，提取题目后存储到向量库"""
    try:
        # ... PDF提取逻辑 ...

        enriched_questions = []
        embeddings = []

        for raw_q in extracted_questions:
            try:
                # 规范化、双表示法、生成 embedding_text 和 description
                q = enrich_question_with_representations(
                    raw_q, llm_client, normalizer, desc_generator
                )
                # 只用 embedding_text 生成向量
                emb = embedding_model.encode_question(q.to_dict())
                # 把完整的规范化和增强后的题目存储起来
                enriched_questions.append(q.to_dict())
                embeddings.append(emb)
            except Exception as e:
                print(f"⚠ 处理题目失败: {e}")
                continue

        if enriched_questions:
            result = vector_db.add_questions(enriched_questions, embeddings)
            # 记录导入历史（时间戳、模型、维度、数量等）
            _append_import_history({
                "timestamp": datetime.now().isoformat(),
                "pdf_file": pdf_file.filename,
                "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
                "questions_added": result.get("added", 0),
                "questions_skipped": result.get("skipped_existing", 0),
            })
            print(f"✓ 添加 {result['added']} 道题目到向量库")

        return {
            "status": "success",
            "total": len(extracted_questions),
            "added": len(enriched_questions),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 7. 完整使用示例

```python
from src.data_models.question import Question
from src.utils.latex_normalizer import LaTeXNormalizer
from src.utils.description_generator import DescriptionGenerator
from src.vector_store.embedding import EmbeddingModel
from src.vector_store.vector_db import VectorDatabase

# 初始化
normalizer = LaTeXNormalizer()
desc_gen = DescriptionGenerator()
embedding_model = EmbeddingModel()
vector_db = VectorDatabase()

# 原始题目（来自OCR/LLM提取）
raw_question = {
    'id': 'q_001',
    'stem_raw': '已知直线 l 的斜率为 2，经过点 (2,1)，则直线 l 的方程为',
    'options_raw': {
        'A': '2x-y-5=0', 'B': '2x+y-5=0',
        'C': '2x-y-3=0', 'D': '2x+y-3=0',
    },
    'question_type': '单选题',
    'difficulty': 3,
    'knowledge_points': ['直线方程', '点斜式'],
    'source_meta': {
        'region': '山东省', 'year': 2025, 'grade': '高二',
        'exam_name': '山东名校联盟期中', 'source_type': '期中',
    },
}

# 1. 规范化为LaTeX
stem_latex = normalizer.normalize_latex(raw_question['stem_raw'])
options_latex = {k: normalizer.normalize_latex(v)
                 for k, v in raw_question['options_raw'].items()}

# 2. 转换为普通文本
stem_plain = normalizer.latex_to_plain(stem_latex)
options_plain = {k: normalizer.latex_to_plain(v) for k, v in options_latex.items()}

# 3. 生成 embedding_text（唯一用于向量化的字段）
embedding_text = normalizer.build_embedding_text(stem_latex, options_latex)

# 4. 生成 description（增强RAG，不参与向量化）
question_dict = {
    'content': {
        'stem_plain': stem_plain, 'stem_latex': stem_latex,
        'options_plain': options_plain, 'options_latex': options_latex,
    },
    'question_meta': {
        'question_type': raw_question['question_type'],
        'knowledge_points': raw_question['knowledge_points'],
    },
}
description = desc_gen.generate_basic_description(question_dict)

# 5. 创建 Question 对象
q = Question(
    id=raw_question['id'],
    embedding_text=embedding_text,
    content={
        'stem_latex': stem_latex, 'stem_plain': stem_plain,
        'options_latex': options_latex, 'options_plain': options_plain,
    },
    source_meta=raw_question['source_meta'],  # 只含 region/year/grade/exam_name/source_type
    question_meta={
        'question_type': raw_question['question_type'],
        'difficulty': raw_question['difficulty'],
        'knowledge_points': raw_question['knowledge_points'],
    },
    description=description,
)

# 6. 向量化（只用 embedding_text）
embedding = embedding_model.encode_question(q.to_dict())

# 7. 存储到向量库
result = vector_db.add_questions([q.to_dict()], [embedding])
print(f"添加结果：{result}")
print(f"向量库现有题目数：{vector_db.count()}")

# 8. 检索示例
query_embedding = embedding_model.encode_single("求直线方程")
results = vector_db.search(query_embedding, top_k=5)
for meta, similarity in results:
    print(f"  [{similarity:.2f}] {meta['content']['stem_plain'][:60]}...")
```

---

## 8. 数据流图

```
PDF文件
  ↓
[PDF处理] → 提取原始题目（stem_raw, options_raw, question_type, difficulty, knowledge_points）
  ↓
[LaTeX规范化] → stem_latex, options_latex
  ↓
[转换为普通文本] → stem_plain, options_plain（评测用）
  ↓
[build_embedding_text] → embedding_text（向量化的唯一输入）
  ↓
[生成 description] → description（增强RAG，不参与向量化）
  ↓
[创建 Question 对象]
  ├── id
  ├── embedding_text        ← 向量化
  ├── content               ← 渲染(latex) / 评测(plain)
  ├── source_meta           ← 一级过滤
  ├── question_meta         ← 二级过滤
  └── description           ← 语义增强
  ↓
[encode_question] → 1024维向量（只用 embedding_text）
  ↓
[存储到FAISS]
  ├── index.faiss            ← 向量
  ├── metadata.json          ← Question.to_dict()
  └── import_history.jsonl   ← 时间戳、模型、维度、数量
  ↓
[RAG检索] → 用户查询 → 向量搜索 → source_meta/question_meta过滤
  ↓
[讲义生成] → stem_latex + options_latex 渲染PDF
```

---

## 9. 关键设计决策

| 决策 | 理由 | 收益 |
|------|------|------|
| 只有 `embedding_text` 参与向量化 | 其他字段用于过滤/渲染，不影响语义相似度 | 向量稳定、语义纯净 |
| 先LaTeX后plain | LaTeX是规范化的中间表示 | 统一OCR输出的多种写法 |
| `source_meta` + `question_meta` 分层 | 来源过滤与属性过滤职责分离 | 查询更灵活 |
| `description` 不参与向量化 | 避免引入主观描述噪声 | RAG语义更纯粹 |
| 时间戳/模型信息存 `import_history` | 题目记录不需要重复存储这些全局信息 | 减少冗余字段 |
| 暂不微调Embedding | 数据量不足（<10K）且成本高 | 节省资源 |

---

## 10. 后续优化方向

### 短期（1-2周）
- 实现双表示法存储
- 集成LaTeX规范化
- 生成基础 description

### 中期（1-2个月）
- 按需启用丰富 description（LLM生成）
- 优化 `build_embedding_text` 的构建策略
- 支持解答题（无选项）的 embedding_text 生成

### 长期（3-6个月）
- 积累10K+题目后评估是否微调Embedding
- 支持图片题目的多模态表示
- 支持解答题步骤分解
