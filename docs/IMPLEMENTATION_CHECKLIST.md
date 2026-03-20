# 向量存储功能 - 实现检查清单

## 📋 完整实现清单

### 阶段1：准备工作（1小时）

- [ ] 阅读所有设计文档
  - [ ] VECTOR_STORAGE_DESIGN.md
  - [ ] IMPLEMENTATION_CODE_PART1.md
  - [ ] IMPLEMENTATION_CODE_PART2.md
  - [ ] ARCHITECTURE.md
  - [ ] QUICK_REFERENCE.md

- [ ] 备份现有代码
  ```bash
  git commit -m "backup: before vector storage refactoring"
  ```

- [ ] 创建新分支
  ```bash
  git checkout -b feature/vector-storage-enhancement
  ```

---

### 阶段2：数据模型更新（30分钟）

**文件：`src/data_models/question.py`**

- [ ] 导入必要的模块
  ```python
  from dataclasses import dataclass, field
  from typing import List, Optional, Dict, Any
  from datetime import datetime
  ```

- [ ] 更新Question类
  - [ ] 添加`content`字段（Dict[str, Any]）
  - [ ] 添加`source_meta`字段（Dict[str, Any]）
  - [ ] 添加`description`字段（Optional[str]）
  - [ ] 添加`normalized_text`字段（Optional[str]）
  - [ ] 添加`embedding_model`字段（str）
  - [ ] 添加`embedding_dim`字段（int）

- [ ] 实现`to_dict()`方法
  - [ ] 返回所有字段的字典表示

- [ ] 实现`from_dict()`类方法
  - [ ] 从字典创建Question对象

- [ ] 测试数据模型
  ```python
  q = Question(id="test", question_type="单选题")
  d = q.to_dict()
  q2 = Question.from_dict(d)
  assert q.id == q2.id
  ```

---

### 阶段3：LaTeX规范化器（1小时）

**文件：`src/utils/latex_normalizer.py`（新增）**

- [ ] 创建文件
  ```bash
  touch src/utils/latex_normalizer.py
  ```

- [ ] 实现LaTeXNormalizer类
  - [ ] `__init__(self, llm_client=None)`
  - [ ] `normalize_latex(self, latex_text: str) -> str`
    - [ ] 构建LLM提示词
    - [ ] 调用LLM
    - [ ] 异常处理（降级到正则）
  - [ ] `post_process_latex(self, latex_text: str) -> str`
    - [ ] 分数规范化（\dfrac → \frac）
    - [ ] 上标规范化（x^2 → x^{2}）
    - [ ] 乘号规范化（\times → \cdot）
    - [ ] 除号规范化（/ → \div）
    - [ ] 括号规范化（\left( → (）
    - [ ] 向量规范化（\vec → \overrightarrow）
    - [ ] 空格清理
  - [ ] `latex_to_plain(self, latex_text: str) -> str`
    - [ ] 分数转换（\frac{a}{b} → a/b）
    - [ ] 上标转换（x^{2} → x²）
    - [ ] 向量转换（\overrightarrow{AB} → AB→）
    - [ ] 乘号转换（\cdot → ·）
    - [ ] 除号转换（\div → /）
    - [ ] 根号转换（\sqrt{x} → √x）
    - [ ] 移除$符号
    - [ ] 移除其他LaTeX命令
  - [ ] `normalize_text(self, latex_text: str, options_latex: Dict) -> str`
    - [ ] 转换题干和选项为普通文本
    - [ ] 合并题干和选项
    - [ ] 移除标点符号
    - [ ] 清理多余空格

- [ ] 测试LaTeX规范化
  ```python
  normalizer = LaTeXNormalizer()
  
  # 测试分数
  assert "\\frac" in normalizer.normalize_latex("\\dfrac{x}{y}")
  
  # 测试上标
  assert "^{" in normalizer.normalize_latex("x^2")
  
  # 测试转换
  assert "x/y" in normalizer.latex_to_plain("\\frac{x}{y}")
  ```

---

### 阶段4：描述生成器（30分钟）

**文件：`src/utils/description_generator.py`（新增）**

- [ ] 创建文件
  ```bash
  touch src/utils/description_generator.py
  ```

- [ ] 实现DescriptionGenerator类
  - [ ] `__init__(self, llm_client=None)`
  - [ ] `generate_basic_description(self, question: Dict) -> str`
    - [ ] 提取题干、题型、知识点
    - [ ] 使用模板生成描述
    - [ ] 返回描述字符串
  - [ ] `generate_rich_description(self, question: Dict) -> str`
    - [ ] 构建LLM提示词
    - [ ] 调用LLM生成丰富描述
    - [ ] 异常处理（降级到基础描述）

- [ ] 测试描述生成
  ```python
  gen = DescriptionGenerator()
  
  question = {
      'content': {
          'stem_plain': '已知直线l的斜率为2',
          'options_plain': {'A': '2x-y-5=0'}
      },
      'question_type': '单选题',
      'knowledge_points': ['直线方程']
  }
  
  desc = gen.generate_basic_description(question)
  assert '单选题' in desc
  assert '直线方程' in desc
  ```

---

### 阶段5：PDF处理集成（1小时）

**文件：`src/data_processing/pdf_processor.py`（修改）**

- [ ] 导入新模块
  ```python
  from ..utils.latex_normalizer import LaTeXNormalizer
  from ..utils.description_generator import DescriptionGenerator
  from ..data_models.question import Question
  ```

- [ ] 创建`enrich_question_with_representations()`函数
  - [ ] 参数：question, llm_client, normalizer, desc_generator
  - [ ] 步骤1：规范化为LaTeX
  - [ ] 步骤2：转换为普通文本
  - [ ] 步骤3：生成规范化文本
  - [ ] 步骤4：生成描述
  - [ ] 步骤5：创建Question对象
  - [ ] 返回Question对象

- [ ] 在PDF处理流程中集成
  - [ ] 在提取题目后调用enrichment函数
  - [ ] 处理异常（记录日志，继续处理下一题）

- [ ] 测试集成
  ```python
  raw_question = {
      'id': 'test_001',
      'question_type': '单选题',
      'stem_raw': '已知直线 l 的斜率为 2',
      'options_raw': {'A': '2x-y-5=0'},
      'knowledge_points': ['直线方程'],
      'source_meta': {'region': '山东省', 'year': 2025}
  }
  
  q = enrich_question_with_representations(
      raw_question, llm_client, normalizer, desc_generator
  )
  
  assert q.id == 'test_001'
  assert q.content['stem_latex'] is not None
  assert q.content['stem_plain'] is not None
  assert q.normalized_text is not None
  assert q.description is not None
  ```

---

### 阶段6：向量化逻辑更新（30分钟）

**文件：`src/vector_store/embedding.py`（修改）**

- [ ] 添加`encode_question()`方法
  ```python
  def encode_question(self, question: Dict[str, Any]) -> List[float]:
      """用规范化文本生成向量"""
      # 优先使用规范化文本
      text_to_embed = question.get('normalized_text')
      
      # 如果没有规范化文本，使用普通文本
      if not text_to_embed:
          text_to_embed = question.get('content', {}).get('stem_plain', '')
      
      # 如果还是没有，使用原始题干
      if not text_to_embed:
          text_to_embed = question.get('content', {}).get('stem_latex', '')
      
      return self.encode_single(text_to_embed)
  ```

- [ ] 测试向量化
  ```python
  embedding_model = EmbeddingModel()
  
  question = {
      'normalized_text': '已知直线l的斜率为2经过点2,1',
      'content': {
          'stem_plain': '已知直线 l 的斜率为 2，经过点 (2,1)',
          'stem_latex': '已知直线 $l$ 的斜率为 $2$，经过点 $(2,1)$'
      }
  }
  
  embedding = embedding_model.encode_question(question)
  assert len(embedding) == 1024  # 或其他维度
  ```

---

### 阶段7：API集成（1小时）

**文件：`src/api/teacher_api.py`（修改）**

- [ ] 导入新模块
  ```python
  from ..utils.latex_normalizer import LaTeXNormalizer
  from ..utils.description_generator import DescriptionGenerator
  from ..data_processing.pdf_processor import enrich_question_with_representations
  ```

- [ ] 添加全局变量
  ```python
  normalizer = None
  desc_generator = None
  ```

- [ ] 在`init_agents()`中初始化
  ```python
  normalizer = LaTeXNormalizer(llm_client)
  desc_generator = DescriptionGenerator(llm_client)
  ```

- [ ] 修改`upload_pdf_exam()`函数
  - [ ] 在处理每个题目时调用enrichment函数
  - [ ] 生成向量
  - [ ] 存储到向量库
  - [ ] 返回处理结果

- [ ] 测试API
  ```bash
  curl -X POST "http://localhost:8000/pdf/upload" \
    -F "pdf_file=@test.pdf" \
    -F "ocr_enabled=true" \
    -F "auto_meta=true"
  ```

---

### 阶段8：单元测试（1小时）

**文件：`tests/test_vector_storage.py`（新增）**

- [ ] 创建测试文件
  ```bash
  mkdir -p tests
  touch tests/test_vector_storage.py
  ```

- [ ] 测试LaTeX规范化
  ```python
  def test_latex_normalization():
      normalizer = LaTeXNormalizer()
      
      # 测试分数
      result = normalizer.normalize_latex("\\dfrac{x}{y}")
      assert "\\frac" in result
      
      # 测试上标
      result = normalizer.normalize_latex("x^2")
      assert "^{" in result
  ```

- [ ] 测试LaTeX转普通文本
  ```python
  def test_latex_to_plain():
      normalizer = LaTeXNormalizer()
      
      result = normalizer.latex_to_plain("\\frac{x}{y}")
      assert "x/y" in result or "x/y" in result
  ```

- [ ] 测试描述生成
  ```python
  def test_description_generation():
      gen = DescriptionGenerator()
      
      question = {
          'content': {
              'stem_plain': '已知直线l的斜率为2',
              'options_plain': {'A': '2x-y-5=0'}
          },
          'question_type': '单选题',
          'knowledge_points': ['直线方程']
      }
      
      desc = gen.generate_basic_description(question)
      assert '单选题' in desc
  ```

- [ ] 测试Question数据模型
  ```python
  def test_question_model():
      q = Question(
          id="test_001",
          question_type="单选题",
          content={
              'stem_latex': '$x^2$',
              'stem_plain': 'x²'
          }
      )
      
      d = q.to_dict()
      q2 = Question.from_dict(d)
      assert q.id == q2.id
  ```

- [ ] 运行测试
  ```bash
  pytest tests/test_vector_storage.py -v
  ```

---

### 阶段9：集成测试（1小时）

**文件：`tests/test_integration.py`（新增）**

- [ ] 创建集成测试
  ```python
  def test_full_pipeline():
      # 1. 初始化
      normalizer = LaTeXNormalizer()
      desc_gen = DescriptionGenerator()
      embedding_model = EmbeddingModel()
      vector_db = VectorDatabase()
      
      # 2. 原始题目
      raw_question = {...}
      
      # 3. 规范化
      stem_latex = normalizer.normalize_latex(raw_question['stem_raw'])
      
      # 4. 转换为普通文本
      stem_plain = normalizer.latex_to_plain(stem_latex)
      
      # 5. 生成规范化文本
      normalized_text = normalizer.normalize_text(stem_latex, options_latex)
      
      # 6. 生成描述
      description = desc_gen.generate_basic_description(question_dict)
      
      # 7. 创建Question对象
      q = Question(...)
      
      # 8. 生成向量
      embedding = embedding_model.encode_question(q.to_dict())
      
      # 9. 存储到向量库
      result = vector_db.add_questions([q.to_dict()], [embedding])
      
      # 10. 验证
      assert result['added'] == 1
      assert vector_db.count() >= 1
  ```

- [ ] 运行集成测试
  ```bash
  pytest tests/test_integration.py -v
  ```

---

### 阶段10：性能测试（1小时）

**文件：`tests/test_performance.py`（新增）**

- [ ] 测试处理速度
  ```python
  def test_processing_speed():
      import time
      
      normalizer = LaTeXNormalizer()
      desc_gen = DescriptionGenerator()
      embedding_model = EmbeddingModel()
      
      raw_questions = [...]  # 100道题目
      
      start = time.time()
      for q in raw_questions:
          # 处理每道题目
          ...
      elapsed = time.time() - start
      
      avg_time = elapsed / len(raw_questions)
      print(f"平均处理时间：{avg_time:.2f}秒/题")
      
      # 预期：~2.5秒/题
      assert avg_time < 5.0
  ```

- [ ] 测试检索速度
  ```python
  def test_retrieval_speed():
      import time
      
      vector_db = VectorDatabase()
      embedding_model = EmbeddingModel()
      
      # 添加1000道题目
      ...
      
      query_text = "求椭圆周长"
      query_embedding = embedding_model.encode_single(query_text)
      
      start = time.time()
      results = vector_db.search(query_embedding, top_k=10)
      elapsed = time.time() - start
      
      print(f"检索时间：{elapsed:.3f}秒")
      
      # 预期：<1秒
      assert elapsed < 1.0
  ```

- [ ] 运行性能测试
  ```bash
  pytest tests/test_performance.py -v -s
  ```

---

### 阶段11：验证和优化（1小时）

- [ ] 验证存储结构
  ```bash
  # 检查metadata.json
  cat data/vector_db/metadata.json | head -20
  
  # 验证字段完整性
  python -c "
  import json
  with open('data/vector_db/metadata.json') as f:
      data = json.load(f)
      q = data[0]
      assert 'content' in q
      assert 'normalized_text' in q
      assert 'description' in q
      print('✅ 存储结构正确')
  "
  ```

- [ ] 验证向量质量
  ```python
  # 测试相似题目检索
  query = "求直线方程"
  results = vector_db.search(query_embedding, top_k=5)
  
  for meta, similarity in results:
      print(f"{meta['id']}: {similarity:.2f}")
  
  # 预期：相似度 > 0.7
  ```

- [ ] 验证RAG精度
  ```python
  # 测试RAG讲义生成
  handout = rag_agent.generate_handout(
      topic="直线方程",
      region="山东省",
      grade="高二"
  )
  
  # 检查返回的题目是否相关
  assert len(handout['questions']) > 0
  ```

- [ ] 性能优化建议
  - [ ] 如果处理速度 > 3秒/题，考虑批量处理
  - [ ] 如果检索速度 > 1秒，考虑增加FAISS索引
  - [ ] 如果内存占用过高，考虑分批处理

---

### 阶段12：文档和提交（30分钟）

- [ ] 更新README.md
  - [ ] 添加向量存储功能说明
  - [ ] 添加双表示法存储结构说明
  - [ ] 添加性能指标

- [ ] 添加代码注释
  - [ ] 为所有新增函数添加docstring
  - [ ] 为复杂逻辑添加行注释

- [ ] 提交代码
  ```bash
  git add -A
  git commit -m "feat: implement vector storage with dual representation

  - Add LaTeX normalization (LLM + regex)
  - Add description generation for RAG enhancement
  - Update Question data model with dual representation
  - Integrate into PDF processing pipeline
  - Update embedding logic to use normalized text
  - Add comprehensive tests and documentation
  
  Performance:
  - Processing: ~2.5s per question
  - Retrieval: ~0.6s per query
  - Precision: +20-40% improvement
  "
  ```

- [ ] 创建Pull Request
  - [ ] 添加详细描述
  - [ ] 链接相关issue
  - [ ] 请求代码审查

---

## 📊 进度跟踪

```
阶段1：准备工作          [████████░░] 100%
阶段2：数据模型更新      [████████░░] 100%
阶段3：LaTeX规范化器     [████████░░] 100%
阶段4：描述生成器        [████████░░] 100%
阶段5：PDF处理集成       [████████░░] 100%
阶段6：向量化逻辑更新    [████████░░] 100%
阶段7：API集成           [████████░░] 100%
阶段8：单元测试          [░░░░░░░░░░] 0%
阶段9：集成测试          [░░░░░░░░░░] 0%
阶段10：性能测试         [░░░░░░░░░░] 0%
阶段11：验证和优化       [░░░░░░░░░░] 0%
阶段12：文档和提交       [░░░░░░░░░░] 0%

总进度：[████████░░] 58%
```

---

## ✅ 完成标志

当以下所有条件都满足时，实现完成：

- [ ] 所有代码文件已创建/修改
- [ ] 所有单元测试通过
- [ ] 所有集成测试通过
- [ ] 性能测试满足预期
- [ ] 代码审查通过
- [ ] 文档已更新
- [ ] 代码已提交到主分支

---

## 🚀 后续工作

实现完成后的后续工作：

1. **短期（1-2周）**
   - [ ] 用真实PDF测试
   - [ ] 收集用户反馈
   - [ ] 修复bug

2. **中期（1-2个月）**
   - [ ] 生成丰富描述（可选）
   - [ ] 优化RAG检索精度
   - [ ] 支持多语言搜索

3. **长期（3-6个月）**
   - [ ] 积累10K+题目
   - [ ] 微调Embedding模型
   - [ ] 支持图片题目
   - [ ] 支持解答题的步骤分解

---

## 📞 需要帮助？

如果在实现过程中遇到问题：

1. 查看QUICK_REFERENCE.md中的常见问题
2. 参考IMPLEMENTATION_CODE_PART1/2.md中的代码示例
3. 检查ARCHITECTURE.md中的架构图
4. 查看故障排查指南
