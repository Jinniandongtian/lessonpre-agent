# 向量存储实现代码 - 第1部分

## 1. Question 数据模型

```python
# src/data_models/question.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Question:
    """题目数据模型 - 双表示法 + 分层元数据"""

    # 唯一标识
    id: str

    # ✅ 向量化的唯一输入：移除LaTeX标记和标点，题干+选项合并
    embedding_text: str

    # ✅ 双表示法：LaTeX用于渲染，plain用于评测对比
    content: Dict[str, Any] = field(default_factory=dict)
    # content 结构：
    # {
    #   "stem_latex":    "已知直线 $l$ 的斜率为 $2$，...",
    #   "stem_plain":    "已知直线 l 的斜率为 2，...",
    #   "options_latex": {"A": "$2x-y-5=0$", ...},
    #   "options_plain": {"A": "2x-y-5=0", ...}
    # }

    # ✅ 一级过滤：按来源查询（地区、年份、年级等）
    source_meta: Dict[str, Any] = field(default_factory=dict)
    # source_meta 结构：
    # {
    #   "region": "山东省", "year": 2025, "grade": "高二",
    #   "exam_name": "山东名校联盟期中", "source_type": "期中"
    # }

    # ✅ 二级过滤：按题目属性筛选（题型、难度、知识点）
    question_meta: Dict[str, Any] = field(default_factory=dict)
    # question_meta 结构：
    # {
    #   "question_type": "单选题",
    #   "difficulty": 3,
    #   "knowledge_points": ["直线方程", "点斜式"]
    # }

    # ✅ 可选：自然语言描述，增强RAG语义理解，不参与向量化
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于存储到metadata.json）"""
        return {
            "id": self.id,
            "embedding_text": self.embedding_text,
            "content": self.content,
            "source_meta": self.source_meta,
            "question_meta": self.question_meta,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        """从字典创建 Question 对象"""
        return cls(
            id=data.get("id", ""),
            embedding_text=data.get("embedding_text", ""),
            content=data.get("content", {}),
            source_meta=data.get("source_meta", {}),
            question_meta=data.get("question_meta", {}),
            description=data.get("description"),
        )
```

---

## 2. LaTeX 规范化器

```python
# src/utils/latex_normalizer.py
import re
from typing import Dict
from ..utils.llm_client import get_default_llm_client


class LaTeXNormalizer:
    """LaTeX格式规范化：LLM规范化 + 正则兜底"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or get_default_llm_client()

    def normalize_latex(self, latex_text: str) -> str:
        """使用LLM将各种LaTeX写法统一为标准形式"""
        prompt = f"""
        将以下LaTeX公式规范化为标准形式。规范化规则：
        1. 分数统一用 \\frac{{分子}}{{分母}}（不用 \\dfrac, \\tfrac）
        2. 上标统一用 x^{{2}}（不用 x^2）
        3. 乘号统一用 \\cdot（不用 *, ×, \\times）
        4. 除号统一用 \\div（不用 /）
        5. 根号统一用 \\sqrt{{x}}
        6. 括号统一用 ( ) 而不是 \\left( \\right)
        7. 向量统一用 \\overrightarrow{{AB}}
        8. 移除多余空格

        原LaTeX：{latex_text}
        输出：规范化后的LaTeX（仅输出公式，不要解释）
        """
        try:
            return self.llm_client.generate(prompt).strip()
        except Exception as e:
            print(f"LLM规范化失败: {e}，使用正则兜底")
            return self.post_process_latex(latex_text)

    def post_process_latex(self, latex_text: str) -> str:
        """正则表达式兜底，处理最常见的变体"""
        # dfrac/tfrac → frac
        latex_text = re.sub(r'\\[dt]frac', r'\\frac', latex_text)
        # x^2 → x^{2}
        latex_text = re.sub(r'\^(\d+)(?!\{)', r'^{\1}', latex_text)
        # \times → \cdot
        latex_text = re.sub(r'\\times', r'\\cdot', latex_text)
        # \left( → (，\right) → )
        latex_text = re.sub(r'\\left\(', '(', latex_text)
        latex_text = re.sub(r'\\right\)', ')', latex_text)
        # \vec{AB} → \overrightarrow{AB}
        latex_text = re.sub(r'\\vec\{([^}]+)\}', r'\\overrightarrow{\1}', latex_text)
        # 多余空格
        return re.sub(r'\s+', ' ', latex_text).strip()

    def latex_to_plain(self, latex_text: str) -> str:
        """将LaTeX转换为普通文本（用于评测对比）"""
        plain = latex_text
        # \frac{a}{b} → a/b
        plain = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', plain)
        # x^{2} → x²
        sup_map = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹'}
        plain = re.sub(r'\^\{(\d+)\}', lambda m: ''.join(sup_map.get(d, d) for d in m.group(1)), plain)
        # \overrightarrow{AB} → AB→
        plain = re.sub(r'\\overrightarrow\{([^}]+)\}', r'\1→', plain)
        # \cdot → ·，\div → /，\sqrt{x} → √x
        plain = plain.replace('\\cdot', '·').replace('\\div', '/')
        plain = re.sub(r'\\sqrt\{([^}]+)\}', r'√\1', plain)
        # 移除 $ 和其他LaTeX命令
        plain = plain.replace('$', '')
        plain = re.sub(r'\\[a-zA-Z]+', '', plain)
        return plain.strip()

    def build_embedding_text(self, stem_latex: str, options_latex: Dict[str, str]) -> str:
        """
        生成 embedding_text：移除LaTeX标记和标点，题干+选项合并。
        这是存入向量库、唯一用于向量化的字段。
        """
        # 先转普通文本
        plain_stem = self.latex_to_plain(stem_latex)
        plain_opts = [self.latex_to_plain(v) for v in options_latex.values()]

        # 合并题干与选项
        combined = plain_stem + ' ' + ' '.join(plain_opts)

        # 移除标点符号和多余空格
        combined = re.sub(r'[，,。\.；;：:！!？?（）()【】\[\]《》<>""\'\'、]', '', combined)
        combined = re.sub(r'\s+', ' ', combined).strip()
        return combined
```

---

## 3. 题目描述生成器

```python
# src/utils/description_generator.py
from typing import Dict, Any
from ..utils.llm_client import get_default_llm_client


class DescriptionGenerator:
    """
    生成题目的自然语言描述。
    description 字段不参与向量化，只用于增强RAG语义理解。
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client or get_default_llm_client()

    def generate_basic_description(self, question: Dict[str, Any]) -> str:
        """基础描述：模板生成，0成本。"""
        stem = question.get('content', {}).get('stem_plain', '')[:80]
        q_meta = question.get('question_meta', {})
        question_type = q_meta.get('question_type', '')
        knowledge_points = q_meta.get('knowledge_points', [])

        kp_str = '、'.join(knowledge_points) if knowledge_points else '基础知识'
        return f"{question_type}：{stem}...，考查{kp_str}"

    def generate_rich_description(self, question: Dict[str, Any]) -> str:
        """丰富描述：LLM生成，+40% RAG精度，按需启用。"""
        stem = question.get('content', {}).get('stem_plain', '')
        options = question.get('content', {}).get('options_plain', {})
        q_meta = question.get('question_meta', {})
        knowledge_points = q_meta.get('knowledge_points', [])

        prompt = f"""
        为以下数学题生成100字以内的描述，包括：
        1. 考查的核心概念
        2. 解题关键思路
        3. 常见错误陷阱

        题干：{stem}
        选项：{', '.join(f"{k}: {v}" for k, v in options.items())}
        知识点：{', '.join(knowledge_points)}

        输出格式：核心概念：...\n解题思路：...\n常见错误：...
        """
        try:
            return self.llm_client.generate(prompt).strip()
        except Exception as e:
            print(f"生成丰富描述失败: {e}，降级为基础描述")
            return self.generate_basic_description(question)
```
