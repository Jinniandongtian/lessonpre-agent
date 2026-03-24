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
    #   "stem_raw":      "原始题干文本",
    #   "stem_latex":    "已知直线 $l$ 的斜率为 $2$，...",
    #   "stem_plain":    "已知直线 l 的斜率为 2，...",
    #   "options_raw":   {"A": "原始选项A", ...},
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
        """转换为字典（用于存储到 metadata.json）"""
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
