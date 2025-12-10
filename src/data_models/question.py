"""题目数据模型"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Question:
    """题目数据模型"""
    id: str
    source_type: str          # "月考" / "期中" / "期末" / "高考" / "模拟"
    region: str               # 省/市
    year: int
    grade: str                # "高一"/"高二"/"高三"
    exam_name: Optional[str]  # "2023年XX市一模"
    index: int                # 试卷第几题

    content: str              # 题干（可带 TeX 标记）
    images: List[str]         # 图片路径（如有）
    answer: Optional[str]
    solution: Optional[str]

    knowledge_points: List[str]
    difficulty: int           # 1-5

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "source_type": self.source_type,
            "region": self.region,
            "year": self.year,
            "grade": self.grade,
            "exam_name": self.exam_name,
            "index": self.index,
            "content": self.content,
            "images": self.images,
            "answer": self.answer,
            "solution": self.solution,
            "knowledge_points": self.knowledge_points,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Question":
        """从字典创建"""
        return cls(
            id=data["id"],
            source_type=data["source_type"],
            region=data["region"],
            year=data["year"],
            grade=data["grade"],
            exam_name=data.get("exam_name"),
            index=data["index"],
            content=data["content"],
            images=data.get("images", []),
            answer=data.get("answer"),
            solution=data.get("solution"),
            knowledge_points=data.get("knowledge_points", []),
            difficulty=data.get("difficulty", 3),
        )

