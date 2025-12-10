"""课程/讲义数据模型"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LessonTopic:
    """课程主题"""
    id: str
    title: str              # "高一 平面向量的概念和运算"
    grade: str
    region: str
    knowledge_points: List[str]
    class_level: Optional[str] = None  # "基础薄弱" / "中等" / "较强" / "冲竞赛"
    template_style: Optional[str] = "A"  # "A" / "B" / "C"

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "grade": self.grade,
            "region": self.region,
            "knowledge_points": self.knowledge_points,
            "class_level": self.class_level,
            "template_style": self.template_style,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LessonTopic":
        """从字典创建"""
        return cls(
            id=data["id"],
            title=data["title"],
            grade=data["grade"],
            region=data["region"],
            knowledge_points=data.get("knowledge_points", []),
            class_level=data.get("class_level"),
            template_style=data.get("template_style", "A"),
        )


@dataclass
class Handout:
    """讲义内容"""
    lesson_topic: LessonTopic
    content: str  # Markdown格式的讲义内容
    example_questions: List[str]  # 例题ID列表
    practice_questions: List[str]  # 练习题ID列表
    created_at: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "lesson_topic": self.lesson_topic.to_dict(),
            "content": self.content,
            "example_questions": self.example_questions,
            "practice_questions": self.practice_questions,
            "created_at": self.created_at,
        }

