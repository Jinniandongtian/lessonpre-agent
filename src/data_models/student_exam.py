"""学生考试数据模型"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class StudentExamAnswer:
    """学生单题作答"""
    question_id: str
    student_answer: str
    is_correct: bool
    score_obtained: float
    score_full: float
    error_type: Optional[str] = None  # "计算粗心" / "概念错误" / "审题不清" / None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "question_id": self.question_id,
            "student_answer": self.student_answer,
            "is_correct": self.is_correct,
            "score_obtained": self.score_obtained,
            "score_full": self.score_full,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StudentExamAnswer":
        """从字典创建"""
        return cls(
            question_id=data["question_id"],
            student_answer=data["student_answer"],
            is_correct=data["is_correct"],
            score_obtained=data["score_obtained"],
            score_full=data["score_full"],
            error_type=data.get("error_type"),
        )


@dataclass
class StudentExamReport:
    """学生考试报告"""
    student_id: str
    exam_id: str
    answers: List[StudentExamAnswer]
    total_score: float
    full_score: float
    exam_name: Optional[str] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "student_id": self.student_id,
            "exam_id": self.exam_id,
            "answers": [ans.to_dict() for ans in self.answers],
            "total_score": self.total_score,
            "full_score": self.full_score,
            "exam_name": self.exam_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StudentExamReport":
        """从字典创建"""
        return cls(
            student_id=data["student_id"],
            exam_id=data["exam_id"],
            answers=[StudentExamAnswer.from_dict(ans) for ans in data["answers"]],
            total_score=data["total_score"],
            full_score=data["full_score"],
            exam_name=data.get("exam_name"),
        )

