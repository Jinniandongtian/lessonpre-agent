from dataclasses import dataclass
from typing import List, Dict, Optional, Any


@dataclass
class LessonTopic:
    id: str
    title: str
    grade: str
    region: str
    knowledge_points: List[str]
    class_level: Optional[str] = None
    template_style: str = "A"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "grade": self.grade,
            "region": self.region,
            "knowledge_points": self.knowledge_points,
            "class_level": self.class_level,
            "template_style": self.template_style,
        }
