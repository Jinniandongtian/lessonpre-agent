from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Question:
    content: str
    images: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    solution: Optional[str] = None
