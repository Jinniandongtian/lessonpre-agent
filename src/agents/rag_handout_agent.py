"""基于RAG的讲义生成Agent"""
from typing import List, Dict, Optional, Any
import uuid
from ..vector_store.embedding import EmbeddingModel
from ..vector_store.vector_db import VectorDatabase
from ..utils.llm_client import LLMClient
from ..data_models.lesson import LessonTopic
from ..export.pdf_exporter import handout_markdown_to_pdf


class RAGHandoutAgent:
    """基于RAG技术的讲义生成Agent"""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        vector_db: Optional[VectorDatabase] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        self.llm = llm_client or LLMClient()
        self.vector_db = vector_db or VectorDatabase()
        self.embedding_model = embedding_model or EmbeddingModel()
    
    def generate_lesson_topic(
        self,
        raw_topic_text: str,
        region: str,
        grade: str,
        class_level: Optional[str] = None,
        template_style: str = "A"
    ) -> LessonTopic:
        """生成课程主题（提取知识点）"""
        prompt = f"""
请分析以下课程主题，提取出涉及的知识点列表。

主题：{raw_topic_text}
年级：{grade}

请以JSON格式返回知识点列表：
{{
    "knowledge_points": ["知识点1", "知识点2", ...]
}}
"""
        
        response = self.llm.generate(prompt)
        knowledge_points = self._parse_knowledge_points(response)
        
        return LessonTopic(
            id=str(uuid.uuid4()),
            title=raw_topic_text,
            grade=grade,
            region=region,
            knowledge_points=knowledge_points,
            class_level=class_level,
            template_style=template_style,
        )
    
    def _parse_knowledge_points(self, llm_response: str) -> List[str]:
        """解析LLM返回的知识点"""
        import json
        import re
        
        try:
            json_match = re.search(r'\{[^}]+\}', llm_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data.get("knowledge_points", [])
        except:
            pass
        
        return []
    
    def retrieve_questions(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        使用RAG检索相关题目
        
        Args:
            query: 查询文本（如"平面向量的概念和运算"）
            top_k: 返回前k个结果
            filters: 过滤条件（knowledge_points, difficulty等）
        
        Returns:
            题目列表（包含content, knowledge_points等）
        """
        # 1. 将查询转换为向量
        query_embedding = self.embedding_model.encode_single(query)
        
        # 2. 向量检索
        results = self.vector_db.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        # 3. 返回题目
        questions = []
        for metadata, similarity in results:
            questions.append({
                "content": metadata.get("content", ""),
                "question_type": metadata.get("question_type", ""),
                "knowledge_points": metadata.get("knowledge_points", []),
                "difficulty": metadata.get("difficulty", 3),
                "similarity": similarity,
                "source_meta": metadata.get("source_meta", {}),
            })
        
        return questions
    
    def select_example_and_practice_questions(
        self,
        lesson: LessonTopic,
        num_examples: int = 3,
        num_practice: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        选择例题和练习题
        
        Returns:
        {
          "examples": [题目字典, ...],
          "practice": [题目字典, ...]
        }
        """
        # 构建查询：主题 + 知识点
        query = f"{lesson.title} {', '.join(lesson.knowledge_points)}"
        
        # 设置过滤条件
        filters = {
            "knowledge_points": lesson.knowledge_points,
            "difficulty": (2, 4) if lesson.class_level != "冲竞赛" else (3, 5),
        }
        
        # 检索更多题目以便筛选
        all_questions = self.retrieve_questions(
            query=query,
            top_k=num_examples + num_practice + 5,  # 多检索一些
            filters=filters
        )
        
        # 选择例题（优先选择相似度高、难度适中的）
        examples = sorted(
            [q for q in all_questions if 2 <= q.get("difficulty", 3) <= 4],
            key=lambda x: x.get("similarity", 0),
            reverse=True
        )[:num_examples]
        
        # 选择练习题（排除已选为例题的）
        example_contents = {q["content"] for q in examples}
        practice = [
            q for q in all_questions
            if q["content"] not in example_contents
        ][:num_practice]
        
        return {
            "examples": examples,
            "practice": practice,
        }
    
    def build_handout_markdown(
        self,
        lesson: LessonTopic,
        examples: List[Dict[str, Any]],
        practices: List[Dict[str, Any]],
        template_style: str = "A",
    ) -> str:
        """
        使用RAG检索到的题目生成讲义Markdown
        """
        # 构建例题文本
        examples_text = "\n\n".join([
            f"例{i+1}：\n{q['content']}\n"
            f"知识点：{', '.join(q.get('knowledge_points', []))}\n"
            f"难度：{q.get('difficulty', 3)}/5"
            for i, q in enumerate(examples)
        ])
        
        # 构建练习题文本
        practices_text = "\n\n".join([
            f"练习{i+1}：\n{q['content']}"
            for i, q in enumerate(practices)
        ])
        
        # 构建prompt
        prompt = f"""
请根据以下信息生成一份数学讲义（Markdown格式）：

主题：{lesson.title}
年级：{lesson.grade}
地区：{lesson.region}
知识点：{', '.join(lesson.knowledge_points)}

例题（从本地试卷中检索到的真实题目）：
{examples_text}

练习题（从本地试卷中检索到的真实题目）：
{practices_text}

请按照以下结构生成讲义：
# {lesson.title}

## 一、知识点梳理
（包含定义、性质、常见错误等，要适合{lesson.grade}学生）

## 二、例题精讲
（对检索到的例题进行详细讲解，包括解题思路和步骤）

## 三、课堂练习
（列出检索到的练习题）

## 四、小结与反思
（总结重点，提出思考题）

要求：
1. 内容要适合{lesson.grade}学生，符合{lesson.region}的教学要求
2. 语言清晰易懂，逻辑严密
3. 例题解析要详细，体现解题思路
4. 结合检索到的真实题目，体现本地化特色
"""
        
        # 调用LLM生成讲义
        handout_content = self.llm.generate(prompt)
        
        return handout_content
    
    def generate_handout(
        self,
        topic: str,
        region: str,
        grade: str,
        class_level: Optional[str] = None,
        template_style: str = "A",
        num_examples: int = 3,
        num_practice: int = 5,
    ) -> Dict[str, Any]:
        """
        完整的讲义生成流程
        
        Returns:
        {
            "lesson_topic": LessonTopic,
            "examples": [...],
            "practice": [...],
            "handout_content": "...",
        }
        """
        # 1. 生成课程主题
        lesson = self.generate_lesson_topic(
            raw_topic_text=topic,
            region=region,
            grade=grade,
            class_level=class_level,
            template_style=template_style,
        )
        
        # 2. RAG检索题目
        questions = self.select_example_and_practice_questions(
            lesson=lesson,
            num_examples=num_examples,
            num_practice=num_practice,
        )
        
        # 3. 生成讲义
        handout_md = self.build_handout_markdown(
            lesson=lesson,
            examples=questions["examples"],
            practices=questions["practice"],
            template_style=template_style,
        )
        
        return {
            "lesson_topic": lesson,
            "examples": questions["examples"],
            "practice": questions["practice"],
            "handout_content": handout_md,
        }

