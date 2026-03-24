"""题目描述生成器：模板描述（0成本）+ LLM 丰富描述（按需启用）"""
from typing import Dict, Any, Optional


class DescriptionGenerator:
    """
    生成题目的自然语言描述。
    description 字段不参与向量化，只用于增强 RAG 语义理解。
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def generate_basic_description(self, question: Dict[str, Any]) -> str:
        """
        基础描述：模板生成，0 成本。
        适合批量处理场景。
        """
        content = question.get('content', {})
        stem = content.get('stem_plain', '') or content.get('stem_latex', '')
        stem = (stem or '')[:80]

        q_meta = question.get('question_meta', {})
        question_type = q_meta.get('question_type', '') or question.get('question_type', '')
        knowledge_points = (
            q_meta.get('knowledge_points')
            or question.get('knowledge_points')
            or []
        )

        kp_str = '、'.join(knowledge_points) if knowledge_points else '基础知识'
        return f"{question_type}：{stem}...，考查{kp_str}"

    def generate_rich_description(self, question: Dict[str, Any]) -> str:
        """
        丰富描述：LLM 生成，增强 RAG 精度，按需启用。
        失败时自动降级为 generate_basic_description()。
        """
        if self.llm_client is None:
            return self.generate_basic_description(question)

        content = question.get('content', {})
        stem = content.get('stem_plain', '') or content.get('stem_latex', '')
        options = content.get('options_plain', {})
        q_meta = question.get('question_meta', {})
        knowledge_points = (
            q_meta.get('knowledge_points')
            or question.get('knowledge_points')
            or []
        )

        prompt = (
            "为以下数学题生成100字以内的描述，包括：\n"
            "1. 考查的核心概念\n"
            "2. 解题关键思路\n"
            "3. 常见错误陷阱\n\n"
            f"题干：{stem}\n"
            f"选项：{', '.join(f'{k}: {v}' for k, v in options.items())}\n"
            f"知识点：{', '.join(knowledge_points)}\n\n"
            "输出格式：核心概念：...\n解题思路：...\n常见错误：..."
        )
        try:
            result = self.llm_client.generate(prompt)
            if result and not result.startswith('['):
                return result.strip()
        except Exception as e:
            print(f"生成丰富描述失败: {e}，降级为基础描述")

        return self.generate_basic_description(question)
