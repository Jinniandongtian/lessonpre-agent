"""QuestionExtractor 规则切题回归测试"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing.question_extractor import QuestionExtractor


class DummyLLM:
    def generate(self, prompt):
        raise RuntimeError("rule-based path should avoid llm fallback in these tests")


class TestQuestionExtractorRuleBased:
    def setup_method(self):
        self.extractor = QuestionExtractor(llm_client=DummyLLM())

    def test_extract_with_llm_keeps_rule_based_result_when_count_is_sufficient(self):
        text = """--- 第 1 页 ---
1. 已知函数f(x)=x^2+1，则（ ）
A. 1
B. 2
--- 第 2 页 ---
2. 已知直线l: x+y=0，则（ ）
A. 甲
B. 乙
"""
        questions = self.extractor._extract_with_llm(text, {})

        assert len(questions) == 2
        assert [self.extractor._extract_question_number(q["content"]) for q in questions] == ["1", "2"]
        assert questions[0]["options"] == {"A": "1", "B": "2"}
        assert questions[1]["options"] == {"A": "甲", "B": "乙"}

    def test_extract_question_blocks_across_pages_preserves_following_options(self):
        text = """--- 第 2 页 ---
10. 已知直线l: mx-y+1+3m=0，圆C:(x+2)^2+y^2=9，则（ ）
--- 第 3 页 ---
第3页/共5页
学科网（北京）股份有限公司
A. 选项甲
B. 选项乙
C. 选项丙
D. 选项丁
11. 下一题
"""
        questions = self.extractor._extract_questions_rule_based(text, {})

        assert len(questions) == 1
        q10 = questions[0]
        assert self.extractor._extract_question_number(q10["content"]) == "10"
        assert sorted(q10["options"].keys()) == ["A", "B", "C", "D"]
        assert "第3页/共5页" not in q10["content"]
        assert "学科网" not in q10["content"]
