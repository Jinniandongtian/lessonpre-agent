"""QuestionEnricher 回填选项回归测试"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing.question_enricher import enrich_question_with_representations
from src.utils.description_generator import DescriptionGenerator
from src.utils.latex_normalizer import LaTeXNormalizer
from src.data_processing.question_extractor import QuestionExtractor


class TestQuestionEnricherBackfill:
    def setup_method(self):
        self.normalizer = LaTeXNormalizer(llm_client=None)
        self.desc_generator = DescriptionGenerator(llm_client=None)

    def test_backfills_options_from_content_even_when_stem_exists(self):
        question = {
            "id": "q10",
            "stem": "10. 已知直线l: mx-y+1+3m=0，圆C:(x+2)^2+y^2=9，则（    ）",
            "options": {},
            "content": "10. 已知直线l: mx-y+1+3m=0，圆C:(x+2)^2+y^2=9，则（    ）\nA. 甲\nB. 乙\nC. 丙\nD. 丁",
            "question_type": "选择题",
            "knowledge_points": ["直线与圆"],
            "difficulty": 3,
        }

        enriched = enrich_question_with_representations(
            question,
            llm_client=None,
            normalizer=self.normalizer,
            desc_generator=self.desc_generator,
        )

        assert sorted(enriched["content"]["options_plain"].keys()) == ["A", "B", "C", "D"]
        assert enriched["content"]["options_plain"]["D"] == "丁"
        assert "A." not in enriched["content"]["stem_plain"]

    def test_backfills_more_complete_options_from_content(self):
        question = {
            "id": "q11",
            "stem": "11. 已知椭圆C: ...，则（    ）",
            "options": {"A": "旧选项A"},
            "content": "11. 已知椭圆C: ...，则（    ）\nA. 新选项A\nB. 新选项B\nC. 新选项C\nD. 新选项D",
            "question_type": "选择题",
            "knowledge_points": ["椭圆"],
            "difficulty": 3,
        }

        enriched = enrich_question_with_representations(
            question,
            llm_client=None,
            normalizer=self.normalizer,
            desc_generator=self.desc_generator,
        )

        options_plain = enriched["content"]["options_plain"]
        assert sorted(options_plain.keys()) == ["A", "B", "C", "D"]
        assert options_plain["A"] == "新选项A"
        assert options_plain["D"] == "新选项D"

    def test_stem_latex_is_converted_to_latex_friendly_text(self):
        question = {
            "id": "q_set",
            "stem": "已知集合A={1,2,3}，x∈A，AB→⊥平面α",
            "options": {},
            "content": "已知集合A={1,2,3}，x∈A，AB→⊥平面α",
            "question_type": "填空题",
            "knowledge_points": ["集合"],
            "difficulty": 2,
        }

        enriched = enrich_question_with_representations(
            question,
            llm_client=None,
            normalizer=self.normalizer,
            desc_generator=self.desc_generator,
        )

        assert enriched["content"]["stem_raw"] == "已知集合A={1,2,3}，x∈A，AB→⊥平面α"
        assert enriched["content"]["options_raw"] == {}
        assert r"\{1,2,3\}" in enriched["content"]["stem_latex"]
        assert r"\in" in enriched["content"]["stem_latex"]
        assert r"\overrightarrow{AB}" in enriched["content"]["stem_latex"]
        assert "{1,2,3}" in enriched["content"]["stem_plain"]


class TestQuestionExtractorOptionBoundaries:
    def setup_method(self):
        self.extractor = QuestionExtractor(llm_client=None)

    def test_extract_options_stops_before_next_section_heading(self):
        content = (
            "8. 示例题\n"
            "A. 甲\n"
            "B. 乙\n"
            "C. 丙\n"
            "D. 丁\n"
            "二、选择题：本题共3小题"
        )
        options = self.extractor._extract_options_from_content(content)
        assert options == {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}

    def test_extract_options_trims_scoring_instruction_tail(self):
        content = (
            "8. 示例题\n"
            "A. 甲\n"
            "B. 乙\n"
            "C. 丙\n"
            "D. 丁 要求.全部选对的得6分，部分选对的得部分分，有选错的得0分.\n"
        )
        options = self.extractor._extract_options_from_content(content)
        assert options == {"A": "甲", "B": "乙", "C": "丙", "D": "丁"}

    def test_inline_a_b_in_solution_question_are_not_treated_as_options(self):
        content = (
            "15. 已知集合A={x | x-3 > 1}，集合B={x | (x+a)(x-2a) > 0}．\n"
            "（1）当a=1时，求A∩B；\n"
            "（2）若“x∈A”是“x∈B”的充分不必要条件，求实数a的取值范围．"
        )
        options = self.extractor._extract_options_from_content(content)
        q_type = self.extractor._infer_question_type_heuristic(content)
        assert options == {}
        assert q_type == "解答题"
