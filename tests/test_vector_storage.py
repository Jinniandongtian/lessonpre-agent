"""向量存储功能单元测试"""
import sys
import os

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_models.question import Question
from src.utils.latex_normalizer import LaTeXNormalizer
from src.utils.description_generator import DescriptionGenerator


# ─────────────────────────────────────────
# 1. Question 数据模型
# ─────────────────────────────────────────

class TestQuestionModel:
    def test_to_dict_and_from_dict_roundtrip(self):
        q = Question(
            id="test_001",
            embedding_text="已知直线l的斜率为2经过点2,1",
            content={
                "stem_latex": "已知直线 $l$ 的斜率为 $2$",
                "stem_plain": "已知直线 l 的斜率为 2",
                "options_latex": {"A": "$2x-y-5=0$"},
                "options_plain": {"A": "2x-y-5=0"},
            },
            source_meta={"region": "山东省", "year": 2025, "grade": "高二"},
            question_meta={
                "question_type": "单选题",
                "difficulty": 3,
                "knowledge_points": ["直线方程"],
            },
            description="单选题：已知直线 l 的斜率为 2...，考查直线方程",
        )
        d = q.to_dict()
        q2 = Question.from_dict(d)

        assert q2.id == "test_001"
        assert q2.embedding_text == q.embedding_text
        assert q2.content["stem_plain"] == "已知直线 l 的斜率为 2"
        assert q2.source_meta["region"] == "山东省"
        assert q2.question_meta["question_type"] == "单选题"
        assert q2.description == q.description

    def test_from_dict_handles_missing_fields(self):
        q = Question.from_dict({"id": "x"})
        assert q.id == "x"
        assert q.embedding_text == ""
        assert q.content == {}
        assert q.description is None


# ─────────────────────────────────────────
# 2. LaTeXNormalizer
# ─────────────────────────────────────────

class TestLaTeXNormalizer:
    def setup_method(self):
        self.n = LaTeXNormalizer(llm_client=None)

    def test_inline_math_delimiters_removed(self):
        # \( \) \[ \] 包裹符应被移除，不留反斜杠
        result = self.n.latex_to_plain(r'\(x^{2}\)')
        assert '\\' not in result
        assert 'x' in result

    def test_no_residual_backslash(self):
        text = r'\(\overrightarrow{a} = (1,0,1),\) 则 \(|\overrightarrow{a}|\) 等于'
        result = self.n.latex_to_plain(text)
        assert '\\' not in result

    def test_dfrac_to_frac(self):
        result = self.n.post_process_latex(r"\dfrac{x}{y}")
        assert r"\frac" in result
        assert r"\dfrac" not in result

    def test_bare_superscript_gets_braces(self):
        result = self.n.post_process_latex(r"x^2")
        assert r"^{2}" in result

    def test_times_to_cdot(self):
        result = self.n.post_process_latex(r"a \times b")
        assert r"\cdot" in result
        assert r"\times" not in result

    def test_left_right_brackets_removed(self):
        result = self.n.post_process_latex(r"\left( x \right)")
        assert r"\left" not in result
        assert r"\right" not in result
        assert "(" in result

    def test_vec_to_overrightarrow(self):
        result = self.n.post_process_latex(r"\vec{AB}")
        assert r"\overrightarrow{AB}" in result

    def test_latex_to_plain_frac(self):
        result = self.n.latex_to_plain(r"\frac{x}{y}")
        assert "x/y" in result

    def test_latex_to_plain_nested_frac_preserves_structure(self):
        result = self.n.latex_to_plain(r"\frac{x^{2}}{4}")
        assert "x²/4" in result
        assert "{" not in result

    def test_latex_to_plain_superscript(self):
        result = self.n.latex_to_plain(r"x^{2}")
        assert "x" in result
        assert "2" in result

    def test_latex_to_plain_overrightarrow(self):
        result = self.n.latex_to_plain(r"\overrightarrow{AB}")
        assert "AB→" in result

    def test_latex_to_plain_overrightarrow_with_subscript(self):
        result = self.n.latex_to_plain(r"\overrightarrow{A_{1}C}")
        assert "A_1C→" in result

    def test_latex_to_plain_handles_double_escaped_latex(self):
        result = self.n.latex_to_plain(r"\\(\\overrightarrow{A_{1}C}=\\)")
        assert result == "A_1C→="

    def test_latex_to_plain_sqrt(self):
        result = self.n.latex_to_plain(r"\sqrt{3}")
        assert "3" in result

    def test_latex_to_plain_unwraps_text_and_symbols(self):
        result = self.n.latex_to_plain(r"\text{平面}\alpha \perp \beta")
        assert "平面α" in result
        assert "⊥" in result
        assert "\\" not in result

    def test_latex_to_plain_removes_dollar(self):
        result = self.n.latex_to_plain("$x$")
        assert "$" not in result

    def test_latex_to_plain_strips_code_fence_noise(self):
        text = "2x - 3y - 4 = 0\n\\boxed{2x - 3y - 4 = 0}\n```latex\n2x - 3y - 4 = 0\n```"
        result = self.n.latex_to_plain(text)
        assert "```" not in result
        assert result.count("2x - 3y - 4 = 0") == 1

    def test_build_embedding_text_removes_punctuation(self):
        stem = r"已知直线 $l$ 的斜率为 $2$，经过点 $(2,1)$"
        options = {"A": "$2x-y-5=0$", "B": "$2x+y-5=0$"}
        emb = self.n.build_embedding_text(stem, options)
        assert "$" not in emb
        assert "，" not in emb
        assert len(emb) > 0

    def test_normalize_latex_falls_back_to_regex_without_llm(self):
        result = self.n.normalize_latex(r"\dfrac{a}{b} \times c")
        assert r"\frac" in result
        assert r"\cdot" in result


# ─────────────────────────────────────────
# 3. DescriptionGenerator
# ─────────────────────────────────────────

class TestDescriptionGenerator:
    def setup_method(self):
        self.gen = DescriptionGenerator(llm_client=None)

    def _make_question(self, q_type="单选题", kp=None):
        return {
            "content": {
                "stem_plain": "已知直线l的斜率为2，经过点(2,1)",
                "options_plain": {"A": "2x-y-5=0", "B": "2x+y-5=0"},
            },
            "question_meta": {
                "question_type": q_type,
                "knowledge_points": kp if kp is not None else ["直线方程", "点斜式"],
                "difficulty": 3,
            },
        }

    def test_basic_description_contains_question_type(self):
        desc = self.gen.generate_basic_description(self._make_question())
        assert "单选题" in desc

    def test_basic_description_contains_knowledge_point(self):
        desc = self.gen.generate_basic_description(self._make_question())
        assert "直线方程" in desc

    def test_basic_description_fallback_kp(self):
        desc = self.gen.generate_basic_description(self._make_question(kp=[]))
        assert "基础知识" in desc

    def test_rich_description_falls_back_without_llm(self):
        desc_basic = self.gen.generate_basic_description(self._make_question())
        desc_rich = self.gen.generate_rich_description(self._make_question())
        assert desc_basic == desc_rich


# ─────────────────────────────────────────
# 4. EmbeddingModel.encode_question
# ─────────────────────────────────────────

class TestEncodeQuestion:
    def setup_method(self):
        from src.vector_store.embedding import EmbeddingModel
        os.environ["EMBEDDING_MODEL"] = "hash"
        self.model = EmbeddingModel()

    def test_encode_question_uses_embedding_text(self):
        q = {
            "embedding_text": "已知直线l斜率2经过点2,1",
            "content": {"stem_plain": "SHOULD NOT USE THIS"},
        }
        emb = self.model.encode_question(q)
        assert isinstance(emb, list)
        assert len(emb) > 0

    def test_encode_question_fallback_to_stem_plain(self):
        q = {
            "embedding_text": "",
            "content": {"stem_plain": "已知直线l的斜率为2"},
        }
        emb = self.model.encode_question(q)
        assert isinstance(emb, list)
        assert len(emb) > 0

    def test_encode_question_fallback_to_stem_latex(self):
        q = {
            "embedding_text": "",
            "content": {"stem_plain": "", "stem_latex": "已知直线 $l$ 斜率 $2$"},
        }
        emb = self.model.encode_question(q)
        assert isinstance(emb, list)
        assert len(emb) > 0

    def test_encode_question_old_format_content_string(self):
        q = {"content": "已知直线l的斜率为2"}
        emb = self.model.encode_question(q)
        assert isinstance(emb, list)
        assert len(emb) > 0


# ─────────────────────────────────────────
# 5. VectorDatabase._normalize_question_meta
# ─────────────────────────────────────────

class TestVectorDBNormalize:
    def setup_method(self):
        import tempfile
        from src.vector_store.vector_db import VectorDatabase
        self.tmp = tempfile.mkdtemp()
        self.db = VectorDatabase(storage_path=self.tmp, backend="simple")

    def test_normalize_new_format(self):
        q = {
            "id": "q001",
            "embedding_text": "直线方程",
            "content": {
                "stem_latex": "$x^{2}$",
                "stem_plain": "x2",
                "options_latex": {},
                "options_plain": {},
            },
            "source_meta": {"region": "山东省", "year": 2025},
            "question_meta": {
                "question_type": "单选题",
                "difficulty": 3,
                "knowledge_points": ["直线方程"],
            },
            "description": "单选题：x2...，考查直线方程",
        }
        entry = self.db._normalize_question_meta(q, 0)
        assert entry["id"] == "q001"
        assert entry["embedding_text"] == "直线方程"
        assert entry["question_type"] == "单选题"
        assert entry["description"] == "单选题：x2...，考查直线方程"
        assert entry["source_meta"]["region"] == "山东省"

    def test_normalize_old_format(self):
        q = {
            "id": "q002",
            "content": "已知直线l的斜率为2",
            "question_type": "选择题",
            "knowledge_points": ["直线"],
            "difficulty": 2,
            "source_meta": {"region": "北京"},
        }
        entry = self.db._normalize_question_meta(q, 1)
        assert entry["question_type"] == "选择题"
        assert entry["knowledge_points"] == ["直线"]
        assert entry["difficulty"] == 2


# ─────────────────────────────────────────
# 6. _split_stem_and_options
# ─────────────────────────────────────────

class TestSplitStemAndOptions:
    def setup_method(self):
        from src.data_processing.pdf_processor import _split_stem_and_options
        self.split = _split_stem_and_options

    def test_abcd_options_extracted(self):
        content = "1. 已知 x=2\nA. 1\nB. 2\nC. 3\nD. 4"
        stem, opts = self.split(content)
        assert 'A' in opts and 'B' in opts and 'C' in opts and 'D' in opts
        assert opts['A'] == '1'
        assert opts['D'] == '4'

    def test_stem_does_not_contain_options(self):
        content = "2. 求直线方程\nA. y=x\nB. y=2x\nC. y=3x\nD. y=4x"
        stem, opts = self.split(content)
        assert 'A.' not in stem and 'B.' not in stem

    def test_no_options_returns_full_content_as_stem(self):
        content = "填空题：已知 x=2，则 x²=___"
        stem, opts = self.split(content)
        assert stem == content.strip()
        assert opts == {}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
