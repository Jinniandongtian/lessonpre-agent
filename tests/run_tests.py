"""快速验证脚本 - 不依赖 pytest"""
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_models.question import Question
from src.utils.latex_normalizer import LaTeXNormalizer
from src.utils.description_generator import DescriptionGenerator

passed = 0
failed = 0

def ok(name):
    global passed
    passed += 1
    print(f"  PASS  {name}")

def fail(name, err):
    global failed
    failed += 1
    print(f"  FAIL  {name}: {err}")

def check(name, fn):
    try:
        assert fn(), f"{name} returned False"
        ok(name)
    except Exception as e:
        fail(name, e)

print("\n=== 1. Question 数据模型 ===")
try:
    q = Question(
        id="t1",
        embedding_text="test embedding",
        content={"stem_plain": "x", "stem_latex": "$x$"},
        source_meta={"region": "山东"},
        question_meta={"question_type": "单选题", "difficulty": 3, "knowledge_points": ["函数"]},
        description="单选题：x...，考查函数",
    )
    d = q.to_dict()
    q2 = Question.from_dict(d)
    assert q2.id == "t1"
    assert q2.embedding_text == "test embedding"
    assert q2.content["stem_plain"] == "x"
    assert q2.source_meta["region"] == "山东"
    assert q2.question_meta["question_type"] == "单选题"
    assert q2.description == "单选题：x...，考查函数"
    ok("Question to_dict/from_dict roundtrip")
except Exception as e:
    fail("Question to_dict/from_dict roundtrip", e)

try:
    q = Question.from_dict({"id": "x"})
    assert q.embedding_text == ""
    assert q.content == {}
    assert q.description is None
    ok("Question.from_dict missing fields")
except Exception as e:
    fail("Question.from_dict missing fields", e)

print("\n=== 2. LaTeXNormalizer ===")
n = LaTeXNormalizer(llm_client=None)

check("dfrac→frac",
      lambda: "\\frac" in n.post_process_latex(r"\dfrac{x}{y}") and "\\dfrac" not in n.post_process_latex(r"\dfrac{x}{y}"))
check("x^2→x^{2}",
      lambda: "^{2}" in n.post_process_latex(r"x^2"))
check("times→cdot",
      lambda: "\\cdot" in n.post_process_latex(r"a \times b") and "\\times" not in n.post_process_latex(r"a \times b"))
check("left/right removed",
      lambda: "\\left" not in n.post_process_latex(r"\left( x \right)") and "(" in n.post_process_latex(r"\left( x \right)"))
check("vec→overrightarrow",
      lambda: "\\overrightarrow{AB}" in n.post_process_latex(r"\vec{AB}"))
check("frac→plain x/y",
      lambda: "x/y" in n.latex_to_plain(r"\frac{x}{y}"))
check("sqrt→plain",
      lambda: "3" in n.latex_to_plain(r"\sqrt{3}"))
check("dollar removed",
      lambda: "$" not in n.latex_to_plain("$x$"))
check("build_embedding_text removes 汉字标点",
      lambda: "，" not in n.build_embedding_text("已知，直线l", {}) and len(n.build_embedding_text("已知直线l", {})) > 0)
check("normalize falls back to regex without llm",
      lambda: "\\frac" in n.normalize_latex(r"\dfrac{a}{b} \times c") and "\\cdot" in n.normalize_latex(r"\dfrac{a}{b} \times c"))
check("inline \\( \\) delimiters removed, no backslash",
      lambda: "\\" not in n.latex_to_plain(r"\(x^{2}\)") and "x" in n.latex_to_plain(r"\(x^{2}\)"))
check("no residual backslash in complex expr",
      lambda: "\\" not in n.latex_to_plain(r"\(\overrightarrow{a} = (1,0,1),\) 则 \(|\overrightarrow{a}|\) 等于"))

print("\n=== 3. DescriptionGenerator ===")
g = DescriptionGenerator(llm_client=None)
q_dict = {
    "content": {"stem_plain": "已知直线l的斜率为2"},
    "question_meta": {"question_type": "单选题", "knowledge_points": ["直线方程"], "difficulty": 3},
}
check("basic desc contains question_type",
      lambda: "单选题" in g.generate_basic_description(q_dict))
check("basic desc contains knowledge_point",
      lambda: "直线方程" in g.generate_basic_description(q_dict))
check("basic desc fallback kp",
      lambda: "基础知识" in g.generate_basic_description(
          {"content": {"stem_plain": "x"}, "question_meta": {"question_type": "填空题", "knowledge_points": [], "difficulty": 2}}
      ))
check("rich desc == basic when no llm",
      lambda: g.generate_basic_description(q_dict) == g.generate_rich_description(q_dict))

print("\n=== 4. EmbeddingModel.encode_question ===")
os.environ["EMBEDDING_MODEL"] = "hash"
from src.vector_store.embedding import EmbeddingModel
em = EmbeddingModel()

check("encode_question uses embedding_text",
      lambda: len(em.encode_question({"embedding_text": "直线方程", "content": {"stem_plain": "IGNORE"}})) > 0)
check("encode_question fallback stem_plain",
      lambda: len(em.encode_question({"embedding_text": "", "content": {"stem_plain": "直线方程"}})) > 0)
check("encode_question fallback stem_latex",
      lambda: len(em.encode_question({"embedding_text": "", "content": {"stem_plain": "", "stem_latex": "$l$"}})) > 0)
check("encode_question old str content",
      lambda: len(em.encode_question({"content": "已知直线l"})) > 0)

print("\n=== 5. VectorDatabase._normalize_question_meta ===")
from src.vector_store.vector_db import VectorDatabase
tmp = tempfile.mkdtemp()
db = VectorDatabase(storage_path=tmp, backend="simple")

try:
    q_new = {
        "id": "q001",
        "embedding_text": "直线方程",
        "content": {"stem_latex": "$x$", "stem_plain": "x", "options_latex": {}, "options_plain": {}},
        "source_meta": {"region": "山东省", "year": 2025},
        "question_meta": {"question_type": "单选题", "difficulty": 3, "knowledge_points": ["直线方程"]},
        "description": "单选题：x...，考查直线方程",
    }
    entry = db._normalize_question_meta(q_new, 0)
    assert entry["id"] == "q001"
    assert entry["embedding_text"] == "直线方程"
    assert entry["question_type"] == "单选题"
    assert entry["source_meta"]["region"] == "山东省"
    assert entry["description"] == "单选题：x...，考查直线方程"
    ok("_normalize_question_meta new format")
except Exception as e:
    fail("_normalize_question_meta new format", e)

try:
    q_old = {
        "id": "q002",
        "content": "已知直线l",
        "question_type": "选择题",
        "knowledge_points": ["直线"],
        "difficulty": 2,
        "source_meta": {"region": "北京"},
    }
    entry = db._normalize_question_meta(q_old, 1)
    assert entry["question_type"] == "选择题"
    assert entry["knowledge_points"] == ["直线"]
    assert entry["difficulty"] == 2
    ok("_normalize_question_meta old format")
except Exception as e:
    fail("_normalize_question_meta old format", e)

print("\n=== 6. _split_stem_and_options ===")
from src.data_processing.pdf_processor import _split_stem_and_options

try:
    content = "1. 已知 x=2\nA. 1\nB. 2\nC. 3\nD. 4"
    stem, opts = _split_stem_and_options(content)
    assert 'A' in opts and opts['A'] == '1' and opts['D'] == '4'
    ok("_split_stem_and_options ABCD extracted")
except Exception as e:
    fail("_split_stem_and_options ABCD extracted", e)

try:
    content = "2. 求直线方程\nA. y=x\nB. y=2x\nC. y=3x\nD. y=4x"
    stem, opts = _split_stem_and_options(content)
    assert 'A.' not in stem and 'B.' not in stem
    ok("_split_stem_and_options stem has no options")
except Exception as e:
    fail("_split_stem_and_options stem has no options", e)

try:
    content = "填空题：已知 x=2，则 x²=___"
    stem, opts = _split_stem_and_options(content)
    assert opts == {} and stem == content.strip()
    ok("_split_stem_and_options no-options returns full stem")
except Exception as e:
    fail("_split_stem_and_options no-options returns full stem", e)

print(f"\n===== {passed} passed, {failed} failed =====")
sys.exit(0 if failed == 0 else 1)
