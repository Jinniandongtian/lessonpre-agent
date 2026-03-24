"""题目增强：双表示法、embedding_text、description 生成"""
import re
from typing import Dict, Any

def _split_stem_and_options(content: str):
    """
    从 LLM 提取的纯字符串 content 中拆分题干和选项。
    选项通常以 A. / A、/ A) 开头，单独成行或在题干后。
    返回 (stem_str, options_dict)。
    """
    # 匹配选项行：A/B/C/D 开头，支持 . 、 ) 等分隔符
    option_pattern = re.compile(
        r'(?:^|\n)\s*([A-Da-d])[\.、\)\uff0e\uff09]\s*(.+?)(?=(?:\n\s*[A-Da-d][\.、\)\uff0e\uff09])|$)',
        re.DOTALL,
    )
    options = {}
    for m in option_pattern.finditer(content):
        key = m.group(1).upper()
        val = m.group(2).strip().replace('\n', ' ')
        options[key] = val

    if options:
        # 取第一个选项在原文中的位置，其前面为题干
        first_opt = option_pattern.search(content)
        stem = content[:first_opt.start()].strip() if first_opt else content.strip()
    else:
        stem = content.strip()

    return stem, options

# 是一个标准化选择题选项字典的函数，核心逻辑是过滤并规整选项的「键」和「值」——
# 只保留 A-D 大写字母作为键，且值为非空的有效文本，最终输出格式统一的选项字典。
def _normalize_options_dict(options: Any) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    if not isinstance(options, dict):
        return normalized
    for key, value in options.items():
        key_norm = str(key or "").strip().upper()
        if not re.fullmatch(r"[A-D]", key_norm):
            continue
        value_norm = str(value or "").strip()
        if value_norm:
            normalized[key_norm] = value_norm
    return normalized

# 判断是否为选择题
def _looks_like_choice_question(question: Dict[str, Any], content: str) -> bool:
    question_type = str(question.get("question_type", "") or "")
    if "选择" in question_type:
        return True
    return bool(re.search(r'(?:^|\n)\s*[A-Da-d][\.、\)\uff0e\uff09]\s*', content or ""))

# 判断「是否需要用新拆分的选项补充 / 替换已有选项」的函数，核心逻辑是优先看选项数量、再结合选择题特征判断 —— 新选项更全 / 更完整时，
# 就返回 True（需要补充），否则返回 False。
def _should_backfill_options(
    question: Dict[str, Any],
    content: str,
    existing_options: Dict[str, str],
    split_options: Dict[str, str],
) -> bool:
    if not split_options:
        return False
    if not existing_options:
        return True
    if len(split_options) > len(existing_options):
        return True
    if _looks_like_choice_question(question, content):
        expected = {"A", "B", "C", "D"}
        existing_keys = set(existing_options.keys())
        split_keys = set(split_options.keys())
        if expected.issubset(split_keys) and not expected.issubset(existing_keys):
            return True
    return False


def enrich_question_with_representations(
    question: Dict[str, Any],
    llm_client,
    normalizer,
    desc_generator,
) -> Dict[str, Any]:
    """
    为原始题目添加双表示法、embedding_text 和 description。

    流程：
    1. 保留原始题面文本（不再把 raw 文本强行当作 LaTeX 规范化）
    2. 转换为普通文本（plain）
    3. 生成 embedding_text（向量化的唯一输入）
    4. 生成 description（可选，增强RAG）
    5. 组装并返回 Question 的 to_dict() 结构
    """
    from ..data_models.question import Question

    # 1. 从原始题目中解析 stem 和 options
    # LLM 提取的题目通常把选项塞在 content 字符串里，需要先拆分
    # LLM 提取的题目格式不统一（有时 stem/options 分开，有时混在 content 字符串里，有时是结构化 dict），
    # 这段代码通过多层回填逻辑，确保最终 stem_raw 和 options_raw 都被正确填充，为后续的 LaTeX 规范化做准备
    stem_raw = question.get('stem_raw', '') or question.get('stem', '')
    options_raw = _normalize_options_dict(question.get('options_raw', {}) or question.get('options', {}))
    content_obj = question.get('content', '')

    content_str = ""
    if isinstance(content_obj, str):
        content_str = content_obj
    elif isinstance(content_obj, dict):
        structured_stem = content_obj.get('stem_latex') or content_obj.get('stem_plain', '')
        structured_options = _normalize_options_dict(
            content_obj.get('options_latex') or content_obj.get('options_plain', {})
        )
        if not stem_raw and structured_stem:
            stem_raw = structured_stem
        if _should_backfill_options(question, structured_stem, options_raw, structured_options):
            options_raw = structured_options

    # 即使 stem 已有值，只要 options 缺失/不完整且 content 能拆出更多选项，就从原始 content 回填
    if isinstance(content_str, str) and content_str:
        split_stem, split_options = _split_stem_and_options(content_str)
        split_options = _normalize_options_dict(split_options)
        if not stem_raw and split_stem:
            stem_raw = split_stem
        if _should_backfill_options(question, content_str, options_raw, split_options):
            options_raw = split_options

    # 2. 将原始题面转换为 LaTeX 友好表示，避免 stem_latex 只是“原始脏文本换名”
    stem_latex = normalizer.text_to_latex(str(stem_raw or "").strip())
    options_latex = {
        k: normalizer.text_to_latex(str(v or "").strip())
        for k, v in options_raw.items()
        if str(v or "").strip()
    }

    # 2. 转换为普通文本
    stem_plain = normalizer.latex_to_plain(stem_latex)
    options_plain = {k: normalizer.latex_to_plain(v) for k, v in options_latex.items()}

    # 3. 生成 embedding_text
    embedding_text = normalizer.build_embedding_text(stem_latex, options_latex)

    # 4. 组装 question_dict 用于描述生成
    question_dict = {
        'content': {
            'stem_raw': str(stem_raw or "").strip(),
            'stem_plain': stem_plain,
            'stem_latex': stem_latex,
            'options_raw': {k: str(v or "").strip() for k, v in options_raw.items() if str(v or "").strip()},
            'options_plain': options_plain,
            'options_latex': options_latex,
        },
        'question_meta': {
            'question_type': question.get('question_type', ''),
            'difficulty': question.get('difficulty', 3),
            'knowledge_points': question.get('knowledge_points', []),
        },
    }
    description = desc_generator.generate_basic_description(question_dict)

    # 5. 组装 Question 对象并返回字典
    q = Question(
        id=question.get('id', ''),
        embedding_text=embedding_text,
        content={
            'stem_raw': str(stem_raw or "").strip(),
            'stem_latex': stem_latex,
            'stem_plain': stem_plain,
            'options_raw': {k: str(v or "").strip() for k, v in options_raw.items() if str(v or "").strip()},
            'options_latex': options_latex,
            'options_plain': options_plain,
        },
        source_meta=question.get('source_meta', {}),
        question_meta={
            'question_type': question.get('question_type', ''),
            'difficulty': question.get('difficulty', 3),
            'knowledge_points': question.get('knowledge_points', []),
        },
        description=description,
    )
    return q.to_dict()
