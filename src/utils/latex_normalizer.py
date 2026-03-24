"""LaTeX 规范化器：LLM 规范化 + 正则兜底"""
import os
import re
from typing import Dict, Optional, List


class LaTeXNormalizer:
    """
    LaTeX 格式规范化。
    先尝试 LLM，失败时自动降级到正则表达式兜底。
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    _SUPERSCRIPT_MAP = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
        'n': 'ⁿ', 'i': 'ⁱ',
    }

    _COMMAND_SYMBOLS = {
        'cdot': '·',
        'times': '×',
        'div': '/',
        'pm': '±',
        'mp': '∓',
        'leq': '≤',
        'le': '≤',
        'geq': '≥',
        'ge': '≥',
        'neq': '≠',
        'ne': '≠',
        'approx': '≈',
        'sim': '∼',
        'infty': '∞',
        'triangle': '△',
        'angle': '∠',
        'perp': '⊥',
        'parallel': '∥',
        'mid': '|',
        'cdots': '...',
        'ldots': '...',
        'dots': '...',
        'to': '→',
        'rightarrow': '→',
        'leftarrow': '←',
        'Rightarrow': '⇒',
        'Leftarrow': '⇐',
        'alpha': 'α',
        'beta': 'β',
        'gamma': 'γ',
        'delta': 'δ',
        'Delta': 'Δ',
        'theta': 'θ',
        'lambda': 'λ',
        'mu': 'μ',
        'pi': 'π',
        'sigma': 'σ',
        'phi': 'φ',
        'omega': 'ω',
        'sin': 'sin',
        'cos': 'cos',
        'tan': 'tan',
        'cot': 'cot',
        'sec': 'sec',
        'csc': 'csc',
        'log': 'log',
        'ln': 'ln',
        'exp': 'exp',
        'min': 'min',
        'max': 'max',
        'cap': '∩',
        'cup': '∪',
        'subset': '⊂',
        'subseteq': '⊆',
        'supset': '⊃',
        'supseteq': '⊇',
        'in': '∈',
        'notin': '∉',
        'because': '∵',
        'therefore': '∴',
    }

    _DROP_COMMANDS = {
        'left',
        'right',
        'limits',
        'nolimits',
        'displaystyle',
        'textstyle',
        'scriptstyle',
        'scriptscriptstyle',
        'qquad',
        'quad',
    }

    _UNWRAP_COMMANDS = {
        'text',
        'mathrm',
        'mathbf',
        'mathit',
        'mathbb',
        'mathcal',
        'operatorname',
        'operatorname*',
        'boxed',
        'mbox',
        'textrm',
        'textbf',
        'underline',
        'overline',
    }

    _TEXT_SYMBOL_TO_LATEX = {
        '∀': r'\forall',
        '∃': r'\exists',
        '∈': r'\in',
        '∉': r'\notin',
        '∪': r'\cup',
        '∩': r'\cap',
        '⊂': r'\subset',
        '⊆': r'\subseteq',
        '⊃': r'\supset',
        '⊇': r'\supseteq',
        '⊥': r'\perp',
        '∥': r'\parallel',
        '∠': r'\angle',
        '△': r'\triangle',
        '∞': r'\infty',
        '≤': r'\leq',
        '≥': r'\geq',
        '≠': r'\neq',
        '±': r'\pm',
        '∓': r'\mp',
        '∴': r'\therefore',
        '∵': r'\because',
        'α': r'\alpha',
        'β': r'\beta',
        'γ': r'\gamma',
        'δ': r'\delta',
        'Δ': r'\Delta',
        'θ': r'\theta',
        'λ': r'\lambda',
        'μ': r'\mu',
        'π': r'\pi',
        'σ': r'\sigma',
        'φ': r'\phi',
        'ω': r'\omega',
    }

    def normalize_latex(self, latex_text: str) -> str:
        """
        将各种 LaTeX 写法统一为标准形式。
        优先使用 LLM，失败时降级到 post_process_latex()。
        """
        if not latex_text or not latex_text.strip():
            return latex_text

        if self.llm_client is not None:
            prompt = (
                "将以下 LaTeX 公式规范化为标准形式。规范化规则：\n"
                "1. 分数统一用 \\frac{分子}{分母}（不用 \\dfrac, \\tfrac）\n"
                "2. 上标统一用 x^{2}（不用 x^2）\n"
                "3. 乘号统一用 \\cdot（不用 *, ×, \\times）\n"
                "4. 除号统一用 \\div（不用 /）\n"
                "5. 根号统一用 \\sqrt{x}\n"
                "6. 括号统一用 ( ) 而不是 \\left( \\right)\n"
                "7. 向量统一用 \\overrightarrow{AB}\n"
                "8. 移除多余空格\n\n"
                f"原 LaTeX：{latex_text}\n"
                "输出：规范化后的 LaTeX（仅输出公式，不要解释）"
            )
            try:
                result = self.llm_client.generate(prompt)
                if result and not result.startswith("["):
                    return result.strip()
            except Exception as e:
                print(f"LLM LaTeX 规范化失败: {e}，降级到正则")

        return self.post_process_latex(latex_text)

    def post_process_latex(self, latex_text: str) -> str:
        """正则表达式兜底，处理最常见的变体"""
        t = latex_text
        # dfrac/tfrac → frac
        t = re.sub(r'\\[dt]frac', r'\\frac', t)
        # x^2 → x^{2}（仅数字上标且尚未带花括号）
        t = re.sub(r'\^(\d+)(?!\{)', r'^{\1}', t)
        # \times → \cdot
        t = re.sub(r'\\times', r'\\cdot', t)
        # \left( → (，\right) → )
        t = re.sub(r'\\left\(', '(', t)
        t = re.sub(r'\\right\)', ')', t)
        # \vec{AB} → \overrightarrow{AB}
        t = re.sub(r'\\vec\{([^}]+)\}', r'\\overrightarrow{\1}', t)
        # 多余空格压缩
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    # 让数学符号和字母 / 数字之间自动加空格
    def _normalize_text_to_latex_spacing(self, text: str) -> str:
        t = re.sub(
            r'(\\(?:forall|exists|in|notin|cup|cap|subseteq|subset|supseteq|supset|perp|parallel|angle|triangle|infty|leq|geq|neq|pm|mp|therefore|because|alpha|beta|gamma|delta|Delta|theta|lambda|mu|pi|sigma|phi|omega))(?=[A-Za-z0-9])',
            r'\1 ',
            text,
        )
        t = re.sub(r'\s+', ' ', t)
        t = re.sub(r' *\n *', '\n', t)
        return t.strip()

    # 将普通文本转换为latex格式
    def _text_to_latex_rule_based(self, text: str) -> str:
        if not text:
            return ""

        t = str(text).replace('\r\n', '\n').replace('\r', '\n')

        if "\\" in t:
            t = self.post_process_latex(t)
        else:
            t = re.sub(r'(?<!\\)\{', r'\\{', t)
            t = re.sub(r'(?<!\\)\}', r'\\}', t)
            t = re.sub(r'([A-Za-z0-9)\]])²', r'\1^{2}', t)
            t = re.sub(r'([A-Za-z0-9)\]])³', r'\1^{3}', t)
            t = re.sub(r'([A-Za-z0-9_]+)\s*→', r'\\overrightarrow{\1}', t)
            t = re.sub(r'→\s*([A-Za-z0-9_]+)', r'\\overrightarrow{\1}', t)
            t = re.sub(r'√\s*([A-Za-z0-9_]+)', r'\\sqrt{\1}', t)
            for symbol, latex in self._TEXT_SYMBOL_TO_LATEX.items():
                t = t.replace(symbol, latex)
            t = self.post_process_latex(t)

        return self._normalize_text_to_latex_spacing(t)

    # 检查文本里的括号、大括号是否成对、不缺失、不多余
    def _has_balanced_delimiters(self, text: str) -> bool:
        pairs = {'(': ')', '[': ']'}
        for open_char, close_char in pairs.items():
            if text.count(open_char) != text.count(close_char):
                return False
        if len(re.findall(r'(?<!\\)\{', text)) != len(re.findall(r'(?<!\\)\}', text)):
            return False
        return True

    def _collect_text_to_latex_risks(self, raw_text: str, draft_latex: str) -> List[str]:
        # 存储所有风险标签（比如 "括号不匹配"）
        risks: List[str] = []
        raw = raw_text or ""    # 原始文本
        draft = draft_latex or ""  # 转换后的LaTeX草稿

        # 如果原始文本里 没有任何数字/字母/数学符号 → 直接无风险
        if not re.search(r'[0-9A-Za-zα-ωΑ-Ω+\-*/=<>≤≥≠√∈∉∪∩∥⊥∠→πλμθ{}[\]()]', raw):
            return risks

        # 【可疑模式列表】：出现这些，大概率是OCR识别错 / 格式乱
        suspicious_patterns = [
            r'\|\|',       # 双竖线
            r'ð',          # 奇怪字符
            r'\^\+',       # 错误上标
            r'\+\+',       # 双重加号
            r'[A-Za-z0-9]/\|', # 奇怪分式
            r'\| x \|',    # 奇怪绝对值
            r'[=<>]\s*0\b',# 错误等号格式
            r'\b\d+\s+\d+\b', # 数字中间有空格（可能是分数被拆了）
        ]
        # 命中任意一个 → 标记可疑
        if any(re.search(pat, raw) for pat in suspicious_patterns):
            risks.append("suspicious_raw_pattern")

        # 【括号配对检查】（调用你刚学的方法）
        if not self._has_balanced_delimiters(raw):
            risks.append("raw_unbalanced_delimiters")
        if not self._has_balanced_delimiters(draft):
            risks.append("draft_unbalanced_delimiters")

        # 【危险检查】LaTeX里居然还残留中文Unicode符号（说明没转干净）
        if re.search(r'[∀∃∈∉∪∩⊂⊆⊃⊇⊥∥∠△∞≤≥≠±∓∴∵αβγδΔθλμπσφω]', draft):
            risks.append("unicode_symbol_left_in_draft")

        # 【集合大括号丢失】原始有 {x|x>0}，但LaTeX里没保住 { → 风险
        if re.search(r'(?<!\\)\{[^{}]*[|∣][^{}]*\}', raw) and r'\{' not in draft:
            risks.append("set_brace_not_preserved")

        # 【多子题风险】文本短，但包含 (1) (2) 多个子问题 → 容易解析混乱
        if len(raw) <= 1200 and len(re.findall(r'[（(]\d+[）)]', raw)) >= 2:
            risks.append("multi_subquestion_math_text")

        # 返回所有风险标签
        return risks

    def _should_use_llm_for_text_to_latex(self, raw_text: str, draft_latex: str) -> bool:
        # 1. 没有配置大模型客户端 → 绝对不调用
        if self.llm_client is None:
            return False

        # 2. 环境变量关闭了AI fallback → 不调用
        enabled = str(os.getenv("TEXT_TO_LATEX_LLM_FALLBACK", "1")).strip().lower()
        if enabled in {"0", "false", "no"}:
            return False

        # 3. 文本太长（超过1200字符）→ 不调用（省钱+防超时）
        if len(raw_text or "") > int(os.getenv("TEXT_TO_LATEX_LLM_MAX_LEN", "1200")):
            return False

        # 4. 【核心！】检查之前的风险列表 → 有风险就调用AI
        # 风险：括号不匹配、符号错乱、格式可疑、多子题等
        return bool(self._collect_text_to_latex_risks(raw_text, draft_latex))

    def _repair_text_to_latex_with_llm(self, raw_text: str, draft_latex: str) -> str:
        # 1. 没有AI客户端 → 直接返回草稿，不修复
        if self.llm_client is None:
            return draft_latex

        # 2. 【核心】给AI发的指令（超级严谨）
        prompt = (
            "请将下面数学题面中的数学表达修正为规范 LaTeX，并保持中文原文、句式、标点、换行不变。\n"
            "要求：\n"
            "1. 只修正数学表达，不要改写题意，不要补不存在的信息。\n"
            "2. 集合花括号写成 \\{ \\}；分数写成 \\frac{a}{b}；根号写成 \\sqrt{}；向量写成 \\overrightarrow{AB}。\n"
            "3. 若某段内容无法确定，请保留原文，不要猜。\n"
            "4. 只输出修正后的完整文本，不要解释。\n\n"
            f"原文：\n{raw_text}\n\n"
            f"规则草稿：\n{draft_latex}\n"
        )

        # 3. 调用AI，尝试修复
        try:
            repaired = self.llm_client.generate(prompt).strip()
            # 修复成功且结果正常 → 返回AI修复版
            if repaired and not repaired.startswith("["):
                return repaired
        except Exception as e:
            # AI崩了/超时/报错 → 打印日志，返回草稿
            print(f"LLM text_to_latex 修复失败: {e}，保留规则结果")

        # 4. 兜底：任何情况失败，都退回规则草稿（保证系统不崩）
        return draft_latex

    def text_to_latex(self, text: str) -> str:
        """
        将原始数学文本转换为 LaTeX。
        规则优先；仅在规则结果低置信度时，才调用 LLM 做保守修复。
        """
        if not text:
            return ""
        raw_text = str(text).replace('\r\n', '\n').replace('\r', '\n').strip()
        draft_latex = self._text_to_latex_rule_based(raw_text)
        if self._should_use_llm_for_text_to_latex(raw_text, draft_latex):
            repaired = self._repair_text_to_latex_with_llm(raw_text, draft_latex)
            return self._normalize_text_to_latex_spacing(repaired)
        return draft_latex

    def _read_balanced(self, text: str, start: int, open_char: str, close_char: str):
        """读取成对括号内的原始内容，支持嵌套。"""
        if start >= len(text) or text[start] != open_char:
            return None, start

        depth = 0
        i = start
        chars = []
        while i < len(text):
            ch = text[i]
            if ch == open_char:
                depth += 1
                if depth > 1:
                    chars.append(ch)
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return ''.join(chars), i + 1
                chars.append(ch)
            else:
                chars.append(ch)
            i += 1
        return ''.join(chars), i

    # 从指定位置读取 LaTeX 命令名（如 \frac 中读取 frac），返回命令名 + 下一个字符位置
    def _read_command_name(self, text: str, start: int):
        if start >= len(text):
            return "", start
        if re.match(r'[A-Za-z]', text[start]):
            end = start
            while end < len(text) and (re.match(r'[A-Za-z]', text[end]) or text[end] == '*'):
                end += 1
            return text[start:end], end
        return text[start], start + 1

    # 读取 LaTeX 命令的参数（如 \frac{1}{2} 中的 1 和 2），返回参数文本 + 下一个位置
    def _read_argument_plain(self, text: str, start: int):
        i = start
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            return "", i
        if text[i] == '{':
            raw, end = self._read_balanced(text, i, '{', '}')
            return self._convert_latex_fragment(raw), end
        if text[i] == '[':
            raw, end = self._read_balanced(text, i, '[', ']')
            return self._convert_latex_fragment(raw), end
        if text[i] == '\\':
            plain, end = self._convert_latex_fragment(text[i:], stop_after_token=True)
            return plain, i + end
        return text[i], i + 1

    # 格式化上标（如 x^2 → x² 或 x^2）
    def _format_superscript(self, token: str) -> str:
        token = (token or "").strip()
        if not token:
            return ""
        if all(ch in self._SUPERSCRIPT_MAP for ch in token):
            return ''.join(self._SUPERSCRIPT_MAP[ch] for ch in token)
        if len(token) == 1:
            return f"^{token}"
        return f"^({token})"
    
    # _format_subscript
    def _format_subscript(self, token: str) -> str:
        token = (token or "").strip()
        if not token:
            return ""
        if re.fullmatch(r'[A-Za-z0-9]+', token):
            return f"_{token}"
        return f"_({token})"
    # 格式化向量符号（如 \vec{AB} → AB→）
    def _format_vector(self, token: str, direction: str = "right") -> str:
        token = (token or "").strip()
        if not token:
            return ""
        if direction == "left":
            return f"{token}←"
        return f"{token}→"

    def _dedupe_adjacent_lines(self, text: str) -> str:
        if not text:
            return ""
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if lines and lines[-1] == line:
                continue
            lines.append(line)
        return '\n'.join(lines)

    def _convert_latex_fragment(self, text: str, stop_after_token: bool = False):
        result = []
        i = 0
        while i < len(text):
            ch = text[i]

            if ch == '\\':
                command, next_i = self._read_command_name(text, i + 1)

                if command in {'(', ')', '[', ']'}:
                    i = next_i
                    if stop_after_token:
                        break
                    continue
                if command in {'{', '}', '$', '%', '&', '#', '_'}:
                    result.append(command)
                    i = next_i
                    if stop_after_token:
                        break
                    continue
                if command in {',', ';', ':', '!', ' '}:
                    result.append(' ')
                    i = next_i
                    if stop_after_token:
                        break
                    continue
                if command in self._DROP_COMMANDS:
                    i = next_i
                    if stop_after_token:
                        break
                    continue
                if command in {'frac', 'dfrac', 'tfrac'}:
                    numerator, pos = self._read_argument_plain(text, next_i)
                    denominator, pos = self._read_argument_plain(text, pos)
                    result.append(f"{numerator}/{denominator}".strip('/'))
                    i = pos
                    if stop_after_token:
                        break
                    continue
                if command == 'sqrt':
                    pos = next_i
                    degree = ""
                    while pos < len(text) and text[pos].isspace():
                        pos += 1
                    if pos < len(text) and text[pos] == '[':
                        degree, pos = self._read_argument_plain(text, pos)
                    radicand, pos = self._read_argument_plain(text, pos)
                    prefix = f"{degree}√" if degree else "√"
                    result.append(f"{prefix}{radicand}".strip())
                    i = pos
                    if stop_after_token:
                        break
                    continue
                if command in {'overrightarrow', 'vec'}:
                    arg, pos = self._read_argument_plain(text, next_i)
                    result.append(self._format_vector(arg, direction="right"))
                    i = pos
                    if stop_after_token:
                        break
                    continue
                if command == 'overleftarrow':
                    arg, pos = self._read_argument_plain(text, next_i)
                    result.append(self._format_vector(arg, direction="left"))
                    i = pos
                    if stop_after_token:
                        break
                    continue
                if command in self._UNWRAP_COMMANDS:
                    arg, pos = self._read_argument_plain(text, next_i)
                    result.append(arg)
                    i = pos
                    if stop_after_token:
                        break
                    continue
                if command in {'begin', 'end'}:
                    _, pos = self._read_argument_plain(text, next_i)
                    i = pos
                    if stop_after_token:
                        break
                    continue
                if command in self._COMMAND_SYMBOLS:
                    result.append(self._COMMAND_SYMBOLS[command])
                    i = next_i
                    if stop_after_token:
                        break
                    continue

                lookahead = next_i
                while lookahead < len(text) and text[lookahead].isspace():
                    lookahead += 1
                if lookahead < len(text) and text[lookahead] in '{[':
                    arg, pos = self._read_argument_plain(text, lookahead)
                    result.append(arg)
                    i = pos
                else:
                    result.append(command)
                    i = next_i

                if stop_after_token:
                    break
                continue

            if ch == '{':
                inner, pos = self._read_balanced(text, i, '{', '}')
                result.append("{" + self._convert_latex_fragment(inner) + "}")
                i = pos
                if stop_after_token:
                    break
                continue

            if ch == '[':
                inner, pos = self._read_balanced(text, i, '[', ']')
                result.append(f"[{self._convert_latex_fragment(inner)}]")
                i = pos
                if stop_after_token:
                    break
                continue

            if ch == '^':
                token, pos = self._read_argument_plain(text, i + 1)
                result.append(self._format_superscript(token))
                i = pos
                if stop_after_token:
                    break
                continue

            if ch == '_':
                blank_match = re.match(r'_{2,}', text[i:])
                if blank_match:
                    result.append(blank_match.group(0))
                    i += len(blank_match.group(0))
                    if stop_after_token:
                        break
                    continue
                token, pos = self._read_argument_plain(text, i + 1)
                result.append(self._format_subscript(token))
                i = pos
                if stop_after_token:
                    break
                continue

            if ch == '$':
                i += 1
                if stop_after_token:
                    break
                continue

            result.append(ch)
            i += 1
            if stop_after_token:
                break

        plain = ''.join(result)
        if stop_after_token:
            return plain, i
        return plain

    def latex_to_plain(self, latex_text: str) -> str:
        """将 LaTeX 转换为普通文本（用于向量化前的预处理）"""
        if not latex_text:
            return ""
        plain = str(latex_text)
        plain = plain.replace('\r\n', '\n').replace('\r', '\n')
        plain = plain.replace('\\\\', '\\')
        plain = re.sub(r'```[a-zA-Z]*\n([\s\S]*?)```', r'\1', plain)
        plain = re.sub(r'\\[()\[\]]', '', plain)
        plain = plain.replace('$', '')
        plain = self._convert_latex_fragment(plain)
        plain = re.sub(r'[ \t]+', ' ', plain)
        plain = re.sub(r' *\n *', '\n', plain)
        plain = re.sub(r'\n{3,}', '\n\n', plain)
        plain = re.sub(r'([(\[（【])\s+', r'\1', plain)
        plain = re.sub(r'\s+([)\]）】,，。；;：:!?！？])', r'\1', plain)
        plain = self._dedupe_adjacent_lines(plain)
        return plain.strip()

    def build_embedding_text(self, stem_latex: str, options_latex: Dict[str, str]) -> str:
        """
        生成 embedding_text：移除 LaTeX 标记和标点，题干 + 选项合并。
        这是存入向量库、唯一用于向量化的字段。
        """
        plain_stem = self.latex_to_plain(stem_latex)
        plain_opts = [self.latex_to_plain(v) for v in options_latex.values()]

        combined = plain_stem + ' ' + ' '.join(plain_opts)

        # 移除标点符号
        combined = re.sub(
            r'[，,。\.；;：:！!？?（）()【】\[\]《》<>""\'\'、]',
            '',
            combined,
        )
        combined = re.sub(r'\s+', ' ', combined).strip()
        return combined
