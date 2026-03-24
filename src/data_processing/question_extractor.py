"""题目提取器：从文本中提取并丰富题目"""
import re
import os
import sys
import json
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path

class QuestionExtractor:
    """题目提取器：从文本中提取题目"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def identify_question_type(self, text: str) -> str:
        """识别题型（优先使用LLM，否则使用关键词匹配）"""
        # 如果LLM可用，使用LLM智能识别
        if self.llm_client:
            return self._identify_question_type_with_llm(text)
        
        # 备用方案：关键词匹配
        return self._infer_question_type_heuristic(text)
    
    def _identify_question_type_with_llm(self, text: str) -> str:
        """使用LLM智能识别题型"""
        if not self.llm_client:
            return "未知题型"
        
        # 截取题目前500字符（通常足够判断题型）
        question_preview = text[:500] if len(text) > 500 else text
        
        prompt = (
            f"请判断以下数学题目的题型。\n\n"
            f"题目内容（可能不完整）：\n{question_preview}\n\n"
            "请只返回题型名称，必须是以下之一：\n"
            "- 选择题\n- 填空题\n- 解答题\n- 计算题\n- 证明题\n- 应用题\n- 未知题型\n\n"
            "只返回题型名称，不要其他文字。"
        )
        
        try:
            response = self.llm_client.generate(prompt).strip()
            # 清理响应，提取题型
            response_lower = response.lower()
            if any(k in response_lower for k in ['选择', '单选', '多选']):
                return "选择题"
            elif any(k in response_lower for k in ['填空', '填', '空']):
                return "填空题"
            elif any(k in response_lower for k in ['解答', '应用', '证明', '计算', '求解']):
                return "解答题"
            else:
                # 回退到启发式
                return self._infer_question_type_heuristic(text)
        except Exception as e:
            print(f"LLM识别题型失败: {e}")
            return "未知题型"

    def _infer_question_type_heuristic(self, text: str) -> str:
        """启发式题型识别，避免全部落在未知"""
        t = text.lower()
        has_options = len(self._extract_options_from_content(text)) >= 3
        has_blanks = '____' in text or '___' in text or '（）' in text or '()' in text
        if has_options:
            return "选择题"
        if has_blanks or any(k in t for k in ['填空', '填入', '填在', '空格']):
            return "填空题"
        if any(k in t for k in ['解答', '求', '证明', '计算', '应用']):
            return "解答题"
        return "未知题型"
    
    
    def _is_exam_instruction_with_llm(self, text: str) -> bool:
        """使用LLM判断是否为试卷说明"""
        if not self.llm_client:
            return False
        
        # 截取文本前300字符（通常足够判断）
        text_preview = text[:300] if len(text) > 300 else text
        
        prompt = (
            "请判断以下文本是否是试卷说明、注意事项等非题目内容。\n\n"
            f"文本内容：\n{text_preview}\n\n"
            "试卷说明通常包括：\n"
            "- 答卷前的要求（如\"答卷前考生务必将...\"）\n"
            "- 答题卡填写说明（如\"回答选择题时...\"）\n"
            "- 考试注意事项\n"
            "- 页眉页脚（如\"第X页\"、\"共X页\"）\n"
            "- 考试信息（如\"考试时间\"、\"满分\"等）\n\n"
            "如果是试卷说明、注意事项等非题目内容，请返回\"是\"。\n"
            "如果是数学题目内容，请返回\"否\"。\n\n"
            "只返回\"是\"或\"否\"，不要其他文字。"
        )
        
        try:
            response = self.llm_client.generate(prompt).strip()
            response_lower = response.lower()
            # 判断响应
            if '是' in response_lower or 'yes' in response_lower or 'true' in response_lower:
                return True
            else:
                return False
        except Exception as e:
            print(f"LLM判断试卷说明失败: {e}")
            # 失败时回退到关键词匹配
            return self._is_exam_instruction_fallback(text)
    
    def _is_exam_instruction_fallback(self, text: str) -> bool:
        """备用方案：关键词匹配判断试卷说明"""
        instruction_keywords = [
            '答卷前', '答题卡', '考生号', '考场号', '座位号', '填写在',
            '选择题时', '选出', '涂黑', '如需改动', '用橡皮擦',
            '非选择题', '黑色字迹', '签字笔', '答在', '答题区域',
            '超出答题区域', '在草稿纸', '试卷上', '均无效',
            '考试时间', '满分', '注意事项', '本试卷', '第', '页',
            '共', '页', '姓名', '班级', '学号'
        ]
        text_lower = text.lower()
        keyword_count = sum(1 for kw in instruction_keywords if kw in text_lower)
        if len(text) < 100 and keyword_count >= 2:
            return True
        if any(kw in text_lower for kw in ['答题卡', '考生号', '考场号', '座位号']):
            return True
        # 章节/说明性文字（如"本题共""多项选择"等）在长度较短时判为说明
        if len(text) < 150 and any(kw in text_lower for kw in ['本题共', '多项选择', '部分选对', '全部选对', '有选错', '每小题']):
            return True
        # 大题类型说明：以中文序号开头，包含说明性词语，不限制长度
        if re.match(r'^[一二三四五六七八九十]+[、．]\s*(选择题|填空题|解答题|计算题|证明题|应用题)', text):
            return True
        if re.match(r'^[一二三四五六七八九十]+[、．]', text) and any(
            kw in text for kw in ['本题共', '每小题', '全部选对', '部分选对', '有选错', '共80分', '共24分', '共45分']
        ):
            return True
        return False
    
    def batch_identify_question_types(self, texts: List[str]) -> List[str]:
        """批量识别题型（使用LLM批量处理，提高效率）"""
        if not self.llm_client or len(texts) == 0:
            # 如果LLM不可用，逐个使用关键词匹配
            return [self.identify_question_type(text) for text in texts]
        
        # 如果题目数量少，逐个处理
        if len(texts) <= 3:
            return [self.identify_question_type(text) for text in texts]
        
        # 批量处理：一次性让LLM识别多个题目的题型
        try:
            # 构建批量prompt
            questions_text = "\n\n".join([
                f"题目{i+1}：{text[:200] if len(text) > 200 else text}"
                for i, text in enumerate(texts)
            ])
            
            prompt = (
                "请判断以下数学题目的题型。每个题目请只返回题型名称。\n\n"
                f"{questions_text}\n\n"
                "请以JSON数组格式返回，格式如下：\n"
                "[\n"
                "  {\"index\": 1, \"question_type\": \"选择题\"},\n"
                "  {\"index\": 2, \"question_type\": \"填空题\"},\n"
                "  ...\n"
                "]\n\n"
                "题型必须是以下之一：选择题、填空题、解答题、计算题、证明题、应用题、未知题型\n\n"
                "只返回JSON数组，不要其他文字。"
            )
            
            response = self.llm_client.generate(prompt)
            import json
            json_match = re.search(r'\[[\s\S]*?\]', response, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group(0))
                # 构建结果映射
                type_map = {r.get('index', i+1): r.get('question_type', '未知题型') 
                           for i, r in enumerate(results)}
                # 返回对应的题型列表
                return [type_map.get(i+1, '未知题型') for i in range(len(texts))]
        except Exception as e:
            print(f"批量识别题型失败: {e}，改用逐个识别")
        
        # 失败时逐个识别
        return [self.identify_question_type(text) for text in texts]
    
    def batch_is_exam_instruction(self, texts: List[str]) -> List[bool]:
        """批量判断是否为试卷说明（使用LLM批量处理，提高效率）"""
        if not self.llm_client or len(texts) == 0:
            return [self._is_exam_instruction_with_llm(text) for text in texts]
        
        # 如果数量少，逐个处理
        if len(texts) <= 3:
            return [self._is_exam_instruction_with_llm(text) for text in texts]
        
        # 批量处理
        try:
            texts_preview = [text[:200] if len(text) > 200 else text for text in texts]
            texts_text = "\n\n".join([
                f"文本{i+1}：{text}"
                for i, text in enumerate(texts_preview)
            ])
            
            prompt = (
                "请判断以下文本是否是试卷说明、注意事项等非题目内容。\n\n"
                f"{texts_text}\n\n"
                "**非题目内容包括**：\n"
                "- 答卷前的要求、答题卡填写说明、考试注意事项、页眉页脚、考试信息等\n"
                "- 大题类型说明（如\"一、选择题：本题共16小题，每小题5分，共80分。\"、\"二、填空题：本题共5道小题\"等）\n"
                "- 多选题说明（如\"全部选对的得6分，部分选对的得部分分，有选错的得0分\"等）\n"
                "- 答题规则（如\"在每小题给出的四个选项中，只有一项是符合题目要求的\"等）\n\n"
                "**题目内容**：以阿拉伯数字题号开头（如\"1.\"、\"2、\"），包含具体数学问题的内容。\n\n"
                "请以JSON数组格式返回，格式如下：\n"
                "[\n"
                "  {\"index\": 1, \"is_instruction\": true},\n"
                "  {\"index\": 2, \"is_instruction\": false},\n"
                "  ...\n"
                "]\n\n"
                "只返回JSON数组，不要其他文字。"
            )
            
            response = self.llm_client.generate(prompt)
            import json
            json_match = re.search(r'\[[\s\S]*?\]', response, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group(0))
                result_map = {r.get('index', i+1): r.get('is_instruction', False)
                            for i, r in enumerate(results)}
                return [result_map.get(i+1, False) for i in range(len(texts))]
        except Exception as e:
            print(f"批量判断试卷说明失败: {e}，改用逐个判断")
        
        # 失败时逐个判断
        return [self._is_exam_instruction_with_llm(text) for text in texts]
    
    def _is_question_complete(self, content: str) -> bool:
        """验证题目是否完整（包含题号、题干、选项等）"""
        if not content or len(content.strip()) < 20:
            return False
        
        content = content.strip()
        
        # 1. 必须包含题号（数字开头）
        has_number_prefix = bool(re.match(r'^\d+[\.、\)]|^[一二三四五六七八九十]+[、．]', content))
        if not has_number_prefix:
            # 检查是否以括号数字开头，如"(1)"
            if not re.match(r'^\(?\d+\)', content):
                return False
        
        # 2. 选择题必须包含选项标识（A、B、C、D等）
        # 检查是否包含选项模式
        has_options = len(self._extract_options_from_content(content)) >= 3
        
        # 3. 如果看起来像选择题但没有选项，可能不完整
        question_lower = content.lower()
        if any(keyword in question_lower for keyword in ['选择', '正确的是', '错误的是', '哪个']):
            if not has_options:
                return False
        
        # 4. 题目长度检查（太短可能不完整）
        if len(content) < 30:
            return False
        
        # 5. 检查是否包含明显的数学内容（数字、公式、符号等）
        has_math_content = bool(
            re.search(r'[0-9+\-×÷=<>≤≥≠√∑∏∫]', content) or
            re.search(r'[xya-z]²|[xya-z]³', content) or
            re.search(r'[函数方程集合]', content)
        )
        
        return has_math_content

    def _extract_question_number(self, content: str) -> Optional[str]:
        if not content:
            return None
        c = content.strip()
        c = re.sub(r"^[\s\u3000\"'“”‘’]+", "", c)
        m = re.match(r'^\s*(\d{1,4})\s*(?:[\.、\)．]|\s+)', c)
        if m:
            try:
                n = int(m.group(1))
            except Exception:
                return None
            if 1900 <= n <= 2100:
                return None
            if n <= 0 or n > 200:
                return None
            return str(n)
        m = re.match(r'^\s*\(\s*(\d+)\s*\)\s*', c)
        if m:
            return m.group(1)
        return None

    def get_content_text(self, content: Any) -> str:
        if isinstance(content, dict):
            return content.get("stem_plain", "") or content.get("stem_latex", "") or ""
        return str(content or "")


    # 题目数据结构化字段的自动补全工具，核心目标是「为题目列表中缺失 stem（题干）、options（选项）字段的题目
    def populate_structured_fields(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for q in questions:
            if not isinstance(q, dict):
                continue
            content = (q.get("content") or "").strip()
            if not content:
                continue

            if not q.get("stem"):
                q["stem"] = self._extract_stem_from_content(content)

            if not q.get("options"):
                opts = self._extract_options_from_content(content)
                if opts:
                    q["options"] = opts

        return questions

    def _extract_stem_from_content(self, content: str) -> str:
        c = content.strip()
        c = re.sub(r"^\s*(\d{1,4})\s*(?:[\.、\)．]|\s+)", "", c)
        lines = c.split("\n")
        stem_lines = []
        for line in lines:
            if re.match(r'^\s*[A-Da-d][\.\、\)]', line):
                break
            stem_lines.append(line)
        return "\n".join(stem_lines).strip()

    def _extract_options_from_content(self, content: str) -> Dict[str, str]:
        options: Dict[str, str] = {}
        if not content:
            return options

        pattern = (
            r'(?m)^\s*([A-Da-d])[\.\、\)\uff0e\uff09]\s*(.+?)'
            r'(?=^\s*[A-Da-d][\.\、\)\uff0e\uff09]\s*'
            r'|^\s*[一二三四五六七八九十]+[、．]'
            r'|^\s*\d{1,3}\s*[\.、\)]'
            r'|^\s*[（(]\s*\d+\s*[）)]'
            r'|\Z)'
        )
        for m in re.finditer(pattern, content, re.DOTALL):
            key = (m.group(1) or "").upper()
            val = (m.group(2) or "").strip()
            val = re.sub(r'\n\s*[一二三四五六七八九十]+[、．][\s\S]*$', '', val).strip()
            val = re.sub(r'\n\s*\d{1,3}\s*[\.、\)][\s\S]*$', '', val).strip()
            val = re.sub(r'(?:\s+要求[\.。：:].*|\s+全部选对.*|\s+部分选对.*|\s+有选错.*)$', '', val).strip()
            if key and val:
                options[key] = val

        return options

    def _is_subquestion_only(self, content: str) -> bool:
        if not content:
            return False
        c = content.strip()
        return bool(re.match(r'^\s*\(\s*\d+\s*\)\s*', c))

    def _normalize_for_dedupe(self, content: Any) -> str:
        t = self.get_content_text(content).strip()
        t = re.sub(r"\s+", " ", t)
        t = t.lower()
        t = re.sub(r"[\s\u3000]+", "", t)
        t = re.sub(r"[，,。\.；;：:！!？?（）()【】\[\]《》<>“”\"'‘’、]", "", t)
        return t

    def _is_top_level_question_line(self, line: str) -> bool:
        if not line:
            return False
        m = re.match(r'^\s*(\d{1,3})\s*[\.、．)]', line)
        if not m:
            return False
        try:
            n = int(m.group(1))
        except Exception:
            return False
        if 1900 <= n <= 2100:
            return False
        return 0 < n <= 200

    def _is_option_line(self, line: str) -> bool:
        return bool(re.match(r'^\s*[A-Da-d][\.\、\)]', line or ""))

    def _is_subquestion_line(self, line: str) -> bool:
        return bool(re.match(r'^\s*[（(]\s*\d+\s*[）)]', line or ""))

    # 用正则识别行首的 1. 2、 3) 这类题目，自动把一大段文本切成一道道独立的题。
    def _extract_question_blocks(self, text: str) -> List[str]:
        if not text:
            return []
        starts = []
        # ?m是开启多行模式
        for m in re.finditer(r'(?m)^\s*(\d{1,3})\s*[\.、．)]', text):
            try:
                n = int(m.group(1))
            except Exception:
                continue
            if 1900 <= n <= 2100 or n <= 0 or n > 200:
                continue
            starts.append((n, m.start()))
        if not starts:
            return []
        blocks = []
        for i, (_, start) in enumerate(starts):
            # 每个题的结束位置就是下一个题的起始位置
            end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
            blocks.append(text[start:end])
        return blocks

    def _should_drop_question_block_line(self, line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return True
        if re.match(r'^--- 第 \d+ 页 ---$', s):
            return True
        if re.match(r'^第\s*\d+\s*页\s*/\s*共\s*\d+\s*页$', s):
            return True
        if "学科网" in s and "公司" in s:
            return True
        if re.match(r'^[一二三四五六七八九十]+[、．]\s*(选择题|填空题|解答题|计算题|证明题|应用题)', s):
            return True
        if any(kw in s for kw in ["全部选对", "部分选对", "有选错", "每小题", "本题共"]):
            return True
        if re.match(r'^要求[\.。：:]', s):
            return True
        return False
    # 根据「前一行文本的结尾特征」和「当前行文本的开头特征」，判断是否需要加空格合并，
    # 适配题目文本的排版规则（如选项、公式、标点的连贯书写）。
    def _merge_question_line(self, prev: str, line: str) -> str:
        prev = (prev or "").rstrip()
        line = (line or "").strip()
        if not prev:
            return line
        if not line:
            return prev
        if re.fullmatch(r'[A-Da-d][\.\、\)]', prev):
            return f"{prev} {line}"
        if prev.endswith(('（', '(', '[', '【', '{', '：', ':', '，', ',', '=', '+', '-', '×', '÷', '·', '/', '<', '>', '≤', '≥')):
            return prev + line
        if line[0] in '，,。；;：:!?！？、）)]】}':
            return prev + line
        return f"{prev} {line}"

    # 题目块文本清洗函数，核心逻辑是逐行处理题目块内容 —— 过滤无效行、识别题目 / 选项 / 子题行作为
    # 独立分段、合并普通文本行到前一分段，最终清理多余空行并返回规整的题目文本。
    def _clean_question_block(self, block: str) -> str:
        if not block:
            return ""
        segments: List[str] = []
        for raw_line in block.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            line = raw_line.strip()
            if self._should_drop_question_block_line(line):
                continue
            if self._is_top_level_question_line(line) or self._is_option_line(line) or self._is_subquestion_line(line):
                segments.append(line)
                continue
            if not segments:
                segments.append(line)
                continue
            segments[-1] = self._merge_question_line(segments[-1], line)
        text = "\n".join(seg for seg in segments if seg.strip())
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _build_question_from_block(self, content: str, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        content = (content or "").strip()
        if not content or len(content) < 20:
            return None
        if self._is_exam_instruction_fallback(content):
            return None
        qnum = self._extract_question_number(content)
        if qnum is None:
            return None

        options = self._extract_options_from_content(content)
        stem = self._extract_stem_from_content(content)
        question_type = self._infer_question_type_heuristic(content)
        return {
            "index": int(qnum) if qnum.isdigit() else 0,
            "stem": stem,
            "options": options,
            "content": content,
            "question_type": question_type,
            "knowledge_points": [],
            "difficulty": 3,
            "source_meta": dict(meta or {}),
        }

    def _extract_questions_rule_based(self, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks = self._extract_question_blocks(text)
        if not blocks:
            return []
        print("_extract_questions_rule_based函数中被切分后的题目:\n",blocks)
        by_num: Dict[str, Dict[str, Any]] = {}
        for block in blocks:
            content = self._clean_question_block(block)
            q = self._build_question_from_block(content, meta)
            if not q:
                continue
            qnum = self._extract_question_number(q.get("content", ""))
            if not qnum:
                continue
            prev = by_num.get(qnum)
            # 如果题号没有，直接入库
            if prev is None:
                by_num[qnum] = q
                continue
            # 如果题号已经存在，对比选择最优版本
            prev_opts = len(prev.get("options", {}) or {})
            curr_opts = len(q.get("options", {}) or {})
            prev_len = len((prev.get("content") or "").strip())
            curr_len = len((q.get("content") or "").strip())
            if (curr_opts, curr_len) > (prev_opts, prev_len):
                by_num[qnum] = q

        ordered: List[Dict[str, Any]] = []
        for _, q in sorted(by_num.items(), key=lambda item: int(item[0])):
            ordered.append(q)
        for i, q in enumerate(ordered):
            q["index"] = i + 1
        return ordered

    # 用于补救题目，但是可靠性堪忧
    def _infer_expected_question_numbers(self, text: str) -> List[int]:
        """从全文文本中粗略推断题号范围（仅提取阿拉伯数字题号）。"""
        if not text:
            return []
        nums = []
        for m in re.finditer(r'^\s*(\d+)\s*[\.、\)]', text, flags=re.MULTILINE):
            try:
                n = int(m.group(1))
                # 过滤明显不是题号的数字：年份、过大的数字等
                if 1900 <= n <= 2100:
                    continue
                if n <= 0 or n > 200:
                    continue
                nums.append(n)
            except Exception:
                continue
        if not nums:
            return []
        max_n = max(nums)
        if max_n <= 0:
            return []
        return list(range(1, max_n + 1))

    def _parse_llm_json_array(self, resp: str) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        if not resp:
            return parsed
        cleaned = resp.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"^```\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
        cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        cleaned = re.sub(r",\s*]", "]", cleaned)
        cleaned = re.sub(r",\s*}", "}", cleaned)
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, dict)]
        except Exception:
            pass
        try:
            obj_matches = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
            for obj_str in obj_matches:
                try:
                    parsed_obj = json.loads(obj_str)
                    if isinstance(parsed_obj, dict):
                        parsed.append(parsed_obj)
                except Exception:
                    continue
        except Exception:
            pass
        return parsed

    def _extract_context_for_question_num(self, text: str, num: int, window: int = 600) -> str:
        """
        从全文中提取题号 num 附近的上下文（前后各 window 字符）。
        若找不到该题号，返回空串。
        """
        # 匹配题号出现的位置：行首的 "N." / "N、" / "(N)"
        pattern = re.compile(
            r'(?:^|\n)\s*' + re.escape(str(num)) + r'\s*[\.、\)）]',
        )
        m = pattern.search(text)
        if not m:
            return ""
        start = max(0, m.start() - 100)   # 往前留一点上文
        end = min(len(text), m.start() + window)
        return text[start:end]

    def _recover_missing_questions_with_llm(
        self,
        text: str,
        meta: Dict[str, Any],
        missing_nums: List[int],
    ) -> List[Dict[str, Any]]:
        if not self.llm_client or not text or not missing_nums:
            return []

        # 为每个缺失题号单独定位上下文，避免把全文塞进 prompt
        # 每次最多处理 5 道，防止 prompt 过长
        all_valid: List[Dict[str, Any]] = []
        batch_size = 5
        for i in range(0, len(missing_nums), batch_size):
            batch = missing_nums[i: i + batch_size]
            # 收集每道缺失题的上下文片段，拼在一起
            context_parts = []
            for num in batch:
                ctx = self._extract_context_for_question_num(text, num, window=800)
                if ctx:
                    context_parts.append(f"--- 题号 {num} 附近文本 ---\n{ctx}")
            # 如果所有缺失题号都没找到对应位置，退回全文（截断到前3000字）
            context_text = "\n\n".join(context_parts) if context_parts else text[:3000]

            nums_text = ", ".join(str(n) for n in batch)
            print(f"nums_text: {nums_text}")
            prompt = (
                "你是一个专业的数学试卷题目提取助手。\n\n"
                f"我已经从试卷文本中提取出部分题目，但缺失了题号为：{nums_text} 的题目。\n\n"
                "请你只从下面给出的试卷文本片段中，找出这些缺失题号对应的**完整题目**，并以JSON数组返回。\n\n"
                "**重要 - OCR纠错：**\n"
                "输入文本来自OCR扫描，请根据数学上下文修正常见错误：\n"
                "- 数字与字母混淆：`48CD` → `ABCD`\n"
                "- 符号误识别：`山` → `⊥`，`榄长` → `棱长`，`巳知` → `已知`\n"
                "- 下标修正：`A1B1C1D1` → `A₁B₁C₁D₁`\n"
                "- 删除乱码/噪声字符\n\n"
                "要求：\n"
                f"- 必须严格按题号匹配（只返回题号属于 {nums_text} 的题）\n"
                "- content 必须尽量包含题号、完整题干、以及所有小问/选项（若有）\n"
                "- 如果某个缺失题号在文本中确实找不到，不要编造，不要输出该题号\n\n"
                "返回格式（只返回markdown ```json 代码块包裹的JSON数组，不要其他文字）：\n"
                "```json\n"
                "[\n"
                '  {{\"index\": 1, \"content\": \"6. ...\", \"question_type\": \"选择题\", \"knowledge_points\": [\"...\"], \"difficulty\": 3}}\n'
                "]\n"
                "```\n\n"
                f"试卷文本片段：\n{context_text}\n"
            )

            try:
                response = self.llm_client.generate(prompt)
                recovered = self._parse_llm_json_array(response)
                missing_set = set(int(n) for n in batch)
                for q in recovered:
                    content = (q.get("content") or "").strip()
                    if not content:
                        continue
                    n = self._extract_question_number(content)
                    if n and n.isdigit() and int(n) in missing_set:
                        q["source_meta"] = meta
                        if not q.get("question_type") or q.get("question_type") == "未知题型":
                            q["question_type"] = self._infer_question_type_heuristic(content)
                        kp = q.get("knowledge_points", [])
                        if isinstance(kp, str):
                            kp = [x.strip() for x in re.split(r"[\n,，、;；]+", kp) if x.strip()]
                        if not isinstance(kp, list):
                            kp = []
                        q["knowledge_points"] = [str(x).strip() for x in kp if str(x).strip()]
                        diff = q.get("difficulty", None)
                        try:
                            diff_int = int(diff)
                        except Exception:
                            diff_int = None
                        if diff_int is None or diff_int < 1 or diff_int > 5:
                            diff_int = 3
                        q["difficulty"] = diff_int
                        all_valid.append(q)
            except Exception as e:
                print(f"缺题补救（LLM）batch {batch} 失败: {e}")
                continue

        return all_valid
    
    
    def extract_questions_from_text(
        self,
        text: str,
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        从文本中提取题目（优先规则切题，不足时降级到LLM整块提取，并补救缺失题目）

        Args:
            text: PDF提取的文本
            meta: 元数据（地区、年份、考试类型等）

        Returns:
            题目列表，每个题目包含：content, question_type, index等
        """
        print(f"开始提取题目，文本总长度: {len(text)} 字符")

        if not self.llm_client:
            print("⚠ LLM客户端未初始化，无法提取题目，返回空列表")
            return []

        questions = self._extract_questions_with_segment(text, meta)
        questions = [q for q in questions if not self._is_exam_instruction_with_llm(q.get('content', ''))]

        # 缺题补救：推断预期题号范围，用LLM针对性补救缺失题目
        expected_nums = self._infer_expected_question_numbers(text)
        print(f"推断到预期题号范围: {expected_nums}")
        present_nums = set()
        for q in questions:
            n = self._extract_question_number((q.get('content', '') or '').strip())
            if n and n.isdigit():
                present_nums.add(int(n))

        missing_nums = [n for n in expected_nums if n not in present_nums]
        if missing_nums:
            try:
                recovered_items = self._recover_missing_questions_with_llm(text, meta, missing_nums)
                if recovered_items:
                    questions.extend(recovered_items)
                    print(f"缺题补救：补回 {len(recovered_items)} 道题（缺失题号: {missing_nums}）")
                else:
                    print(f"缺题补救（LLM）无结果，缺失题号: {missing_nums} 无法补回")
            except Exception as e:
                print(f"缺题补救失败: {e}")

        # 验证题目完整性
        complete_questions = []
        for q in questions:
            content = q.get('content', '')
            if self._is_question_complete(content):
                complete_questions.append(q)
            else:
                n = self._extract_question_number(content)
                if n is not None and len((content or '').strip()) >= 20:
                    complete_questions.append(q)
                else:
                    print(f"  警告：题目不完整，已跳过: {q.get('content', '')[:100]}...")

        print(f"✓提取完成，共 {len(complete_questions)} 道完整题目")
        return complete_questions
    

    
    def _extract_with_llm_fallback(self, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """规则切题不足时，退回旧的整块 LLM 提取逻辑。"""
        if not self.llm_client:
            return []

        chunk_size = int(os.getenv("LLM_CHUNK_SIZE", "3000"))
        overlap_pages = 1

        page_pattern = re.compile(r'(?=--- 第 \d+ 页 ---)')
        page_splits = list(page_pattern.finditer(text))

        if len(page_splits) >= 2:
            pages = []
            for i, m in enumerate(page_splits):
                start = m.start()
                end = page_splits[i + 1].start() if i + 1 < len(page_splits) else len(text)
                pages.append(text[start:end])

            chunks = []
            current = ""
            current_pages = []
            for page in pages:
                if current and len(current) + len(page) > chunk_size:
                    chunks.append((current, current_pages[:]))
                    overlap_text = "".join(current_pages[-overlap_pages:])
                    current = overlap_text + page
                    current_pages = current_pages[-overlap_pages:] + [page]
                else:
                    current += page
                    current_pages.append(page)
            if current:
                chunks.append((current, current_pages[:]))
            chunks = [c for c, _ in chunks]
            print(f"按页分块：{len(pages)} 页 → {len(chunks)} 块")
        else:
            char_chunk = chunk_size
            overlap = int(os.getenv("LLM_CHUNK_OVERLAP", "800"))
            chunks = []
            if len(text) <= char_chunk:
                chunks = [text]
            else:
                start = 0
                while start < len(text):
                    end = min(start + char_chunk, len(text))
                    chunks.append(text[start:end])
                    if end == len(text):
                        break
                    start = end - overlap
            print(f"按字符分块（无页标记）：{len(chunks)} 块")

        all_questions = []
        print(f"开始使用LLM提取题目，文本分为 {len(chunks)} 块处理...")

        for i, chunk in enumerate(chunks):
            print(f"正在处理第 {i+1}/{len(chunks)} 块（长度: {len(chunk)} 字符）...")

            prompt = (
                "你是一个专业的数学试卷题目提取助手。请从以下试卷文本中提取所有数学题目。\n\n"
                "**重要提示 - OCR纠错：**\n"
                "输入文本来自OCR扫描，可能存在识别错误，请根据数学上下文智能修正：\n"
                "- 数字与字母混淆：`48CD` → `ABCD`，`kBCD` → `A₁B₁C₁D₁`，`P−ABCD` → `P-ABCD`\n"
                "- 符号误识别：`山` → `⊥`（垂直），`|` → `1`，`O` → `0`，`榄长` → `棱长`，`巳知` → `已知`\n"
                "- 下标丢失或乱码：`A1B1C1D1` → `A₁B₁C₁D₁`，`F1F2` → `F₁F₂`\n"
                "- 公式符号：`≤` `≥` `∑` `∏` `√` `π` `∈` `∉` `⊂` `⊃` `∩` `∪` `∠` `△` `⊥` `∥`\n"
                "- 常见数学术语修正：`焕点` → `焦点`，`余玄` → `余弦`，`正玄` → `正弦`\n"
                "- 乱码/噪声字符（如`j 门 az`、`msu`、无意义符号串）应删除\n\n"
                "**核心要求 - 题目必须完整：**\n"
                '1. **必须包含题号**：如"1."、"2、"、"(1)"、"一、"等\n'
                "2. **必须包含完整题干**：题目的完整描述和条件\n"
                "3. **选择题必须包含所有选项**：如A、B、C、D等所有选项\n"
                "4. **填空题必须包含所有空格**：所有需要填空的位置\n"
                "5. **解答题必须包含完整问题**：所有需要解答的问题\n"
                "6. **保留公式**：保留指数/分数/根号等符号（如 x^2, 1/2, √3, ≤, ≥, ∑, ∏）\n\n"
                "**提取规则：**\n"
                "- **只提取真正的数学题目**，忽略试卷说明、注意事项、标题等非题目内容\n"
                "- **题目必须完整**：不能只提取题干的一部分，必须包含题号、完整题干、所有选项（如果有）\n"
                '- **识别题目编号**：题目通常以数字开头（如"1."、"2、"、"(1)"等）\n'
                '- **一题一条**：每个JSON对象只能包含一道题；如果文本中出现了下一个题号（如"20."/"20、"/"(20)"），必须在该题号处切分，绝不能把两道题合并到同一个content里\n'
                "- **过滤掉以下内容**：\n"
                '  - 试卷说明（如"答卷前考生务必将..."、"回答选择题时..."等）\n'
                '  - 页眉页脚（如"第X页"、"共X页"等）\n'
                '  - 考试信息（如"考试时间"、"满分"等）\n'
                "  - 注意事项\n\n"
                "**字段说明：**\n"
                "- stem：题号 + 完整题干（不含选项）\n"
                "- options：选择题填 {\"A\": \"...\", \"B\": \"...\", ...}，填空/解答题填 {}\n"
                "- content：题号 + 题干 + 所有选项完整原文（供完整性校验，选择题必须A/B/C/D）\n\n"
                f"试卷文本：\n{chunk}\n\n"
                "请以JSON数组格式返回所有提取到的**完整题目**，格式如下，务必使用 markdown ```json 代码块包裹，且只输出这个数组：\n"
                "```json\n"
                "[\n"
                "  {\n"
                '    "index": 1,\n'
                '    "stem": "1. 已知函数f(x)=x²+1，则f(2)的值为（    ）",\n'
                '    "options": {"A": "3", "B": "4", "C": "5", "D": "6"},\n'
                '    "content": "1. 已知函数f(x)=x²+1，则f(2)的值为（    ）\\nA. 3\\nB. 4\\nC. 5\\nD. 6",\n'
                '    "question_type": "选择题",\n'
                '    "knowledge_points": ["函数", "代入求值"],\n'
                '    "difficulty": 2\n'
                "  }\n"
                "]\n"
                "```\n\n"
                "**重要：stem 字段只含题干（不含选项），options 字段按字母分开，content 字段保留完整原文。**\n\n"
                "只返回JSON数组，不要其他解释文字。"
            )

            try:
                print("  调用LLM API...")
                response = self.llm_client.generate(prompt)
                print(f"  LLM响应长度: {len(response)} 字符")

                questions = self._parse_llm_json_array(response)
                if questions:
                    valid_questions = []
                    for q in questions:
                        content = q.get('content', '').strip()
                        if len(content) < 30:
                            continue
                        if not self._is_question_complete(content):
                            n = self._extract_question_number(content)
                            if n is None and not self._is_subquestion_only(content):
                                print(f"    跳过不完整题目: {content[:80]}...")
                                continue
                        q["source_meta"] = dict(meta or {})
                        if not q.get("question_type") or q.get("question_type") == "未知题型":
                            q["question_type"] = self._infer_question_type_heuristic(content)
                        kp = q.get("knowledge_points", [])
                        if isinstance(kp, str):
                            kp = [x.strip() for x in re.split(r"[\n,，、;；]+", kp) if x.strip()]
                        if not isinstance(kp, list):
                            kp = []
                        q["knowledge_points"] = [str(x).strip() for x in kp if str(x).strip()]
                        diff = q.get("difficulty", None)
                        try:
                            diff_int = int(diff)
                        except Exception:
                            diff_int = None
                        if diff_int is None or diff_int < 1 or diff_int > 5:
                            diff_int = 3
                        q["difficulty"] = diff_int
                        valid_questions.append(q)

                    if valid_questions:
                        contents = [q.get('content', '') for q in valid_questions]
                        is_instructions = self.batch_is_exam_instruction(contents)
                        valid_count = 0
                        for qi, q in enumerate(valid_questions):
                            if not is_instructions[qi]:
                                all_questions.append(q)
                                valid_count += 1
                        print(f"  第 {i+1} 块提取到 {valid_count} 道完整有效题目（共{len(valid_questions)}道，过滤{len(valid_questions)-valid_count}道说明/无效）")
                    else:
                        print(f"  第 {i+1} 块未提取到有效题目")
                else:
                    print(f"  ⚠ 第 {i+1} 块LLM响应未能解析为有效JSON，跳过。响应预览: {response[:200]}")

            except Exception as e:
                print(f"  ⚠ LLM提取题目失败（第{i+1}块）: {e}")
                import traceback
                traceback.print_exc()

        unique_questions = []
        seen_nums = set()
        seen_contents = set()
        for q in all_questions:
            content = q.get('content', '').strip()
            qnum = self._extract_question_number(content)
            if qnum and qnum.isdigit():
                if qnum in seen_nums:
                    continue
                seen_nums.add(qnum)
            norm = self._normalize_for_dedupe(content)
            content_key = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_questions.append(q)

        for q in unique_questions:
            if not q.get("question_type") or q.get("question_type") == "未知题型":
                q["question_type"] = self._infer_question_type_heuristic(q.get("content", ""))
            if not q.get("stem"):
                q["stem"] = self._extract_stem_from_content(q.get("content", ""))
            if not q.get("options"):
                q["options"] = self._extract_options_from_content(q.get("content", ""))

        for i, q in enumerate(unique_questions):
            q["index"] = i + 1

        print(f"✓ LLM提取完成，共提取到 {len(unique_questions)} 道有效题目")
        return unique_questions

    def _extract_questions_with_segment(self, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优先规则切题保留原文，规则不足时再退回整块 LLM 提取。"""
        if not self.llm_client:
            return []

        rule_based_questions = self._extract_questions_rule_based(text, meta)
        expected_count = len(self._infer_expected_question_numbers(text))
        if expected_count <= 3:
            threshold = expected_count or 1
        else:
            threshold = max(3, int(expected_count * 0.6))

        if rule_based_questions:
            print(f"规则切题提取到 {len(rule_based_questions)} 道题，预期题量 {expected_count or '未知'}")
        else:
            print("规则切题未提取到有效题目")

        if rule_based_questions and (expected_count == 0 or len(rule_based_questions) >= threshold):
            print("规则切题结果充足，直接保留原始 content，不再让 LLM 重写题面")
            return rule_based_questions

        print("规则切题结果不足，降级到整块 LLM 提取兜底")
        fallback_questions = self._extract_with_llm_fallback(text, meta)
        return fallback_questions or rule_based_questions
    
    def enrich_question_with_llm(
        self,
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用LLM丰富题目信息（知识点、难度等）"""
        if not self.llm_client:
            return question

        enriched = self.batch_enrich_questions_with_llm([question])
        return enriched[0] if enriched else question

    def batch_enrich_questions_with_llm(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not questions:
            return []
        if not self.llm_client:
            return questions

        previews = []
        for i, q in enumerate(questions):
            content = (q.get("content") or "").strip()
            if len(content) > 1200:
                content = content[:1200]
            previews.append(f"题目{i+1}：{content}")

        prompt = (
            "你是一个数学教研助手。\n\n"
            "请对下面每一道数学题给出：\n"
            "- knowledge_points：知识点列表（尽量具体，2~6个）\n"
            "- difficulty：难度等级（1-5的整数，1最简单，5最难）\n\n"
            + chr(10).join(previews) +
            "\n\n要求：\n"
            "- 必须只返回一个 JSON 数组\n"
            "- 数组长度必须与题目数量一致\n"
            "- 每个元素的 index 从 1 开始\n"
            "- 必须用 markdown ```json 代码块包裹\n"
            "- 不要输出任何额外解释文字\n\n"
            "返回格式示例：\n"
            "```json\n"
            "[\n"
            '  {{\"index\": 1, \"knowledge_points\": [\"集合\", \"不等式\"], \"difficulty\": 3}},\n'
            '  {{\"index\": 2, \"knowledge_points\": [\"向量\", \"数量积\"], \"difficulty\": 2}}\n'
            "]\n"
            "```\n"
        )

        try:
            response = self.llm_client.generate(prompt)
            results = self._parse_llm_json_array(response)
            if not results:
                print(f"LLM批量丰富题目信息：未能解析JSON数组，响应预览: {(response or '')[:200]}")
                return questions

            by_index: Dict[int, Dict[str, Any]] = {}
            for r in results:
                try:
                    idx = int(r.get("index"))
                except Exception:
                    continue
                if idx <= 0:
                    continue
                by_index[idx] = r

            enriched: List[Dict[str, Any]] = []
            for i, q in enumerate(questions):
                r = by_index.get(i + 1)
                if not r:
                    enriched.append(q)
                    continue

                kp = r.get("knowledge_points", [])
                if isinstance(kp, str):
                    kp = [x.strip() for x in re.split(r"[\n,，、;；]+", kp) if x.strip()]
                if not isinstance(kp, list):
                    kp = []
                kp = [str(x).strip() for x in kp if str(x).strip()]

                diff = r.get("difficulty", None)
                try:
                    diff_int = int(diff)
                except Exception:
                    diff_int = None
                if diff_int is None or diff_int < 1 or diff_int > 5:
                    diff_int = 3

                q["knowledge_points"] = kp
                q["difficulty"] = diff_int
                enriched.append(q)

            return enriched
        except Exception as e:
            print(f"LLM批量丰富题目信息失败: {e}")
            return questions
