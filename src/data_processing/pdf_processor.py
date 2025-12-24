"""PDF处理模块：OCR、文本提取、题目提取"""
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import sys
import json
import hashlib

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("警告：OCR功能不可用，请安装 pytesseract 和 pdf2image")


from .meta_extractor import ExamMetaExtractor

class PDFProcessor:
    """PDF处理器：支持OCR和文本提取"""
    
    def __init__(self, ocr_enabled: bool = True):
        self.ocr_enabled = ocr_enabled and OCR_AVAILABLE
        self.poppler_path = self._resolve_poppler_path()
        self._setup_library_path()

    def _setup_library_path(self):
        """
        设置 DYLD_LIBRARY_PATH 以确保 pdfinfo 加载正确架构的库（MacOS Arm64/x86_64 混合环境问题修复）
        优先将 poppler_path 的同级 lib 目录加入环境变量
        """
        if not self.poppler_path:
            return
            
        # 推断 lib 目录： bin/pdfinfo -> ../lib
        bin_dir = Path(self.poppler_path)
        lib_dir = bin_dir.parent / "lib"
        
        if lib_dir.exists():
            lib_path = str(lib_dir.absolute())
            current_ld_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            
            # 如果尚未添加，则添加到最前面
            if lib_path not in current_ld_path:
                print(f"Adding to DYLD_LIBRARY_PATH: {lib_path}")
                os.environ["DYLD_LIBRARY_PATH"] = f"{lib_path}:{current_ld_path}"


    def _resolve_poppler_path(self) -> Optional[str]:
        """
        返回poppler可执行所在目录，避免系统PATH指到不兼容的pdfinfo。
        优先顺序：
        1) 环境变量 POPPLER_PATH（若指向bin目录）
        2) CONDA_PREFIX/bin
        3) None（由pdf2image自行查找）
        """
        # 1) 显式指定
        poppler_env = os.getenv("POPPLER_PATH")
        if poppler_env and os.path.isdir(poppler_env):
            return poppler_env
        # 2) sys.executable 所在目录 (Prioritize Active Env)
        # 优先使用当前 Python 解释器所在的 bin 目录，确保环境一致性
        bin_path = os.path.dirname(sys.executable)
        pdfinfo_path = os.path.join(bin_path, "pdfinfo")
        if os.path.exists(pdfinfo_path):
            return bin_path

        # 3) conda环境 (CONDA_PREFIX) - Fallback
        conda_prefix = os.getenv("CONDA_PREFIX")
        if conda_prefix:
            bin_path = os.path.join(conda_prefix, "bin")
            pdfinfo_path = os.path.join(bin_path, "pdfinfo")
            if os.path.exists(pdfinfo_path):
                return bin_path


        return None
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        判断PDF是否为扫描版（图片格式）
        
        简单判断：如果PDF中文本层很少或为空，可能是扫描版
        """
        try:
            doc = fitz.open(pdf_path)
            text_count = 0
            meaningful_ratio_sum = 0.0
            checked_pages = 0
            for page_num in range(min(3, len(doc))):  # 检查前3页
                page = doc[page_num]
                text = page.get_text()
                text_count += len(text.strip())
                # 一些加密/占位文本常见模式，直接视为需OCR
                if re.search(r'\{#\{.*?\}#\}', text):
                    doc.close()
                    return True
                if text:
                    checked_pages += 1
                    # 计算“有意义字符”占比（中英文、数字占比低则认为可能是扫描/加密文本）
                    meaningful = re.findall(r'[A-Za-z0-9\u4e00-\u9fa5]', text)
                    meaningful_ratio = len(meaningful) / max(len(text), 1)
                    meaningful_ratio_sum += meaningful_ratio
            doc.close()
            
            # 如果前3页文本很少，可能是扫描版
            if text_count < 100:
                return True
            # 如果有意义字符占比很低，可能是扫描版或有加密字体
            if checked_pages > 0 and (meaningful_ratio_sum / checked_pages) < 0.2:
                return True
            return False
        except Exception as e:
            print(f"判断PDF类型失败: {e}")
            return True  # 默认按扫描版处理
    
    # 新增最大页数，方便识别前1-2页的元数据
    def extract_text_with_ocr(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR功能未启用或未安装相关依赖")

        # pdf2image 支持 first_page / last_page（从1开始）
        first_page = 1
        last_page = max_pages if max_pages else None
        images = convert_from_path(
            pdf_path,
            dpi=300,
            poppler_path=self.poppler_path,
            first_page=first_page,
            last_page=last_page
        )
        print(f"OCR: 已将 PDF 转换为 {len(images)} 张图片")
        all_text = []
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            all_text.append(f"--- 第 {i+1} 页 ---\n{text}\n")
        return "\n".join(all_text)
    
    def extract_text_native(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        doc = fitz.open(pdf_path)
        all_text = []
        page_count = len(doc) if max_pages is None else min(max_pages, len(doc))
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()
            all_text.append(f"--- 第 {page_num + 1} 页 ---\n{text}\n")
        doc.close()
        return "\n".join(all_text)
    
    def extract_text(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        """
        提取PDF文本（自动判断扫描版或原生版）
        
        Returns:
            提取的文本内容
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        is_scanned = self.is_scanned_pdf(str(pdf_path))
        if is_scanned:
            return self.extract_text_with_ocr(str(pdf_path), max_pages=max_pages)
        else:
            return self.extract_text_native(str(pdf_path), max_pages=max_pages) 
    
    def clean_text(self, text: str) -> str:
        """清洗提取的文本，尽量保留数学符号并去掉页码/水印"""
        # 去掉页码、水印、分隔线
        text = re.sub(r'---\s*第\s*\d+\s*页\s*---', '', text)
        text = re.sub(r'第\s*\d+\s*页\s*/\s*共\s*\d+\s*页', '', text)
        text = re.sub(r'第\s*\d+\s*页', '', text)
        text = re.sub(r'共\s*\d+\s*页', '', text)
        # 合并多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 保留更多数学符号：^ / % × ÷ √ ≤ ≥ ≈ ∑ ∏ π · _
        text = re.sub(
            r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.\,\;\:\!\?\(\)\[\]\{\}\+\-\*\/\=\>\<\^％%×÷√≤≥≈∑∏π·_]',
            '',
            text
        )
        return text.strip()


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
        
        prompt = f"""请判断以下数学题目的题型。

题目内容（可能不完整）：
{question_preview}

请只返回题型名称，必须是以下之一：
- 选择题
- 填空题
- 解答题
- 计算题
- 证明题
- 应用题
- 未知题型

只返回题型名称，不要其他文字。"""
        
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
        has_options = bool(re.search(r'[a-dＡ-Ｄ]\s*[\\.|、|\\)]', t))
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
        
        prompt = f"""请判断以下文本是否是试卷说明、注意事项等非题目内容。

文本内容：
{text_preview}

试卷说明通常包括：
- 答卷前的要求（如"答卷前考生务必将..."）
- 答题卡填写说明（如"回答选择题时..."）
- 考试注意事项
- 页眉页脚（如"第X页"、"共X页"）
- 考试信息（如"考试时间"、"满分"等）

如果是试卷说明、注意事项等非题目内容，请返回"是"。
如果是数学题目内容，请返回"否"。

只返回"是"或"否"，不要其他文字。"""
        
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
        # 章节/说明性文字（如“本题共”“多项选择”等）在长度较短时判为说明
        if len(text) < 150 and any(kw in text_lower for kw in ['本题共', '多项选择', '部分选对', '全部选对', '有选错', '每小题']):
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
            
            prompt = f"""请判断以下数学题目的题型。每个题目请只返回题型名称。

{questions_text}

请以JSON数组格式返回，格式如下：
[
  {{"index": 1, "question_type": "选择题"}},
  {{"index": 2, "question_type": "填空题"}},
  ...
]

题型必须是以下之一：选择题、填空题、解答题、计算题、证明题、应用题、未知题型

只返回JSON数组，不要其他文字。"""
            
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
            return [self._is_exam_instruction(text) for text in texts]
        
        # 如果数量少，逐个处理
        if len(texts) <= 3:
            return [self._is_exam_instruction(text) for text in texts]
        
        # 批量处理
        try:
            texts_preview = [text[:200] if len(text) > 200 else text for text in texts]
            texts_text = "\n\n".join([
                f"文本{i+1}：{text}"
                for i, text in enumerate(texts_preview)
            ])
            
            prompt = f"""请判断以下文本是否是试卷说明、注意事项等非题目内容。

{texts_text}

试卷说明通常包括：答卷前的要求、答题卡填写说明、考试注意事项、页眉页脚、考试信息等。

请以JSON数组格式返回，格式如下：
[
  {{"index": 1, "is_instruction": true}},
  {{"index": 2, "is_instruction": false}},
  ...
]

只返回JSON数组，不要其他文字。"""
            
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
        return [self._is_exam_instruction(text) for text in texts]
    
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
        has_options = bool(re.search(r'[A-Z][\.、\)]\s*', content))
        
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
        m = re.match(r'^\s*(\d+)\s*[\.、\)]', c)
        if m:
            return m.group(1)
        m = re.match(r'^\s*\(\s*(\d+)\s*\)\s*', c)
        if m:
            return m.group(1)
        return None

    def _is_subquestion_only(self, content: str) -> bool:
        if not content:
            return False
        c = content.strip()
        return bool(re.match(r'^\s*\(\s*\d+\s*\)\s*', c))

    def _normalize_for_dedupe(self, content: str) -> str:
        t = (content or "").strip()
        t = re.sub(r"\s+", " ", t)
        t = t.lower()
        t = re.sub(r"[\s\u3000]+", "", t)
        t = re.sub(r"[，,。\.；;：:！!？?（）()【】\[\]《》<>“”\"'‘’、]", "", t)
        return t

    def _infer_expected_question_numbers(self, text: str) -> List[int]:
        """从全文文本中粗略推断题号范围（仅提取阿拉伯数字题号）。"""
        if not text:
            return []
        nums = []
        for m in re.finditer(r'^\s*(\d+)\s*[\.、\)]', text, flags=re.MULTILINE):
            try:
                nums.append(int(m.group(1)))
            except Exception:
                continue
        if not nums:
            return []
        max_n = max(nums)
        if max_n <= 0:
            return []
        return list(range(1, max_n + 1))

    def _merge_questions_by_number(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not questions:
            return []

        # buckets：按序号分桶，key=序号（如"1"），value=该序号下的所有问题条目
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        passthrough: List[Dict[str, Any]] = []
        last_number: Optional[str] = None

        # 第一遍遍历：给问题按序号分桶，无序号的暂存到passthrough
        for q in questions:
            content = (q.get("content") or "").strip()
            if not content:
                continue

            # 如果没有题号且是子题目，加上上一个题的题号
            number = self._extract_question_number(content)
            if number is None and self._is_subquestion_only(content) and last_number is not None:
                number = last_number

            # 无有效序号：暂存到passthrough（最终保留）
            if number is None:
                passthrough.append(q)
                continue
            # 有有效序号，即不是子题目：更新last_number，放入对应桶
            last_number = number
            buckets.setdefault(number, []).append(q)

        merged: List[Dict[str, Any]] = []
        used_numbers = set()

        for q in questions:
            content = (q.get("content") or "").strip()
            number = self._extract_question_number(content)
            if number is None:
                continue
            if number in used_numbers:
                continue
            used_numbers.add(number)

            parts = buckets.get(number, [])
            if not parts:
                continue

            base = max(parts, key=lambda x: len((x.get("content") or "")))
            base_text = (base.get("content") or "").strip()
            base_norm = self._normalize_for_dedupe(base_text)

            extras: List[str] = []
            for p in parts:
                t = (p.get("content") or "").strip()
                if not t:
                    continue
                tn = self._normalize_for_dedupe(t)
                if tn == base_norm:
                    continue
                if tn in base_norm:
                    continue
                extras.append(t)

            if extras:
                base["content"] = "\n".join([base_text] + extras)
            merged.append(base)

        merged.extend(passthrough)

        def _sort_key(item: Dict[str, Any]) -> int:
            n = self._extract_question_number((item.get("content") or "").strip())
            return int(n) if n and n.isdigit() else 10**9

        merged.sort(key=_sort_key)
        for i, q in enumerate(merged):
            q["index"] = i + 1
        return merged
    
    def extract_questions_from_text(
        self,
        text: str,
        meta: Dict[str, Any],
        use_regex_fallback: bool = False
    ) -> List[Dict[str, Any]]:
        """
        从文本中提取题目（优先使用LLM，确保题目完整性）
        
        Args:
            text: PDF提取的文本
            meta: 元数据（地区、年份、考试类型等）
            use_regex_fallback: 是否在LLM失败时启用正则备用（默认False，防止低质量污染）
        
        Returns:
            题目列表，每个题目包含：content, question_type, index等
        """
        print(f"开始提取题目，文本总长度: {len(text)} 字符")
        
        # 优先使用LLM提取（更准确、更完整）
        if self.llm_client:
            print("优先使用LLM智能提取题目...")
            llm_questions = self._extract_with_llm(text, meta)
            llm_questions = [q for q in llm_questions if not self.is_exam_instruction_with_llm(q.get('content', ''))]

            llm_questions = self._merge_questions_by_number(llm_questions)

            # 缺题补救：用全文题号推断 expected，再用正则切题补齐缺失题号
            expected_nums = self._infer_expected_question_numbers(text)
            print(f"推断到预期题号范围: {expected_nums}")
            present_nums = set()
            for q in llm_questions:
                n = self._extract_question_number((q.get('content', '') or '').strip())
                if n and n.isdigit():
                    present_nums.add(int(n))

            missing_nums = [n for n in expected_nums if n not in present_nums]
            if missing_nums:
                try:
                    print("LLM提取题目失败，开始使用正则备用...")
                    regex_candidates = self._extract_with_regex(text, meta)
                    by_num: Dict[int, Dict[str, Any]] = {}
                    for rq in regex_candidates:
                        rn = self._extract_question_number((rq.get('content', '') or '').strip())
                        if rn and rn.isdigit():
                            by_num[int(rn)] = rq
                    recovered = 0
                    for mn in missing_nums:
                        if mn in by_num:
                            llm_questions.append(by_num[mn])
                            recovered += 1
                    if recovered:
                        llm_questions = self._merge_questions_by_number(llm_questions)
                        print(f"缺题补救：补回 {recovered} 道题（缺失题号: {missing_nums}）")
                    else:
                        print(f"缺题补救：未能从正则切题补回缺失题号: {missing_nums}")
                except Exception as e:
                    print(f"缺题补救失败: {e}")
            
            # 验证题目完整性
            complete_questions = []
            for q in llm_questions:
                content = q.get('content', '')
                if self._is_question_complete(content):
                    complete_questions.append(q)
                else:
                    n = self._extract_question_number(content)
                    if n is not None and len((content or '').strip()) >= 20:
                        complete_questions.append(q)
                    else:
                        print(f"  警告：题目不完整，已跳过: {q.get('content', '')[:100]}...")
            
            if complete_questions:
                print(f"✓ LLM提取完成，共 {len(complete_questions)} 道完整题目")
                return complete_questions
            else:
                print("⚠ LLM未提取到完整题目，尝试使用正则表达式备用方案...")
        
        # 如果LLM不可用或提取失败，根据开关决定是否使用正则表达式备用方案
        if use_regex_fallback:
            print("使用正则表达式提取（备用方案）...")
            regex_questions = self._extract_with_regex(text, meta)
            regex_questions = [q for q in regex_questions if not self._is_exam_instruction(q.get('content', ''))]
            print(f"✓ 正则表达式提取到 {len(regex_questions)} 道题目")
            return regex_questions
        else:
            print("已禁用正则备用方案，返回空列表以避免低质量题目污染。")
            return []
    
    def _extract_with_regex(self, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """使用正则表达式提取题目（备用方案）"""
        questions = []
        
        # 改进的正则表达式：匹配题目编号和完整内容
        # 匹配格式：数字. 或 数字、 开头的题目
        # 改进：匹配到下一个题目编号或明显的题目结束标志
        
        # 先找到所有题目编号的位置
        question_starts = []
        # 匹配：1. 或 1、 或 (1) 或 一、 等格式
        patterns = [
            (r'^(\d+)[\.、]\s+', re.MULTILINE),  # 1. 或 1、
            (r'^\((\d+)\)\s+', re.MULTILINE),   # (1)
            (r'^[一二三四五六七八九十]+[、．]\s+', re.MULTILINE),  # 一、
        ]
        
        for pattern, flags in patterns:
            for match in re.finditer(pattern, text, flags):
                start_pos = match.start()
                question_starts.append((start_pos, match.group(0)))
        
        # 按位置排序
        question_starts.sort(key=lambda x: x[0])
        
        # 提取每道题目的完整内容
        for i, (start_pos, prefix) in enumerate(question_starts):
            # 确定结束位置：下一个题目开始或文本结束
            end_pos = question_starts[i + 1][0] if i + 1 < len(question_starts) else len(text)
            
            question_text = text[start_pos:end_pos].strip()
            
            # 过滤太短的文本（可能是误匹配）
            if len(question_text) < 30:
                continue
            
            # 清理题目文本：移除多余的空白
            question_text = re.sub(r'\s+', ' ', question_text)
            question_text = re.sub(r'\n{2,}', '\n', question_text)
            
            questions.append({
                "content": question_text,
                "question_type": "未知题型",  # 稍后批量识别
                "index": len(questions) + 1,
                "source_meta": meta,
            })
        
        # 批量识别题型和过滤试卷说明
        if questions:
            question_contents = [q["content"] for q in questions]
            question_types = self.batch_identify_question_types(question_contents)
            is_instructions = self.batch_is_exam_instruction(question_contents)
            
            # 更新题型并过滤试卷说明
            filtered_questions = []
            for i, q in enumerate(questions):
                if not is_instructions[i]:  # 只保留非试卷说明的题目
                    q["question_type"] = question_types[i]
                    filtered_questions.append(q)
            
            return filtered_questions
        
        return questions
    
    def _extract_with_llm(self, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """使用LLM提取题目（智能提取，优先使用）"""
        if not self.llm_client:
            return []
        
        # 如果文本太长，分块处理（每块更小，避免LLM响应阶段过长被截断）
        chunk_size = 2000
        overlap = 400
        chunks = []
        
        if len(text) <= chunk_size:
            chunks = [text]
        else:
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end]
                chunks.append(chunk)
                if end == len(text):
                    break
                start = end - overlap  # 重叠部分避免题目被截断
        
        
        all_questions = []
        
        print(f"开始使用LLM提取题目，文本分为 {len(chunks)} 块处理...")

        def _parse_llm_json(resp: str) -> List[Dict[str, Any]]:
            """对LLM响应做容错解析，尽量恢复JSON数组"""
            parsed = []
            cleaned = resp.strip()
            # 去掉markdown代码块
            cleaned = re.sub(r"^```json\\s*", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
            cleaned = re.sub(r"^```\\s*", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"```\\s*$", "", cleaned, flags=re.MULTILINE)
            # 截取第一个'['到最后一个']'
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1 and end > start:
                cleaned = cleaned[start:end+1]
            # 替换智能引号/修剪尾随逗号
            cleaned = cleaned.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
            cleaned = re.sub(r",\\s*]", "]", cleaned)
            cleaned = re.sub(r",\\s*}", "}", cleaned)
            try:
                return json.loads(cleaned)
            except Exception:
                # 尝试补一个右中括号
                if not cleaned.endswith("]"):
                    try:
                        return json.loads(cleaned + "]")
                    except Exception:
                        pass
            # 回退：尝试逐个解析对象，忽略损坏尾巴
            try:
                obj_matches = re.findall(r"\{[^{}]*\}", cleaned, flags=re.DOTALL)
                for obj_str in obj_matches:
                    try:
                        parsed_obj = json.loads(obj_str)
                        parsed.append(parsed_obj)
                    except Exception:
                        continue
            except Exception:
                pass
            return parsed
        
        for i, chunk in enumerate(chunks):
            print(f"正在处理第 {i+1}/{len(chunks)} 块（长度: {len(chunk)} 字符）...")
            
            prompt = f"""你是一个专业的数学试卷题目提取助手。请从以下试卷文本中提取所有数学题目。

**核心要求 - 题目必须完整：**
1. **必须包含题号**：如"1."、"2、"、"(1)"、"一、"等
2. **必须包含完整题干**：题目的完整描述和条件
3. **选择题必须包含所有选项**：如A、B、C、D等所有选项
4. **填空题必须包含所有空格**：所有需要填空的位置
5. **解答题必须包含完整问题**：所有需要解答的问题
6. **保留公式**：保留指数/分数/根号等符号（如 x^2, 1/2, √3, ≤, ≥, ∑, ∏）

**提取规则：**
- **只提取真正的数学题目**，忽略试卷说明、注意事项、标题等非题目内容
- **题目必须完整**：不能只提取题干的一部分，必须包含题号、完整题干、所有选项（如果有）
- **识别题目编号**：题目通常以数字开头（如"1."、"2、"、"(1)"等）
- **过滤掉以下内容**：
  - 试卷说明（如"答卷前考生务必将..."、"回答选择题时..."等）
  - 页眉页脚（如"第X页"、"共X页"等）
  - 考试信息（如"考试时间"、"满分"等）
  - 注意事项

**示例格式：**
- 选择题应包含：题号 + 题干 + 选项A + 选项B + 选项C + 选项D
- 填空题应包含：题号 + 完整题干（包括所有空格位置）
- 解答题应包含：题号 + 完整题干 + 所有问题

试卷文本：
{chunk}

请以JSON数组格式返回所有提取到的**完整题目**，格式如下，务必使用 markdown ```json 代码块包裹，且只输出这个数组：
```json
[
  {{
    "index": 1,
    "content": "1. 已知函数f(x)=x²+1，则f(2)的值为（    ）\nA. 3\nB. 4\nC. 5\nD. 6",
    "question_type": "选择题"
  }},
  {{
    "index": 2,
    "content": "2. 若a+b=5，a-b=1，则a=____，b=____",
    "question_type": "填空题"
  }}
]
```

**重要：content字段必须包含题号、完整题干和所有选项（如果有），不要遗漏任何部分！**

只返回JSON数组，不要其他解释文字。"""
            
            try:
                print(f"  调用LLM API...")
                response = self.llm_client.generate(prompt)
                print(f"  LLM响应长度: {len(response)} 字符")
                
                questions = _parse_llm_json(response)
                if questions:
                    # 验证和清理题目
                    valid_questions = []
                    for q in questions:
                        content = q.get('content', '').strip()
                        
                        # 1. 过滤太短的内容
                        if len(content) < 30:
                            continue
                        
                        # 2. 验证题目完整性
                        if not self._is_question_complete(content):
                            n = self._extract_question_number(content)
                            if n is None and not self._is_subquestion_only(content):
                                print(f"    跳过不完整题目: {content[:80]}...")
                                continue
                        
                        # 3. 添加元数据
                        q["source_meta"] = meta
                        # 若LLM未给出题型或给出未知，使用启发式补全
                        if not q.get("question_type") or q.get("question_type") == "未知题型":
                            q["question_type"] = self._infer_question_type_heuristic(content)
                        valid_questions.append(q)
                    
                    # 批量判断并过滤试卷说明
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
        
        # 去重（基于内容相似度）
        unique_questions = []
        seen_contents = set()
        for q in all_questions:
            content = q.get('content', '').strip()
            norm = self._normalize_for_dedupe(content)
            content_key = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_questions.append(q)

        unique_questions = self._merge_questions_by_number(unique_questions)
        
        # 补全题型（若缺失或未知）
        for q in unique_questions:
            if not q.get("question_type") or q.get("question_type") == "未知题型":
                q["question_type"] = self._infer_question_type_heuristic(q.get("content", ""))
        
        # 重新编号
        for i, q in enumerate(unique_questions):
            q["index"] = i + 1
        
        print(f"✓ LLM提取完成，共提取到 {len(unique_questions)} 道有效题目")
        return unique_questions
    
    def enrich_question_with_llm(
        self,
        question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """使用LLM丰富题目信息（知识点、难度等）"""
        if not self.llm_client:
            return question
        
        prompt = f"""
请分析以下数学题目，并给出：
1. 涉及的知识点列表
2. 难度等级（1-5，1最简单，5最难）

题目内容：
{question['content']}

请以JSON格式返回：
{{
    "knowledge_points": ["知识点1", "知识点2"],
    "difficulty": 3
}}
"""
        
        try:
            response = self.llm_client.generate(prompt)
            import json
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                question["knowledge_points"] = result.get("knowledge_points", [])
                question["difficulty"] = result.get("difficulty", 3)
        except Exception as e:
            print(f"LLM丰富题目信息失败: {e}")
        
        return question


def process_pdf_to_questions(
    pdf_path: str,
    meta: Dict[str, Any],
    ocr_enabled: bool = True,
    llm_client=None,
    use_regex_fallback: bool = False,
    auto_meta: bool = True,            # 新增
    meta_pages: int = 2,               # 新增：只扫前2页做元数据
) -> Dict[str, Any]:
    processor = PDFProcessor(ocr_enabled=ocr_enabled)

    meta_report = {"meta": {}, "confidence": {}, "evidence": {}}
    meta_merged = dict(meta or {})

    # 1) 自动识别元数据（只在字段缺失或显式开启时）
    if auto_meta:
        preview_text = processor.extract_text(pdf_path, max_pages=meta_pages)
        preview_text = processor.clean_text(preview_text)
        meta_extractor = ExamMetaExtractor(llm_client=llm_client)
        meta_report = meta_extractor.extract(preview_text)

        # 合并策略：用户显式传入优先；否则用自动识别补齐
        inferred = meta_report.get("meta", {})
        for k, v in inferred.items():
            if not meta_merged.get(k):
                meta_merged[k] = v

    # 2) 全文提取与清洗（用于提题）
    text = processor.extract_text(pdf_path)       # 默认全量
    text = processor.clean_text(text)

    extractor = QuestionExtractor(llm_client=llm_client)
    questions = extractor.extract_questions_from_text(text, meta_merged, use_regex_fallback=use_regex_fallback)

    enriched_questions = [extractor.enrich_question_with_llm(q) for q in questions]

    return {
        "questions": enriched_questions,
        "meta_used": meta_merged,
        "meta_inferred": meta_report.get("meta", {}),
        "meta_confidence": meta_report.get("confidence", {}),
        "meta_evidence": meta_report.get("evidence", {}),
    }

