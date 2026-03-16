"""PDF处理模块：OCR、文本提取、题目提取"""
import re
import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from pathlib import Path
try:
    import fitz  # PyMuPDF，用于提取原生pdf文本
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False
    print("警告：PyMuPDF(fitz)不可用，将无法直接提取文本层，只能依赖OCR")
from PIL import Image
import io
import os
import sys
import json
import hashlib

try:
    from pdf2image import convert_from_path  # 用于将PDF转换为图片
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    convert_from_path = None
    print("警告：pdf2image 不可用，扫描版 PDF 将无法进行 OCR")

try:
    import pytesseract  # 用于OCR，处理扫描版PDF
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False
    print("警告：pytesseract 不可用，Tesseract OCR 后端不可用")

OCR_AVAILABLE = bool(PDF2IMAGE_AVAILABLE and TESSERACT_AVAILABLE)


from .meta_extractor import ExamMetaExtractor

class PDFProcessor:
    """PDF处理器：支持OCR和文本提取"""
    
    def __init__(self, ocr_enabled: bool = True, vision_llm_client=None):
        self.ocr_enabled = ocr_enabled and OCR_AVAILABLE
        self.poppler_path = self._resolve_poppler_path()
        self._setup_library_path()
        # 视觉大模型客户端，用于替代 Tesseract 识别数学公式
        self.vision_llm_client = vision_llm_client

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
        if not PYMUPDF_AVAILABLE:
            return True

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

    def detect_ocr_need(self, pdf_path: str, max_pages: int = 3) -> Dict[str, Any]:
        if not PYMUPDF_AVAILABLE:
            return {
                "need_ocr": True,
                "native_text_len": 0,
                "meaningful_ratio": 0.0,
                "checked_pages": 0,
                "reason": "pymupdf_unavailable",
            }

        try:
            doc = fitz.open(pdf_path)
            text_count = 0
            meaningful_ratio_sum = 0.0
            checked_pages = 0
            page_count = min(max_pages, len(doc))
            for page_num in range(page_count):
                page = doc[page_num]
                text = page.get_text() or ""
                if re.search(r'\{#\{.*?\}#\}', text):
                    doc.close()
                    return {
                        "need_ocr": True,
                        "native_text_len": len(text.strip()),
                        "meaningful_ratio": 0.0,
                        "checked_pages": page_count,
                        "reason": "encrypted_or_placeholder_text",
                    }
                t = text.strip()
                text_count += len(t)
                if t:
                    checked_pages += 1
                    meaningful = re.findall(r'[A-Za-z0-9\u4e00-\u9fa5]', t)
                    meaningful_ratio_sum += len(meaningful) / max(len(t), 1)
            doc.close()

            avg_ratio = (meaningful_ratio_sum / checked_pages) if checked_pages else 0.0

            if text_count < 100:
                return {
                    "need_ocr": True,
                    "native_text_len": text_count,
                    "meaningful_ratio": avg_ratio,
                    "checked_pages": page_count,
                    "reason": "native_text_too_short",
                }

            if checked_pages > 0 and avg_ratio < 0.2:
                return {
                    "need_ocr": True,
                    "native_text_len": text_count,
                    "meaningful_ratio": avg_ratio,
                    "checked_pages": page_count,
                    "reason": "meaningful_ratio_too_low",
                }

            return {
                "need_ocr": False,
                "native_text_len": text_count,
                "meaningful_ratio": avg_ratio,
                "checked_pages": page_count,
                "reason": "native_text_ok",
            }
        except Exception as e:
            return {
                "need_ocr": True,
                "native_text_len": 0,
                "meaningful_ratio": 0.0,
                "checked_pages": 0,
                "reason": f"detect_failed:{e}",
            }
    
    def extract_text_with_vision_llm(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        """
        使用视觉大模型识别 PDF 页面图片，专为含数学公式/向量符号的试卷设计。
        将每页渲染为图片后发给视觉 LLM，输出带 LaTeX 格式的完整文本。
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image 不可用，无法将 PDF 渲染为图片")
        if self.vision_llm_client is None:
            raise RuntimeError("未配置视觉 LLM 客户端（vision_llm_client）")
        if not self.vision_llm_client.supports_vision():
            raise RuntimeError(f"{self.vision_llm_client.__class__.__name__} 不支持视觉输入")

        import base64

        try:
            dpi = int(os.getenv("OCR_DPI", "200"))
        except Exception:
            dpi = 200

        first_page = 1
        last_page = max_pages if max_pages else None

        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            poppler_path=self.poppler_path,
            first_page=first_page,
            last_page=last_page,
        )
        print(f"视觉OCR: 已将 PDF 转换为 {len(images)} 张图片")

        vision_prompt = """请识别图片中的数学试卷文本，完整输出所有内容。

要求：
1. 保留所有题号、题干、选项，一字不漏
2. 数学公式用 LaTeX 格式表示：
   - 向量用 \\vec{} 或 \\overrightarrow{}，如 $\\vec{a}$、$\\overrightarrow{AB}$
   - 分数用 \\frac{}{}，如 $\\frac{1}{2}$
   - 根号用 \\sqrt{}，上标用 ^{}，下标用 _{}
   - 垂直符号 \\perp，平行 \\parallel，角 \\angle
3. 下标字母（如 A₁B₁）写成 $A_1B_1$
4. 选项 A、B、C、D 单独成行
5. 不要添加任何分析或解释，只输出原文"""

        all_text = []
        for i, image in enumerate(images):
            # 将 PIL Image 转为 base64
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            try:
                text = self.vision_llm_client.generate_with_image(
                    prompt=vision_prompt,
                    image_base64=img_b64,
                    image_media_type="image/png",
                )
                # 过滤掉视觉LLM调用失败的错误信息
                if text.startswith("[视觉LLM调用失败"):
                    print(f"  第 {i+1} 页视觉识别失败: {text}")
                    text = ""
            except Exception as e:
                print(f"  第 {i+1} 页视觉识别异常: {e}")
                text = ""

            all_text.append(f"--- 第 {i+1} 页 ---\n{text}\n")
            print(f"  第 {i+1}/{len(images)} 页识别完成，字符数: {len(text)}")

        return "\n".join(all_text)

    # 新增最大页数，方便识别前1-2页的元数据
    def extract_text_with_ocr(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        if not OCR_AVAILABLE:
            raise RuntimeError("OCR功能未启用或未安装相关依赖")
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image 不可用，无法将 PDF 渲染为图片进行 OCR")

        
        # 通过灰度化+二值化提升对比度，让OCR看得更清楚
        def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
            """OCR预处理：去噪 + 对比度增强 + Otsu二值化"""
            arr = np.array(img.convert("L"), dtype=np.uint8)

            # 1. 轻度去噪（高斯模糊，保留文字笔画，不破坏字形）
            arr = cv2.GaussianBlur(arr, (3, 3), 0)

            # 2. 对比度增强（CLAHE自适应均衡化）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            arr = clahe.apply(arr)

            # 3. Otsu 自动阈值二值化
            _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return Image.fromarray(arr, mode="L")

        # OCR后的文本后处理：清理空格/换行/特殊字符
        def _postprocess_ocr_text(t: str) -> str:
            if not t:
                return ""
            t = t.replace("\u00a0", " ")
            t = re.sub(r"[ \t]{2,}", " ", t)
            t = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", t)
            t = re.sub(r"\n{3,}", "\n\n", t)
            return t

        # pdf2image 支持 first_page / last_page（从1开始）
        first_page = 1
        last_page = max_pages if max_pages else None
        try:
            dpi = int(os.getenv("OCR_DPI", "300"))
        except Exception:
            dpi = 300
        # oem 1代表使用LSTM 深度学习引擎
        tesseract_config = os.getenv("TESSERACT_CONFIG", "--oem 1 --psm 4")
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("未安装 pytesseract，无法进行 OCR")

        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            poppler_path=self.poppler_path,
            first_page=first_page,
            last_page=last_page
        )
        print(f"OCR: 已将 PDF 转换为 {len(images)} 张图片")
        all_text = []
        for i, image in enumerate(images):
            processed = _preprocess_for_ocr(image)
            text = pytesseract.image_to_string(processed, lang='chi_sim+eng', config=tesseract_config)
            text = _postprocess_ocr_text(text)
            all_text.append(f"--- 第 {i+1} 页 ---\n{text}\n")
        return "\n".join(all_text)
    
    def extract_text_native(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF(fitz)不可用，无法提取PDF文本层")
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

        detect_pages = 3
        try:
            if max_pages is not None:
                detect_pages = min(detect_pages, max_pages)
        except Exception:
            detect_pages = 3

        detect = self.detect_ocr_need(str(pdf_path), max_pages=detect_pages)
        need_ocr = bool(detect.get("need_ocr", True))

        self.last_extraction_mode = "ocr" if need_ocr else "native"
        self.last_extraction_stats = detect

        if need_ocr:
            # 优先使用视觉大模型（对数学公式/向量符号识别更准确）
            if self.vision_llm_client is not None and self.vision_llm_client.supports_vision():
                print("使用视觉大模型进行OCR识别（支持数学公式/向量符号）...")
                self.last_extraction_mode = "vision_llm"
                return self.extract_text_with_vision_llm(str(pdf_path), max_pages=max_pages)
            # 降级到 Tesseract OCR
            if not self.ocr_enabled:
                raise RuntimeError("PDF需要OCR（扫描版/无有效文本层），但当前已禁用OCR")
            return self.extract_text_with_ocr(str(pdf_path), max_pages=max_pages)

        return self.extract_text_native(str(pdf_path), max_pages=max_pages)
    
    def clean_text(self, text: str) -> str:
        """保留原始文本，仅压缩多余空行，让LLM处理所有噪声"""
        text = re.sub(r'\n{3,}', '\n\n', text)
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

        pattern = r'([A-Da-d])[\.\、\)]\s*(.+?)(?=(?:[A-Da-d][\.\、\)]|$))'
        for m in re.finditer(pattern, content, re.DOTALL):
            key = (m.group(1) or "").upper()
            val = (m.group(2) or "").strip()
            if key and val:
                options[key] = val

        return options

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
        从文本中提取题目（使用LLM提取，LLM补救缺失题目）

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

        print("使用LLM智能提取题目...")
        llm_questions = self._extract_with_llm(text, meta)
        llm_questions = [q for q in llm_questions if not self._is_exam_instruction_with_llm(q.get('content', ''))]

        # 缺题补救：推断预期题号范围，用LLM针对性补救缺失题目
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
                recovered_items = self._recover_missing_questions_with_llm(text, meta, missing_nums)
                if recovered_items:
                    llm_questions.extend(recovered_items)
                    print(f"缺题补救：补回 {len(recovered_items)} 道题（缺失题号: {missing_nums}）")
                else:
                    print(f"缺题补救（LLM）无结果，缺失题号: {missing_nums} 无法补回")
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

        print(f"✓ LLM提取完成，共 {len(complete_questions)} 道完整题目")
        return complete_questions
    

    
    def _extract_with_llm(self, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """使用LLM提取题目（智能提取，优先使用）"""
        if not self.llm_client:
            return []

        # 优先按页分块（OCR文本每页有 --- 第 N 页 --- 分隔符）
        # 页内题目完整，不会被截断；相邻页合并到不超过 chunk_size
        chunk_size = int(os.getenv("LLM_CHUNK_SIZE", "3000"))
        overlap_pages = 1  # 相邻块共享1页，防止跨页题目丢失

        # 尝试按页分割
        page_pattern = re.compile(r'(?=--- 第 \d+ 页 ---)')
        page_splits = list(page_pattern.finditer(text))

        if len(page_splits) >= 2:
            # 按页切分
            pages = []
            for i, m in enumerate(page_splits):
                start = m.start()
                end = page_splits[i + 1].start() if i + 1 < len(page_splits) else len(text)
                pages.append(text[start:end])

            # 合并相邻页到不超过 chunk_size
            chunks = []
            current = "" # 当前合并好的文本块（比如 “页 1 + 页 2” 的完整文本）
            current_pages = [] # 当前文本块包含的页文本列表（比如 [页1文本, 页2文本]）
            # 核心判断： 每来一个新页，看"当前块 + 新页"是否超过 chunk_size：
            # 超过 → 当前块封存，把最后1页作为重叠，开启新块
            # 不超过 → 新页直接追加进当前块
            # 假设有4页，chunk_size=3000，overlap_pages=1：
            # 页1(1000字) + 页2(1000字) + 页3(1000字) = 3000 ✅ → 继续合并
            # 页1+页2+页3 + 页4(1000字) = 4000 > 3000 ❌ → 封存块1
            # 块1 = [页1, 页2, 页3]
            # 新块开始：
            # overlap_text = 页3（最后1页）
            # current = 页3 + 页4
            # 块2 = [页3, 页4]  ← 页3被两个块共享，防止页3末尾的题目丢失
            for page in pages:
                if current and len(current) + len(page) > chunk_size:
                    # 将 “当前合并好的文本块” 和 “该块包含的页列表副本” 打包成元组，存入最终的 chunks 列表，既记录文本块内容，也记录其对应的页来源
                    chunks.append((current, current_pages[:]))
                    # 保留最后 overlap_pages 页作为下一块的开头，即“重叠页文本”
                    overlap_text = "".join(current_pages[-overlap_pages:])
                    current = overlap_text + page # 将 “重叠页文本” 和 “当前页文本” 合并，得到新的 “当前合并好的文本块”
                    current_pages = current_pages[-overlap_pages:] + [page]
                else:
                    current += page
                    current_pages.append(page)
            if current:
                chunks.append((current, current_pages[:]))
            chunks = [c for c, _ in chunks]
            print(f"按页分块：{len(pages)} 页 → {len(chunks)} 块")
        else:
            # 降级为按字符分块（固定2000字符/800字符重叠）
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
                "**示例格式：**\n"
                "- 选择题应包含：题号 + 题干 + 选项A + 选项B + 选项C + 选项D\n"
                "- 填空题应包含：题号 + 完整题干（包括所有空格位置）\n"
                "- 解答题应包含：题号 + 完整题干 + 所有问题\n\n"
                f"试卷文本：\n{chunk}\n\n"
                "请以JSON数组格式返回所有提取到的**完整题目**，格式如下，务必使用 markdown ```json 代码块包裹，且只输出这个数组：\n"
                "```json\n"
                "[\n"
                "  {{\n"
                '    \"index\": 1,\n'
                '    \"content\": \"1. 已知函数f(x)=x²+1，则f(2)的值为（    ）\\nA. 3\\nB. 4\\nC. 5\\nD. 6\",\n'
                '    \"question_type\": \"选择题\",\n'
                '    \"knowledge_points\": [\"函数\", \"代入求值\"],\n'
                '    \"difficulty\": 2\n'
                "  }},\n"
                "  {{\n"
                '    \"index\": 2,\n'
                '    \"content\": \"2. 若a+b=5，a-b=1，则a=____，b=____\",\n'
                '    \"question_type\": \"填空题\",\n'
                '    \"knowledge_points\": [\"方程组\"],\n'
                '    \"difficulty\": 2\n'
                "  }}\n"
                "]\n"
                "```\n\n"
                "**重要：content字段必须包含题号、完整题干和所有选项（如果有），不要遗漏任何部分！**\n\n"
                "只返回JSON数组，不要其他解释文字。"
            )
            
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
        
        # 去重：先按题号去重（处理重叠页重复提取），再按内容哈希去重
        unique_questions = []
        seen_nums = set()
        seen_contents = set()
        for q in all_questions:
            content = q.get('content', '').strip()
            # 1. 题号去重：同一题号只保留第一次
            qnum = self._extract_question_number(content)
            if qnum and qnum.isdigit():
                if qnum in seen_nums:
                    continue
                seen_nums.add(qnum)
            # 2. 内容哈希去重：兜底处理无题号或题号相同但内容不同的情况
            norm = self._normalize_for_dedupe(content)
            content_key = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_questions.append(q)

        # 补全题型（若缺失或未知）
        for q in unique_questions:
            if not q.get("question_type") or q.get("question_type") == "未知题型":
                q["question_type"] = self._infer_question_type_heuristic(q.get("content", ""))
        
        # 重新编号是一个“清洗/归一化”步骤，index是一个“展示用序号”，并不一定等于题号
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

        enriched = self.batch_enrich_questions_with_llm([question])
        return enriched[0] if enriched else question

    def batch_enrich_questions_with_llm(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not questions:
            return []
        if not self.llm_client:
            return questions
        if str(os.getenv("ENRICH_WITH_LLM", "0")).strip() in {"0", "false", "False", "no", "No"}:
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


def process_pdf_to_questions(
    pdf_path: str,
    meta: Dict[str, Any],
    ocr_enabled: bool = True, # 视觉模型失效时降级回Tesseract OCR
    llm_client=None,
    vision_llm_client=None,
    auto_meta: bool = True,
    meta_pages: int = 2,
) -> Dict[str, Any]:
    """
    Args:
        vision_llm_client: 支持视觉输入的 LLM 客户端（如 OpenAIClient 配合 gpt-4o）。
            传入后扫描版 PDF 将优先使用视觉大模型识别，可正确处理向量符号和数学公式。
            若不传，则降级使用 Tesseract OCR。
    """
    processor = PDFProcessor(ocr_enabled=ocr_enabled, vision_llm_client=vision_llm_client)

    meta_report = {"meta": {}, "confidence": {}, "evidence": {}}
    meta_merged = dict[str, Any](meta or {})

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
    questions = extractor.extract_questions_from_text(text, meta_merged)

    if getattr(processor, "last_extraction_mode", None) == "native":
        questions = extractor.populate_structured_fields(questions)

    need_enrich = False
    for q in questions:
        if not q.get("knowledge_points") or q.get("difficulty") in (None, 3, "3"):
            need_enrich = True
            break

    enriched_questions = extractor.batch_enrich_questions_with_llm(questions) if need_enrich else questions

    return {
        "questions": enriched_questions,
        "text_extraction": {
            "mode": getattr(processor, "last_extraction_mode", None),
            "stats": getattr(processor, "last_extraction_stats", None),
        },
        "meta_used": meta_merged,
        "meta_inferred": meta_report.get("meta", {}),
        "meta_confidence": meta_report.get("confidence", {}),
        "meta_evidence": meta_report.get("evidence", {}),
    }

