"""PDF处理器：OCR 与文本提取"""
import base64
import re
import os
import sys
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path
from statistics import median
import numpy as np
import cv2
from PIL import Image

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False
    print("警告：PyMuPDF(fitz)不可用，将无法直接提取文本层，只能依赖OCR")

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    convert_from_path = None
    print("警告：pdf2image 不可用，扫描版 PDF 将无法进行 OCR")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None
    TESSERACT_AVAILABLE = False
    print("警告：pytesseract 不可用，Tesseract OCR 后端不可用")

OCR_AVAILABLE = bool(PDF2IMAGE_AVAILABLE and TESSERACT_AVAILABLE)

class PDFProcessor:
    """PDF处理器：支持OCR和文本提取"""

    # 部分试卷的文本层会把 Symbol/MathType 字形写进 Private Use Area。
    # 这里先做稳定的一一映射，避免后续链路继续携带这些脏字符。
    _PDF_GLYPH_MAP = {
        "\uf022": "∀",
        "\uf024": "∃",
        "\uf028": "(",
        "\uf029": ")",
        "\uf02b": "+",
        "\uf02d": "-",
        "\uf03d": "=",
        "\uf03e": ">",
        "\uf055": "∪",
        "\uf056": "△",
        "\uf05e": "⊥",
        "\uf061": "α",
        "\uf062": "β",
        "\uf06c": "λ",
        "\uf06f": "°",
        "\uf071": "θ",
        "\uf072": "→",
        "\uf075": "",
        "\uf0a5": "∞",
        "\uf0ce": "∈",
        "\uf0d0": "∠",
        "\uf0e6": "(",
        "\uf0e7": "(",
        "\uf0e8": "(",
        "\uf0f6": ")",
        "\uf0f7": ")",
        "\uf0f8": ")",
    }
    
    def __init__(self, ocr_enabled: bool = True, vision_llm_client=None):
        self.ocr_enabled = ocr_enabled and OCR_AVAILABLE
        self.poppler_path = self._resolve_poppler_path()
        self._setup_library_path()
        # 视觉大模型客户端，用于替代 Tesseract 识别数学公式
        self.vision_llm_client = vision_llm_client
        # 最近一次提取状态，供上层接口回传调试信息
        self.last_extraction_mode: Optional[str] = None
        self.last_extraction_stats: Optional[Dict[str, Any]] = None

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
    
    # def is_scanned_pdf(self, pdf_path: str) -> bool:
    #     """
    #     判断PDF是否为扫描版（图片格式）
        
    #     简单判断：如果PDF中文本层很少或为空，可能是扫描版
    #     """
    #     detect = self.detect_ocr_need(pdf_path, max_pages=3)
    #     return bool(detect.get("need_ocr", True))

    def _render_pdf_page_paths(
        self,
        pdf_path: str,
        dpi: int,
        max_pages: Optional[int] = None,
    ) -> tuple[tempfile.TemporaryDirectory, list[str]]:
        """
        将 PDF 页面渲染到临时目录并返回图片路径列表。

        使用 `paths_only=True`，避免一次性把所有 PIL 图片加载进内存。
        调用方负责在用完后清理返回的 TemporaryDirectory。
        """
        if not PDF2IMAGE_AVAILABLE:
            raise RuntimeError("pdf2image 不可用，无法将 PDF 渲染为图片")
        # 在系统临时目录下创建一个以 "pdf_pages_" 开头的临时文件夹，用于存放后续生成的PDF页面图片。
        temp_dir = tempfile.TemporaryDirectory(prefix="pdf_pages_")
        # pdf2imagek库的把pdf转换成图片的函数，当paths_only=True时，返回图片文件路径列表
        image_paths = convert_from_path(
            pdf_path,
            dpi=dpi,
            poppler_path=self.poppler_path,
            first_page=1,
            last_page=max_pages if max_pages else None,
            output_folder=temp_dir.name,
            fmt="png",
            paths_only=True,
        )
        return temp_dir, [str(p) for p in image_paths]

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
            with fitz.open(pdf_path) as doc:
                text_count = 0
                meaningful_ratio_sum = 0.0
                checked_pages = 0
                page_count = min(max_pages, len(doc))
                for page_num in range(page_count):
                    page = doc[page_num]
                    text = page.get_text() or ""
                    if re.search(r'\{#\{.*?\}#\}', text):
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

    def _normalize_pdf_glyphs(self, text: str) -> str:
        """清洗 PDF 文本层里的私有区数学字形。"""
        if not text:
            return ""

        normalized = str(text).replace("\u00a0", " ")
        normalized = normalized.translate(str.maketrans(self._PDF_GLYPH_MAP))

        # 某些 PDF 会把向量箭头拆成前置箭头 + 变量，如 "→v"；统一整理为 "v→"。
        normalized = re.sub(r"→\s*([A-Za-z][A-Za-z0-9_]*)", r"\1→", normalized)

        # 坐标经常被提成 "( ) 2,1" 这种顺序，先做一个保守重排。
        normalized = re.sub(
            r"\(\s*\)\s*([A-Za-z0-9+\-./]+\s*(?:,\s*[A-Za-z0-9+\-./]+)+)",
            r"(\1)",
            normalized,
        )

        # 去掉仍未识别的私有区字符，至少不要让脏字形继续流到后面。
        normalized = re.sub(r"[\uf000-\uf8ff]", " ", normalized)

        normalized = re.sub(r"\(\s+", "(", normalized)
        normalized = re.sub(r"\s+\)", ")", normalized)
        normalized = re.sub(r"[ \t]{2,}", " ", normalized)
        normalized = re.sub(r" *\n *", "\n", normalized)
        return normalized.strip()

    def _group_words_into_visual_lines(self, words: list[tuple]) -> list[list[dict[str, Any]]]:
        """按视觉行重建 words，避免 page.get_text() 的阅读顺序把数学式打散。"""
        tokens: list[dict[str, Any]] = []
        for word in words or []:
            if len(word) < 5:
                continue
            x0, y0, x1, y1, raw_text = word[:5]
            text = self._normalize_pdf_glyphs(raw_text)
            if not text or not text.strip():
                continue
            tokens.append(
                {
                    "x0": float(x0),
                    "y0": float(y0),
                    "x1": float(x1),
                    "y1": float(y1),
                    "cx": (float(x0) + float(x1)) / 2,
                    "cy": (float(y0) + float(y1)) / 2,
                    "w": max(float(x1) - float(x0), 0.0),
                    "h": max(float(y1) - float(y0), 0.0),
                    "text": text.strip(),
                }
            )

        if not tokens:
            return []

        tokens.sort(key=lambda t: (t["cy"], t["x0"]))
        line_merge_tol = max(2.0, median(t["h"] for t in tokens) * 0.18)

        lines: list[list[dict[str, Any]]] = []
        current_line: list[dict[str, Any]] = []
        line_top = 0.0
        line_bottom = 0.0

        for token in tokens:
            if not current_line:
                current_line = [token]
                line_top = token["y0"]
                line_bottom = token["y1"]
                continue

            overlaps_current_line = (
                token["y0"] <= line_bottom + line_merge_tol
                and token["y1"] >= line_top - line_merge_tol
            )
            if overlaps_current_line and self._starts_new_structural_line(token["text"]):
                current_cy = median(t["cy"] for t in current_line)
                if token["cy"] - current_cy > max(8.0, median(t["h"] for t in current_line) * 0.6):
                    overlaps_current_line = False
            if overlaps_current_line:
                current_line.append(token)
                line_top = min(line_top, token["y0"])
                line_bottom = max(line_bottom, token["y1"])
            else:
                lines.append(sorted(current_line, key=lambda t: (t["x0"], t["cy"])))
                current_line = [token]
                line_top = token["y0"]
                line_bottom = token["y1"]

        if current_line:
            lines.append(sorted(current_line, key=lambda t: (t["x0"], t["cy"])))

        return lines

    def _starts_new_structural_line(self, text: str) -> bool:
        if not text:
            return False
        return bool(
            re.match(r"^\d{1,3}\s*[\.、．)]", text)
            or re.match(r"^[A-Da-d][\.\、\)]", text)
            or re.match(r"^[（(]\s*\d+\s*[）)]", text)
        )

    def _split_structural_visual_line(self, line_tokens: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if not line_tokens:
            return []

        anchors = sorted(
            [t for t in line_tokens if self._starts_new_structural_line(t["text"])],
            key=lambda t: (t["cy"], t["x0"]),
        )
        if len(anchors) <= 1:
            return [sorted(line_tokens, key=lambda t: (t["x0"], t["cy"]))]

        gaps = [anchors[i + 1]["cy"] - anchors[i]["cy"] for i in range(len(anchors) - 1)]
        if not any(gap > 8.0 for gap in gaps):
            return [sorted(line_tokens, key=lambda t: (t["x0"], t["cy"]))]

        boundaries = [
            (anchors[i]["cy"] + anchors[i + 1]["cy"]) / 2.0
            for i in range(len(anchors) - 1)
        ]
        groups: list[list[dict[str, Any]]] = [[] for _ in anchors]
        for token in line_tokens:
            idx = 0
            while idx < len(boundaries) and token["cy"] > boundaries[idx]:
                idx += 1
            groups[idx].append(token)

        return [sorted(group, key=lambda t: (t["x0"], t["cy"])) for group in groups if group]

    def _is_simple_math_anchor(self, text: str) -> bool:
        if not text:
            return False
        return bool(re.fullmatch(r"[A-Za-zΑ-Ωα-ω]+|\d+|[+\-=]", text))

    def _should_attach_vertically(self, anchor: dict[str, Any], other: dict[str, Any]) -> bool:
        x_overlap = min(anchor["x1"], other["x1"]) - max(anchor["x0"], other["x0"])
        x_gap = max(other["x0"] - anchor["x1"], anchor["x0"] - other["x1"], 0.0)
        cy_gap = abs(anchor["cy"] - other["cy"])
        vertical_tol = max(2.5, min(anchor["h"], other["h"]) * 0.18)
        return (x_overlap >= 0 or x_gap <= 1.5) and cy_gap >= vertical_tol

    def _looks_like_subscript(self, base_text: str, lower_tokens: list[dict[str, Any]]) -> bool:
        if not base_text or not lower_tokens:
            return False
        lower_text = "".join(t["text"] for t in lower_tokens)
        return bool(re.fullmatch(r"[A-Za-zΑ-Ωα-ω]+", base_text) and re.fullmatch(r"\d+", lower_text))

    def _render_vertical_stack(self, anchor: dict[str, Any], attached: list[dict[str, Any]]) -> str:
        if not attached:
            return anchor["text"]

        upper_tokens = sorted(
            [t for t in attached if t["cy"] < anchor["cy"] - 1.5],
            key=lambda t: (t["x0"], t["cy"]),
        )
        lower_tokens = sorted(
            [t for t in attached if t["cy"] > anchor["cy"] + 1.5],
            key=lambda t: (t["x0"], t["cy"]),
        )

        text = anchor["text"]
        if upper_tokens:
            text += "^" + "".join(t["text"] for t in upper_tokens)

        if lower_tokens:
            lower_text = "".join(t["text"] for t in lower_tokens)
            if self._looks_like_subscript(text, lower_tokens) and not upper_tokens:
                text += "_" + lower_text
            else:
                text += "/" + lower_text

        return text

    def _should_insert_space(self, prev: str, curr: str) -> bool:
        if not prev or not curr:
            return False

        prev_last = prev[-1]
        curr_first = curr[0]

        if prev_last in "([{（【" or curr_first in ")]}）】，,。；;：:!?！？":
            return False
        if prev_last in "+-*/=<>≤≥×÷·,^" or curr_first in "+-*/=<>≤≥×÷·,^":
            return False
        if prev_last == "_" or curr_first == "_":
            return False
        if re.fullmatch(r"\d+", prev) and re.fullmatch(r"[A-Za-zΑ-Ωα-ω]", curr):
            return False
        if re.fullmatch(r"[A-Za-zΑ-Ωα-ω]", prev) and re.fullmatch(r"\d+", curr):
            return False
        if re.fullmatch(r"[A-Za-zΑ-Ωα-ω0-9_]+", prev) and curr_first == "(":
            return False
        if prev_last in "，,。；;：:!?！？":
            return False
        if re.search(r"[\u4e00-\u9fff]$", prev) and re.match(r"^[A-Za-zΑ-Ωα-ω0-9(（]", curr):
            return False
        if re.search(r"[A-Za-zΑ-Ωα-ω0-9)）]$", prev) and re.match(r"^[\u4e00-\u9fff]", curr):
            return False
        return True

    # 行内文本拼接器
    def _render_visual_line(self, line_tokens: list[dict[str, Any]]) -> str:
        if not line_tokens:
            return ""

        tokens = sorted(line_tokens, key=lambda t: (t["x0"], t["cy"]))
        rendered_parts: list[str] = []
        i = 0

        while i < len(tokens):
            token = tokens[i]
            if not token["text"]:
                i += 1
                continue

            if self._is_simple_math_anchor(token["text"]):
                attached: list[dict[str, Any]] = []
                j = i + 1
                while j < len(tokens):
                    candidate = tokens[j]
                    if candidate["x0"] > token["x1"] + 14:
                        break
                    if self._should_attach_vertically(token, candidate):
                        attached.append(candidate)
                    j += 1

                if attached:
                    rendered_parts.append(self._render_vertical_stack(token, attached))
                    i += 1 + len(attached)
                    continue

            rendered_parts.append(token["text"])
            i += 1

        if not rendered_parts:
            return ""

        line = rendered_parts[0]
        prev_part = rendered_parts[0]
        for part in rendered_parts[1:]:
            if self._should_insert_space(prev_part, part):
                line += " "
            line += part
            prev_part = part

        line = re.sub(r"\(\s+", "(", line)
        line = re.sub(r"\s+\)", ")", line)
        line = re.sub(r"([A-D])\.\s*", r"\1. ", line)
        line = re.sub(r"(?<=\d)\s+(?=[A-Za-zΑ-Ωα-ω])", "", line)
        line = re.sub(r"(?<=[A-Za-zΑ-Ωα-ω0-9_])\s+(?=\()", "", line)
        line = re.sub(r"([+\-])=\s*([0-9]+)\s*0\b", r"\1\2=0", line)
        line = re.sub(r"\s{2,}", " ", line).strip()
        return self._postprocess_rendered_line(line)

    # 对渲染后的单行文本进行精细化清洗和格式化
    def _postprocess_rendered_line(self, line: str) -> str:
        if not line:
            return ""

        line = line.strip()
        line = re.sub(r"\(([A-Z])\s*([+\-]?\d+(?:\s*,\s*[+\-]?\d+)+)\)", r"\1(\2)", line)
        line = re.sub(r"(?<![A-Za-z0-9_])([12])F(?![A-Za-z0-9_])", r"F_\1", line)
        line = re.sub(r"([A-Z]{2,})\s+([12])(?=(?://|→|⊥|=|[，,;；:.。？！!? ]|$))", r"\1_\2", line)
        line = re.sub(r"→\s*([A-Z]{1,3}(?:_[0-9]+)?)", r"\1→", line)
        line = re.sub(r"\(\s*([+\-∞]?)\^?(\d+)\s+(\d+)\s*,", r"(\1\2/\3,", line)
        line = re.sub(r",\s*([+\-∞]?)\^?(\d+)\s+(\d+)\s*\)", r",\1\2/\3)", line)
        line = re.sub(r"\({2,}", "(", line)
        line = re.sub(r"\){2,}", ")", line)
        line = re.sub(r"-∞-\s*,", "-∞,-", line)
        line = re.sub(r"(?<=\))\s*2\b", "^2", line)
        line = re.sub(r"直线:\s*l\s+", "直线l: ", line)
        line = re.sub(r"([A-Za-zΑ-Ωα-ω0-9_]+)\^→", r"\1→", line)
        line = re.sub(r"/\s*([A-Za-z])\s*/(?=平面)", r"\1∥", line)
        line = re.sub(r"\s*//\s*", "∥", line)
        line = re.sub(r"\+\+(\d+)\s+([0-9]*[A-Za-z])", r"+\1+\2", line)
        line = re.sub(r"\(--", "(-", line)
        line = re.sub(r"([A-Z]{2}|OP|PA|PB|PC|PD|AB|AC|AD|BC|BD|CD)\s*=\s*(\d+)\^2\b", r"\1=\2√2", line)
        line = re.sub(r"(离心率为)2\^2\b", r"\1√2/2", line)
        line = re.sub(r"π\s+(\d+)(?=(?:[，,。；;：:）)]|[\u4e00-\u9fff]|$))", r"π/\1", line)
        line = re.sub(
            r"(离心率为)(\d+)\s+(\d+)(?=(?:[，,。；;：:）)]|[\u4e00-\u9fff]|$))",
            r"\1\2/\3",
            line,
        )
        line = re.sub(
            r"((?:cos|sin|tan)\s*[^=]{0,20}=)\s*(\d+)\s+(\d+)(?=(?:[，,。；;：:）)]|[\u4e00-\u9fff]|$))",
            r"\1\2/\3",
            line,
        )
        line = re.sub(
            r"(\d+)\s*/\s*(\d+)\s+(\d+)(?=[，,。；;：:）)]|$)",
            lambda m: f"{m.group(1)}√{m.group(3)}/{m.group(2)}" if m.group(2) == m.group(3) else m.group(0),
            line,
        )
        line = re.sub(
            r"(斜率为)(\d+)\s+(\d+)(?=[，,。；;：:）)]|$)",
            lambda m: f"{m.group(1)}{m.group(2)}/{m.group(3)}",
            line,
        )
        line = re.sub(
            r"((?:距离|弦长|周长|最小值|最大值|长度|半径|边长|面积)为)(\d+)\s+(\d+)(?=[，,。；;：:）)]|$)",
            lambda m: f"{m.group(1)}{m.group(2)}√{m.group(3)}",
            line,
        )
        line = re.sub(
            r"((?:[+\-])\d+)\s+(\d+)(?=[，,。；;：:）)]|$)",
            lambda m: f"{m.group(1)}√{m.group(2)}",
            line,
        )
        line = re.sub(
            r"^([A-D][\.\、\)]\s*)(\d+)([+\-])(\d+)$",
            lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}√{m.group(4)}",
            line,
        )

        # 典型圆锥曲线标准式：x^2/a^2 + y^2/b^2 = 1
        line = re.sub(
            r"(?<![A-Za-z0-9_])(\d+)\^x2\s*\+\s*y\^2/(\d+)\s*=\s*1",
            r"x^2/\1+y^2/\2=1",
            line,
        )
        line = re.sub(
            r"(?<![A-Za-z0-9_])(\d+)\^x2\s*\+\s*(\d+)\^y2\s*=\s*1",
            r"x^2/\1+y^2/\2=1",
            line,
        )
        line = re.sub(
            r"(?<![A-Za-z0-9_])([A-Za-z])\^x22\s*\+\s*([A-Za-z])\^y22\s*=\s*1",
            r"x^2/\1^2+y^2/\2^2=1",
            line,
        )

        # 同行选项、子问强制拆行，方便后续 question_extractor 结构化。
        if re.match(r"^\s*[A-D][\.\、\)]", line):
            line = re.sub(r"(?<!^)\s*(?=[B-D][\.\、\)])", "\n", line)
        line = re.sub(r"(?<!^)\s*(?=[（(]\d+[）)])", "\n", line)
        line = re.sub(r"\n{2,}", "\n", line)

        if "\n" in line:
            parts = [part.strip() for part in line.split("\n") if part.strip()]
            if len(parts) > 1:
                return "\n".join(self._postprocess_rendered_line(part) for part in parts)

        line = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", line)
        line = re.sub(r"[ \t]{2,}", " ", line).strip()
        return line

    def _reconstruct_page_text_from_words(self, words: list[tuple]) -> str:
        # 把散落的单词 → 按视觉位置拼成一行
        lines = self._group_words_into_visual_lines(words)
        split_lines: list[list[dict[str, Any]]] = []
        for line in lines:
            # 把挤在同一行的多个题目 / 选项 → 拆开
            split_lines.extend(self._split_structural_visual_line(line))
        # 把一行单词 → 拼成通顺文本（核心！处理数学公式）
        rendered_lines = [self._render_visual_line(line) for line in split_lines]
        expanded_lines: list[str] = []
        for line in rendered_lines:
            if not line:
                continue
            expanded_lines.extend(part for part in str(line).split("\n") if part.strip())
        return "\n".join(expanded_lines)
    
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

        try:
            dpi = int(os.getenv("OCR_DPI", "200"))
        except Exception:
            dpi = 200

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

        temp_dir, image_paths = self._render_pdf_page_paths(pdf_path, dpi=dpi, max_pages=max_pages)
        try:
            print(f"视觉OCR: 已将 PDF 转换为 {len(image_paths)} 张图片")
            all_text = []
            non_empty_pages = 0

            for i, image_path in enumerate(image_paths, start=1):
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")

                try:
                    text = self.vision_llm_client.generate_with_image(
                        prompt=vision_prompt,
                        image_base64=img_b64,
                        image_media_type="image/png",
                    )
                    # 过滤掉视觉LLM调用失败的错误信息
                    if text.startswith("[视觉LLM调用失败"):
                        print(f"  第 {i} 页视觉识别失败: {text}")
                        text = ""
                except Exception as e:
                    print(f"  第 {i} 页视觉识别异常: {e}")
                    text = ""

                if text.strip():
                    non_empty_pages += 1

                all_text.append(f"--- 第 {i} 页 ---\n{text}\n")
                print(f"  第 {i}/{len(image_paths)} 页识别完成，字符数: {len(text)}")

            if image_paths and non_empty_pages == 0:
                raise RuntimeError("视觉大模型未提取到任何有效文本")

            return "\n".join(all_text)
        finally:
            temp_dir.cleanup()

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

        try:
            dpi = int(os.getenv("OCR_DPI", "300"))
        except Exception:
            dpi = 300
        # oem 1代表使用LSTM 深度学习引擎
        tesseract_config = os.getenv("TESSERACT_CONFIG", "--oem 1 --psm 4")
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("未安装 pytesseract，无法进行 OCR")

        temp_dir, image_paths = self._render_pdf_page_paths(pdf_path, dpi=dpi, max_pages=max_pages)
        try:
            print(f"OCR: 已将 PDF 转换为 {len(image_paths)} 张图片")
            all_text = []
            for i, image_path in enumerate(image_paths, start=1):
                with Image.open(image_path) as image:
                    processed = _preprocess_for_ocr(image)
                try:
                    text = pytesseract.image_to_string(processed, lang='chi_sim+eng', config=tesseract_config)
                finally:
                    processed.close()
                text = _postprocess_ocr_text(text)
                all_text.append(f"--- 第 {i} 页 ---\n{text}\n")
            return "\n".join(all_text)
        finally:
            temp_dir.cleanup()
    
    def extract_text_native(self, pdf_path: str, max_pages: Optional[int] = None) -> str:
        if not PYMUPDF_AVAILABLE:
            raise RuntimeError("PyMuPDF(fitz)不可用，无法提取PDF文本层")
        
        with fitz.open(pdf_path) as doc:
            all_text = []
            page_count = len(doc) if max_pages is None else min(max_pages, len(doc))
            for page_num in range(page_count):
                page = doc[page_num]
                words = page.get_text("words", sort=True) or []
                text = self._reconstruct_page_text_from_words(words)
                if not text.strip():
                    text = self._normalize_pdf_glyphs(page.get_text("text", sort=True))
                all_text.append(f"--- 第 {page_num + 1} 页 ---\n{text}\n")
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
            vision_llm_available = (
                self.vision_llm_client is not None
                and hasattr(self.vision_llm_client, "supports_vision")
                and self.vision_llm_client.supports_vision()
            )
            if vision_llm_available:
                try:
                    print("使用视觉大模型进行OCR识别（支持数学公式/向量符号）...")
                    self.last_extraction_mode = "vision_llm"
                    llm_text = self.extract_text_with_vision_llm(str(pdf_path), max_pages=max_pages)
                    return llm_text
                except Exception as e:
                    print(f"视觉大模型OCR失败，降级回Tesseract OCR：{e}")
                    self.last_extraction_mode = "ocr"
            # 降级到 Tesseract OCR
            if not self.ocr_enabled:
                raise RuntimeError("PDF需要OCR（扫描版/无有效文本层），但当前已禁用OCR")
            return self.extract_text_with_ocr(str(pdf_path), max_pages=max_pages)
        else:
            print("检测到是原生版pdf,将跳过ocr和vison_llm直接提取")
            return self.extract_text_native(str(pdf_path), max_pages=max_pages)
    
    def clean_text(self, text: str) -> str:
        """清洗提取文本：先做常见 PDF glyph 映射，再压缩空白。"""
        text = self._normalize_pdf_glyphs(text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
