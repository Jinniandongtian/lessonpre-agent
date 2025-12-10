"""PDF处理模块：OCR、文本提取、题目提取"""
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("警告：OCR功能不可用，请安装 pytesseract 和 pdf2image")


class PDFProcessor:
    """PDF处理器：支持OCR和文本提取"""
    
    def __init__(self, ocr_enabled: bool = True):
        self.ocr_enabled = ocr_enabled and OCR_AVAILABLE
    
    def is_scanned_pdf(self, pdf_path: str) -> bool:
        """
        判断PDF是否为扫描版（图片格式）
        
        简单判断：如果PDF中文本层很少或为空，可能是扫描版
        """
        try:
            doc = fitz.open(pdf_path)
            text_count = 0
            for page_num in range(min(3, len(doc))):  # 检查前3页
                page = doc[page_num]
                text = page.get_text()
                text_count += len(text.strip())
            doc.close()
            
            # 如果前3页文本很少，可能是扫描版
            return text_count < 100
        except Exception as e:
            print(f"判断PDF类型失败: {e}")
            return True  # 默认按扫描版处理
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """使用OCR提取扫描版PDF的文本"""
        if not self.ocr_enabled:
            raise RuntimeError("OCR功能未启用或未安装相关依赖")
        
        try:
            # 将PDF转换为图片
            images = convert_from_path(pdf_path, dpi=300)
            all_text = []
            
            for i, image in enumerate(images):
                # OCR识别
                text = pytesseract.image_to_string(image, lang='chi_sim+eng')
                all_text.append(f"--- 第 {i+1} 页 ---\n{text}\n")
            
            return "\n".join(all_text)
        except Exception as e:
            raise RuntimeError(f"OCR处理失败: {e}")
    
    def extract_text_native(self, pdf_path: str) -> str:
        """提取原生PDF的文本"""
        try:
            doc = fitz.open(pdf_path)
            all_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                all_text.append(f"--- 第 {page_num + 1} 页 ---\n{text}\n")
            
            doc.close()
            return "\n".join(all_text)
        except Exception as e:
            raise RuntimeError(f"PDF文本提取失败: {e}")
    
    def extract_text(self, pdf_path: str) -> str:
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
            print(f"检测到扫描版PDF，使用OCR提取文本...")
            return self.extract_text_with_ocr(str(pdf_path))
        else:
            print(f"检测到原生PDF，直接提取文本...")
            return self.extract_text_native(str(pdf_path))
    
    def clean_text(self, text: str) -> str:
        """清洗提取的文本"""
        # 移除多余的空白字符
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除特殊字符（保留中文、英文、数字、基本标点）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.\,\;\:\!\?\(\)\[\]\{\}\+\-\*\/\=\>\<]', '', text)
        return text.strip()


class QuestionExtractor:
    """题目提取器：从文本中提取题目"""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def identify_question_type(self, text: str) -> str:
        """识别题型"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['选择', '以下哪个', '正确的是', '错误的是']):
            return "选择题"
        elif any(keyword in text_lower for keyword in ['填空', '填写', '填入']):
            return "填空题"
        elif any(keyword in text_lower for keyword in ['解答', '证明', '计算', '求解']):
            return "解答题"
        else:
            return "未知题型"
    
    def extract_questions_from_text(
        self,
        text: str,
        meta: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        从文本中提取题目
        
        Args:
            text: PDF提取的文本
            meta: 元数据（地区、年份、考试类型等）
        
        Returns:
            题目列表，每个题目包含：content, question_type, index等
        """
        questions = []
        
        # 使用正则表达式提取题目编号
        # 匹配：1. 或 (1) 或 一、 等格式
        patterns = [
            r'(\d+)[\.、]\s*([^\d]+?)(?=\d+[\.、]|$)',
            r'\((\d+)\)\s*([^\(]+?)(?=\(\d+\)|$)',
            r'[一二三四五六七八九十]+[、．]\s*([^一二三四五六七八九十]+?)(?=[一二三四五六七八九十]+[、．]|$)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                question_text = match.group(0).strip()
                if len(question_text) > 20:  # 过滤太短的文本
                    question_type = self.identify_question_type(question_text)
                    
                    questions.append({
                        "content": question_text,
                        "question_type": question_type,
                        "index": len(questions) + 1,
                        "source_meta": meta,
                    })
        
        # 如果没有匹配到，尝试使用LLM提取
        if not questions and self.llm_client:
            questions = self._extract_with_llm(text, meta)
        
        return questions
    
    def _extract_with_llm(self, text: str, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        """使用LLM提取题目"""
        if not self.llm_client:
            return []
        
        prompt = f"""
请从以下文本中提取所有数学题目，并以JSON格式返回。

文本内容：
{text[:2000]}  # 限制长度

请返回格式：
[
  {{
    "content": "题目内容",
    "question_type": "选择题/填空题/解答题",
    "index": 1
  }}
]
"""
        
        try:
            response = self.llm_client.generate(prompt)
            import json
            # 尝试解析JSON
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                questions = json.loads(json_match.group())
                for q in questions:
                    q["source_meta"] = meta
                return questions
        except Exception as e:
            print(f"LLM提取题目失败: {e}")
        
        return []
    
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
    llm_client=None
) -> List[Dict[str, Any]]:
    """
    处理PDF文件，提取题目
    
    Args:
        pdf_path: PDF文件路径
        meta: 元数据（region, year, grade, exam_name等）
        ocr_enabled: 是否启用OCR
        llm_client: LLM客户端（用于题目提取和标注）
    
    Returns:
        题目列表
    """
    # 1. 提取文本
    processor = PDFProcessor(ocr_enabled=ocr_enabled)
    text = processor.extract_text(pdf_path)
    text = processor.clean_text(text)
    
    # 2. 提取题目
    extractor = QuestionExtractor(llm_client=llm_client)
    questions = extractor.extract_questions_from_text(text, meta)
    
    # 3. 丰富题目信息（知识点、难度）
    enriched_questions = []
    for q in questions:
        enriched = extractor.enrich_question_with_llm(q)
        enriched_questions.append(enriched)
    
    return enriched_questions

