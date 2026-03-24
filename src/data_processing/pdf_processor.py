"""PDF处理入口：process_pdf_to_questions 主函数"""
import re
from typing import List, Dict, Any, Optional

from .pdf_extractor import PDFProcessor
from .question_extractor import QuestionExtractor
from .question_enricher import enrich_question_with_representations, _split_stem_and_options
from .meta_extractor import ExamMetaExtractor

# 原生和扫描版都走的函数
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

    # 1) 全文提取与清洗（只做一次 OCR，避免重复调用视觉模型）
    text = processor.extract_text(pdf_path)
    print("process_pdf_to_questions函数中新鲜提取出来的text:\n",text)
    text = processor.clean_text(text)

    meta_report = {"meta": {}, "confidence": {}, "evidence": {}}
    meta_merged = dict(meta or {})

    # 2) 自动识别元数据（从已提取的全文中截取前几页内容，避免二次 OCR）
    if auto_meta:
        # 按页分隔符截取前 meta_pages 页；若无分隔符则取前 2000 字符
        import re as _re
        page_sep = _re.compile(r'--- 第 \d+ 页 ---')
        page_splits = list(page_sep.finditer(text))
        if len(page_splits) > meta_pages:
            preview_text = text[:page_splits[meta_pages].start()]
        else:
            preview_text = text[:2000]
        meta_extractor = ExamMetaExtractor(llm_client=llm_client)
        meta_report = meta_extractor.extract(preview_text)

        # 合并策略：用户显式传入优先；否则用自动识别补齐
        inferred = meta_report.get("meta", {})
        for k, v in inferred.items():
            if not meta_merged.get(k):
                meta_merged[k] = v

    extractor = QuestionExtractor(llm_client=llm_client)
    questions = extractor.extract_questions_from_text(text, meta_merged)
    
    # 调试代码
    print("pdf提取出的原问题文本如下：\n")
    for q in questions:
        print(q,'\n')

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

