"""教师API接口 - PDF处理与RAG讲义生成"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import Optional, Dict, Any, List
import uuid
import re
import hashlib
import difflib # 比较库
import traceback
from pathlib import Path
import os
import json
from datetime import datetime

from ..utils.config import Config
from ..utils.llm_client import get_default_llm_client, get_vision_llm_client
from ..utils.latex_normalizer import LaTeXNormalizer
from ..utils.description_generator import DescriptionGenerator
from ..data_processing.pdf_processor import process_pdf_to_questions, enrich_question_with_representations
from ..data_processing.question_extractor import QuestionExtractor
from ..vector_store.embedding import EmbeddingModel
from ..vector_store.vector_db import VectorDatabase
from ..agents.rag_handout_agent import RAGHandoutAgent
from ..export.pdf_exporter import handout_markdown_to_pdf

# 初始化
Config.init_directories()
Config.check_system_dependencies()
app = FastAPI(title="独立老师备课助手API - PDF处理与RAG讲义生成", version="2.0.0")

# 全局对象（延迟初始化）
llm_client = None
vision_llm_client = None
vector_db = None
embedding_model = None
rag_agent = None
normalizer: Optional[LaTeXNormalizer] = None
desc_generator: Optional[DescriptionGenerator] = None


def _append_import_history(record: Dict[str, Any]) -> None:
    try:
        p = Config.DATA_DIR / "import_history.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠ 写入导入历史失败: {e}")


def init_agents():
    """初始化所有Agent（延迟初始化）"""
    global llm_client, vision_llm_client, vector_db, embedding_model, rag_agent
    global normalizer, desc_generator
    try:
        llm_client = get_default_llm_client()
        vision_llm_client = get_vision_llm_client()
        if vision_llm_client:
            print(f"✓ 视觉模型已配置: {getattr(vision_llm_client, 'model', vision_llm_client.__class__.__name__)}")
        vector_db = VectorDatabase(str(Config.VECTOR_DB_PATH))
        embedding_model = EmbeddingModel()
        rag_agent = RAGHandoutAgent(
            llm_client=llm_client,
            vector_db=vector_db,
            embedding_model=embedding_model
        )
        normalizer = LaTeXNormalizer(llm_client)
        desc_generator = DescriptionGenerator(llm_client)
        print("✓ 所有Agent初始化成功")
        print(f"✓ 向量库中有 {vector_db.count()} 道题目")
    except Exception as e:
        print(f"⚠ 警告：Agent初始化失败: {e}")
        print("服务将继续运行，但部分功能可能不可用")


# 启动时初始化
init_agents()


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "独立老师备课助手API - PDF处理与RAG讲义生成",
        "version": "2.0.0",
        "endpoints": {
            "上传PDF试卷": "/pdf/upload",
            "生成讲义": "/lesson/handout",
            "向量库统计": "/vector_db/stats",
        }
    }


@app.post("/pdf/extract_text")
async def extract_pdf_text(
    pdf_file: UploadFile = File(...),
    ocr_enabled: bool = Form(True),
    max_pages: Optional[int] = Form(None),
):
    """调试接口：返回PDF原始提取文本（不入库、不提题）"""
    from ..data_processing.pdf_processor import PDFProcessor
    pdf_filename = f"dbg_{uuid.uuid4().hex[:8]}_{pdf_file.filename}"
    pdf_path = Config.PDF_STORAGE_PATH / pdf_filename
    try:
        with open(pdf_path, "wb") as f:
            f.write(await pdf_file.read())
        processor = PDFProcessor(ocr_enabled=ocr_enabled)
        text = processor.extract_text(str(pdf_path), max_pages=max_pages)
        return {"length": len(text), "text": text}
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass


@app.post("/pdf/upload")
async def upload_pdf_exam(
    pdf_file: UploadFile = File(...),

    # ✅ 改为可选：用户不传也能跑 auto meta
    region: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    grade: Optional[str] = Form(None),
    exam_name: Optional[str] = Form(None),
    source_type: Optional[str] = Form(None),  # 期中/期末/高考/模拟/一模/二模...

    # ✅ 新增：自动识别元数据
    auto_meta: bool = Form(True),
    meta_pages: int = Form(2),  # 只取前N页做元数据识别（建议 1~2）
    ocr_enabled: bool = Form(True),
):
    """
    上传PDF试卷，自动提取题目并存储到向量库

    支持：
    - 扫描版PDF（自动OCR）
    - 原生PDF（直接提取文本）
    - 自动识别题型和知识点
    - 自动向量化存储
    - ✅ 自动识别元数据（region/year/grade/exam_name/source_type），并给出置信度/需确认字段
    """
    if vector_db is None or embedding_model is None:
        raise HTTPException(status_code=500, detail="服务未正确初始化，请检查日志")

    pdf_path: Optional[Path] = None
    question_helper = QuestionExtractor()

    try:
        # 1. 保存上传的PDF
        file_ext = Path(pdf_file.filename).suffix.lower()
        if file_ext != ".pdf":
            raise HTTPException(status_code=400, detail="只支持PDF文件")

        pdf_filename = f"{uuid.uuid4().hex[:16]}.pdf"
        pdf_path = Config.PDF_STORAGE_PATH / pdf_filename

        with open(pdf_path, "wb") as f:
            content = await pdf_file.read()
            f.write(content)

        # 稳定标识：同一份 PDF（字节级相同）重复上传，pdf_hash 恒定
        pdf_hash = hashlib.md5(content).hexdigest()

        # 2. 组装 meta（用户提供的字段优先；其余由 auto_meta 补全）
        user_meta: Dict[str, Any] = {
            "region": region,
            "year": year,
            "grade": grade,
            "exam_name": exam_name,
            "source_type": source_type,
        }
        # 去掉 None/空串，避免覆盖自动识别结果
        user_meta = {k: v for k, v in user_meta.items() if v is not None and str(v).strip() != ""}

        print(f"开始处理PDF: {pdf_filename}")
        print(f"PDF文件大小: {pdf_path.stat().st_size / 1024:.2f} KB")

        result = process_pdf_to_questions(
            pdf_path=str(pdf_path),
            meta=user_meta,
            ocr_enabled=ocr_enabled,
            llm_client=llm_client,
            vision_llm_client=vision_llm_client,
            auto_meta=auto_meta,
            meta_pages=meta_pages,
        )
        if not isinstance(result, dict):
            raise RuntimeError(
                f"process_pdf_to_questions 返回格式错误: {type(result).__name__}"
            )

        questions = result.get("questions", []) or []
        meta_used = result.get("meta_used", user_meta) or user_meta
        meta_inferred = result.get("meta_inferred", {}) or {}
        meta_confidence = result.get("meta_confidence", {}) or {}
        meta_evidence = result.get("meta_evidence", {}) or {}
        text_extraction = result.get("text_extraction", None)

        # 5. 如果 meta 仍不完整，给一个兜底 exam_name（避免下游存储缺字段）
        #    注意：这个兜底不会覆盖用户/自动识别的 exam_name
        if not meta_used.get("exam_name"):
            y = meta_used.get("year", "")
            r = meta_used.get("region", "")
            st = meta_used.get("source_type", "")
            parts = []
            if str(y).strip() != "":
                parts.append(f"{y}年")
            if str(r).strip() != "":
                parts.append(str(r).strip())
            if str(st).strip() != "":
                parts.append(str(st).strip())
            fallback = "".join(parts).strip()
            meta_used["exam_name"] = fallback if fallback else "未命名试卷"

        print(f"题目提取完成，共提取 {len(questions)} 道题目")

        extracted_total = len(questions)

        # (1) 第一次去重：先用pdfhash+题号在同pdf内部去重
        deduped_questions = []
        seen_keys = set()
        duplicates_removed = 0
        internal_duplicates = []
        for q in questions:
            content = q.get("content", "")

            # 优先使用“pdf_hash + 题号”作为稳定 id（OCR 噪声/空格变化不会影响）
            qnum = question_helper._extract_question_number(content)
            stable_id = f"{pdf_hash}:{qnum}" if qnum else None

            # fallback：无题号才退回到内容归一化 hash
            if stable_id:
                dedupe_key = stable_id
            else:
                key_src = question_helper._normalize_for_dedupe(content)
                dedupe_key = f"{pdf_hash}:h:{hashlib.md5(key_src.encode('utf-8')).hexdigest()}"

            if dedupe_key in seen_keys:
                duplicates_removed += 1
                internal_duplicates.append({
                    "reason": "internal_exact",
                    "dedupe_key": dedupe_key,
                    "content_preview": question_helper.get_content_text(content)[:160],
                })
                continue
            seen_keys.add(dedupe_key)
            # 始终使用稳定 id，避免 extractor/LLM 生成的 id 因噪声波动
            q["id"] = stable_id or dedupe_key
            deduped_questions.append(q)
        questions = deduped_questions
        if duplicates_removed:
            print(f"去重完成：本次提取内部移除重复题目 {duplicates_removed} 道")
        else:
            print("去重完成：本次提取内部未发现重复题目")

        after_internal_exact = len(questions)

        if not questions:
            return {
                "success": False,
                # 应该是pdfhash去重后没有了有效题目？
                "message": "未能从PDF中提取到题目，请检查PDF格式",
                "questions_extracted": extracted_total,
                "duplicates_removed": duplicates_removed,
                "text_extraction": text_extraction,
                "dedupe_report": {
                    "extracted_total": extracted_total,
                    "after_internal_exact": after_internal_exact,
                    "after_db_existing_id": 0,
                    "after_similarity": 0,
                    "internal_exact": duplicates_removed,
                    "db_existing_id": 0,
                    "similarity": 0,
                    "items": internal_duplicates,
                },
                "meta_used": meta_used,
                "meta_inferred": meta_inferred,
                "meta_confidence": meta_confidence,
                "meta_evidence": meta_evidence,
            }

        # 6. 入库前去重：过滤掉向量库中已存在的题目（避免重复生成向量）
        # (2) 第二次去重：查询向量库中已存在的题目 ID，过滤掉重复 ID 的题目
        existing_ids = vector_db.get_existing_ids([q.get("id", "") for q in questions if q.get("id")])
        skipped_existing = 0
        db_existing_duplicates = []
        if existing_ids:
            filtered = []
            for q in questions:
                qid = q.get("id")
                if qid and qid in existing_ids:
                    skipped_existing += 1
                    db_existing_duplicates.append({
                        "reason": "db_existing_id",
                        "id": qid,
                        "content_preview": question_helper.get_content_text(q.get("content") or "")[:160],
                    })
                    continue
                filtered.append(q)
            questions = filtered

        if skipped_existing:
            print(f"入库前去重：向量库已存在，跳过 {skipped_existing} 道题")
        else:
            print("入库前去重：向量库未发现重复题")

        after_db_existing_id = len(questions)

        # 7. needs_confirmation（仅当你实现了 meta_confidence 时才有意义）
        needs_confirmation = []
        if meta_confidence:
            # 你可以调整阈值，比如 0.7
            threshold = 0.7
            keys = ["region", "year", "grade", "exam_name", "source_type"]
            needs_confirmation = [k for k in keys if meta_confidence.get(k, 0.0) < threshold]
        # (3) 第三次去重：语义相似度去重
        similarity_duplicates = []
        embeddings: list = []
        if questions:
            print(f"为 {len(questions)} 道题目生成向量...")
            if normalizer is None or desc_generator is None:
                raise RuntimeError("题目表示增强器未初始化，无法按新格式入库")

            enriched = []
            for q in questions:
                enriched_q = enrich_question_with_representations(
                    q, llm_client, normalizer, desc_generator
                )
                # 保留原始题目的 id（enrich 函数已写入，但以防万一）
                enriched_q["id"] = q.get("id", enriched_q.get("id", ""))
                enriched.append(enriched_q)
            questions = enriched
            embeddings = [embedding_model.encode_question(q) for q in questions]

            enable_similarity = str(os.getenv("DEDUPE_SIMILARITY", "1")).strip().lower() not in {"0", "false", "no"}
            try:
                text_thr = float(os.getenv("DEDUPE_TEXT_SIM_THRESHOLD", "0.92"))
            except Exception:
                text_thr = 0.92
            try:
                emb_thr = float(os.getenv("DEDUPE_EMB_SIM_THRESHOLD", "0.80"))
            except Exception:
                emb_thr = 0.80

            if enable_similarity and vector_db.count() > 0:
                filtered_q = []
                filtered_e = []
                # 遍历题目和向量
                for q, emb in zip(questions, embeddings):
                    new_norm = question_helper._normalize_for_dedupe(q.get("content", ""))
                    best = None
                    try:
                        # 从向量库搜索Top3相似题目（平衡精度和性能）
                        candidates = vector_db.search(query_embedding=emb, top_k=3)
                    except Exception as search_error:
                        raise RuntimeError(
                            f"相似度去重检索失败，已中止本次导入: {search_error!r}"
                        ) from search_error
                    for meta, sim in candidates:
                        old_norm = question_helper._normalize_for_dedupe(meta.get("content", ""))
                        if not new_norm or not old_norm:
                            continue
                        # # 计算文本层面的相似度（difflib）
                        tr = difflib.SequenceMatcher(None, new_norm, old_norm).ratio()
                        item = {
                            "match_id": meta.get("id"),
                            "embedding_similarity": float(sim), # 向量余弦相似度
                            "text_similarity": float(tr), # 文本层面的相似度
                            "match_preview": question_helper.get_content_text(meta.get("content") or "")[:160],
                        }
                        if best is None or (item["text_similarity"], item["embedding_similarity"]) > (best["text_similarity"], best["embedding_similarity"]):
                            best = item
                    # 双阈值判断：文本相似度≥阈值 OR 向量相似度≥阈值 → 判定为重复        
                    if best and (best["text_similarity"] >= text_thr or best["embedding_similarity"] >= emb_thr):
                        similarity_duplicates.append({
                            "reason": "similarity",
                            "id": q.get("id"),
                            "content_preview": question_helper.get_content_text(q.get("content") or "")[:160],
                            **best,
                        })
                        continue
                    filtered_q.append(q)
                    filtered_e.append(emb)

                if similarity_duplicates:
                    print(f"相似度去重：跳过疑似重复题 {len(similarity_duplicates)} 道")
                # 更新题目和向量列表（过滤掉相似重复题）
                questions = filtered_q
                embeddings = filtered_e
        else:
            print("本次无新增题目，无需生成向量")

        after_similarity = len(questions)

        # 9. 存储到向量库（仅新增题目）
        print(f"存储到向量库...")
        add_stats = {"added": 0, "skipped_existing": 0}
        if questions:
            add_stats = vector_db.add_questions(questions, embeddings)
        added_count = int((add_stats or {}).get("added", 0) or 0)
        # 将“本次上传内部去重” + “入库前过滤（库内已存在）”合并
        duplicates_removed_total = duplicates_removed + skipped_existing + len(similarity_duplicates)

        print(
            f"向量库写入统计：本次输入 {len(questions)} 道，新增入库 {added_count} 道，库内已存在跳过 {skipped_existing} 道，"
            f"累计去重 {duplicates_removed_total} 道，当前向量库总量 {vector_db.count()}"
        )

        dedupe_report = {
            "extracted_total": extracted_total,
            "after_internal_exact": after_internal_exact,
            "after_db_existing_id": after_db_existing_id,
            "after_similarity": after_similarity,
            "internal_exact": duplicates_removed,
            "db_existing_id": skipped_existing,
            "similarity": len(similarity_duplicates),
            "similarity_enabled": bool(enable_similarity) if "enable_similarity" in locals() else False,
            "thresholds": {
                "text_similarity": float(text_thr) if "text_thr" in locals() else None,
                "embedding_similarity": float(emb_thr) if "emb_thr" in locals() else None,
            },
            "items": internal_duplicates + db_existing_duplicates + similarity_duplicates,
        }

        _append_import_history({
            "ts": datetime.utcnow().isoformat() + "Z",
            "pdf_filename": pdf_filename,
            "pdf_hash": pdf_hash,
            "meta_used": meta_used,
            "extracted_total": extracted_total,
            "added": added_count,
            "vector_db_total": vector_db.count(),
            "duplicates_removed": duplicates_removed_total,
            "dedupe_report": dedupe_report,
        })

        return {
            "success": True,
            "message": f"成功提取并存储 {added_count} 道题目",
            "questions_extracted": extracted_total,
            "duplicates_removed": duplicates_removed_total,
            "dedupe_report": dedupe_report,
            "vector_db_total": vector_db.count(),
            "text_extraction": text_extraction,

            # ✅ 新增：把元数据结果返回给前端/调用方，方便确认与回流
            # "meta_used": meta_used,
            # "meta_inferred": meta_inferred,
            # "meta_confidence": meta_confidence,
            # "needs_confirmation": needs_confirmation,
            # "meta_evidence": meta_evidence,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理失败: {repr(e)}")
    finally:
        if pdf_path and pdf_path.exists():
            try:
                os.remove(pdf_path)
            except Exception as cleanup_error:
                print(f"⚠ 清理临时PDF失败: {cleanup_error}")


@app.post("/lesson/handout")
async def create_handout(
    topic: str = Form(...),
    region: str = Form(...),
    grade: str = Form(...),
    class_level: Optional[str] = Form("中等"),
    template_style: Optional[str] = Form("A"),
    num_examples: int = Form(3),
    num_practice: int = Form(5),
):
    """
    使用RAG技术生成讲义
    
    流程：
    1. 解析主题，提取知识点
    2. 使用RAG从向量库检索相关题目
    3. 生成讲义内容
    4. 导出PDF
    """
    if rag_agent is None:
        raise HTTPException(status_code=500, detail="服务未正确初始化，请检查日志")
    
    try:
        # 生成讲义
        result = rag_agent.generate_handout(
            topic=topic,
            region=region,
            grade=grade,
            class_level=class_level,
            template_style=template_style,
            num_examples=num_examples,
            num_practice=num_practice,
        )
        
        # 导出PDF
        pdf_filename = f"handout_{uuid.uuid4().hex[:8]}.pdf"
        pdf_path = Config.EXPORT_DIR / pdf_filename
        handout_markdown_to_pdf(result["handout_content"], str(pdf_path))
        
        return {
            "success": True,
            "lesson_topic": result["lesson_topic"].to_dict(),
            "handout_content": result["handout_content"],
            "pdf_url": f"/exports/{pdf_filename}",
            "examples_count": len(result["examples"]),
            "practice_count": len(result["practice"]),
            "examples": result["examples"][:3],  # 返回前3个例题预览
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/exports/{filename}")
async def download_export(filename: str):
    """下载导出的文件"""
    file_path = Config.EXPORT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(file_path), filename=filename)


@app.get("/vector_db/stats")
async def get_vector_db_stats():
    """获取向量库统计信息"""
    if vector_db is None:
        raise HTTPException(status_code=500, detail="向量库未初始化")
    
    total = vector_db.count()
    
    return {
        "total_questions": total,
        "backend": vector_db.backend,
        "status": "ready" if total > 0 else "empty",
    }


@app.get("/vector_db/query")
async def vector_db_query(
    exam_name: Optional[str] = None,
    region: Optional[str] = None,
    year: Optional[int] = None,
    grade: Optional[str] = None,
    source_type: Optional[str] = None,
    offset: int = 0,
    limit: int = 20,
    include_content: bool = False,
):
    if vector_db is None:
        raise HTTPException(status_code=500, detail="向量库未初始化")
    filters: Dict[str, Any] = {
        "exam_name": exam_name,
        "region": region,
        "year": year,
        "grade": grade,
        "source_type": source_type,
    }
    filters = {k: v for k, v in filters.items() if v is not None and str(v).strip() != ""}
    return {
        "filters": filters,
        **vector_db.query_by_source_meta(filters=filters, offset=offset, limit=limit, include_content=include_content),
    }


@app.get("/vector_db/groups")
async def vector_db_groups(
    group_by: str,
    exam_name: Optional[str] = None,
    region: Optional[str] = None,
    year: Optional[int] = None,
    grade: Optional[str] = None,
    source_type: Optional[str] = None,
    top_k: int = 50,
):
    if vector_db is None:
        raise HTTPException(status_code=500, detail="向量库未初始化")
    filters: Dict[str, Any] = {
        "exam_name": exam_name,
        "region": region,
        "year": year,
        "grade": grade,
        "source_type": source_type,
    }
    filters = {k: v for k, v in filters.items() if v is not None and str(v).strip() != ""}
    return {
        "group_by": group_by,
        "filters": filters,
        "groups": vector_db.group_counts_by_source_meta(group_by=group_by, filters=filters, top_k=top_k),
    }


@app.post("/vector_db/delete")
async def vector_db_delete(payload: Dict[str, Any]):
    if vector_db is None:
        raise HTTPException(status_code=500, detail="向量库未初始化")
    filters = payload.get("filters") or {}
    dry_run = bool(payload.get("dry_run", True))
    sample = int(payload.get("sample", 20))
    try:
        result = vector_db.delete_by_source_meta(filters=filters, dry_run=dry_run, sample=sample)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"dry_run": dry_run, "filters": filters, **result}


@app.post("/vector_db/reset")
async def vector_db_reset(payload: Dict[str, Any] = None):
    if vector_db is None:
        raise HTTPException(status_code=500, detail="向量库未初始化")
    payload = payload or {}
    dry_run = bool(payload.get("dry_run", False))
    # 试运行模式
    if dry_run:
        return {"dry_run": True, "would_delete": vector_db.count(), "backend": vector_db.backend}
    vector_db.reset()
    return {"success": True, "backend": vector_db.backend, "total_questions": vector_db.count()}

# 
@app.get("/vector_db/import_history")
async def vector_db_import_history(limit: int = 50, offset: int = 0):
    # 定义导入历史文件路径（JSONL格式：每行是一个独立的JSON导入记录）
    p = Config.DATA_DIR / "import_history.jsonl"
    if not p.exists():
        return {"total": 0, "items": []}
    lines: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                continue
    total = len(lines)
    sliced = lines[max(0, total - offset - limit): max(0, total - offset)]
    sliced.reverse()
    return {"total": total, "items": sliced}


# ============ 质量评测 API ============
@app.get("/evaluation/vector_db")
async def evaluate_vector_db(limit: int = 0, use_llm: bool = False):
    """
    评测向量库中题目的提取质量
    
    Args:
        limit: 限制评测题目数量 (0=不限制)
        use_llm: 是否使用 LLM 判断填空题完整性
    """
    from ..evaluation.quality_metrics import evaluate_questions
    
    init_agents()
    
    if vector_db is None:
        raise HTTPException(status_code=500, detail="向量库未初始化")
    
    questions = vector_db.metadata if hasattr(vector_db, 'metadata') else []
    if limit > 0 and len(questions) > limit:
        questions = questions[:limit]
    
    if not questions:
        return {"error": "向量库为空，无题目可评测"}
    
    report = evaluate_questions(questions, llm_client=llm_client if use_llm else None)
    return report
    

@app.post("/evaluation/questions")
async def evaluate_questions_api(payload: Dict[str, Any]):
    """
    评测传入的题目列表
    
    Body:
        questions: List[Dict] - 题目列表
        raw_llm_responses: List[str] (可选) - LLM 原始响应
        use_llm: bool (可选) - 是否使用 LLM 判断填空题完整性
    """
    from ..evaluation.quality_metrics import evaluate_questions
    
    init_agents()
    
    questions = payload.get("questions", [])
    raw_responses = payload.get("raw_llm_responses")
    use_llm = payload.get("use_llm", False)
    
    if not questions:
        raise HTTPException(status_code=400, detail="questions 不能为空")
    
    report = evaluate_questions(questions, raw_responses, llm_client=llm_client if use_llm else None)
    return report


@app.post("/evaluation/pdf")
async def evaluate_pdf(
    pdf_file: UploadFile = File(...),
    ocr_enabled: bool = Form(True),
):
    """
    上传 PDF，提取题目并评测质量（不入库）
    """
    from ..evaluation.quality_metrics import evaluate_questions
    
    init_agents()
    
    # 保存临时文件
    pdf_filename = f"eval_{uuid.uuid4().hex[:8]}_{pdf_file.filename}"
    pdf_path = Config.PDF_STORAGE_PATH / pdf_filename
    
    with open(pdf_path, "wb") as f:
        content = await pdf_file.read()
        f.write(content)
    
    try:
        result = process_pdf_to_questions(
            pdf_path=str(pdf_path),
            meta={},
            ocr_enabled=ocr_enabled,
            llm_client=llm_client,
            vision_llm_client=vision_llm_client,
            auto_meta=True,
        )
        
        questions = result.get("questions", []) if isinstance(result, dict) else result
        
        report = evaluate_questions(questions, llm_client=llm_client)
        report["extraction_info"] = {
            "total_extracted": len(questions),
            "meta_inferred": result.get("meta_inferred", {}) if isinstance(result, dict) else {},
            "text_extraction": result.get("text_extraction", None) if isinstance(result, dict) else None,
        }
        
        return report
    finally:
        # 清理临时文件
        try:
            os.remove(pdf_path)
        except Exception:
            pass


@app.post("/evaluation/gold_standard")
async def evaluate_with_gold_standard(
    pdf_file: UploadFile = File(...),
    gold_file: UploadFile = File(...),
    strategy: str = Form("llm_only"),
    ocr_enabled: bool = Form(True),
):
    """
    使用金标数据评测 PDF 提取质量
    
    Args:
        pdf_file: 待评测的 PDF 文件
        gold_file: 金标数据 JSON 文件
        strategy: 提取策略标识 (llm_only, llm+regex_fallback, rule_based 等)
        ocr_enabled: 是否启用 OCR
    
    金标格式:
        {"paper_id": "...", "questions": [{"num": 1, "type": "选择题"}, ...]}
    """
    from ..evaluation.coverage_metrics import evaluate_with_gold
    
    init_agents()
    
    # 保存临时文件
    pdf_filename = f"gold_eval_{uuid.uuid4().hex[:8]}_{pdf_file.filename}"
    pdf_path = Config.PDF_STORAGE_PATH / pdf_filename
    
    with open(pdf_path, "wb") as f:
        pdf_content = await pdf_file.read()
        f.write(pdf_content)
    
    # 读取金标数据
    gold_content = await gold_file.read()
    try:
        gold_data = json.loads(gold_content.decode('utf-8'))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"金标文件 JSON 解析失败: {e}")
    
    try:
        # 提取题目
        result = process_pdf_to_questions(
            pdf_path=str(pdf_path),
            meta={},
            ocr_enabled=ocr_enabled,
            llm_client=llm_client,
            auto_meta=True,
        )
        
        questions = result.get("questions", []) if isinstance(result, dict) else result
        
        # 金标评测
        report = evaluate_with_gold(gold_data, questions, strategy=strategy)
        
        # 添加提取信息
        report["extraction_info"] = {
            "total_extracted": len(questions),
            "strategy": strategy,
            "ocr_enabled": ocr_enabled,
            "text_extraction": result.get("text_extraction", None) if isinstance(result, dict) else None,
        }
        
        return report
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass


@app.post("/evaluation/gold_standard_json")
async def evaluate_gold_standard_json(payload: Dict[str, Any]):
    """
    使用金标数据评测已提取的题目列表（不需要上传 PDF）
    
    Body:
        gold_data: Dict - 金标数据 {"paper_id": "...", "questions": [...]}
        predictions: List[Dict] - 预测的题目列表
        strategy: str - 提取策略标识
    """
    from ..evaluation.coverage_metrics import evaluate_with_gold
    
    gold_data = payload.get("gold_data")
    predictions = payload.get("predictions", [])
    strategy = payload.get("strategy", "unknown")
    
    if not gold_data:
        raise HTTPException(status_code=400, detail="gold_data 不能为空")
    if not predictions:
        raise HTTPException(status_code=400, detail="predictions 不能为空")
    
    report = evaluate_with_gold(gold_data, predictions, strategy=strategy)
    return report


# ============ PDF预览与质量检查 API ============
@app.post("/pdf/preview")
async def preview_pdf_extraction(
    pdf_file: UploadFile = File(...),
    ocr_enabled: bool = Form(True),
    auto_meta: bool = Form(True),
    meta_pages: int = Form(2),
    ):
    """
    预览PDF提取效果（不入库，仅用于检查质量）
    
    返回：
    - 提取的题目列表（带题号、题型、内容预览）
    - 元数据识别结果
    - 文本提取模式（OCR/原生）
    - 质量评测指标
    - 潜在问题提示
    
    用途：上传前先预览，确认无误后再正式入库
    """
    from ..evaluation.quality_metrics import evaluate_questions
    
    init_agents()
    
    if llm_client is None:
        raise HTTPException(status_code=500, detail="LLM客户端未初始化")
    
    # 保存临时文件
    pdf_filename = f"preview_{uuid.uuid4().hex[:8]}_{pdf_file.filename}"
    pdf_path = Config.PDF_STORAGE_PATH / pdf_filename
    
    try:
        with open(pdf_path, "wb") as f:
            content = await pdf_file.read()
            f.write(content)
        
        # 提取题目
        result = process_pdf_to_questions(
            pdf_path=str(pdf_path),
            meta={},
            ocr_enabled=ocr_enabled,
            llm_client=llm_client,
            auto_meta=auto_meta,
            meta_pages=meta_pages,
        )
        
        questions = result.get("questions", []) if isinstance(result, dict) else result
        
        # 质量评测
        quality_report = evaluate_questions(questions, llm_client=None)  # 不使用LLM判断填空题
        
        # 生成题目摘要（便于快速浏览）
        question_summaries = []
        for i, q in enumerate(questions):
            content = q.get("content", "")
            # 提取题号
            qnum_match = re.match(r'^\s*(\d+)[\.、\)]\s*', content)
            qnum = qnum_match.group(1) if qnum_match else str(i+1)
            
            # 内容预览（前150字符）
            preview = content[:150] + "..." if len(content) > 150 else content
            
            question_summaries.append({
                "index": i + 1,
                "question_number": qnum,
                "question_type": q.get("question_meta").get("question_type",""),
                "content_preview": preview,
                "content_length": len(content),
                "knowledge_points": q.get("knowledge_points", []),
                "difficulty": q.get("difficulty"),
            })
        
        # 检测潜在问题
        issues = []
        
        # 1. 题号不连续
        question_numbers = []
        for q in questions:
            content = q.get("content", "")
            qnum_match = re.match(r'^\s*(\d+)[\.、\)]\s*', content)
            if qnum_match:
                try:
                    question_numbers.append(int(qnum_match.group(1)))
                except:
                    pass
        
        if question_numbers:
            question_numbers.sort()
            expected = list(range(question_numbers[0], question_numbers[-1] + 1))
            missing = [n for n in expected if n not in question_numbers]
            if missing:
                issues.append({
                    "type": "missing_questions",
                    "severity": "warning",
                    "message": f"题号不连续，可能缺失题目：{missing}",
                    "missing_numbers": missing,
                })
        
        # 2. 题目过短
        short_questions = [
            {"index": i+1, "length": len(q.get("content", ""))}
            for i, q in enumerate(questions)
            if len(q.get("content", "")) < 50
        ]
        if short_questions:
            issues.append({
                "type": "short_questions",
                "severity": "warning",
                "message": f"发现 {len(short_questions)} 道题目内容过短，可能提取不完整",
                "questions": short_questions,
            })
        
        # 3. 选择题缺少选项
        incomplete_choices = []
        for i, q in enumerate(questions):
            if q.get("question_meta").get("question_type","") == "选择题":
                content = q.get("content", "")
                # 检查是否包含A、B、C、D选项
                has_options = bool(re.search(r'[A-D][\.、\)]\s*', content))
                if not has_options:
                    incomplete_choices.append({
                        "index": i+1,
                        "content_preview": content[:100],
                    })
        
        if incomplete_choices:
            issues.append({
                "type": "incomplete_choices",
                "severity": "error",
                "message": f"发现 {len(incomplete_choices)} 道选择题缺少选项",
                "questions": incomplete_choices,
            })
        
        # 4. OCR质量问题（常见错误模式）
        ocr_issues = []
        if result.get("text_extraction", {}).get("mode") == "ocr":
            for i, q in enumerate(questions):
                content = q.get("content", "")
                # 检测常见OCR错误
                if re.search(r'\d{2,}[A-Z]{2,}', content):  # 如 48CD
                    ocr_issues.append({
                        "index": i+1,
                        "issue": "数字字母混淆",
                        "pattern": "数字+字母连写",
                    })
                if '山' in content and '⊥' not in content:  # 山可能是⊥
                    ocr_issues.append({
                        "index": i+1,
                        "issue": "符号误识别",
                        "pattern": "山 → ⊥",
                    })
        
        if ocr_issues:
            issues.append({
                "type": "ocr_errors",
                "severity": "warning",
                "message": f"检测到 {len(ocr_issues)} 处可能的OCR错误",
                "details": ocr_issues[:5],  # 只显示前5个
            })
        
        return {
            "success": True,
            "pdf_filename": pdf_file.filename,
            "extraction_info": {
                "mode": result.get("text_extraction", {}).get("mode"),
                "stats": result.get("text_extraction", {}).get("stats"),
                "total_questions": len(questions),
            },
            "meta_info": {
                "meta_used": result.get("meta_used", {}),
                "meta_inferred": result.get("meta_inferred", {}),
                "meta_confidence": result.get("meta_confidence", {}),
                "needs_confirmation": [
                    k for k, v in result.get("meta_confidence", {}).items()
                    if v < 0.7
                ],
            },
            "question_summaries": question_summaries,
            "quality_metrics": {
                "A1_json_parse_rate": quality_report.get("A1_json_parse_rate"),
                "A2_schema_valid_rate": quality_report.get("A2_schema_valid_rate"),
                "A3_structure_complete_rate": quality_report.get("A3_structure_complete_rate"),
                "A4_question_number_consistency": quality_report.get("A4_question_number_consistency"),
            },
            "issues": issues,
            "recommendation": _generate_recommendation(issues, quality_report),
        }
    
    finally:
        # 清理临时文件
        try:
            os.remove(pdf_path)
        except Exception:
            pass


def _generate_recommendation(issues: List[Dict], quality_report: Dict) -> str:
    """生成处理建议"""
    if not issues:
        return "✅ 提取质量良好，可以直接入库"
    
    error_count = sum(1 for i in issues if i.get("severity") == "error")
    warning_count = sum(1 for i in issues if i.get("severity") == "warning")
    
    if error_count > 0:
        return f"❌ 发现 {error_count} 个严重问题，建议修正后再入库"
    elif warning_count > 3:
        return f"⚠️ 发现 {warning_count} 个警告，建议检查后再入库"
    else:
        return "⚠️ 发现少量问题，可以入库但建议人工复核"


# ============ 统一评测 API ============
@app.get("/evaluation/full/vector_db")
async def full_evaluation_vector_db(
    gold_file_path: Optional[str] = None,
    strategy: str = "llm_only",
    use_llm: bool = False,
    save_report: bool = True,
    ):
    """
    对当前题库运行完整评测（A1-A4 + 金标覆盖/切分）
    
    Args:
        gold_file_path: 金标文件路径（可选，如 data/gold_standards/xxx.json）
        strategy: 提取策略标识
        use_llm: 是否使用 LLM 判断填空题完整性
        save_report: 是否保存 Markdown 报告
    """
    from ..evaluation.full_evaluator import run_full_evaluation
    
    init_agents()
    
    if vector_db is None:
        raise HTTPException(status_code=500, detail="向量库未初始化")
    
    questions = vector_db.metadata if hasattr(vector_db, 'metadata') else []
    if not questions:
        return {"error": "向量库为空，无题目可评测"}
    
    # 加载金标数据（可选）
    gold_data = None
    if gold_file_path:
        try:
            with open(gold_file_path, 'r', encoding='utf-8') as f:
                gold_data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"金标文件加载失败: {e}")
    
    report = run_full_evaluation(
        questions=questions,
        gold_data=gold_data,
        strategy=strategy,
        source_name="vector_db",
        llm_client=llm_client if use_llm else None,
        llm_model_name=getattr(llm_client, "model", None),
        save_report=save_report,
    )
    
    return report


@app.post("/evaluation/full/pdf")
async def full_evaluation_pdf(
    pdf_file: UploadFile = File(...),
    gold_file: Optional[UploadFile] = File(None),
    strategy: str = Form("llm_only"),
    ocr_enabled: bool = Form(True),
    use_llm: bool = Form(False),
    save_report: bool = Form(True),
):
    """
    对单个 PDF 运行完整评测（提取 + A1-A4 + 金标覆盖/切分）
    
    Args:
        pdf_file: 待评测的 PDF 文件
        gold_file: 金标数据 JSON 文件（可选）
        strategy: 提取策略标识
        ocr_enabled: 是否启用 OCR
        use_llm: 是否使用 LLM 判断填空题完整性
        save_report: 是否保存 Markdown 报告
    """
    from ..evaluation.full_evaluator import run_full_evaluation
    
    init_agents()
    
    # 保存临时 PDF
    pdf_filename = f"full_eval_{uuid.uuid4().hex[:8]}_{pdf_file.filename}"
    pdf_path = Config.PDF_STORAGE_PATH / pdf_filename
    
    with open(pdf_path, "wb") as f:
        pdf_content = await pdf_file.read()
        f.write(pdf_content)
    
    # 加载金标数据（可选）
    gold_data = None
    if gold_file:
        gold_content = await gold_file.read()
        try:
            gold_data = json.loads(gold_content.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"金标文件 JSON 解析失败: {e}")
    
    try:
        # 提取题目
        result = process_pdf_to_questions(
            pdf_path=str(pdf_path),
            meta={},
            ocr_enabled=ocr_enabled,
            llm_client=llm_client,
            vision_llm_client=vision_llm_client,
            auto_meta=True,
        )
        
        questions = result.get("questions", []) if isinstance(result, dict) else result
        # print("评测时提取出来的题目：",questions)
        # 生成评测对象名称
        source_name = pdf_file.filename.replace(".pdf", "").replace(" ", "_")
        
        # 运行完整评测
        report = run_full_evaluation(
            questions=questions,
            gold_data=gold_data,
            strategy=strategy,
            source_name=source_name,
            llm_client=llm_client if use_llm else None,
            llm_model_name=getattr(llm_client, "model", None),
            save_report=save_report,
        )
        
        # 添加提取信息
        report["extraction_info"] = {
            "total_extracted": len(questions),
            "strategy": strategy,
            "ocr_enabled": ocr_enabled,
            "meta_inferred": result.get("meta_inferred", {}) if isinstance(result, dict) else {},
            "text_extraction": result.get("text_extraction", None) if isinstance(result, dict) else None,
        }
        
        return report
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass
