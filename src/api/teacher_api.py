"""教师API接口 - PDF处理与RAG讲义生成"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import uuid
import re
import hashlib
import traceback
from pathlib import Path

from ..utils.config import Config
from ..utils.llm_client import get_default_llm_client
from ..data_processing.pdf_processor import process_pdf_to_questions
from ..vector_store.embedding import EmbeddingModel
from ..vector_store.vector_db import VectorDatabase
from ..agents.rag_handout_agent import RAGHandoutAgent
from ..export.pdf_exporter import handout_markdown_to_pdf

# 初始化
Config.init_directories()
app = FastAPI(title="独立老师备课助手API - PDF处理与RAG讲义生成", version="2.0.0")

# 全局对象（延迟初始化）
llm_client = None
vector_db = None
embedding_model = None
rag_agent = None


def init_agents():
    """初始化所有Agent（延迟初始化）"""
    global llm_client, vector_db, embedding_model, rag_agent
    try:
        llm_client = get_default_llm_client()
        vector_db = VectorDatabase(str(Config.VECTOR_DB_PATH))
        embedding_model = EmbeddingModel()
        rag_agent = RAGHandoutAgent(
            llm_client=llm_client,
            vector_db=vector_db,
            embedding_model=embedding_model
        )
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
    use_regex_fallback: bool = Form(False),
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

    try:
        def _normalize_question_content(text: str) -> str:
            t = (text or "").strip()
            t = re.sub(r"^\s*\(?\s*\d+\s*\)?\s*[\.、]\s*", "", t)
            t = re.sub(r"\s+", " ", t)
            t = t.lower()
            t = re.sub(r"[\s\u3000]+", "", t)
            t = re.sub(r"[，,。\.；;：:！!？?（）()【】\[\]《》<>“”\"'‘’、]", "", t)
            return t

        # 1. 保存上传的PDF
        file_ext = Path(pdf_file.filename).suffix.lower()
        if file_ext != ".pdf":
            raise HTTPException(status_code=400, detail="只支持PDF文件")

        pdf_filename = f"{uuid.uuid4().hex[:16]}.pdf"
        pdf_path = Config.PDF_STORAGE_PATH / pdf_filename

        with open(pdf_path, "wb") as f:
            content = await pdf_file.read()
            f.write(content)

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

        # 3. 调用 pdf 处理（兼容新旧 process_pdf_to_questions 返回类型）
        result = None
        try:
            # 如果你已经按方案给 process_pdf_to_questions 加了 auto_meta/meta_pages 参数
            result = process_pdf_to_questions(
                pdf_path=str(pdf_path),
                meta=user_meta,
                ocr_enabled=ocr_enabled,
                llm_client=llm_client,
                use_regex_fallback=use_regex_fallback,
                auto_meta=auto_meta,
                meta_pages=meta_pages,
            )
        except TypeError:
            # 兼容旧版本签名（还没加 auto_meta/meta_pages）
            result = process_pdf_to_questions(
                pdf_path=str(pdf_path),
                meta=user_meta,
                ocr_enabled=ocr_enabled,
                llm_client=llm_client,
                use_regex_fallback=use_regex_fallback,
            )

        # 4. 统一解析 result
        # - 新版：result 是 dict: {"questions":..., "meta_used":..., "meta_confidence":...}
        # - 旧版：result 是 list[question]
        if isinstance(result, dict):
            questions = result.get("questions", []) or []
            meta_used = result.get("meta_used", user_meta) or user_meta
            meta_inferred = result.get("meta_inferred", {}) or {}
            meta_confidence = result.get("meta_confidence", {}) or {}
            meta_evidence = result.get("meta_evidence", {}) or {}
        else:
            questions = result or []
            meta_used = user_meta
            meta_inferred = {}
            meta_confidence = {}
            meta_evidence = {}

        # 5. 如果 meta 仍不完整，给一个兜底 exam_name（避免下游存储缺字段）
        #    注意：这个兜底不会覆盖用户/自动识别的 exam_name
        if not meta_used.get("exam_name"):
            y = meta_used.get("year", "")
            r = meta_used.get("region", "")
            st = meta_used.get("source_type", "")
            fallback = f"{y}年{r}{st}".strip()
            meta_used["exam_name"] = fallback if fallback else "未命名试卷"

        print(f"题目提取完成，共提取 {len(questions)} 道题目")

        deduped_questions = []
        seen_keys = set()
        duplicates_removed = 0
        for q in questions:
            content = q.get("content", "")
            key_src = _normalize_question_content(content)
            key = hashlib.md5(key_src.encode("utf-8")).hexdigest()
            if key in seen_keys:
                duplicates_removed += 1
                continue
            seen_keys.add(key)
            q["id"] = q.get("id") or key
            deduped_questions.append(q)
        questions = deduped_questions
        if duplicates_removed:
            print(f"去重完成：移除重复题目 {duplicates_removed} 道")

        if not questions:
            return {
                "success": False,
                "message": "未能从PDF中提取到题目，请检查PDF格式",
                "questions_extracted": 0,
                "duplicates_removed": duplicates_removed,
                "meta_used": meta_used,
                "meta_inferred": meta_inferred,
                "meta_confidence": meta_confidence,
                "meta_evidence": meta_evidence,
            }

        # 6. needs_confirmation（仅当你实现了 meta_confidence 时才有意义）
        needs_confirmation = []
        if meta_confidence:
            # 你可以调整阈值，比如 0.7
            threshold = 0.7
            keys = ["region", "year", "grade", "exam_name", "source_type"]
            needs_confirmation = [k for k in keys if meta_confidence.get(k, 0.0) < threshold]

        # 7. 为题目生成向量
        print(f"为 {len(questions)} 道题目生成向量...")
        question_contents = [q["content"] for q in questions]
        embeddings = embedding_model.encode(question_contents)

        # 8. 存储到向量库
        print(f"存储到向量库...")
        vector_db.add_questions(questions, embeddings)

        return {
            "success": True,
            "message": f"成功提取并存储 {len(questions)} 道题目",
            "questions_extracted": len(questions),
            "duplicates_removed": duplicates_removed,
            "vector_db_total": vector_db.count(),

            # ✅ 新增：把元数据结果返回给前端/调用方，方便确认与回流
            "meta_used": meta_used,
            "meta_inferred": meta_inferred,
            "meta_confidence": meta_confidence,
            "needs_confirmation": needs_confirmation,
            "meta_evidence": meta_evidence,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理失败: {repr(e)}")


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
