"""教师API接口 - PDF处理与RAG讲义生成"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
import uuid
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
    region: str = Form(...),
    year: int = Form(...),
    grade: str = Form(...),
    exam_name: Optional[str] = Form(None),
    source_type: str = Form("期中"),  # 期中/期末/高考/模拟
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
    """
    if vector_db is None or embedding_model is None:
        raise HTTPException(status_code=500, detail="服务未正确初始化，请检查日志")
    
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
        
        # 2. 处理PDF，提取题目
        meta = {
            "region": region,
            "year": year,
            "grade": grade,
            "exam_name": exam_name or f"{year}年{region}{source_type}",
            "source_type": source_type,
        }
        
        print(f"开始处理PDF: {pdf_filename}")
        print(f"PDF文件大小: {pdf_path.stat().st_size / 1024:.2f} KB")
        questions = process_pdf_to_questions(
            pdf_path=str(pdf_path),
            meta=meta,
            ocr_enabled=ocr_enabled,
            llm_client=llm_client,
            use_regex_fallback=use_regex_fallback
        )
        print(f"题目提取完成，共提取 {len(questions)} 道题目")
        
        if not questions:
            return {
                "success": False,
                "message": "未能从PDF中提取到题目，请检查PDF格式",
                "questions_extracted": 0,
            }
        
        # 3. 为题目生成向量
        print(f"为 {len(questions)} 道题目生成向量...")
        question_contents = [q["content"] for q in questions]
        embeddings = embedding_model.encode(question_contents)
        
        # 4. 存储到向量库
        print(f"存储到向量库...")
        vector_db.add_questions(questions, embeddings)
        
        return {
            "success": True,
            "message": f"成功提取并存储 {len(questions)} 道题目",
            "questions_extracted": len(questions),
            "vector_db_total": vector_db.count(),
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


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
