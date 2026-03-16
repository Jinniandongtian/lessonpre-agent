#!/usr/bin/env python3
"""
题目提取质量评测 CLI 工具

使用方式:
1. 评测向量库中的题目:
   python -m src.evaluation.cli --source vector_db

2. 评测 JSON 文件:
   python -m src.evaluation.cli --source file --input questions.json

3. 评测单个 PDF 并输出报告:
   python -m src.evaluation.cli --source pdf --input path/to/test.pdf --ocr
"""

import argparse
import json
import sys
import os

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.evaluation.quality_metrics import evaluate_questions, print_evaluation_report


def load_from_vector_db() -> list:
    """从向量库加载题目"""
    from src.vector_store.vector_db import VectorDatabase
    db = VectorDatabase()
    # 获取所有题目的元数据
    return db.metadata if hasattr(db, 'metadata') else []


def load_from_json(path: str) -> list:
    """从 JSON 文件加载题目"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 支持两种格式：直接是列表，或者 {"questions": [...]}
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "questions" in data:
        return data["questions"]
    else:
        raise ValueError("JSON 格式不支持，需要是列表或包含 'questions' 字段的对象")


def load_from_pdf(path: str, ocr_enabled: bool = True) -> list:
    """从 PDF 提取题目并返回（用于评测）"""
    from src.data_processing.pdf_processor import process_pdf_to_questions
    from src.utils.llm_client import get_default_llm_client, get_vision_llm_client

    llm_client = get_default_llm_client()
    vision_llm_client = get_vision_llm_client()
    if vision_llm_client:
        print(f"使用视觉模型进行 OCR: {getattr(vision_llm_client, 'model', vision_llm_client.__class__.__name__)}")

    result = process_pdf_to_questions(
        pdf_path=path,
        meta={},
        ocr_enabled=ocr_enabled,
        llm_client=llm_client,
        vision_llm_client=vision_llm_client,
        auto_meta=True,
    )

    questions = result.get("questions", []) if isinstance(result, dict) else result
    return questions


def main():
    parser = argparse.ArgumentParser(description="题目提取质量评测工具")
    parser.add_argument(
        "--source",
        choices=["vector_db", "file", "pdf"],
        default="vector_db",
        help="数据来源: vector_db (向量库), file (JSON文件), pdf (PDF文件)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="输入文件路径 (当 source=file 或 source=pdf 时必填)",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="启用 OCR (当 source=pdf 时有效)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出报告到 JSON 文件 (可选)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制评测题目数量 (0=不限制)",
    )
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"正在加载数据 (source={args.source})...")
    
    if args.source == "vector_db":
        questions = load_from_vector_db()
    elif args.source == "file":
        if not args.input:
            parser.error("--input 参数必填 (当 source=file)")
        questions = load_from_json(args.input)
    elif args.source == "pdf":
        if not args.input:
            parser.error("--input 参数必填 (当 source=pdf)")
        questions = load_from_pdf(args.input, ocr_enabled=args.ocr)
    else:
        parser.error(f"未知的 source: {args.source}")
    
    # 限制数量
    if args.limit > 0 and len(questions) > args.limit:
        print(f"限制评测前 {args.limit} 道题目 (共 {len(questions)} 道)")
        questions = questions[:args.limit]
    
    print(f"加载完成，共 {len(questions)} 道题目")
    
    if not questions:
        print("错误: 没有题目可评测")
        sys.exit(1)
    
    # 评测
    print("正在评测...")
    report = evaluate_questions(questions)
    
    # 输出
    print_evaluation_report(report)
    
    # 保存到文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n报告已保存到: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
