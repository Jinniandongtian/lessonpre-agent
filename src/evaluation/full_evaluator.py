"""
统一评测系统入口

整合：
- A1. JSON 解析成功率
- A2. Schema 合格率
- A3. 结构完整率
- A4. 题号一致性
- B1. 覆盖指标（Precision/Recall/F1）
- B2. 切分指标（疑似合并/拆分）
- C1. 选项文本准确率（Option Exact/Soft Match）
- C2. 题干相似度（Stem Similarity）

支持：
- 针对当前题库评测
- 针对单个 PDF 评测
- 生成 Markdown 报告
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .quality_metrics import QualityEvaluator, evaluate_questions
from .coverage_metrics import CoverageEvaluator, evaluate_with_gold
from .content_metrics import ContentEvaluator, evaluate_content


def _extract_question_number(content: str) -> Optional[int]:
    """从题目内容中提取题号"""
    if not content:
        return None
    c = content.strip()
    # 匹配 "1." "1、" "1)" 等格式
    m = re.match(r'^(\d{1,4})\s*[\.、\)）]', c)
    if m:
        try:
            n = int(m.group(1))
            # 过滤明显不是题号的数字
            if 1900 <= n <= 2100 or n <= 0 or n > 200:
                return None
            return n
        except Exception:
            return None
    # 匹配 "(1)" 格式
    m = re.match(r'^\(\s*(\d{1,4})\s*\)', c)
    if m:
        try:
            n = int(m.group(1))
            if 1900 <= n <= 2100 or n <= 0 or n > 200:
                return None
            return n
        except Exception:
            return None
    return None


def _extract_stem_from_content(content: str) -> str:
    """从题目内容中提取题干（去掉题号和选项）"""
    c = content.strip()
    # 去掉题号
    c = re.sub(r'^(\d{1,4})\s*[\.、\)）]\s*', '', c)
    # 按行分割，去掉选项部分
    lines = c.split('\n')
    stem_lines = []
    for line in lines:
        # 如果遇到选项行（A. B. C. D. 等），停止
        if re.match(r'^[A-Da-d][\.\、\)）]\s*', line):
            break
        stem_lines.append(line)
    return '\n'.join(stem_lines).strip()


def _extract_options_from_content(content: str) -> Dict[str, str]:
    """从题目内容中提取选项"""
    options = {}
    # 匹配 "A. xxx" "B. xxx" 等格式
    pattern = r'([A-Da-d])[\.\、\)）]\s*(.+?)(?=(?:[A-Da-d][\.\、\)）]|$))'
    for m in re.finditer(pattern, content, re.DOTALL):
        key = m.group(1).upper()
        val = m.group(2).strip()
        if key and val:
            options[key] = val
    return options


def _convert_to_gold_format(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将入库题目格式转换为金标格式（用于 C 类指标评测）
    
    入库格式: {content, question_type, knowledge_points, difficulty, source_meta, ...}
    金标格式: {num, type, stem, options}
    """
    gold_questions = []
    
    for q in questions:
        content = q.get("content", "").strip()
        if not content:
            continue
        
        # 提取题号
        num = _extract_question_number(content)
        if num is None:
            continue
        
        # 提取题型（转换为金标格式）
        qtype = q.get("question_type", "未知题型")
        type_map = {
            "选择题": "单选题",
            "多选题": "多选题",
            "填空题": "填空题",
            "解答题": "解答题",
            "其他": "解答题",
        }
        gold_type = type_map.get(qtype, "解答题")
        
        # 提取题干
        stem = _extract_stem_from_content(content)
        if not stem:
            continue
        
        gold_q = {
            "num": num,
            "type": gold_type,
            "stem": stem,
        }
        
        # 提取选项（仅选择题）
        if gold_type in ["单选题", "多选题"]:
            options = _extract_options_from_content(content)
            if options:
                gold_q["options"] = options
        
        gold_questions.append(gold_q)
    
    return gold_questions


def _resolve_llm_model_name(llm_client=None, llm_model_name: Optional[str] = None) -> str:
    if llm_model_name and str(llm_model_name).strip():
        return str(llm_model_name).strip()
    if llm_client is not None:
        m = getattr(llm_client, "model", None)
        if m and str(m).strip():
            return str(m).strip()
        cname = llm_client.__class__.__name__
        if cname and cname != "LLMClient":
            return cname
    env_m = os.getenv("SILICONFLOW_MODEL") or os.getenv("LLM_MODEL")
    if env_m and str(env_m).strip():
        return str(env_m).strip()
    return "unknown"


class FullEvaluator:
    """统一评测器"""
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 客户端（用于填空题完整性判断）
        """
        self.quality_evaluator = QualityEvaluator(llm_client=llm_client)
        self.coverage_evaluator = CoverageEvaluator()
        self.content_evaluator = ContentEvaluator()
        self.llm_client = llm_client
    
    def evaluate_all(
        self,
        questions: List[Dict[str, Any]],
        gold_data: Optional[Dict[str, Any]] = None,
        raw_llm_responses: Optional[List[str]] = None,
        strategy: str = "unknown",
        source_name: str = "unknown",
        llm_model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行所有评测指标
        
        Args:
            questions: 题目列表
            gold_data: 金标数据（可选，用于覆盖/切分评测）
            raw_llm_responses: LLM 原始响应（可选，用于 A1 指标）
            strategy: 提取策略标识
            source_name: 评测对象名称（用于报告）
        
        Returns:
            完整评测报告
        """
        report = {
            "source_name": source_name,
            "strategy": strategy,
            "llm_model": _resolve_llm_model_name(self.llm_client, llm_model_name=llm_model_name),
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
        }
        
        # A 类指标：格式与可用性（不需要金标）
        quality_report = self.quality_evaluator.evaluate(questions, raw_llm_responses)
        report["quality_metrics"] = {
            "a1_json_parse": quality_report.metrics.get("a1_json_parse").to_dict() if "a1_json_parse" in quality_report.metrics else None,
            "a2_schema_compliance": quality_report.metrics.get("a2_schema_compliance").to_dict() if "a2_schema_compliance" in quality_report.metrics else None,
            "a3_structure_completeness": quality_report.metrics.get("a3_structure_completeness").to_dict() if "a3_structure_completeness" in quality_report.metrics else None,
            "a4_number_consistency": quality_report.metrics.get("a4_number_consistency").to_dict() if "a4_number_consistency" in quality_report.metrics else None,
        }
        report["quality_summary"] = quality_report.summary
        
        # B 类指标：覆盖与切分（需要金标）
        if gold_data:
            coverage_report = self.coverage_evaluator.evaluate(gold_data, questions, strategy)
            report["coverage_metrics"] = coverage_report.coverage.to_dict()
            report["segmentation_metrics"] = coverage_report.segmentation.to_dict()
            report["gold_summary"] = coverage_report.summary
        else:
            report["coverage_metrics"] = None
            report["segmentation_metrics"] = None
            report["gold_summary"] = None
        
        # C 类指标：结构化内容正确性（需要金标含 stem/options）
        if gold_data and self._gold_has_content(gold_data):
            # 转换题目格式为金标格式用于 C 类评测
            questions_for_c_eval = _convert_to_gold_format(questions)
            print("即将用于评测的题目：",questions_for_c_eval)
            content_report = self.content_evaluator.evaluate(gold_data, questions_for_c_eval)
            report["content_metrics"] = content_report.to_dict()
        else:
            report["content_metrics"] = None
        
        return report
    
    def _gold_has_content(self, gold_data: Dict[str, Any]) -> bool:
        """检查金标是否包含内容字段（stem/options）"""
        questions = gold_data.get("questions", [])
        if not questions:
            return False
        # 只要有一题含 stem 或 options 就认为可用
        for q in questions:
            if q.get("stem") or q.get("options"):
                return True
        return False
    
    def generate_markdown_report(
        self,
        report: Dict[str, Any],
        output_dir: str = "src/evaluation/results",
    ) -> str:
        """
        生成 Markdown 格式的评测报告
        
        Args:
            report: 评测报告（来自 evaluate_all）
            output_dir: 输出目录
        
        Returns:
            生成的报告文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 文件名：评测对象-模型-时间
        source_name = report.get("source_name", "unknown").replace("/", "_").replace(" ", "_")
        llm_model = str(report.get("llm_model", "unknown") or "unknown")
        llm_model = llm_model.replace("/", "_").replace(" ", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source_name}-{llm_model}-{timestamp}.md"
        filepath = output_path / filename
        
        # 生成 Markdown 内容
        md_content = self._build_markdown(report)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)
        
        return str(filepath)
    
    def _build_markdown(self, report: Dict[str, Any]) -> str:
        """构建 Markdown 报告内容"""
        lines = []
        
        # 标题
        lines.append(f"# 题目提取质量评测报告")
        lines.append("")
        lines.append(f"**评测对象**: {report.get('source_name', 'unknown')}")
        lines.append(f"**提取策略**: {report.get('strategy', 'unknown')}")
        lines.append(f"**LLM模型**: {report.get('llm_model', 'unknown')}")
        lines.append(f"**评测时间**: {report.get('timestamp', '')}")
        lines.append(f"**题目总数**: {report.get('total_questions', 0)}")
        lines.append("")
        
        # 指标说明
        lines.append("---")
        lines.append("")
        lines.append("## 指标说明")
        lines.append("")
        lines.append("### A 类指标（格式与可用性，不需要金标）")
        lines.append("")
        lines.append("| 指标 | 说明 |")
        lines.append("|------|------|")
        lines.append("| **A1. JSON 解析成功率** | LLM 输出中能被解析为合法 JSON 的比例。反映 LLM 输出格式的稳定性。 |")
        lines.append("| **A2. Schema 合格率** | 题目通过 Schema 校验的比例（必填字段存在、类型正确）。反映数据结构的规范性。 |")
        lines.append("| **A3. 结构完整率** | 按题型分别评测：选择题有 ABCD 选项、填空题有空标记、解答题有小问或足够内容。反映题目能否直接用于检索/组卷。 |")
        lines.append("| **A4. 题号一致性** | 检测重复题号、跳号、非法题号。反映题目切分的准确性。 |")
        lines.append("")
        lines.append("### B 类指标（覆盖与切分，需要金标）")
        lines.append("")
        lines.append("| 指标 | 说明 |")
        lines.append("|------|------|")
        lines.append("| **B1. 覆盖指标** | Precision（提取的题里有多少是真的）、Recall（真实题覆盖了多少）、F1（综合）。 |")
        lines.append("| **B2. 切分指标** | 疑似合并（两题粘一起）、疑似拆分（一题拆成两题）的数量。 |")
        lines.append("| **诊断指标** | 题号不可解析率（OCR/LLM 把题号吃掉）、题号重复率（拆分或重复提取信号）。 |")
        lines.append("")
        lines.append("### C 类指标（结构化内容正确性，需要金标含 stem/options）")
        lines.append("")
        lines.append("| 指标 | 说明 |")
        lines.append("|------|------|")
        lines.append("| **C1. 选项文本准确率** | 选择题 A-D 选项与金标的匹配程度。严格匹配率 + 字符级相似度（容忍 OCR 噪声）。 |")
        lines.append("| **C2. 题干相似度** | 题干文本与金标的匹配程度。用字符级相似度/编辑距离，不依赖模型。 |")
        lines.append("")
        
        # A 类指标结果
        lines.append("---")
        lines.append("")
        lines.append("## A 类指标结果")
        lines.append("")
        
        quality_metrics = report.get("quality_metrics", {})
        quality_summary = report.get("quality_summary", {})
        
        # A2 Schema 合格率
        a2 = quality_metrics.get("a2_schema_compliance")
        if a2:
            lines.append("### A2. Schema 合格率")
            lines.append("")
            lines.append(f"- **得分**: {a2.get('score', 0):.2%}")
            lines.append(f"- **通过/总数**: {a2.get('passed', 0)}/{a2.get('total', 0)}")
            lines.append("")
            field_presence = a2.get("details", {}).get("field_presence", {})
            if field_presence:
                lines.append("**字段存在率**:")
                lines.append("")
                lines.append("| 字段 | 存在率 |")
                lines.append("|------|--------|")
                for field, rate in field_presence.items():
                    lines.append(f"| {field} | {rate:.1%} |")
                lines.append("")
        
        # A3 结构完整率
        a3 = quality_metrics.get("a3_structure_completeness")
        if a3:
            lines.append("### A3. 结构完整率")
            lines.append("")
            lines.append(f"- **得分**: {a3.get('score', 0):.2%}")
            lines.append(f"- **通过/总数**: {a3.get('passed', 0)}/{a3.get('total', 0)}")
            lines.append("")
            by_type = a3.get("details", {}).get("by_type", {})
            if by_type:
                lines.append("**按题型**:")
                lines.append("")
                lines.append("| 题型 | 完整/总数 | 完整率 |")
                lines.append("|------|-----------|--------|")
                for qt, stats in by_type.items():
                    if stats.get("total", 0) > 0:
                        lines.append(f"| {qt} | {stats.get('complete', 0)}/{stats.get('total', 0)} | {stats.get('rate', 0):.1%} |")
                lines.append("")
        
        # A4 题号一致性
        a4 = quality_metrics.get("a4_number_consistency")
        if a4:
            lines.append("### A4. 题号一致性")
            lines.append("")
            lines.append(f"- **得分**: {a4.get('score', 0):.2%}")
            lines.append(f"- **通过/总数**: {a4.get('passed', 0)}/{a4.get('total', 0)}")
            lines.append("")
            details = a4.get("details", {})
            lines.append(f"- **重复题号率**: {details.get('duplicate_rate', 0):.2%}")
            lines.append(f"- **跳号率**: {details.get('gap_rate', 0):.2%}")
            lines.append(f"- **非法题号率**: {details.get('invalid_rate', 0):.2%}")
            if details.get("duplicates"):
                lines.append(f"- **重复题号**: {details.get('duplicates')}")
            if details.get("missing_numbers"):
                missing = details.get("missing_numbers", [])[:10]
                lines.append(f"- **缺失题号**: {missing}")
            lines.append("")
        
        # A 类汇总
        if quality_summary:
            lines.append("### A 类指标汇总")
            lines.append("")
            lines.append("| 指标 | 得分 |")
            lines.append("|------|------|")
            for k, v in quality_summary.items():
                lines.append(f"| {k} | {v:.2%} |")
            lines.append("")
        
        # B 类指标结果
        coverage = report.get("coverage_metrics")
        segmentation = report.get("segmentation_metrics")
        gold_summary = report.get("gold_summary")
        
        if coverage or segmentation:
            lines.append("---")
            lines.append("")
            lines.append("## B 类指标结果（金标评测）")
            lines.append("")
        
        if coverage:
            lines.append("### B1. 覆盖指标")
            lines.append("")
            lines.append(f"- **金标题数**: {coverage.get('gold_count', 0)}")
            lines.append(f"- **预测题数**: {coverage.get('pred_count', 0)}")
            lines.append(f"- **TP（正确提取）**: {coverage.get('tp', 0)}")
            lines.append(f"- **FP（多提/误提）**: {coverage.get('fp', 0)}")
            lines.append(f"- **FN（漏题）**: {coverage.get('fn', 0)}")
            lines.append("")
            lines.append(f"- **Precision**: {coverage.get('precision', 0):.2%}")
            lines.append(f"- **Recall**: {coverage.get('recall', 0):.2%}")
            lines.append(f"- **F1**: {coverage.get('f1', 0):.2%}")
            lines.append("")
            lines.append("**诊断指标**:")
            lines.append("")
            lines.append(f"- **题号不可解析率**: {coverage.get('unparsable_rate', 0):.2%} ({coverage.get('unparsable_count', 0)}/{coverage.get('pred_count', 0)})")
            lines.append(f"- **题号重复率**: {coverage.get('duplicate_rate', 0):.2%}")
            if coverage.get("duplicate_nums"):
                lines.append(f"- **重复题号详情**: {coverage.get('duplicate_nums')}")
            if coverage.get("missing_nums"):
                lines.append(f"- **漏题题号**: {coverage.get('missing_nums')}")
            if coverage.get("extra_nums"):
                lines.append(f"- **多提题号**: {coverage.get('extra_nums')}")
            lines.append("")
        
        if segmentation:
            lines.append("### B2. 切分指标")
            lines.append("")
            lines.append(f"- **疑似合并**: {segmentation.get('merge_count', 0)} 处")
            merge_suspects = segmentation.get("merge_suspects", [])
            if merge_suspects:
                for m in merge_suspects[:5]:
                    lines.append(f"  - 题号 {m.get('pred_num')} 可能合并了 {m.get('missing_next')}，内容长度 {m.get('content_length')}")
            lines.append(f"- **疑似拆分**: {segmentation.get('split_count', 0)} 处")
            split_suspects = segmentation.get("split_suspects", [])
            if split_suspects:
                for s in split_suspects[:5]:
                    lines.append(f"  - 题号 {s.get('num')} 出现 {s.get('occurrences')} 次")
            lines.append("")
        
        if gold_summary:
            lines.append("### B 类指标汇总")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| Precision | {gold_summary.get('precision', 0):.2%} |")
            lines.append(f"| Recall | {gold_summary.get('recall', 0):.2%} |")
            lines.append(f"| F1 | {gold_summary.get('f1', 0):.2%} |")
            lines.append(f"| 题号不可解析率 | {gold_summary.get('unparsable_rate', 0):.2%} |")
            lines.append(f"| 题号重复率 | {gold_summary.get('duplicate_rate', 0):.2%} |")
            lines.append(f"| 疑似合并数 | {gold_summary.get('merge_count', 0)} |")
            lines.append(f"| 疑似拆分数 | {gold_summary.get('split_count', 0)} |")
            lines.append("")
        
        if not coverage and not segmentation:
            lines.append("---")
            lines.append("")
            lines.append("## B 类指标结果")
            lines.append("")
            lines.append("*未提供金标数据，跳过覆盖/切分评测*")
            lines.append("")
        
        # C 类指标结果
        content_metrics = report.get("content_metrics")
        if content_metrics:
            lines.append("---")
            lines.append("")
            lines.append("## C 类指标结果（结构化内容正确性）")
            lines.append("")
            
            lines.append("### C1. 选项文本准确率")
            lines.append("")
            lines.append(f"- **选择题数量**: {content_metrics.get('choice_count', 0)}")
            lines.append(f"- **选项严格匹配率**: {content_metrics.get('c1_option_exact_match_rate', 0):.2%}")
            lines.append(f"- **选项平均相似度**: {content_metrics.get('c1_option_avg_similarity', 0):.2%}")
            lines.append("")
            
            lines.append("### C2. 题干相似度")
            lines.append("")
            lines.append(f"- **匹配题目数**: {content_metrics.get('total_matched', 0)}/{content_metrics.get('total_gold', 0)}")
            lines.append(f"- **题干严格匹配率**: {content_metrics.get('c2_stem_exact_match_rate', 0):.2%}")
            lines.append(f"- **题干平均相似度**: {content_metrics.get('c2_stem_avg_similarity', 0):.2%}")
            lines.append("")
            
            # 未匹配诊断
            unmatched_gold = content_metrics.get("unmatched_gold_nums", [])
            unmatched_pred = content_metrics.get("unmatched_pred_nums", [])
            if unmatched_gold or unmatched_pred:
                lines.append("**诊断**:")
                lines.append("")
                if unmatched_gold:
                    lines.append(f"- 金标有但预测没有: {unmatched_gold[:10]}")
                if unmatched_pred:
                    lines.append(f"- 预测有但金标没有: {unmatched_pred[:10]}")
                lines.append("")
            
            lines.append("### C 类指标汇总")
            lines.append("")
            lines.append("| 指标 | 值 |")
            lines.append("|------|-----|")
            lines.append(f"| 选项严格匹配率 | {content_metrics.get('c1_option_exact_match_rate', 0):.2%} |")
            lines.append(f"| 选项平均相似度 | {content_metrics.get('c1_option_avg_similarity', 0):.2%} |")
            lines.append(f"| 题干严格匹配率 | {content_metrics.get('c2_stem_exact_match_rate', 0):.2%} |")
            lines.append(f"| 题干平均相似度 | {content_metrics.get('c2_stem_avg_similarity', 0):.2%} |")
            lines.append("")
        elif report.get("coverage_metrics") is not None:
            # 有金标但没有 stem/options
            lines.append("---")
            lines.append("")
            lines.append("## C 类指标结果")
            lines.append("")
            lines.append("*金标数据未包含 stem/options 字段，跳过内容正确性评测*")
            lines.append("")
        
        # 结尾
        lines.append("---")
        lines.append("")
        lines.append("*报告由评测系统自动生成*")
        
        return "\n".join(lines)


# ============ 便捷函数 ============
def run_full_evaluation(
    questions: List[Dict[str, Any]],
    gold_data: Optional[Dict[str, Any]] = None,
    raw_llm_responses: Optional[List[str]] = None,
    strategy: str = "unknown",
    source_name: str = "unknown",
    llm_client=None,
    llm_model_name: Optional[str] = None,
    output_dir: str = "src/evaluation/results",
    save_report: bool = True,
) -> Dict[str, Any]:
    """
    运行完整评测并生成报告
    
    Args:
        questions: 题目列表
        gold_data: 金标数据（可选）
        raw_llm_responses: LLM 原始响应（可选）
        strategy: 提取策略
        source_name: 评测对象名称
        llm_client: LLM 客户端
        output_dir: 报告输出目录
        save_report: 是否保存 Markdown 报告
    
    Returns:
        评测报告（含报告文件路径）
    """
    evaluator = FullEvaluator(llm_client=llm_client)
    report = evaluator.evaluate_all(
        questions=questions,
        gold_data=gold_data,
        raw_llm_responses=raw_llm_responses,
        strategy=strategy,
        source_name=source_name,
        llm_model_name=llm_model_name,
    )
    
    if save_report:
        report_path = evaluator.generate_markdown_report(report, output_dir)
        report["report_path"] = report_path
    
    return report
