"""
题号级切分/覆盖评测模块

需要金标数据，用于评测：
B1. 覆盖指标：Precision/Recall/F1
B2. 切分指标：疑似合并/拆分检测
"""

import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class CoverageResult:
    """覆盖评测结果"""
    # 基础统计
    gold_count: int          # 金标题目数
    pred_count: int          # 预测题目数
    
    # 集合
    gold_nums: Set[int]      # 金标题号集合 G
    pred_nums: Set[int]      # 预测题号集合 P (不含 None)
    
    # TP/FP/FN
    tp: int                  # |G ∩ P| 正确提到的题号
    fp: int                  # |P - G| 多提/误提的题号
    fn: int                  # |G - P| 漏题
    
    # 指标
    precision: float         # TP / (TP + FP)
    recall: float            # TP / (TP + FN)
    f1: float                # 2PR / (P + R)
    
    # 诊断指标
    unparsable_count: int    # 题号不可解析的题目数
    unparsable_rate: float   # 题号不可解析率
    duplicate_nums: Dict[int, int]  # 重复题号及出现次数
    duplicate_rate: float    # 题号重复率
    
    # 详情
    correct_nums: List[int]  # TP 的具体题号
    extra_nums: List[int]    # FP 的具体题号（多提）
    missing_nums: List[int]  # FN 的具体题号（漏题）
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "gold_count": self.gold_count,
            "pred_count": self.pred_count,
            "gold_nums": sorted(list(self.gold_nums)),
            "pred_nums": sorted(list(self.pred_nums)),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "unparsable_count": self.unparsable_count,
            "unparsable_rate": self.unparsable_rate,
            "duplicate_nums": self.duplicate_nums,
            "duplicate_rate": self.duplicate_rate,
            "correct_nums": self.correct_nums,
            "extra_nums": self.extra_nums,
            "missing_nums": self.missing_nums,
        }


@dataclass
class SegmentationResult:
    """切分评测结果"""
    # 疑似合并
    merge_suspects: List[Dict[str, Any]]  # [{pred_num, content_preview, detected_nums}]
    merge_count: int
    
    # 疑似拆分
    split_suspects: List[Dict[str, Any]]  # [{num, occurrences, content_previews}]
    split_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "merge_count": self.merge_count,
            "merge_suspects": self.merge_suspects,
            "split_count": self.split_count,
            "split_suspects": self.split_suspects,
        }


@dataclass
class GoldStandardEvalReport:
    """金标评测完整报告"""
    paper_id: str
    strategy: str            # 提取策略标识
    coverage: CoverageResult
    segmentation: SegmentationResult
    summary: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "strategy": self.strategy,
            "coverage": self.coverage.to_dict(),
            "segmentation": self.segmentation.to_dict(),
            "summary": self.summary,
        }


class CoverageEvaluator:
    """金标数据评测器"""
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        gold_data: Dict[str, Any],
        predictions: List[Dict[str, Any]],
        strategy: str = "unknown",
    ) -> GoldStandardEvalReport:
        """
        评测预测结果与金标数据
        
        Args:
            gold_data: 金标数据 {"paper_id": "...", "questions": [{"num": 1, "type": "..."}, ...]}
            predictions: 预测的题目列表 [{"content": "...", ...}, ...]
            strategy: 提取策略标识，如 "llm_only", "llm+regex_fallback", "rule_based"
        
        Returns:
            GoldStandardEvalReport
        """
        paper_id = gold_data.get("paper_id", "unknown")
        gold_questions = gold_data.get("questions", [])
        
        # 1. 覆盖评测
        coverage = self._eval_coverage(gold_questions, predictions)
        
        # 2. 切分评测
        segmentation = self._eval_segmentation(gold_questions, predictions)
        
        # 3. 汇总
        summary = {
            "precision": coverage.precision,
            "recall": coverage.recall,
            "f1": coverage.f1,
            "unparsable_rate": coverage.unparsable_rate,
            "duplicate_rate": coverage.duplicate_rate,
            "merge_count": segmentation.merge_count,
            "split_count": segmentation.split_count,
        }
        
        return GoldStandardEvalReport(
            paper_id=paper_id,
            strategy=strategy,
            coverage=coverage,
            segmentation=segmentation,
            summary=summary,
        )
    
    def _eval_coverage(
        self,
        gold_questions: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
    ) -> CoverageResult:
        """覆盖指标评测"""
        
        # 金标题号集合
        gold_nums = set()
        for gq in gold_questions:
            num = gq.get("num")
            if num is not None:
                gold_nums.add(int(num))
        
        # 预测题号（重要：从 content 解析，不用 index！）
        pred_nums_list = []  # 保留重复，用于统计重复率
        unparsable_count = 0
        
        for pq in predictions:
            content = pq.get("content", "")
            num = self._extract_question_number_from_content(content)
            if num is not None:
                pred_nums_list.append(num)
            else:
                unparsable_count += 1
        
        pred_nums = set(pred_nums_list)
        
        # TP / FP / FN
        tp_set = gold_nums & pred_nums
        fp_set = pred_nums - gold_nums
        fn_set = gold_nums - pred_nums
        
        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)
        
        # Precision / Recall / F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # 诊断指标：题号不可解析率
        pred_count = len(predictions)
        unparsable_rate = unparsable_count / pred_count if pred_count > 0 else 0.0
        
        # 诊断指标：题号重复率
        from collections import Counter
        num_counts = Counter(pred_nums_list)
        duplicate_nums = {k: v for k, v in num_counts.items() if v > 1}
        duplicate_count = sum(v - 1 for v in duplicate_nums.values())  # 多出来的次数
        duplicate_rate = duplicate_count / len(pred_nums_list) if pred_nums_list else 0.0
        
        return CoverageResult(
            gold_count=len(gold_questions),
            pred_count=pred_count,
            gold_nums=gold_nums,
            pred_nums=pred_nums,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            unparsable_count=unparsable_count,
            unparsable_rate=unparsable_rate,
            duplicate_nums=duplicate_nums,
            duplicate_rate=duplicate_rate,
            correct_nums=sorted(list(tp_set)),
            extra_nums=sorted(list(fp_set)),
            missing_nums=sorted(list(fn_set)),
        )
    
    def _eval_segmentation(
        self,
        gold_questions: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
    ) -> SegmentationResult:
        """切分指标评测"""
        
        gold_nums = set()
        for gq in gold_questions:
            num = gq.get("num")
            if num is not None:
                gold_nums.add(int(num))
        
        # 按题号分组预测
        pred_by_num: Dict[int, List[Dict[str, Any]]] = {}
        for pq in predictions:
            content = pq.get("content", "")
            num = self._extract_question_number_from_content(content)
            if num is not None:
                if num not in pred_by_num:
                    pred_by_num[num] = []
                pred_by_num[num].append(pq)
        
        # 疑似拆分：同一题号出现多次
        split_suspects = []
        for num, pqs in pred_by_num.items():
            if len(pqs) > 1:
                split_suspects.append({
                    "num": num,
                    "occurrences": len(pqs),
                    "content_previews": [pq.get("content", "")[:100] for pq in pqs],
                })
        
        # 疑似合并：金标有 k 和 k+1，预测只有 k，且 k 的 content 很长或包含多个题号
        merge_suspects = []
        sorted_gold = sorted(gold_nums)
        pred_nums = set(pred_by_num.keys())
        
        for i, k in enumerate(sorted_gold):
            if i + 1 < len(sorted_gold):
                k_next = sorted_gold[i + 1]
                # 金标有 k 和 k+1，但预测缺 k+1
                if k in pred_nums and k_next not in pred_nums:
                    # 检查 k 的 content 是否包含多个题号或很长
                    for pq in pred_by_num.get(k, []):
                        content = pq.get("content", "")
                        detected_nums = self._detect_multiple_question_nums(content)
                        
                        # 条件：content 长度 > 500 或检测到多个题号
                        if len(content) > 500 or len(detected_nums) > 1:
                            merge_suspects.append({
                                "pred_num": k,
                                "missing_next": k_next,
                                "content_length": len(content),
                                "detected_nums": detected_nums,
                                "content_preview": content[:200],
                            })
        
        return SegmentationResult(
            merge_suspects=merge_suspects,
            merge_count=len(merge_suspects),
            split_suspects=split_suspects,
            split_count=len(split_suspects),
        )
    
    def _extract_question_number_from_content(self, content: str) -> Optional[int]:
        """
        从题目内容中解析真实题号
        
        重要：不用 index 字段！必须从 content 解析！
        """
        if not content:
            return None
        
        # 取前 80 字符
        head = content[:80]
        
        # 常见题号格式（按优先级）
        patterns = [
            r'^[\s\n]*(\d{1,2})\s*[.、．:：\)\]】]',      # 1. 2、 3）
            r'^[\s\n]*[（\(]\s*(\d{1,2})\s*[）\)]',       # (1) （1）- 小问形式
            r'^[\s\n]*第\s*(\d{1,2})\s*题',              # 第1题
            r'^\s*(\d{1,2})\s*[\.\、]',                  # 简单的 1. 2、
        ]
        
        for pat in patterns:
            m = re.match(pat, head)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    continue
        
        return None
    
    def _detect_multiple_question_nums(self, content: str) -> List[int]:
        """
        检测 content 中是否包含多个题号（用于合并检测）
        """
        # 匹配所有可能的题号
        pattern = r'(?:^|\n)\s*(\d{1,2})\s*[.、．:：\)\]】]'
        matches = re.findall(pattern, content)
        
        nums = []
        for m in matches:
            try:
                nums.append(int(m))
            except ValueError:
                continue
        
        return sorted(set(nums))


# ============ 便捷函数 ============
def load_gold_standard(path: str) -> Dict[str, Any]:
    """加载金标数据文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_with_gold(
    gold_data: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    strategy: str = "unknown",
) -> Dict[str, Any]:
    """
    使用金标数据评测
    
    Args:
        gold_data: 金标数据
        predictions: 预测的题目列表
        strategy: 提取策略标识
    
    Returns:
        评测报告（dict 格式）
    """
    evaluator = GoldStandardEvaluator()
    report = evaluator.evaluate(gold_data, predictions, strategy)
    return report.to_dict()


def print_gold_eval_report(report: Dict[str, Any]):
    """打印金标评测报告"""
    print("\n" + "=" * 60)
    print(f"金标评测报告 - {report['paper_id']}")
    print(f"提取策略: {report['strategy']}")
    print("=" * 60)
    
    cov = report["coverage"]
    print("\n【覆盖指标】")
    print(f"  金标题数: {cov['gold_count']}, 预测题数: {cov['pred_count']}")
    print(f"  TP: {cov['tp']}, FP: {cov['fp']}, FN: {cov['fn']}")
    print(f"  Precision: {cov['precision']:.2%}")
    print(f"  Recall: {cov['recall']:.2%}")
    print(f"  F1: {cov['f1']:.2%}")
    print(f"\n  【诊断】")
    print(f"  题号不可解析率: {cov['unparsable_rate']:.2%} ({cov['unparsable_count']}/{cov['pred_count']})")
    print(f"  题号重复率: {cov['duplicate_rate']:.2%}")
    if cov['duplicate_nums']:
        print(f"  重复题号: {cov['duplicate_nums']}")
    if cov['missing_nums']:
        print(f"  漏题: {cov['missing_nums']}")
    if cov['extra_nums']:
        print(f"  多提: {cov['extra_nums']}")
    
    seg = report["segmentation"]
    print("\n【切分指标】")
    print(f"  疑似合并: {seg['merge_count']} 处")
    for m in seg['merge_suspects'][:3]:
        print(f"    - 题号 {m['pred_num']} 可能合并了 {m['missing_next']}，内容长度 {m['content_length']}")
    print(f"  疑似拆分: {seg['split_count']} 处")
    for s in seg['split_suspects'][:3]:
        print(f"    - 题号 {s['num']} 出现 {s['occurrences']} 次")
    
    print("\n" + "=" * 60)
