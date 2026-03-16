"""
结构化内容正确性评测模块 (C类指标)

需要金标数据（含题干/选项文本），用于评测：
C1. 选项文本准确率（Option Exact/Soft Match）
C2. 题干相似度（Stem Similarity）

支持：
- 严格匹配 (Exact Match)
- 字符级相似度 (Character-level Similarity)
- 编辑距离归一化 (Normalized Edit Distance)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class OptionMatchResult:
    """单个选项的匹配结果"""
    option_key: str          # A/B/C/D
    gold_text: str           # 金标选项文本
    pred_text: str           # 预测选项文本
    exact_match: bool        # 严格匹配
    similarity: float        # 字符级相似度 [0, 1]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "option_key": self.option_key,
            "gold_text": self.gold_text[:50] + "..." if len(self.gold_text) > 50 else self.gold_text,
            "pred_text": self.pred_text[:50] + "..." if len(self.pred_text) > 50 else self.pred_text,
            "exact_match": self.exact_match,
            "similarity": round(self.similarity, 4),
        }


@dataclass
class QuestionContentResult:
    """单题的内容评测结果"""
    question_num: int
    question_type: str
    
    # 题干
    stem_gold: str
    stem_pred: str
    stem_exact_match: bool
    stem_similarity: float
    
    # 选项（仅选择题）
    option_results: List[OptionMatchResult]
    option_exact_match_rate: float   # 选项严格匹配率
    option_avg_similarity: float     # 选项平均相似度
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_num": self.question_num,
            "question_type": self.question_type,
            "stem_exact_match": self.stem_exact_match,
            "stem_similarity": round(self.stem_similarity, 4),
            "option_exact_match_rate": round(self.option_exact_match_rate, 4),
            "option_avg_similarity": round(self.option_avg_similarity, 4),
            "option_details": [o.to_dict() for o in self.option_results],
        }


@dataclass
class ContentEvalResult:
    """C类指标评测结果"""
    # 汇总统计
    total_matched: int           # 成功匹配的题目数（金标与预测都有）
    total_gold: int              # 金标总题数
    total_pred: int              # 预测总题数
    
    # C1: 选项文本准确率
    choice_count: int            # 选择题数量
    option_exact_match_rate: float   # 选项严格匹配率（所有选项）
    option_avg_similarity: float     # 选项平均相似度
    
    # C2: 题干相似度
    stem_exact_match_rate: float     # 题干严格匹配率
    stem_avg_similarity: float       # 题干平均相似度
    
    # 详细结果
    per_question: List[QuestionContentResult]
    
    # 诊断
    unmatched_gold_nums: List[int]   # 金标有但预测没有的题号
    unmatched_pred_nums: List[int]   # 预测有但金标没有的题号
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_matched": self.total_matched,
            "total_gold": self.total_gold,
            "total_pred": self.total_pred,
            "choice_count": self.choice_count,
            "c1_option_exact_match_rate": round(self.option_exact_match_rate, 4),
            "c1_option_avg_similarity": round(self.option_avg_similarity, 4),
            "c2_stem_exact_match_rate": round(self.stem_exact_match_rate, 4),
            "c2_stem_avg_similarity": round(self.stem_avg_similarity, 4),
            "unmatched_gold_nums": self.unmatched_gold_nums,
            "unmatched_pred_nums": self.unmatched_pred_nums,
            "per_question": [q.to_dict() for q in self.per_question],
        }


class ContentEvaluator:
    """结构化内容评测器"""
    
    def __init__(self):
        pass
    
    def evaluate(
        self,
        gold_data: Dict[str, Any],
        predictions: List[Dict[str, Any]],
    ) -> ContentEvalResult:
        """
        评测预测内容与金标内容
        
        Args:
            gold_data: 金标数据，格式：
                {
                    "paper_id": "...",
                    "questions": [
                        {
                            "num": 1,
                            "type": "选择题",
                            "stem": "题干文本...",
                            "options": {"A": "选项A文本", "B": "...", "C": "...", "D": "..."}
                        },
                        ...
                    ]
                }
            predictions: 预测的题目列表
        
        Returns:
            ContentEvalResult
        """
        gold_questions = gold_data.get("questions", [])
        
        # 构建金标题号 -> 题目映射
        gold_by_num = {}
        for q in gold_questions:
            num = q.get("num")
            if num is not None:
                gold_by_num[num] = q
        
        # 构建预测题号 -> 题目映射
        pred_by_num = {}
        for q in predictions:
            # num = self._parse_question_num(q)
            num = q.get("num")
            if num is not None:
                pred_by_num[num] = q
        print("预测题号映射：",pred_by_num)
        # 找到匹配的题号
        gold_nums = set(gold_by_num.keys())
        pred_nums = set(pred_by_num.keys())
        matched_nums = gold_nums & pred_nums
        print("匹配的题号：",matched_nums)
        print("金标题号：",gold_nums)
        print("预测题号：",pred_nums)
        # 逐题评测
        per_question_results = []
        all_option_exact = []
        all_option_sim = []
        all_stem_exact = []
        all_stem_sim = []
        choice_count = 0
        
        for num in sorted(matched_nums):
            gold_q = gold_by_num[num]
            pred_q = pred_by_num[num]
            
            result = self._eval_single_question(gold_q, pred_q)
            per_question_results.append(result)
            
            # 汇总题干
            all_stem_exact.append(1 if result.stem_exact_match else 0)
            all_stem_sim.append(result.stem_similarity)
            
            # 汇总选项（仅选择题）
            if result.option_results:
                choice_count += 1
                for opt_r in result.option_results:
                    all_option_exact.append(1 if opt_r.exact_match else 0)
                    all_option_sim.append(opt_r.similarity)
        
        # 计算汇总指标
        option_exact_rate = sum(all_option_exact) / len(all_option_exact) if all_option_exact else 0.0
        option_avg_sim = sum(all_option_sim) / len(all_option_sim) if all_option_sim else 0.0
        stem_exact_rate = sum(all_stem_exact) / len(all_stem_exact) if all_stem_exact else 0.0
        stem_avg_sim = sum(all_stem_sim) / len(all_stem_sim) if all_stem_sim else 0.0
        
        return ContentEvalResult(
            total_matched=len(matched_nums),
            total_gold=len(gold_nums),
            total_pred=len(pred_nums),
            choice_count=choice_count,
            option_exact_match_rate=option_exact_rate,
            option_avg_similarity=option_avg_sim,
            stem_exact_match_rate=stem_exact_rate,
            stem_avg_similarity=stem_avg_sim,
            per_question=per_question_results,
            unmatched_gold_nums=sorted(list(gold_nums - pred_nums)),
            unmatched_pred_nums=sorted(list(pred_nums - gold_nums)),
        )
    
    def _eval_single_question(
        self,
        gold_q: Dict[str, Any],
        pred_q: Dict[str, Any],
    ) -> QuestionContentResult:
        """评测单个题目"""
        num = gold_q.get("num", 0)
        qtype = gold_q.get("type", "未知")
        
        # 题干
        gold_stem = self._extract_stem(gold_q)
        pred_stem = self._extract_stem(pred_q)
        stem_exact, stem_sim = self._compare_text(gold_stem, pred_stem)
        
        # 选项
        option_results = []
        gold_options = gold_q.get("options", {})
        pred_options = self._extract_options(pred_q)
        
        if gold_options:
            for key in ["A", "B", "C", "D"]:
                gold_opt = gold_options.get(key, "")
                pred_opt = pred_options.get(key, "")
                exact, sim = self._compare_text(gold_opt, pred_opt)
                option_results.append(OptionMatchResult(
                    option_key=key,
                    gold_text=gold_opt,
                    pred_text=pred_opt,
                    exact_match=exact,
                    similarity=sim,
                ))
        
        # 选项汇总
        if option_results:
            opt_exact_rate = sum(1 for o in option_results if o.exact_match) / len(option_results)
            opt_avg_sim = sum(o.similarity for o in option_results) / len(option_results)
        else:
            opt_exact_rate = 0.0
            opt_avg_sim = 0.0
        
        return QuestionContentResult(
            question_num=num,
            question_type=qtype,
            stem_gold=gold_stem,
            stem_pred=pred_stem,
            stem_exact_match=stem_exact,
            stem_similarity=stem_sim,
            option_results=option_results,
            option_exact_match_rate=opt_exact_rate,
            option_avg_similarity=opt_avg_sim,
        )
    
    def _extract_stem(self, q: Dict[str, Any]) -> str:
        """从题目中提取题干"""
        # 优先取 stem 字段
        if q.get("stem"):
            return str(q["stem"]).strip()
        # 否则取 content 字段，去掉选项部分
        content = q.get("content", "")
        if not content:
            return ""
        # 简单去掉选项部分（A. B. C. D. 开头的行）
        lines = content.split("\n")
        stem_lines = []
        for line in lines:
            if re.match(r'^\s*[A-Da-d][\.\、\)]', line):
                break
            stem_lines.append(line)
        return "\n".join(stem_lines).strip()
    
    def _extract_options(self, q: Dict[str, Any]) -> Dict[str, str]:
        """从题目中提取选项"""
        # 优先取 options 字段
        if q.get("options") and isinstance(q["options"], dict):
            return q["options"]
        
        # 否则从 content 中解析
        content = q.get("content", "")
        if not content:
            return {}
        
        options = {}
        # 匹配 A. xxx 或 A、xxx 或 A) xxx
        pattern = r'([A-Da-d])[\.\、\)]\s*(.+?)(?=(?:[A-Da-d][\.\、\)]|$))'
        for m in re.finditer(pattern, content, re.DOTALL):
            key = m.group(1).upper()
            val = m.group(2).strip()
            options[key] = val
        
        return options
    
    def _compare_text(self, gold: str, pred: str) -> Tuple[bool, float]:
        """
        比较两段文本
        
        Returns:
            (exact_match, similarity)
        """
        gold_norm = self._normalize_text(gold)
        pred_norm = self._normalize_text(pred)
        
        exact_match = (gold_norm == pred_norm)
        
        # 字符级相似度（SequenceMatcher）
        if not gold_norm and not pred_norm:
            similarity = 1.0
        elif not gold_norm or not pred_norm:
            similarity = 0.0
        else:
            similarity = SequenceMatcher(None, gold_norm, pred_norm).ratio()
        
        return exact_match, similarity
    
    def _normalize_text(self, text: str) -> str:
        """归一化文本（用于比较，容忍OCR噪声和格式差异）"""
        if not text:
            return ""
        t = str(text).strip()
        # 统一全角/半角空格
        t = re.sub(r'\s+', ' ', t)
        # 统一标点（去掉比较时影响匹配的标点）
        t = re.sub(r'[，,。\.；;：:\s！!？?（）()【】\[\]《》<>""\"\'\'、·…—~～]', '', t)
        # 数学符号归一化
        t = t.replace('√', 'sqrt').replace('∞', 'inf')
        t = t.replace('×', '*').replace('÷', '/').replace('²', '^2').replace('³', '^3')
        t = t.replace('≤', '<=').replace('≥', '>=').replace('≠', '!=')
        t = t.replace('π', 'pi').replace('∈', 'in').replace('⊆', 'subset')
        # 下标数字归一化（A₁→A1）
        sub_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
        t = t.translate(sub_map)
        # 统一大小写
        t = t.lower()
        return t
    
    def _parse_question_num(self, q: Dict[str, Any]) -> Optional[int]:
        """从题目中解析题号"""
        # 优先取 num 字段
        if q.get("num") is not None:
            try:
                return int(q["num"])
            except (ValueError, TypeError):
                pass
        
        # 否则从 content 中解析
        content = q.get("content", "")
        if not content:
            return None
        
        m = re.match(r'^\s*(\d+)\s*[\.、\)]', content)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                pass
        
        return None


# ============ 便捷函数 ============
def evaluate_content(
    gold_data: Dict[str, Any],
    predictions: List[Dict[str, Any]],
) -> ContentEvalResult:
    """
    评测结构化内容正确性
    
    Args:
        gold_data: 金标数据（需含 stem/options）
        predictions: 预测的题目列表
    
    Returns:
        ContentEvalResult
    """
    evaluator = ContentEvaluator()
    return evaluator.evaluate(gold_data, predictions)
