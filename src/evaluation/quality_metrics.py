"""
题目提取质量评测指标（不需要金标数据）

A1. JSON 解析成功率
A2. Schema 合格率
A3. 结构完整率（按题型）
A4. 题号一致性
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ============ Schema 定义 ============
QUESTION_SCHEMA = {
    "required_fields": ["content"],  # 最基础的必填字段
    "recommended_fields": ["question_type", "knowledge_points", "difficulty"],
    "optional_fields": ["id", "index", "source_meta", "answer", "solution"],
}

# 题型枚举（用于校验）
VALID_QUESTION_TYPES = {
    "选择题", "多选题", "单选题",  # 选择类
    "填空题",                        # 填空类
    "解答题",                        # 解答类
    "未知题型",                      # 兜底
}


@dataclass
class MetricResult:
    """单个指标的评测结果"""
    name: str
    score: float  # 0~1
    total: int
    passed: int
    failed: int
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """完整的评测报告"""
    total_questions: int
    metrics: Dict[str, MetricResult]
    summary: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "summary": self.summary,
        }


class QualityEvaluator:
    """题目提取质量评测器"""
    
    
    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: 可选，LLM 客户端，用于填空题完整性判断
        """
        self.llm_client = llm_client
    
    def evaluate(self, questions: List[Dict[str, Any]], raw_llm_responses: Optional[List[str]] = None) -> EvaluationReport:
        """
        评测题目列表
        
        Args:
            questions: 已解析的题目列表（dict 格式）
            raw_llm_responses: 可选，LLM 原始响应（用于计算 JSON 解析成功率）
        
        Returns:
            EvaluationReport
        """
        metrics = {}
        
        # A1. JSON 解析成功率（如果提供了原始响应）
        if raw_llm_responses:
            metrics["a1_json_parse"] = self._eval_json_parse_rate(raw_llm_responses)
        
        # A2. Schema 合格率
        metrics["a2_schema_compliance"] = self._eval_schema_compliance(questions)
        
        # A3. 结构完整率
        metrics["a3_structure_completeness"] = self._eval_structure_completeness(questions)
        
        # A4. 题号一致性
        metrics["a4_number_consistency"] = self._eval_number_consistency(questions)
        
        # 汇总
        summary = {}
        for k, v in metrics.items():
            summary[k] = v.score
        summary["overall"] = sum(summary.values()) / len(summary) if summary else 0.0
        
        return EvaluationReport(
            total_questions=len(questions),
            metrics=metrics,
            summary=summary,
        )
    
    # ============ A1: JSON 解析成功率 ============
    def _eval_json_parse_rate(self, raw_responses: List[str]) -> MetricResult:
        """评测 LLM 原始响应的 JSON 解析成功率"""
        total = len(raw_responses)
        passed = 0
        failed_samples = []
        
        for i, resp in enumerate(raw_responses):
            success, _ = self._try_parse_json(resp)
            if success:
                passed += 1
            else:
                if len(failed_samples) < 5:
                    failed_samples.append({"index": i, "preview": resp[:200]})
        
        score = passed / total if total > 0 else 0.0
        return MetricResult(
            name="JSON解析成功率",
            score=score,
            total=total,
            passed=passed,
            failed=total - passed,
            details={"failed_samples": failed_samples},
        )
    
    def _try_parse_json(self, text: str) -> Tuple[bool, Any]:
        """尝试解析 JSON（支持 markdown 代码块）"""
        if not text:
            return False, None
        
        # 尝试提取 markdown 代码块
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\[\s*\{[\s\S]*\}\s*\]',
        ]
        
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                try:
                    data = json.loads(m.group(1) if '```' in pat else m.group(0))
                    return True, data
                except json.JSONDecodeError:
                    continue
        
        # 直接尝试解析
        try:
            data = json.loads(text)
            return True, data
        except json.JSONDecodeError:
            return False, None
    
    # ============ A2: Schema 合格率 ============
    def _eval_schema_compliance(self, questions: List[Dict[str, Any]]) -> MetricResult:
        """评测题目是否符合定义的 schema"""
        total = len(questions)
        passed = 0
        violations = []
        
        for i, q in enumerate(questions):
            is_valid, errors = self._check_schema(q)
            if is_valid:
                passed += 1
            else:
                if len(violations) < 10:
                    # schema有错误的题目
                    violations.append({
                        "index": i,
                        "errors": errors,
                        "content_preview": str(q.get("content", ""))[:100],
                    })
        
        score = passed / total if total > 0 else 0.0
        
        # 分项统计
        field_stats = self._count_field_presence(questions)
        
        return MetricResult(
            name="Schema合格率",
            score=score,
            total=total,
            passed=passed,
            failed=total - passed,
            details={
                "violations": violations,
                "field_presence": field_stats,
            },
        )
    
    def _check_schema(self, q: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """检查单个题目是否符合 schema"""
        errors = []
        
        # 必填字段
        for field in QUESTION_SCHEMA["required_fields"]:
            if field not in q or not q[field]:
                errors.append(f"缺少必填字段: {field}")
        
        # content 长度检查
        content = q.get("content", "")
        if len(content) < 4:
            errors.append(f"content 过短 ({len(content)} 字符)")
        
        # question_type 校验
        qt = q.get("question_type")
        if qt and qt not in VALID_QUESTION_TYPES:
            errors.append(f"未知题型: {qt}")
        
        # difficulty 范围校验
        diff = q.get("difficulty")
        if diff is not None:
            try:
                d = int(diff)
                if d < 1 or d > 5:
                    errors.append(f"difficulty 超出范围 [1,5]: {diff}")
            except (ValueError, TypeError):
                errors.append(f"difficulty 非整数: {diff}")
        
        # knowledge_points 类型校验
        kp = q.get("knowledge_points")
        if kp is not None and not isinstance(kp, list):
            errors.append(f"knowledge_points 应为列表: {type(kp)}")
        
        return len(errors) == 0, errors
    
    def _count_field_presence(self, questions: List[Dict[str, Any]]) -> Dict[str, float]:
        """统计各字段存在率"""
        if not questions:
            return {}
        # 所有字段的集合(包括必填、推荐、可选，如id、content、question_type等)
        all_fields = (
            QUESTION_SCHEMA["required_fields"] +
            QUESTION_SCHEMA["recommended_fields"] +
            QUESTION_SCHEMA["optional_fields"]
        )
        # 如果f字段存在，则初始化为0
        counts = {f: 0 for f in all_fields}
        # 遍历每道题
        for q in questions:
            # 遍历每道题的每个字段
            for f in all_fields:
                # 如果字段存在且不为空
                if f in q and q[f]:
                    counts[f] += 1
        # 计算存在率 → 有效数 / 总题数（转为浮点数，体现“比例”而非绝对数）
        return {f: counts[f] / len(questions) for f in all_fields}
    
    # ============ A3: 结构完整率 ============
    def _eval_structure_completeness(self, questions: List[Dict[str, Any]]) -> MetricResult:
        """按题型评测结构完整性"""
        total = len(questions)
        
        # 分题型统计
        choice_stats = {"total": 0, "complete": 0, "issues": []}
        blank_stats = {"total": 0, "complete": 0, "issues": []}
        answer_stats = {"total": 0, "complete": 0, "issues": []}
        other_stats = {"total": 0, "complete": 0}
        # enumerate返回一个(索引，值)的迭代对象
        for i, q in enumerate(questions):
            content = q.get("content", "")
            qt = q.get("question_type", "未知题型")
            
            if qt == "选择题":
                choice_stats["total"] += 1
                has_options, option_count = self._check_choice_options(content)
                if has_options and option_count >= 4:
                    choice_stats["complete"] += 1
                else:
                    if len(choice_stats["issues"]) < 5:
                        choice_stats["issues"].append({
                            "index": i,
                            "option_count": option_count,
                            "preview": content[:100],
                        })
            
            elif qt == "填空题":
                blank_stats["total"] += 1
                has_blank = self._check_blank_markers(content)
                if has_blank:
                    blank_stats["complete"] += 1
                else:
                    if len(blank_stats["issues"]) < 5:
                        blank_stats["issues"].append({
                            "index": i,
                            "preview": content[:100],
                        })
            
            elif qt == "解答题":
                answer_stats["total"] += 1
                has_subq = self._check_subquestions(content)
                # 解答题可以没有小问，只要内容足够长就算完整
                if has_subq or len(content) > 50:
                    answer_stats["complete"] += 1
                else:
                    if len(answer_stats["issues"]) < 5:
                        answer_stats["issues"].append({
                            "index": i,
                            "preview": content[:100],
                        })
            
            else:
                other_stats["total"] += 1
                if len(content) > 30:
                    other_stats["complete"] += 1
        
        # 计算各类完整率
        type_scores = {}
        if choice_stats["total"] > 0:
            type_scores["选择题"] = choice_stats["complete"] / choice_stats["total"]
        if blank_stats["total"] > 0:
            type_scores["填空题"] = blank_stats["complete"] / blank_stats["total"]
        if answer_stats["total"] > 0:
            type_scores["解答题"] = answer_stats["complete"] / answer_stats["total"]
        if other_stats["total"] > 0:
            type_scores["其他"] = other_stats["complete"] / other_stats["total"]
        
        # 总体完整率
        total_complete = (
            choice_stats["complete"] + blank_stats["complete"] +
            answer_stats["complete"] + other_stats["complete"]
        )
        score = total_complete / total if total > 0 else 0.0
        
        return MetricResult(
            name="结构完整率",
            score=score,
            total=total,
            passed=total_complete,
            failed=total - total_complete,
            details={
                "by_type": {
                    "选择题": {
                        "total": choice_stats["total"],
                        "complete": choice_stats["complete"],
                        "rate": type_scores.get("选择题", 0),
                        "issues": choice_stats["issues"],
                    },
                    "填空题": {
                        "total": blank_stats["total"],
                        "complete": blank_stats["complete"],
                        "rate": type_scores.get("填空题", 0),
                        "issues": blank_stats["issues"],
                    },
                    "解答题": {
                        "total": answer_stats["total"],
                        "complete": answer_stats["complete"],
                        "rate": type_scores.get("解答题", 0),
                        "issues": answer_stats["issues"],
                    },
                    "其他": {
                        "total": other_stats["total"],
                        "complete": other_stats["complete"],
                        "rate": type_scores.get("其他", 0),
                    },
                },
            },
        )
    
    def _check_choice_options(self, content: str) -> Tuple[bool, int]:
        """检查选择题是否有 ABCD 选项"""
        # 匹配 A. B. C. D. 或 A、B、C、D 或 A B C D 等
        patterns = [
            r'[A-D]\s*[.、．:：)\]】]',  # A. A、 A）
            r'[（\(]\s*[A-D]\s*[）\)]',  # (A) （A）
        ]
        
        found_options = set()
        for pat in patterns:
            matches = re.findall(pat, content, re.IGNORECASE)
            for m in matches:
                for c in "ABCD":
                    if c in m.upper():
                        found_options.add(c)
        
        return len(found_options) >= 2, len(found_options)
    
    def _check_blank_markers(self, content: str) -> bool:
        """使用 LLM 判断填空题是否完整（是否有明确的填空位置）"""
        if not self.llm_client:
            # 没有 LLM 时默认认为完整
            return True
        
        prompt = f"""请判断以下填空题是否完整。

        完整的填空题应该：
        1. 有明确的填空位置（如 ____、（ ）、空格等）
        2. 或者题目明确要求填写某个结果（如"方程为"、"取值范围是"、"的值为"等结尾）

        填空题内容：
        {content[:500]}

        请只回复 "YES" 或 "NO"：
        - YES: 填空题完整，有明确的填空位置或要求
        - NO: 填空题不完整，缺少填空位置或要求不明确"""
        
        try:
            response = self.llm_client.generate(prompt)
            response = response.strip().upper()
            return "YES" in response
        except Exception as e:
            print(f"填空题 LLM 判断失败: {e}")
            return True  # 失败时默认完整
    
    def _check_subquestions(self, content: str) -> bool:
        """检查解答题是否有小问"""
        # 小问标记：(1) (2) ① ② 等
        subq_patterns = [
            r'[\(（]\s*[1-9]\s*[\)）]',   # (1) （1）
            r'[①②③④⑤⑥⑦⑧⑨⑩]',
            r'\b[1-9]\s*[）\)]\s*',        # 1) 2)
        ]
        
        for pat in subq_patterns:
            if re.search(pat, content):
                return True
        return False
    
    # ============ A4: 题号一致性 ============
    def _eval_number_consistency(self, questions: List[Dict[str, Any]]) -> MetricResult:
        """评测题号的一致性（重复/跳号/非法）"""
        total = len(questions)
        
        numbers = []
        invalid_numbers = []
        
        for i, q in enumerate(questions):
            content = q.get("content", "")
            num = self._extract_question_number(content)
            
            if num is None:
                invalid_numbers.append({
                    "index": i,
                    "reason": "未检测到题号",
                    "preview": content[:80],
                })
                numbers.append(None)
            elif not num.isdigit():
                invalid_numbers.append({
                    "index": i,
                    "reason": f"非法题号: {num}",
                    "preview": content[:80],
                })
                numbers.append(num)
            else:
                numbers.append(int(num))
        
        # 统计
        valid_nums = [n for n in numbers if isinstance(n, int)]
        
        # 重复题号
        from collections import Counter
        num_counts = Counter(valid_nums)
        # 如果v大于1，则说明有重复
        duplicates = {k: v for k, v in num_counts.items() if v > 1}
        duplicate_rate = len(duplicates) / len(set(valid_nums)) if valid_nums else 0.0
        
        # 跳号检测
        if valid_nums:
            sorted_nums = sorted(set(valid_nums))
            # 生成从第一个元素到最后一个元素的列表，-1是最后一个元素，
            expected = list(range(sorted_nums[0], sorted_nums[-1] + 1))
            missing = set(expected) - set(sorted_nums)
            gap_rate = len(missing) / len(expected) if expected else 0.0
        else:
            missing = set()
            gap_rate = 0.0
        
        # 非法题号率
        invalid_rate = len(invalid_numbers) / total if total > 0 else 0.0
        
        # 综合得分：1 - (重复率 + 跳号率 + 非法率) / 3
        score = max(0.0, 1.0 - (duplicate_rate + gap_rate + invalid_rate) / 3)
        
        passed = total - len(invalid_numbers) - len(duplicates)
        
        return MetricResult(
            name="题号一致性",
            score=score,
            total=total,
            passed=max(0, passed),
            failed=total - max(0, passed),
            details={
                "duplicates": duplicates,
                "duplicate_rate": duplicate_rate,
                "missing_numbers": list(missing)[:20],  # 最多列 20 个
                "gap_rate": gap_rate,
                "invalid_numbers": invalid_numbers[:10],  # 最多列 10 个
                "invalid_rate": invalid_rate,
            },
        )
    
    def _extract_question_number(self, content: str) -> Optional[str]:
        """从题目内容中提取题号"""
        if not content:
            return None
        
        # 取前 50 字符
        head = content[:50]
        
        # 常见题号格式
        patterns = [
            r'^[\s\n]*(\d{1,2})\s*[.、．:：\)\]】]',  # 1. 2、 3）
            r'^[\s\n]*[（\(]\s*(\d{1,2})\s*[）\)]',   # (1) （1）
            r'^[\s\n]*第\s*(\d{1,2})\s*题',          # 第1题
        ]
        
        for pat in patterns:
            m = re.match(pat, head)
            if m:
                return m.group(1)
        
        return None


# ============ 便捷函数 ============
def evaluate_questions(
    questions: List[Dict[str, Any]],
    raw_llm_responses: Optional[List[str]] = None,
    llm_client=None,
) -> Dict[str, Any]:
    """
    评测题目列表，返回评测报告（dict 格式）
    
    Args:
        questions: 题目列表
        raw_llm_responses: LLM 原始响应（用于 A1 指标）
        llm_client: LLM 客户端（用于填空题完整性判断）
    """
    evaluator = QualityEvaluator(llm_client=llm_client)
    report = evaluator.evaluate(questions, raw_llm_responses)
    return report.to_dict()


def print_evaluation_report(report: Dict[str, Any]):
    """打印评测报告"""
    print("\n" + "=" * 60)
    print("题目提取质量评测报告")
    print("=" * 60)
    print(f"总题目数: {report['total_questions']}")
    print()
    
    for key, metric in report["metrics"].items():
        print(f"【{metric['name']}】")
        print(f"  得分: {metric['score']:.2%}")
        print(f"  通过: {metric['passed']}/{metric['total']}")
        
        # 打印关键细节
        details = metric.get("details", {})
        if key == "a2_schema_compliance" and "field_presence" in details:
            print("  字段存在率:")
            for f, rate in details["field_presence"].items():
                print(f"    - {f}: {rate:.1%}")
        
        if key == "a3_structure_completeness" and "by_type" in details:
            print("  按题型:")
            for qt, stats in details["by_type"].items():
                if stats["total"] > 0:
                    print(f"    - {qt}: {stats['complete']}/{stats['total']} ({stats['rate']:.1%})")
        
        if key == "a4_number_consistency":
            if details.get("duplicates"):
                print(f"  重复题号: {details['duplicates']}")
            if details.get("missing_numbers"):
                print(f"  缺失题号: {details['missing_numbers'][:10]}")
        
        print()
    
    print("-" * 60)
    print("汇总得分:")
    for k, v in report["summary"].items():
        print(f"  {k}: {v:.2%}")
    print("=" * 60)
