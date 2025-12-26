from typing import List, Dict, Any, Optional
import re
class ExamMetaExtractor:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def extract(self, text: str) -> Dict[str, Any]:
        """
        返回：
        {
          "meta": {...},
          "confidence": {...},
          "evidence": {...}
        }
        """
        meta, conf, evidence = {}, {}, {}

        # 1) 取前若干行当“标题候选”
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        head = "\n".join(lines[:30])
        evidence["title_candidates"] = lines[:10]

        # 2) year
        m = re.search(r'(20\d{2})\s*年', head)
        if not m:
            m = re.search(r'(20\d{2})\s*[-—–~至一]\s*(20\d{2})\s*学年', head)
        if not m:
            m = re.search(r'(20\d{2})\s*[-—–~]\s*(20\d{2})', head)
        if m:
            try:
                meta["year"] = int(m.group(1))
                conf["year"] = 0.95
            except Exception:
                conf["year"] = 0.0
        else:
            conf["year"] = 0.0

        # 3) grade（可扩展初中/小学）
        grade_patterns = ["高一", "高二", "高三", "初一", "初二", "初三"]
        for g in grade_patterns:
            if g in head:
                meta["grade"] = g
                conf["grade"] = 0.9
                break
        conf.setdefault("grade", 0.0)

        # 4) source_type
        st_map = ["期中", "期末", "月考", "模拟", "一模", "二模", "三模", "联考"]
        for st in st_map:
            if st in head:
                meta["source_type"] = st
                conf["source_type"] = 0.85
                break
        conf.setdefault("source_type", 0.0)

        # 5) region（先抓“XX市/省/区”这类）
        # 简化：抓“北京市/上海市/…省/自治区/区/县”等
        m = re.search(r'([\u4e00-\u9fa5]{2,8}(市|省|自治区|区|县))', head)
        if m:
            meta["region"] = m.group(1)
            conf["region"] = 0.75  # 规则命中但可能误抓“某某学校”，所以别给满
        else:
            conf["region"] = 0.0

        # 6) exam_name：尽量取包含 year/region/source_type 的那一行
        exam_line = None
        keywords = ("考试", "试卷", "试题", "检测", "测验", "联考", "月考", "期中", "期末")
        grade_hint = meta.get("grade")
        st_hint = meta.get("source_type")
        year_hint = str(meta.get("year")) if meta.get("year") else None

        best_score = -1
        for ln in lines[:25]:
            if not any(k in ln for k in keywords):
                continue
            if len(ln) > 90:
                continue

            score = 0
            if grade_hint and grade_hint in ln:
                score += 2
            if st_hint and st_hint in ln:
                score += 2
            if year_hint and year_hint in ln:
                score += 2
            if "学年" in ln:
                score += 1
            if "数学" in ln:
                score += 1

            if score > best_score:
                best_score = score
                exam_line = ln

        if exam_line:
            cleaned = re.sub(r"[,，]\s*msu.*$", "", exam_line, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            meta["exam_name"] = cleaned
            conf["exam_name"] = 0.7 if best_score >= 2 else 0.6
            evidence["exam_name_line"] = cleaned
        else:
            conf["exam_name"] = 0.0

        # 7) LLM兜底：当关键字段缺失/置信度低时才调用（省钱）
        need_llm = any(conf.get(k, 0.0) < 0.6 for k in ["region", "exam_name", "year", "grade"])
        if self.llm_client and need_llm:
            llm_meta = self._extract_with_llm(head, meta)
            # 合并：只补空缺，或提升低置信字段
            for k, v in llm_meta.get("meta", {}).items():
                if (k not in meta) or conf.get(k, 0.0) < llm_meta.get("confidence", {}).get(k, 0.0):
                    meta[k] = v
                    conf[k] = llm_meta.get("confidence", {}).get(k, conf.get(k, 0.0))
            evidence["llm_used"] = True
        else:
            evidence["llm_used"] = False

        return {"meta": meta, "confidence": conf, "evidence": evidence}

    def _extract_with_llm(self, text_head: str, meta_hint: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""你是“试卷元数据抽取器”。请从下面文本中抽取字段并返回严格 JSON。

文本（来自试卷前部）：
{text_head}

已抽取到的候选（可能有误，仅供参考）：
{meta_hint}

需要返回字段：
- region（地区，如“北京市”）
- year（年份整数）
- grade（如“高一/初二”）
- exam_name（考试名称，尽量完整）
- source_type（如“期中/期末/一模/二模/月考”等）

请输出：
{{
  "meta": {{...}},
  "confidence": {{"region":0-1, "year":0-1, "grade":0-1, "exam_name":0-1, "source_type":0-1}}
}}

只输出 JSON，不要解释。
"""
        resp = self.llm_client.generate(prompt)
        # 复用你现有的“容错 JSON 解析思路”即可（你在题目提取里已经写了 _parse_llm_json 的容错逻辑 :contentReference[oaicite:6]{index=6}）
        import json
        m = re.search(r'\{[\s\S]*\}', resp)
        if not m:
            return {"meta": {}, "confidence": {}}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {"meta": {}, "confidence": {}}
