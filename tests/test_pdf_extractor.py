"""PDF 文本层 glyph 清洗回归测试"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_processing.pdf_extractor import PDFProcessor


class TestPDFGlyphNormalization:
    def setup_method(self):
        self.processor = PDFProcessor(ocr_enabled=False, vision_llm_client=None)

    def test_normalizes_common_private_use_math_glyphs(self):
        raw = "点\uf028 \uf029 2,1，m \uf024 \uf0ceR，若平面\uf062\uf05e平面\uf061，则△ABC"
        cleaned = self.processor._normalize_pdf_glyphs(raw)

        assert "(2,1)" in cleaned
        assert "∃ ∈R" in cleaned
        assert "β⊥平面α" in cleaned
        assert "△ABC" in cleaned
        assert "\uf028" not in cleaned
        assert "\uf0ce" not in cleaned

    def test_normalizes_vector_arrows_and_strips_unknown_pua(self):
        raw = "设方向向量为\uf072v，法向量为n\uf072，且AC\uf03d\uf075\uf075\uf075\uf072，\uf123"
        cleaned = self.processor._normalize_pdf_glyphs(raw)

        assert "v→" in cleaned
        assert "n→" in cleaned
        assert "AC=→" in cleaned
        assert "\uf123" not in cleaned


class TestNativeTextReconstruction:
    def setup_method(self):
        self.processor = PDFProcessor(ocr_enabled=False, vision_llm_client=None)

    def test_groups_stem_words_by_visual_row_and_restores_point_order(self):
        words = [
            (214.3, 369.4, 229.5, 385.5, "2,1", 0, 0, 0),
            (229.0, 366.3, 233.1, 386.2, "\uf029", 0, 0, 1),
            (54.0, 378.2, 61.9, 391.9, "1.", 0, 0, 2),
            (67.1, 373.6, 113.8, 392.5, "已知直线l", 0, 0, 3),
            (116.0, 366.3, 213.4, 392.5, "的斜率为2，经过点\uf028", 0, 0, 4),
            (235.0, 373.6, 281.7, 392.5, "，则直线l", 0, 0, 5),
            (283.9, 380.1, 336.4, 392.5, "的方程为（", 0, 0, 6),
            (357.4, 380.1, 367.8, 392.5, "）", 0, 0, 7),
        ]

        text = self.processor._reconstruct_page_text_from_words(words)

        assert "1. 已知直线l" in text
        assert "经过点(2,1)" in text
        assert "方程为（）" in text

    def test_restores_linear_equation_and_simple_stacked_fraction(self):
        words = [
            (54.0, 404.9, 64.2, 418.6, "A.", 0, 0, 0),
            (71.3, 402.5, 77.3, 418.2, "2", 0, 0, 1),
            (78.0, 402.3, 83.3, 418.2, "x", 0, 0, 2),
            (85.5, 402.6, 92.0, 417.1, "\uf02d", 0, 0, 3),
            (95.2, 402.3, 100.5, 418.2, "y", 0, 0, 4),
            (102.8, 402.6, 109.4, 417.1, "\uf02d", 0, 0, 5),
            (111.1, 402.5, 117.0, 418.2, "5", 0, 0, 6),
            (119.5, 402.6, 126.0, 417.1, "\uf03d", 0, 0, 7),
            (128.6, 402.5, 134.6, 418.2, "0", 0, 0, 8),
        ]
        line = self.processor._render_visual_line(self.processor._group_words_into_visual_lines(words)[0])
        assert line == "A. 2x-y-5=0"

        frac_words = [
            (110.7, 474.1, 118.7, 489.9, "C", 0, 0, 0),
            (121.2, 474.3, 124.5, 489.9, ":", 0, 0, 1),
            (127.9, 466.5, 133.2, 482.4, "x", 0, 0, 2),
            (133.8, 466.4, 137.3, 475.4, "2", 0, 0, 3),
            (130.0, 483.7, 135.9, 499.3, "4", 0, 0, 4),
            (141.4, 474.3, 147.9, 488.9, "\uf02b", 0, 0, 5),
            (152.1, 466.5, 157.4, 482.4, "y", 0, 0, 6),
            (158.2, 466.4, 161.6, 475.4, "2", 0, 0, 7),
            (154.0, 483.7, 160.0, 499.3, "3", 0, 0, 8),
            (166.5, 474.3, 173.1, 488.9, "\uf03d", 0, 0, 9),
            (174.5, 474.3, 180.4, 489.9, "1", 0, 0, 10),
        ]
        line = self.processor._render_visual_line(self.processor._group_words_into_visual_lines(frac_words)[0])
        assert "C:" in line
        assert "x^2/4" in line
        assert "+y^2/3=1" in line

    def test_postprocesses_conics_points_and_inline_options(self):
        line = "已知椭圆C:a^x22+b^y22=1(a>b>0)，点(A 2,0,0)，1F，直线AC 1//平面α"
        fixed = self.processor._postprocess_rendered_line(line)
        assert "x^2/a^2+y^2/b^2=1" in fixed
        assert "A(2,0,0)" in fixed
        assert "F_1" in fixed
        assert "AC_1∥平面α" in fixed

        option_line = self.processor._postprocess_rendered_line("A. 2x-y-5=0 B. 2x+y-5=0 C. 2x-y-3=0")
        assert option_line.splitlines() == ["A. 2x-y-5=0", "B. 2x+y-5=0", "C. 2x-y-3=0"]

    def test_postprocesses_interval_like_fraction(self):
        fixed = self.processor._postprocess_rendered_line("A. (((-^12 5,0)))")
        assert "(-12/5,0)" in fixed

    def test_postprocesses_radicals_and_contextual_fractions(self):
        fixed = self.processor._postprocess_rendered_line("B. ∃ m ∈R，使得圆心C到l的距离为3 2")
        assert fixed.endswith("距离为3√2")

        fixed = self.processor._postprocess_rendered_line("C. 3/14 14")
        assert fixed == "C. 3√14/14"

        fixed = self.processor._postprocess_rendered_line("（2）若直线AF_1的斜率为1 2，求AF_1+BF_2的长度；")
        assert "斜率为1/2" in fixed

    def test_postprocesses_pi_fraction_vectors_parallel_and_split_chinese(self):
        fixed = self.processor._postprocess_rendered_line("若二面角P-AC-B的大小为π 3，则BP的长度为（）")
        assert "π/3" in fixed

        fixed = self.processor._postprocess_rendered_line("椭圆C的离心率为1 2")
        assert fixed.endswith("离心率为1/2")

        fixed = self.processor._postprocess_rendered_line(
            "OP→=mOA^→+nOB^→+lOC^→，直线AC_1//平面α，若v^→=-(2,4,-6)，则直线/l/平面α，且PA ⊥平 面ABCD，求PA的 长"
        )
        assert "OA→" in fixed
        assert "OB→" in fixed
        assert "OC→" in fixed
        assert "v→=-(2,4,-6)" in fixed
        assert "AC_1∥平面α" in fixed
        assert "直线l∥平面α" in fixed
        assert "平面ABCD" in fixed
        assert "PA的长" in fixed

    def test_keeps_subquestions_on_separate_visual_lines(self):
        words = [
            (54.0, 251.3, 150.6, 267.1, "（1）证明：直线OC", 0, 0, 0),
            (154.4, 250.0, 208.0, 266.6, "\uf05e平面\uf061；", 0, 0, 1),
            (54.0, 280.1, 101.2, 293.9, "（2）已知", 0, 0, 2),
            (104.0, 278.6, 118.6, 294.4, "AP", 0, 0, 3),
            (121.7, 278.8, 128.2, 293.4, "\uf03d", 0, 0, 4),
            (130.5, 278.6, 133.8, 294.4, "t", 0, 0, 5),
        ]
        text = self.processor._reconstruct_page_text_from_words(words)
        assert text.splitlines() == ["（1）证明：直线OC ⊥平面α；", "（2）已知AP=t"]
