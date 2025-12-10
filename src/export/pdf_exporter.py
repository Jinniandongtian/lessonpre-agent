"""PDF导出模块"""
from pathlib import Path
from typing import List
from ..data_models.question import Question
from ..utils.config import Config


def _resolve_image_path(img_path: str) -> str:
    """解析图片路径，返回可用的文件路径或URL"""
    from pathlib import Path
    
    # 如果是URL路径（以/images/开头）
    if img_path.startswith('/images/'):
        filename = Path(img_path).name
        full_path = Config.IMAGES_DIR / filename
        if full_path.exists():
            return str(full_path)
    
    # 如果是相对路径
    elif not Path(img_path).is_absolute():
        # 尝试相对于images目录
        full_path = Config.IMAGES_DIR / img_path
        if full_path.exists():
            return str(full_path)
        # 尝试相对于data目录
        full_path = Config.DATA_DIR / img_path
        if full_path.exists():
            return str(full_path)
    
    # 如果是绝对路径且存在
    elif Path(img_path).exists():
        return img_path
    
    return None


def handout_markdown_to_pdf(md_text: str, output_path: str) -> None:
    """
    把讲义 Markdown 转为 PDF
    
    需要安装：pip install markdown pdfkit 或使用其他库
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import markdown
        from markdown.extensions import Extension
        from markdown.preprocessors import Preprocessor
        import re
        
        # 自定义扩展：处理图片路径
        class ImagePathExtension(Extension):
            def extendMarkdown(self, md):
                md.preprocessors.register(ImagePathProcessor(md), 'image_path', 175)
        
        class ImagePathProcessor(Preprocessor):
            def run(self, lines):
                new_lines = []
                for line in lines:
                    # 查找Markdown图片语法 ![alt](path)
                    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
                    def replace_img(match):
                        alt = match.group(1)
                        path = match.group(2)
                        resolved = _resolve_image_path(path)
                        if resolved:
                            return f'<img src="{resolved}" alt="{alt}" style="max-width: 100%; height: auto;" />'
                        return match.group(0)
                    line = re.sub(img_pattern, replace_img, line)
                    new_lines.append(line)
                return new_lines
        
        # Markdown转HTML
        html = markdown.markdown(md_text, extensions=['extra', 'codehilite', ImagePathExtension()])
        
        # HTML转PDF
        try:
            import pdfkit
            pdfkit.from_string(html, str(output_path))
        except ImportError:
            try:
                from weasyprint import HTML, CSS
                HTML(string=html).write_pdf(output_path)
            except ImportError:
                # 保存为HTML
                output_path = Path(output_path).with_suffix('.html')
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"<html><head><meta charset='utf-8'></head><body>{html}</body></html>")
                print(f"已保存为HTML文件（需要安装pdfkit或weasyprint才能导出PDF）: {output_path}")
    except ImportError:
        # 如果没有安装pdfkit，使用替代方案
        try:
            from markdown import markdown
            from weasyprint import HTML, CSS
            
            html = markdown(md_text, extensions=['extra', 'codehilite'])
            HTML(string=html).write_pdf(output_path)
        except ImportError:
            # 如果都没有，保存为HTML
            output_path = output_path.with_suffix('.html')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"<html><head><meta charset='utf-8'></head><body>{html}</body></html>")
            print(f"已保存为HTML文件（需要安装pdfkit或weasyprint才能导出PDF）: {output_path}")


def assignment_to_pdf(questions: List[Question], output_path: str, include_answers: bool = False) -> None:
    """
    把作业题目渲染为试卷样式 PDF
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 构建HTML内容
    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>",
        "body { font-family: 'SimSun', serif; padding: 20px; }",
        "h1 { text-align: center; }",
        ".question { margin-bottom: 30px; page-break-inside: avoid; }",
        ".answer { margin-top: 10px; padding: 10px; background: #f0f0f0; }",
        ".question-image { max-width: 100%; height: auto; margin: 10px 0; }",
        ".image-container { text-align: center; margin: 15px 0; }",
        "</style>",
        "</head><body>",
        "<h1>作业</h1>",
    ]
    
    for i, question in enumerate(questions, 1):
        html_parts.append(f'<div class="question">')
        html_parts.append(f'<h3>第{i}题</h3>')
        html_parts.append(f'<p>{question.content}</p>')
        
        # 添加图片
        if question.images:
            html_parts.append('<div class="image-container">')
            for img_path in question.images:
                # 处理图片路径
                img_url = _resolve_image_path(img_path)
                if img_url:
                    html_parts.append(f'<img src="{img_url}" class="question-image" alt="题目图片" />')
            html_parts.append('</div>')
        
        if include_answers:
            if question.answer:
                html_parts.append(f'<div class="answer"><strong>答案：</strong>{question.answer}</div>')
            if question.solution:
                html_parts.append(f'<div class="answer"><strong>解析：</strong>{question.solution}</div>')
        
        html_parts.append('</div>')
    
    html_parts.append('</body></html>')
    html = ''.join(html_parts)
    
    # 转换为PDF
    try:
        import pdfkit
        pdfkit.from_string(html, str(output_path))
    except ImportError:
        try:
            from weasyprint import HTML
            HTML(string=html).write_pdf(output_path)
        except ImportError:
            # 保存为HTML
            output_path = output_path.with_suffix('.html')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"已保存为HTML文件（需要安装pdfkit或weasyprint才能导出PDF）: {output_path}")

