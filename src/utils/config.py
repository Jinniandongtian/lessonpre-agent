"""配置管理"""
import os
import shutil
from pathlib import Path
from typing import Optional

# 加载 .env 文件
def load_env_file():
    """加载项目根目录的 .env 文件"""
    try:
        from dotenv import load_dotenv
        # 获取项目根目录（src/utils/config.py 的父目录的父目录）
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
    except ImportError:
        # 如果没有安装 python-dotenv，跳过
        pass

# 在模块加载时自动加载 .env 文件
load_env_file()


class Config:
    """应用配置"""
    
    # 数据存储路径
    DATA_DIR = Path("data")
    VECTOR_DB_PATH = DATA_DIR / "vector_db"
    PDF_STORAGE_PATH = DATA_DIR / "pdfs"
    
    # LLM配置
    LLM_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
    SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    SILICONFLOW_MODEL = os.getenv("SILICONFLOW_MODEL", "deepseek-chat")
    
    # 导出配置
    EXPORT_DIR = DATA_DIR / "exports"
    
    # 上传文件配置
    UPLOAD_DIR = DATA_DIR / "uploads"
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
    
    # 图片存储配置
    IMAGES_DIR = DATA_DIR / "images"
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg"}
    MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB per image
    
    @classmethod
    def init_directories(cls):
        """初始化必要的目录"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.EXPORT_DIR.mkdir(exist_ok=True)
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        cls.IMAGES_DIR.mkdir(exist_ok=True)
        cls.VECTOR_DB_PATH.mkdir(exist_ok=True)
        cls.PDF_STORAGE_PATH.mkdir(exist_ok=True)

    @classmethod
    def check_system_dependencies(cls):
        missing = []

        if shutil.which("tesseract") is None:
            missing.append("tesseract")

        poppler_ok = any(shutil.which(x) is not None for x in ("pdftoppm", "pdfinfo"))
        if not poppler_ok:
            missing.append("poppler")

        weasyprint_ok = True
        try:
            from weasyprint import HTML  # noqa: F401
        except Exception:
            weasyprint_ok = False

        pdfkit_ok = True
        try:
            import pdfkit  # noqa: F401
            if shutil.which("wkhtmltopdf") is None:
                pdfkit_ok = False
        except Exception:
            pdfkit_ok = False

        if missing:
            print(f"⚠ 系统依赖缺失：{', '.join(missing)}")
            if "tesseract" in missing:
                print("  - OCR 需要 tesseract：macOS 可用 'brew install tesseract tesseract-lang'；Ubuntu 可用 'sudo apt-get install tesseract-ocr'")
            if "poppler" in missing:
                print("  - pdf2image 需要 poppler：macOS 可用 'brew install poppler'；Ubuntu 可用 'sudo apt-get install poppler-utils'")

        if not weasyprint_ok and not pdfkit_ok:
            print("⚠ PDF 导出依赖未就绪：pdfkit(wkhtmltopdf) 与 weasyprint 均不可用，将降级为导出 HTML")
            print("  - pdfkit: 需要 pip 安装 pdfkit 且系统安装 wkhtmltopdf")
            print("  - weasyprint: 需要 pip 安装 weasyprint 且系统安装 cairo/pango 等依赖")

