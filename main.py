"""主入口文件"""
import uvicorn
from pathlib import Path

# 确保在导入其他模块前加载 .env 文件
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

if __name__ == "__main__":
    uvicorn.run(
        "src.api.teacher_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

