"""主入口文件"""
import uvicorn
from pathlib import Path
import os

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
    reload_env = str(os.getenv("UVICORN_RELOAD", "true")).strip().lower()
    reload_enabled = reload_env not in {"0", "false", "no"}
    uvicorn.run(
        "src.api.teacher_api:app",
        host="0.0.0.0",
        port=8000,
        reload=reload_enabled,
    )

