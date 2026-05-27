# 雀魂AI Vision Service 入口
# 使用方式：python server.py

import sys
from pathlib import Path

# 把项目根目录加入 sys.path，使 vision_service / game_engine 等包可被直接导入
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from vision_service.main import app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
