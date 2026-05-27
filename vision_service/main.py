# vision_service/main.py
# FastAPI 图像识别服务入口

import base64
import io
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel as PydanticBaseModel
from PIL import Image

from .detector import TileDetector
from .models import AnalyzeRequest, AnalyzeResponse
from game_engine.state import build_game_state
from llm_advisor.advisor import LLMAdvisor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== 全局组件 ====================

detector: TileDetector = None
advisor: LLMAdvisor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, advisor
    logger.info("正在加载 TileDetector...")
    detector = TileDetector()
    logger.info("正在初始化 LLMAdvisor...")
    advisor = LLMAdvisor()
    logger.info("服务启动完成")
    yield
    # 关闭连接
    if advisor:
        await advisor.close()
    logger.info("服务关闭")


app = FastAPI(
    title="雀魂AI Vision Service",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "chrome-extension://*",     # Chrome 扩展
        "http://localhost:*",       # 本地开发
        "http://127.0.0.1:*",      # 本地开发
    ],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ==================== 接口 ====================


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "detector_ready": detector is not None and detector.ready,
        "advisor_ready": advisor is not None
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    if not detector or not detector.ready:
        raise HTTPException(503, "模型未就绪，请检查 YOLO 权重文件")

    # 1. Base64 解码 -> PIL Image
    try:
        img_bytes = base64.b64decode(req.image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        raise HTTPException(400, f"图像解码失败: {e}")

    # 2. 目标检测
    detection = detector.detect(img_np)
    if not detection.hand_tiles:
        return AnalyzeResponse(error="未检测到手牌，请确认游戏画面可见")

    # 3. 构建游戏状态
    game_state = build_game_state(detection)

    # 4. LLM 推理
    advice = await advisor.get_advice(game_state)

    return AnalyzeResponse(
        hand_tiles=detection.hand_tiles,
        discard_pool=detection.discard_pool,
        shanten=game_state.shanten,
        effective_tiles=game_state.effective_tiles,
        advice=advice
    )


@app.post("/analyze/debug")
async def analyze_debug(req: AnalyzeRequest):
    """调试接口：返回原始检测结果（不调用 LLM）"""
    try:
        img_bytes = base64.b64decode(req.image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        raise HTTPException(400, f"图像解码失败: {e}")

    detection = detector.detect(img_np)
    return {
        "hand_tiles": detection.hand_tiles,
        "discard_pool": detection.discard_pool,
        "raw_boxes": detection.raw_boxes,
        "confidence_scores": detection.confidence_scores
    }


# ==================== LLM 动态配置 ====================


class LLMConfigRequest(PydanticBaseModel):
    backend: str = "ollama"               # ollama | openai | claude
    api_key: str = ""                      # openai-compatible 或 claude 的 key
    base_url: str = ""                     # openai-compatible 的 base URL
    model: str = ""                        # 模型名称
    ollama_url: str = ""                   # ollama 地址（可选覆盖）
    ollama_model: str = ""                 # ollama 模型名（可选覆盖）


@app.post("/config/llm")
async def update_llm_config(req: LLMConfigRequest):
    """
    接收前端发来的 LLM 配置，动态更新 advisor 参数。
    前端 popup 设置面板调用此接口保存自定义模型配置。
    注意：永不回显 API Key，仅返回是否已配置。
    """
    if advisor is None:
        raise HTTPException(503, "advisor 未初始化")

    advisor.backend = req.backend

    key_configured = False
    if req.backend == 'openai':
        if req.api_key:
            advisor.openai_api_key = req.api_key
        key_configured = bool(advisor.openai_api_key)
        if req.base_url:
            advisor.openai_base_url = req.base_url
        if req.model:
            advisor.openai_model = req.model
    elif req.backend == 'ollama':
        if req.ollama_url:
            advisor.ollama_url = req.ollama_url
        if req.ollama_model:
            advisor.ollama_model = req.ollama_model
        key_configured = True  # Ollama 不需要 key
    elif req.backend == 'claude':
        if req.api_key:
            advisor.claude_api_key = req.api_key
        key_configured = bool(advisor.claude_api_key)

    logger.info(f"LLM 配置已更新: backend={req.backend}, model={req.model or advisor.openai_model}")
    return {"ok": True, "backend": req.backend, "key_configured": key_configured}
