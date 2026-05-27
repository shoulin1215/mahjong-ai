# vision_service/models.py
# Pydantic 数据模型

from typing import Optional
from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    image: str           # base64 编码的 JPEG 截图
    tab_id: Optional[int] = None


class AdviceResult(BaseModel):
    discard: Optional[str] = None        # 推荐打出的牌，如 "5m"
    action: Optional[str] = None         # 特殊操作：chi/pon/kan/tsumo/ron
    reason: str = ""                     # LLM 给出的理由
    confidence: float = 0.0              # 置信度 0-1
    alternative: Optional[str] = None   # 备选出牌


class AnalyzeResponse(BaseModel):
    hand_tiles: list[str] = []
    discard_pool: list[list[str]] = [[], [], [], []]
    shanten: Optional[int] = None
    effective_tiles: list[str] = []
    advice: Optional[AdviceResult] = None
    error: Optional[str] = None
