# vision_service/recognizer.py
# 识别流水线编排
#
# 将检测器和分类器组合为完整的识别流程：
# 截图 -> YOLO检测(位置) -> 裁剪牌面 -> CNN分类 -> 结构化输出
#
# 从 mahjong-ai 迁移，适配 quehun 的模块结构。
# MahjongHand 作为 CV 识别的最终输出，也是 game_engine 和 LLM 的输入。

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class MahjongHand:
    """结构化麻将手牌数据

    这是 CV 识别的最终输出，也是 game_engine 和 LLM 的输入。
    """
    # 手牌 (13或14张)
    hand: List[str] = field(default_factory=list)
    # 每张牌的置信度
    hand_confidences: List[float] = field(default_factory=list)

    # 副露 (吃/碰/杠) - 暂不支持自动识别，预留字段
    melds: List[Dict[str, Any]] = field(default_factory=list)

    # 牌河 - 四家弃牌
    river: List[str] = field(default_factory=list)
    discard_pool: List[List[str]] = field(
        default_factory=lambda: [[], [], [], []]
    )  # 四家弃牌 [自家, 右家, 对家, 左家]

    # 场况信息 (部分需要从UI其他区域识别)
    dora_indicator: str = ""           # 宝牌指示牌
    round_wind: str = "1z"            # 场风（1z=东, 2z=南）
    seat_wind: str = "1z"             # 自风
    round_number: int = 1             # 局目
    honba_sticks: int = 0             # 本场棒
    riichi_sticks: int = 0            # 立直棒
    scores: List[int] = field(
        default_factory=lambda: [25000, 25000, 25000, 25000]
    )

    # 元数据
    timestamp: float = 0.0
    recognition_time_ms: float = 0.0

    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=indent)

    def to_dict(self) -> dict:
        """转为字典"""
        return asdict(self)

    @property
    def tile_count(self) -> int:
        """手牌总数"""
        return len(self.hand)

    @property
    def is_valid(self) -> bool:
        """基本合法性检查"""
        if not self.hand:
            return False
        if not (13 <= len(self.hand) <= 14):
            logger.warning(f"手牌数量异常: {len(self.hand)}")
        return True


@dataclass
class RecognizerConfig:
    """识别流水线配置"""
    use_dummy: bool = False       # 是否使用虚拟模型（开发测试）
    sort_hand: bool = True        # 手牌是否排序（按花色+数字）


class MahjongRecognizer:
    """麻将牌型识别流水线

    编排 检测器 + 分类器的完整流程，
    输出结构化的 MahjongHand 数据。
    """

    def __init__(
        self,
        detector=None,
        classifier=None,
        config: Optional[RecognizerConfig] = None,
    ):
        self.config = config or RecognizerConfig()

        if self.config.use_dummy or detector is None:
            from .detector import TileDetector
            # 无模型时使用 detector 自带的 mock 模式
            self.detector = TileDetector()  # model 不存在会自动 mock
            from .classifier import DummyClassifier
            self.classifier = DummyClassifier()
            logger.warning("使用虚拟模型模式 (dummy mode)")
        else:
            self.detector = detector
            self.classifier = classifier

    def recognize(
        self,
        image,
    ) -> MahjongHand:
        """完整识别流程

        Args:
            image: 手牌区域的 PIL Image 或 numpy array

        Returns:
            MahjongHand 结构化结果
        """
        t_start = time.perf_counter()

        if image is None:
            logger.warning("recognize 收到空图像")
            return MahjongHand()

        # Step 1: YOLO 检测
        # quehun 的 TileDetector.detect() 返回 DetectionResult
        # 包含 hand_tiles 和 discard_pool
        from .detector import DetectionResult as QDetectionResult

        if isinstance(self.detector, TileDetector) or hasattr(self.detector, 'detect'):
            detection = self.detector.detect(image if isinstance(image, np.ndarray) else np.array(image))
        else:
            logger.error("检测器不支持 detect() 方法")
            return MahjongHand()

        # 如果检测器直接返回了 hand_tiles（quehun 模式），直接用
        if isinstance(detection, QDetectionResult) and detection.hand_tiles:
            elapsed = (time.perf_counter() - t_start) * 1000
            return MahjongHand(
                hand=detection.hand_tiles,
                discard_pool=detection.discard_pool,
                timestamp=time.time(),
                recognition_time_ms=round(elapsed, 1),
            )

        # 如果检测器返回的是 mahjong-ai 风格的 DetectionResult（含 boxes）
        # 则需要用分类器进一步分类
        if hasattr(detection, 'boxes') and detection.count > 0:
            # mahjong-ai 风格: boxes + class_ids -> 需要分类器
            if isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image

            rgb = img_array[:, :, :3] if len(img_array.shape) == 3 and img_array.shape[2] >= 3 else img_array

            tile_images: List[Image.Image] = []
            for box in detection.boxes:
                x, y, w, h = [int(v) for v in box[:4]]
                H, W = rgb.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(W, x + w), min(H, y + h)
                if x2 > x1 and y2 > y1:
                    tile_images.append(Image.fromarray(rgb[y1:y2, x1:x2]))

            # 批量分类
            valid_imgs = [img for img in tile_images if img is not None]
            if valid_imgs and self.classifier:
                results = self.classifier.classify_batch(valid_imgs)
                hand_tiles = [r.tile for r in results]
                confidences = [round(r.confidence, 3) for r in results]
            else:
                hand_tiles = []
                confidences = []

            elapsed = (time.perf_counter() - t_start) * 1000
            return MahjongHand(
                hand=hand_tiles,
                hand_confidences=confidences,
                timestamp=time.time(),
                recognition_time_ms=round(elapsed, 1),
            )

        logger.debug("未检测到任何牌面")
        return MahjongHand(timestamp=time.time())


# 为了避免循环导入，延迟引用
from .detector import TileDetector  # noqa: E402


# ============================================
# 工具函数
# ============================================
def format_hand_for_llm(hand: MahjongHand) -> str:
    """将手牌格式化为适合 LLM prompt 的文本表示

    示例输出:
        手牌: 1m 2m 3m 4p 5p 6p 7s 8s 1z 1z
        副露: 无
        宝牌指示: 3p
        自风: 1z, 场风: 1z
    """
    tiles_str = " ".join(hand.hand)

    melds_str = "无"
    if hand.melds:
        parts = []
        for m in hand.melds:
            mtype = m.get("type", "")
            tiles = m.get("tiles", [])
            parts.append(f"{mtype}{' '.join(tiles)}")
        melds_str = ", ".join(parts)

    lines = [
        f"手牌: {tiles_str}",
        f"副露: {melds_str}",
        f"牌河: {' '.join(hand.river) if hand.river else '(待识别)'}",
        f"宝牌指示: {hand.dora_indicator or '未知'}",
        f"自风: {hand.seat_wind}, 场风: {hand.round_wind}",
        f"点数: {hand.scores}",
        f"本场: {hand.honba_sticks}, 立直棒: {hand.riichi_sticks}",
    ]
    return "\n".join(lines)


def compare_hands(prev: MahjongHand, curr: MahjongHand) -> bool:
    """比较两次识别结果是否相同（用于缓存判断）"""
    return prev.hand == curr.hand
