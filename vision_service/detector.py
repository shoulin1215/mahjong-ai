# vision_service/detector.py
# YOLOv8 麻将牌检测器

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 34 种标准麻将牌标签（顺序与 YOLO 训练标签一致）
TILE_LABELS = [
    # 万子 1-9
    '1m', '2m', '3m', '4m', '5m', '6m', '7m', '8m', '9m',
    # 饼子 1-9
    '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
    # 索子 1-9
    '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s',
    # 字牌：东南西北中发白
    '1z', '2z', '3z', '4z', '5z', '6z', '7z'
]


@dataclass
class DetectionResult:
    hand_tiles: list[str] = field(default_factory=list)
    discard_pool: list[list[str]] = field(default_factory=lambda: [[], [], [], []])  # 四家弃牌
    raw_boxes: list[dict] = field(default_factory=list)
    confidence_scores: list[float] = field(default_factory=list)


class TileDetector:
    """
    使用 YOLOv8 检测雀魂截图中的麻将牌。

    区域划分（基于 1920x1080 分辨率）：
      手牌区域：y=[820, 980], x=[280, 1650]
      弃牌区域（自家）：y=[540, 720], x=[480, 1440]

    如分辨率不同，会自动按比例缩放。
    """

    # 手牌区域（归一化坐标）
    HAND_REGION = (0.145, 0.758, 0.859, 0.907)    # x1, y1, x2, y2
    DISCARD_REGIONS = {
        'self':  (0.250, 0.500, 0.750, 0.667),    # 自家弃牌
        'right': (0.720, 0.278, 0.880, 0.611),    # 右家弃牌
        'across':(0.250, 0.111, 0.750, 0.278),    # 对家弃牌
        'left':  (0.120, 0.278, 0.280, 0.611),    # 左家弃牌
    }

    def __init__(self, model_path: Optional[str] = None):
        self.ready = False
        self.model = None

        if model_path is None:
            model_path = str(Path(__file__).parent / 'models' / 'tiles_yolov8.pt')

        self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            path = Path(model_path)
            if path.exists():
                self.model = YOLO(str(path))
                self.ready = True
                logger.info(f"YOLOv8 模型加载成功: {model_path}")
            else:
                logger.warning(
                    f"模型文件不存在: {model_path}\n"
                    "请训练或下载模型文件放置到 vision_service/models/tiles_yolov8.pt\n"
                    "调试模式：将使用模拟数据"
                )
                self.ready = False
        except ImportError:
            logger.error("未安装 ultralytics，请运行: pip install ultralytics")
            self.ready = False

    def detect(self, img: np.ndarray) -> DetectionResult:
        """
        对完整游戏截图进行检测。

        Args:
            img: RGB numpy array，形状 (H, W, 3)

        Returns:
            DetectionResult
        """
        result = DetectionResult()

        if not self.ready or self.model is None:
            # 无模型时返回模拟数据（方便开发调试）
            result.hand_tiles = self._mock_hand()
            return result

        h, w = img.shape[:2]

        # ---- 检测手牌区域 ----
        hand_crop = self._crop(img, self.HAND_REGION, h, w)
        hand_detections = self._run_yolo(hand_crop, conf=0.5)
        result.hand_tiles = self._sort_hand(hand_detections, hand_crop.shape[1])
        result.raw_boxes = [d['box'] for d in hand_detections]
        result.confidence_scores = [d['conf'] for d in hand_detections]

        # ---- 检测弃牌区域 ----
        for player, region in self.DISCARD_REGIONS.items():
            crop = self._crop(img, region, h, w)
            detections = self._run_yolo(crop, conf=0.45)
            tiles = [d['label'] for d in detections]
            if player == 'self':
                result.discard_pool[0] = tiles
            elif player == 'right':
                result.discard_pool[1] = tiles
            elif player == 'across':
                result.discard_pool[2] = tiles
            elif player == 'left':
                result.discard_pool[3] = tiles

        return result

    def _run_yolo(self, img: np.ndarray, conf: float = 0.5) -> list[dict]:
        """运行 YOLO 推理，返回检测结果列表"""
        results = self.model.predict(img, conf=conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id < len(TILE_LABELS):
                    detections.append({
                        'label': TILE_LABELS[cls_id],
                        'conf': float(box.conf[0]),
                        'box': box.xyxy[0].tolist(),   # [x1, y1, x2, y2]
                        'cx': float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                    })
        return detections

    def _sort_hand(self, detections: list[dict], img_width: int) -> list[str]:
        """按 x 坐标从左到右排列手牌"""
        sorted_det = sorted(detections, key=lambda d: d['cx'])
        return [d['label'] for d in sorted_det]

    @staticmethod
    def _crop(img: np.ndarray, region: tuple, h: int, w: int) -> np.ndarray:
        """按归一化区域裁剪图像"""
        x1, y1, x2, y2 = region
        return img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]

    @staticmethod
    def _mock_hand() -> list[str]:
        """无模型时的模拟手牌，用于开发调试"""
        return ['1m', '2m', '3m', '4p', '5p', '6p', '7s', '8s', '9s', '1z', '1z', '1z', '5m']
