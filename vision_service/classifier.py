# vision_service/classifier.py
# 牌面分类模块 - 34类 CNN 分类（ResNet-18 / MobileNetV3-Small）
#
# 对 YOLO 检测到的每张牌面进行 34 类分类：
# 34 类 = 万子9 + 筒子9 + 索子9 + 风牌4 + 箭牌3
#
# 从 mahjong-ai 迁移，适配 quehun 的字牌编码体系（1z~7z）。

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================
# 34 类麻将牌标签定义 (日麻标准)
# 使用与 quehun/detector.py 一致的编码:
#   万子: 1m~9m, 筒子: 1p~9p, 索子: 1s~9s, 字牌: 1z~7z
# ============================================
MAHJONG_TILES_34 = [
    # 万子 1-9
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    # 筒子 1-9
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    # 索子 1-9
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    # 字牌: 东(1z) 南(2z) 西(3z) 北(4z) 中(5z) 发(6z) 白(7z)
    "1z", "2z", "3z", "4z", "5z", "6z", "7z",
]

# 标签到索引的映射
TILE_TO_IDX = {name: i for i, name in enumerate(MAHJONG_TILES_34)}
IDX_TO_TILE = {i: name for i, name in enumerate(MAHJONG_TILES_34)}

# 赤宝牌映射 (5m/5p/5s 的红色版本 -> 对应普通版)
RED_DORA_MAP = {
    "5m_red": "5m", "0m": "5m",
    "5p_red": "5p", "0p": "5p",
    "5s_red": "5s", "0s": "5s",
}


@dataclass
class ClassificationResult:
    """单张牌的分类结果"""
    tile: str              # 如 "3p", "1z"
    confidence: float      # 置信度 0-1
    tile_index: int        # 34类中的索引

    @property
    def is_red_dora(self) -> bool:
        return self.tile in RED_DORA_MAP

    @property
    def is_honor(self) -> bool:
        """是否为字牌(风+箭)"""
        return self.tile_index >= 27

    def __str__(self):
        return f"{self.tile}({self.confidence:.2f})"


@dataclass
class ClassifierConfig:
    """分类器配置"""
    model_path: str = "models/resnet_mahjong.pt"
    num_classes: int = 34
    conf_threshold: float = 0.7
    input_size: Tuple[int, int] = (56, 80)  # (height, width)
    device: str = "auto"  # auto | cuda | cpu


# ============================================
# 模型定义
# ============================================
def create_classifier_model(
    num_classes: int = 34,
    pretrained: bool = False,
) -> "nn.Module":
    """创建 MobileNetV3-Small 牌面分类模型

    Args:
        num_classes: 分类数（默认34类）
        pretrained: 是否用 ImageNet 预训练权重

    Returns:
        PyTorch 模型
    """
    import torchvision.models as models
    import torch.nn as nn

    weights = (
        models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    )
    model = models.mobilenet_v3_small(
        weights=weights,
        num_classes=num_classes,
    )

    if pretrained:
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        logger.info("使用 ImageNet 预训练 MobileNetV3-Small")

    return model


def create_resnet18_classifier(
    num_classes: int = 34,
    pretrained: bool = False,
    weights_path: str = "",
) -> "nn.Module":
    """ResNet-18 分类模型

    Args:
        num_classes: 分类数
        pretrained: 是否使用 ImageNet 预训练
        weights_path: 自定义权重文件路径
    """
    import torchvision.models as models
    import torch
    import torch.nn as nn

    weights = (
        models.ResNet18_Weights.DEFAULT if pretrained else None
    )
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if weights_path and os.path.exists(weights_path):
        logger.info(f"加载自定义分类器权重: {weights_path}")
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=False)
    return model


class TileClassifier:
    """牌面分类器

    对 YOLO 检测到的每个边界框裁剪出的牌面图片
    进行 34 类分类识别。
    """

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self._model = None
        self._device = None
        self._transform = None

    @property
    def device(self):
        if self._device is None:
            import torch
            if self.config.device == "auto":
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self._device = torch.device(self.config.device)
        return self._device

    @property
    def transform(self):
        """图像预处理 transform"""
        if self._transform is None:
            from torchvision import transforms
            h, w = self.config.input_size
            self._transform = transforms.Compose([
                transforms.Resize((h, w)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        return self._transform

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            import torch

            model = create_resnet18_classifier(self.config.num_classes)

            if os.path.exists(self.config.model_path):
                logger.info(f"加载分类器权重: {self.config.model_path}")
                state_dict = torch.load(
                    self.config.model_path,
                    map_location=self.device,
                    weights_only=True,
                )
                model.load_state_dict(state_dict)
            else:
                logger.warning(
                    f"分类器权重文件不存在: {self.config.model_path}\n"
                    f"请先运行 tools/train_classifier.py 训练模型"
                )

            model = model.to(self.device)
            model.eval()
            self._model = model

        return self._model

    def classify(
        self, image
    ) -> ClassificationResult:
        """对单张牌进行分类

        Args:
            image: 单张牌的 PIL Image 或 numpy array (RGB)

        Returns:
            ClassificationResult
        """
        import torch

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        try:
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, dim=1)

            tile_idx = pred.item()
            confidence = conf.item()
            tile_name = MAHJONG_TILES_34[tile_idx]

            # 赤宝牌归一化到普通牌
            if tile_name in RED_DORA_MAP:
                tile_name = RED_DORA_MAP[tile_name]

            return ClassificationResult(
                tile=tile_name,
                confidence=confidence,
                tile_index=TILE_TO_IDX.get(tile_name, -1),
            )

        except Exception as e:
            logger.error(f"分类失败: {e}")
            return ClassificationResult(
                tile="unknown",
                confidence=0.0,
                tile_index=-1,
            )

    def classify_batch(
        self, images: List[Image.Image]
    ) -> List[ClassificationResult]:
        """批量分类多张牌（比逐张调用更快）

        Args:
            images: 多张牌的 PIL Image 列表

        Returns:
            分类结果列表
        """
        import torch

        if not images:
            return []

        try:
            batch_tensor = torch.stack([
                self.transform(img) for img in images
            ]).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, dim=1)

            results = []
            for i in range(len(images)):
                tile_idx = preds[i].item()
                tile_name = MAHJONG_TILES_34[tile_idx]
                # 赤宝牌归一化
                if tile_name in RED_DORA_MAP:
                    tile_name = RED_DORA_MAP[tile_name]
                results.append(ClassificationResult(
                    tile=tile_name,
                    confidence=confs[i].item(),
                    tile_index=TILE_TO_IDX.get(tile_name, -1),
                ))
            return results

        except Exception as e:
            logger.error(f"批量分类失败: {e}")
            return [
                ClassificationResult("unknown", 0.0, -1)
                for _ in images
            ]


class DummyClassifier:
    """虚拟分类器 - 用于开发测试，无需真实模型

    返回随机的有效牌名。
    """

    def __init__(self, seed: int = 42):
        import random
        self.rng = random.Random(seed)

    def classify(self, image=None) -> ClassificationResult:
        idx = self.rng.randint(0, 33)
        return ClassificationResult(
            tile=MAHJONG_TILES_34[idx],
            confidence=self.rng.uniform(0.85, 0.99),
            tile_index=idx,
        )

    def classify_batch(
        self, images: list = None
    ) -> List[ClassificationResult]:
        n = len(images) if images else 13
        return [self.classify() for _ in range(n)]
