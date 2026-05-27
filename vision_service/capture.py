# vision_service/capture.py
# 屏幕捕获模块（本地桌面截图模式）
#
# 使用 mss 库实现高效屏幕区域截图，支持：
# - 固定区域裁剪（手牌区）
# - 指定窗口截图
# - 多分辨率适配
#
# 适用于本地桌面模式（mahjong-ai 架构迁移）。
# Chrome 扩展模式不需要此模块（扩展直接截图 -> Base64）。

import time
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class CaptureConfig:
    """截图配置"""
    top: int = 585
    left: int = 260
    width: int = 800
    height: int = 100
    interval: float = 0.5
    window_title: str = ""

    @property
    def region(self) -> dict:
        """返回 mss 兼容的区域字典"""
        return {
            "top": self.top,
            "left": self.left,
            "width": self.width,
            "height": self.height,
        }


class ScreenCapture:
    """屏幕捕获器

    使用 mss 实现高性能屏幕截图，
    支持固定区域裁剪以减少数据量。
    """

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._sct = None
        self._last_capture_time = 0.0
        self._last_frame: Optional[np.ndarray] = None

    def __enter__(self):
        self._sct = mss.mss()
        return self

    def __exit__(self, *args):
        if self._sct:
            self._sct.close()

    def grab(self, force: bool = False) -> Optional[np.ndarray]:
        """截取指定区域

        Args:
            force: 强制截取，忽略间隔限制

        Returns:
            BGRA 格式的 numpy 数组 (H, W, 4)，失败返回 None
        """
        now = time.time()
        if not force and (now - self._last_capture_time) < self.config.interval:
            return self._last_frame

        try:
            import mss
            if self._sct is None:
                self._sct = mss.mss()

            screenshot = np.array(self._sct.grab(self.config.region))
            self._last_frame = screenshot
            self._last_capture_time = now
            return screenshot

        except Exception as e:
            logger.error(f"截图失败: {e}")
            return None

    def grab_pil(self, force: bool = False) -> Optional[Image.Image]:
        """截取并返回 PIL Image (RGB格式)

        Returns:
            RGB PIL Image，失败返回 None
        """
        frame = self.grab(force=force)
        if frame is None:
            return None
        # BGRA -> RGB
        return Image.fromarray(frame[:, :, :3])

    def grab_tile(
        self, bbox: Tuple[int, int, int, int]
    ) -> Optional[Image.Image]:
        """从当前帧中裁剪单张牌区域

        Args:
            bbox: (x, y, width, height) YOLO 格式

        Returns:
            单张牌的 PIL Image
        """
        frame = self.grab()
        if frame is None:
            return None

        rgb = frame[:, :, :3]
        x, y, w, h = [int(v) for v in bbox]

        # 边界安全检查
        H, W = rgb.shape[:2]
        x1 = max(0, min(x, W))
        y1 = max(0, min(y, H))
        x2 = max(0, min(x + w, W))
        y2 = max(0, min(y + h, H))

        tile = rgb[y1:y2, x1:x2]
        return Image.fromarray(tile)

    @property
    def last_frame(self) -> Optional[np.ndarray]:
        """获取上一帧截图"""
        return self._last_frame


def find_game_window(window_title: str = "") -> Optional[dict]:
    """查找游戏窗口位置（仅 Windows）

    Args:
        window_title: 窗口标题关键词

    Returns:
        窗口区域字典或 None
    """
    import ctypes
    import ctypes.wintypes

    EnumWindows = ctypes.windll.user32.EnumWindows
    GetWindowTextW = ctypes.windll.user32.GetWindowTextW
    GetWindowRect = ctypes.windll.user32.GetWindowRect
    IsVisible = ctypes.windll.user32.IsWindowVisible

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.HWND, ctypes.LPARAM)

    results = []

    def cb(hwnd, lp):
        if not IsVisible(hwnd):
            return True
        length = GetWindowTextW(hwnd, 0, 0)
        if length == 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if window_title and window_title.lower() in title.lower():
            rect = ctypes.wintypes.RECT()
            GetWindowRect(hwnd, ctypes.byref(rect))
            results.append({
                "title": title,
                "hwnd": hwnd,
                "left": rect.left,
                "top": rect.top,
                "right": rect.right,
                "bottom": rect.bottom,
                "width": rect.right - rect.left,
                "height": rect.bottom - rect.top,
            })
        return True

    EnumWindows(WNDENUMPROC(cb), 0)
    return results[0] if results else None
