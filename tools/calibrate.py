"""雀魂麻将 AI 助手 - 坐标校准工具

帮助用户在雀魂游戏窗口上标定手牌区域、副露区域、河区域的坐标。
标定结果直接写入 config.yaml。

Usage:
    python tools/calibrate.py
    python -m tools.calibrate
"""

import json
import time
import tkinter as tk
from pathlib import Path
from typing import List, Tuple

# 尝试导入 pyautogui，失败则提示安装
try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False


# ============================================================
# 简易十字准星覆盖层
# ============================================================

class CrosshairOverlay:
    """全屏透明覆盖层，显示十字准星和坐标"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.6)
        self.root.overrideredirect(True)
        self.root.config(bg="systemTransparent" if hasattr(self.root, "attributes") else "white")
        # 用黑色背景 + 低透明度模拟透明
        self.root.configure(bg="black")
        self.root.attributes("-alpha", 0.4)

        self.canvas = tk.Canvas(
            self.root,
            bg="black",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 状态提示
        self.info_var = tk.StringVar(value="移动鼠标至目标位置，按 Enter 记录坐标，按 Esc 完成")
        info_bar = tk.Label(
            self.root,
            textvariable=self.info_var,
            font=("Microsoft YaHei UI", 14),
            fg="white",
            bg="#222",
            anchor="w",
            height=2,
        )
        info_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 坐标显示
        self.pos_var = tk.StringVar(value="(0, 0)")
        pos_lbl = tk.Label(
            self.root,
            textvariable=self.pos_var,
            font=("Consolas", 12),
            fg="#00ff88",
            bg="#111",
            anchor="e",
        )
        pos_lbl.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        # 绑定事件
        self.root.bind("<Motion>", self._on_mouse_move)
        self.root.bind("<Return>", self._on_enter)
        self.root.bind("<Escape>", self._on_escape)
        self.root.bind("<Key>", self._on_key)

        self.points: List[Tuple[int, int]] = []
        self.step = 0
        self.labels = [
            "① 点击手牌区域 左上角 (Enter确认)",
            "② 点击手牌区域 右下角 (Enter确认)",
            "③ 点击河区域 左上角 (Enter确认)",
            "④ 点击河区域 右下角 (Enter确认)",
            "⑤ 点击副露区域 左上角 (Enter确认)",
            "⑥ 点击副露区域 右下角 (Enter确认)",
            "校准完成！按 Esc 保存退出",
        ]
        self.info_var.set(self.labels[0])

        # 绘制初始十字线
        self._draw_crosshair(
            self.root.winfo_screenwidth() // 2,
            self.root.winfo_screenheight() // 2,
        )

    def _on_mouse_move(self, event):
        x, y = event.x_root, event.y_root
        self.pos_var.set(f"({x}, {y})")
        self._draw_crosshair(x, y)

    def _draw_crosshair(self, x: int, y: int):
        self.canvas.delete("all")
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()

        # 横线
        self.canvas.create_line(0, y, w, y, fill="#00ff88", width=1, tags="cross")
        # 竖线
        self.canvas.create_line(x, 0, x, h, fill="#00ff88", width=1, tags="cross")
        # 中心点
        r = 6
        self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="#ff4444", width=2, tags="cross")
        # 坐标标签
        self.canvas.create_text(
            x + 12, y - 12,
            text=f"({x}, {y})",
            fill="#00ff88",
            font=("Consolas", 11),
            anchor="nw",
            tags="cross",
        )

        # 已记录的点的标记
        for i, (px, py) in enumerate(self.points):
            self.canvas.create_oval(px - 4, py - 4, px + 4, py + 4, fill="#ff8800", tags="cross")
            self.canvas.create_text(px + 10, py, text=f"P{i+1}", fill="#ff8800", anchor="w", tags="cross")

    def _on_enter(self, event):
        x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()
        self.points.append((x, y))
        print(f"[校准] 记录点 P{len(self.points)}: ({x}, {y})")

        if self.step < len(self.labels) - 1:
            self.step += 1
            self.info_var.set(self.labels[min(self.step, len(self.labels) - 1)])

        self._draw_crosshair(x, y)

    def _on_escape(self, event):
        self.root.quit()
        self.root.destroy()

    def _on_key(self, event):
        # 支持空格键重新开始
        if event.keysym == "space":
            self.points.clear()
            self.step = 0
            self.info_var.set(self.labels[0])
            self._draw_crosshair(
                self.root.winfo_pointerx(),
                self.root.winfo_pointery(),
            )

    def run(self) -> List[Tuple[int, int]]:
        self.root.mainloop()
        return self.points


# ============================================================
# 配置写入
# ============================================================

def points_to_region(points: List[Tuple[int, int]]) -> List[int]:
    """将对角两点转换为 [x1, y1, x2, y2] 区域"""
    if len(points) < 2:
        return [0, 0, 100, 100]
    x1, y1 = points[0]
    x2, y2 = points[1]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def save_to_config(points: List[Tuple[int, int]], config_path: str = "config.yaml"):
    """将校准点保存到 config.yaml"""
    import yaml

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        print(f"⚠️ 配置文件不存在: {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # 手牌区域 (points[0], points[1])
    if len(points) >= 2:
        hand_region = points_to_region(points[:2])
        cfg.setdefault("capture", {})["region"] = hand_region
        print(f"[配置] 手牌区域: {hand_region}")

    # 河区域 (points[2], points[3])
    if len(points) >= 4:
        river_region = points_to_region(points[2:4])
        cfg["capture"]["river_region"] = river_region
        print(f"[配置] 河区域: {river_region}")

    # 副露区域 (points[4], points[5])
    if len(points) >= 6:
        meld_region = points_to_region(points[4:6])
        cfg["capture"]["meld_region"] = meld_region
        print(f"[配置] 副露区域: {meld_region}")

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    print(f"\n✅ 校准结果已保存到 {config_path}")


# ============================================================
# CLI 入口
# ============================================================

def main():
    print("=" * 50)
    print("  雀魂麻将 AI - 坐标校准工具")
    print("=" * 50)
    print()
    print("操作说明：")
    print("  移动鼠标到目标位置 → 按 Enter 记录坐标")
    print("  需要记录 6 个点（手牌/河/副露 各2个对角点）")
    print("  完成后按 Esc 保存退出")
    print("  按 空格键 清空已记录的点重新开始")
    print()

    overlay = CrosshairOverlay()
    points = overlay.run()

    if points:
        print(f"\n记录到 {len(points)} 个点:")
        for i, (x, y) in enumerate(points):
            print(f"  P{i+1}: ({x}, {y})")

        save_to_config(points)
    else:
        print("\n未记录任何点，未保存。")


if __name__ == "__main__":
    main()
