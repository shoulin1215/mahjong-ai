"""雀魂麻将 AI 助手 - YOLO 检测器训练脚本

使用 Ultralytics YOLOv8 训练牌位检测模型。
需要准备已标注的数据集（YOLO 格式）。

数据集目录结构::

    datasets/
      mahjong-tiles/
        images/
          train/*.jpg
          val/*.jpg
        labels/
          train/*.txt
          val/*.txt
        data.yaml   # YOLO 数据集配置

Usage:
    python tools/train_detector.py --data datasets/mahjong-tiles/data.yaml
    python tools/train_detector.py --model yolo11n.pt --epochs 50
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def check_ultralytics():
    """检查 ultralytics 是否可用"""
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("❌ 未安装 ultralytics，请运行: pip install ultralytics")
        sys.exit(1)


def train(
    data_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    project: str = "runs/detect",
    name: str = "mahjong_tiles",
    exist_ok: bool = True,
):
    """训练 YOLO 牌位检测模型

    Args:
        data_yaml: 数据集配置文件路径
        model: 预训练模型或起始权重
        epochs: 训练轮数
        imgsz: 输入图像尺寸
        batch: 批次大小
        device: 训练设备 (空=自动, 'cpu', '0', '0,1')
        project: 输出目录
        name: 实验名称
        exist_ok: 是否允许覆盖已有实验
    """
    YOLO = check_ultralytics()

    logger.info(f"加载模型: {model}")
    model_obj = YOLO(model)

    logger.info(f"开始训练，数据集: {data_yaml}")
    results = model_obj.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=exist_ok,
        verbose=True,
        # 关闭数据增强中的 Mosaic（小目标检测更有效）
        mosaic=0.0,
        # 提高小目标检测性能
        box=7.5,
        cls=0.5,
    )

    # 保存最终模型路径
    best_model = Path(project) / name / "weights" / "best.pt"
    logger.info(f"训练完成! 最佳模型保存在: {best_model}")

    # 验证
    metrics = model_obj.val()
    logger.info(f"mAP50: {metrics.box.map50:.4f}")
    logger.info(f"mAP50-95: {metrics.box.map:.4f}")

    return str(best_model)


def export_to_onnx(model_path: str, opset: int = 12):
    """将训练好的模型导出为 ONNX 格式（可选）"""
    YOLO = check_ultralytics()
    model = YOLO(model_path)
    out = model.export(format="onnx", opset=opset)
    logger.info(f"已导出 ONNX: {out}")
    return out


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="训练雀魂麻将牌位检测 YOLO 模型")
    parser.add_argument(
        "--data", "-d", required=True,
        help="数据集 data.yaml 文件路径",
    )
    parser.add_argument(
        "--model", "-m", default="yolov8n.pt",
        help="起始模型 (默认: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=50,
        help="训练轮数 (默认: 50)",
    )
    parser.add_argument(
        "--imgsz", "-s", type=int, default=640,
        help="图像尺寸 (默认: 640)",
    )
    parser.add_argument(
        "--batch", "-b", type=int, default=16,
        help="批次大小 (默认: 16)",
    )
    parser.add_argument(
        "--device", default="",
        help="训练设备 (默认自动选择，可选: cpu, 0, 0,1)",
    )
    parser.add_argument(
        "--project", "-p", default="runs/detect",
        help="输出目录 (默认: runs/detect)",
    )
    parser.add_argument(
        "--name", "-n", default="mahjong_tiles",
        help="实验名称 (默认: mahjong_tiles)",
    )
    parser.add_argument(
        "--export-onnx", action="store_true",
        help="训练完成后导出 ONNX 格式",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    best_model = train(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )

    if args.export_onnx and best_model:
        export_to_onnx(best_model)


if __name__ == "__main__":
    main()
