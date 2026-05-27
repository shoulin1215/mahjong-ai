"""雀魂麻将 AI 助手 - 牌面分类模型训练脚本

训练 34 类牌面分类模型（ResNet-18 / MobileNetV3）。
使用 PyTorch 进行训练，支持 GPU 加速。

数据集目录结构::

    datasets/
      mahjong-tiles-classify/
        1m/
          *.jpg / *.png
        2m/
          ...
        9s/
          ...
        East/
          ...
        ...

Usage:
    python tools/train_classifier.py --data datasets/mahjong-tiles-classify
    python tools/train_classifier.py --arch resnet18 --epochs 30 --batch 64
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

logger = logging.getLogger(__name__)

# 34 类牌面对应的目录名（与 classifier.py MAHJONG_TILES_34 一致，全小写）
TILE_CLASSES_34 = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "east", "south", "west", "north", "white", "green", "red",
]


def get_transforms(train: bool = True, img_size: int = 224):
    """获取数据预处理/增强变换"""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])


def build_model(arch: str = "resnet18", num_classes: int = 34, pretrained: bool = True):
    """构建分类模型

    Args:
        arch: 架构名 (resnet18, resnet34, mobilenetv3)
        num_classes: 分类数（34 类）
        pretrained: 是否使用 ImageNet 预训练权重
    """
    weights = "IMAGENET1K_V1" if pretrained else None

    if arch == "resnet18":
        model = models.resnet18(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)

    elif arch == "resnet34":
        model = models.resnet34(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)

    elif arch in ("mobilenetv3", "mobilenet_v3_small"):
        model = models.mobilenet_v3_small(weights=weights)
        in_feat = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_feat, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes),
        )

    else:
        raise ValueError(f"不支持的架构: {arch}")

    return model


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 20 == 0:
            logger.info(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion, device):
    """评估一个 epoch"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), correct / total


def train_main(
    data_dir: str,
    arch: str = "resnet18",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    img_size: int = 224,
    device: str = "auto",
    output_dir: str = "runs/classify",
    save_name: str = "mahjong_classifier.pth",
):
    """主训练流程"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ 数据集目录不存在: {data_dir}")
        sys.exit(1)

    # 设备
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    # 数据集
    train_dir = data_path / "train"
    val_dir = data_path / "val"

    if not train_dir.exists():
        # 尝试自动分割
        logger.warning(f"未找到 train/val 子目录，使用完整数据集训练")
        full_dataset = datasets.ImageFolder(
            str(data_path),
            transform=get_transforms(train=True, img_size=img_size),
        )
        # 80/20 分割
        n_train = int(len(full_dataset) * 0.8)
        n_val = len(full_dataset) - n_train
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [n_train, n_val]
        )
        # 验证集用 eval transform
        val_ds.dataset.transform = get_transforms(train=False, img_size=img_size)
    else:
        train_ds = datasets.ImageFolder(
            str(train_dir),
            transform=get_transforms(train=True, img_size=img_size),
        )
        val_ds = datasets.ImageFolder(
            str(val_dir),
            transform=get_transforms(train=False, img_size=img_size),
        )

    logger.info(f"训练集: {len(train_ds)} 张 | 验证集: {len(val_ds)} 张")
    logger.info(f"类别数: {len(train_ds.dataset.classes if hasattr(train_ds, 'dataset') else train_ds.classes)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 模型
    model = build_model(arch=arch, num_classes=len(TILE_CLASSES_34), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 训练循环
    best_acc = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = output_path / f"best_{save_name}"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"✅ 最佳模型已保存: {ckpt_path} (Acc: {val_acc:.4f})")

        # 每个 epoch 也保存最新权重
        latest_path = output_path / f"latest_{save_name}"
        torch.save(model.state_dict(), latest_path)

    # 最终模型
    final_path = output_path / f"final_{save_name}"
    torch.save(model.state_dict(), final_path)
    logger.info(f"\n🎉 训练完成! 最终模型: {final_path}")
    logger.info(f"最佳验证准确率: {best_acc:.4f}")

    return str(final_path)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="训练雀魂麻将牌面分类模型 (34类)")
    parser.add_argument(
        "--data", "-d", required=True,
        help="数据集根目录（包含各牌面子目录）",
    )
    parser.add_argument(
        "--arch", "-a", default="resnet18",
        choices=["resnet18", "resnet34", "mobilenetv3"],
        help="模型架构 (默认: resnet18)",
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=30,
        help="训练轮数 (默认: 30)",
    )
    parser.add_argument(
        "--batch", "-b", type=int, default=64,
        help="批次大小 (默认: 64)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="学习率 (默认: 0.001)",
    )
    parser.add_argument(
        "--imgsz", "-s", type=int, default=224,
        help="图像尺寸 (默认: 224)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="训练设备 (auto/cpu/cuda)",
    )
    parser.add_argument(
        "--output-dir", "-o", default="runs/classify",
        help="输出目录 (默认: runs/classify)",
    )
    parser.add_argument(
        "--save-name", "-n", default="mahjong_classifier.pth",
        help="模型文件名 (默认: mahjong_classifier.pth)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    train_main(
        data_dir=args.data,
        arch=args.arch,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        img_size=args.imgsz,
        device=args.device,
        output_dir=args.output_dir,
        save_name=args.save_name,
    )


if __name__ == "__main__":
    main()
