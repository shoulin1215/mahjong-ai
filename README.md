# 雀魂AI助手

自动识别浏览器中的雀魂麻将画面，接入大模型提供实时出牌建议。

## 项目结构

```
quehun/
├── extension/              # Chrome 扩展
│   ├── manifest.json       # MV3 扩展清单
│   ├── background.js       # Service Worker（定时截图）
│   ├── content.js          # 页面注入（浮层展示）
│   ├── overlay.css         # 浮层样式
│   ├── popup.html/js       # 控制面板
│   └── icons/              # 图标（需自行添加）
│
├── vision_service/         # 图像识别服务（Python + FastAPI）
│   ├── main.py             # API 入口
│   ├── detector.py         # YOLOv8 牌检测器
│   ├── models.py           # Pydantic 数据模型
│   └── models/             # 存放 YOLO 权重文件
│
├── game_engine/            # 麻将规则引擎（纯 Python）
│   ├── shanten.py          # 向听数 & 有效进张计算
│   ├── danger.py           # 危险牌评估
│   └── state.py            # 游戏状态管理
│
├── llm_advisor/            # 大模型推理
│   ├── advisor.py          # Ollama / OpenAI / Claude 路由
│   └── prompt.py           # Prompt 构造引擎
│
├── server.py               # 启动入口
├── test_engine.py          # 引擎自测脚本
├── requirements.txt
└── .env.example
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，选择 LLM_BACKEND（默认 ollama）
```

### 3. 启动后端服务

```bash
python server.py
# 或者：uvicorn server:app --port 8000 --reload
```

### 4. 安装 Chrome 扩展

1. 打开 `chrome://extensions/`
2. 启用"开发者模式"
3. 点击"加载已解压的扩展程序"
4. 选择 `extension/` 目录

### 5. 训练/放置 YOLO 模型

将训练好的 YOLOv8 权重放到：
```
vision_service/models/tiles_yolov8.pt
```

> 没有模型时系统会使用模拟数据，可先调试其他流程。

### 6. 测试游戏引擎

```bash
python test_engine.py
```

## 技术选型

| 层级 | 技术 | 说明 |
|------|------|------|
| 浏览器截图 | Chrome Extension MV3 | `captureVisibleTab` API |
| 图像识别 | YOLOv8 + OpenCV | 34 类麻将牌目标检测 |
| 服务接口 | FastAPI + uvicorn | localhost:8000 |
| 规则引擎 | 纯 Python | 向听数、进张、危险度 |
| 大模型 | Ollama / OpenAI / Claude | 策略推理与解释 |
| 浮层 UI | Vanilla JS + CSS | 注入到游戏页面 |

## 训练数据说明

需要自行采集雀魂截图并标注 34 类牌面，推荐使用 Roboflow 标注工具。
训练命令：
```bash
yolo train data=tiles.yaml model=yolov8n.pt epochs=100 imgsz=640
```

## 注意事项

- 本项目仅供学习研究，请遵守游戏服务条款
- 不建议在排位赛中使用，以免影响游戏体验
