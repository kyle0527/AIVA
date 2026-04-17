# function_steganography - 隱寫術分析模組

> **版本**: v1.0.1 | **狀態**: ✅ 引擎完成 | **語言**: Python

## 模組概述

隱寫術嵌入與偵測模組，提供 StegX 演算法嵌入/提取，以及 AI 模型驅動的隱寫偵測能力。適合用於圖片中隱藏資訊的提取與分析。

### 功能完成狀態

| 功能 | 說明 |
|------|------|
| StegX 嵌入/提取 | StegXEngine 圖片隱藏資訊的嵌入與提取 |
| AI 隱寫偵測 | AIStegDetectionEngine 透過卷積神經網路分析圖片特徵 |
| 批次掃描 | 目錄遞迴掃描與分析 |
| 綜合偵測 | EnhancedSteganographyDetector 提供完整整合介面 |

## 架構設計

```
function_steganography/
├── __init__.py           # 模組入口匯出
├── manager.py            # 綜合管理介面 (SteganographyManager)
├── models.py             # 資料模型與整合偵測器 (EnhancedSteganographyDetector)
└── engines/
    ├── __init__.py
    ├── stegx_engine.py       # StegX LSB 嵌入/提取實作
    └── ai_steg_engine.py     # AI 隱寫偵測實作 (CNN 模型)
```

## 執行方式

### 作為 Python 模組匯入

可透過 `SteganographyManager` 或 `EnhancedSteganographyDetector` 來進行操作：

```python
from services.features.function_steganography.manager import SteganographyManager
from services.features.function_steganography.models import EnhancedSteganographyDetector

# 使用偵測器
detector = EnhancedSteganographyDetector()
result = detector.analyze_image("suspicious_image.png")

# 使用 Manager 進行操作
manager = SteganographyManager()
manager.stegx_extract_file("stego.png", "extracted.txt", password="pass")
```

## 可調用方法（內部 API）

| 類別 / 方法 | 說明 |
|------|------|
| `EnhancedSteganographyDetector.analyze_image(file_path)` | 綜合偵測圖片中的隱藏資料 |
| `SteganographyManager.stegx_hide_file(carrier_image, secret_file, output_image, password, compress)` | StegX 嵌入資訊 |
| `SteganographyManager.stegx_extract_file(stego_image, output_file, password)` | StegX 提取資訊 |
| `SteganographyManager.stegx_analyze_image(image_path)` | 快速 StegX 特徵分析 |
| `AIStegDetectionEngine.detect_steganography(image_path)` | 呼叫神經網路模型預測是否有隱寫 |
| `AIStegDetectionEngine.batch_scan(directory)` | 針對整個目錄進行批量 AI 偵測 |
| `AIStegDetectionEngine.train_model(training_data_dir, output_model_path)` | 自定義神經網路模型訓練 |

## 注意事項

- AI 偵測模型預設只是一個基礎 CNN 實作 (使用 Tensorflow/Keras)，精確的結果需要搭配充分標註的資料集來訓練。
- 圖片若經過壓縮 (如 JPEG) 會影響基於 LSB 的隱寫演算法 (如 StegX)。
- 系統無提供獨立的 CLI 入口。
