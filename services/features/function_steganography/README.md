# function_steganography - 隱寫術分析模組

> **版本**: v1.0.0 | **狀態**: ✅ 引擎完成 | **語言**: Python | **能力登錄**: ⬜ 待登錄（對應 `steganography_detect`）

## 模組概述

隱寫術嵌入與偵測模組，提供 StegX 演算法嵌入/提取，以及 AI 模型驅動的隱寫偵測能力。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| StegX 嵌入/提取 | ✅ 完成 | StegXEngine（圖片隱寫） |
| AI 隱寫偵測 | ✅ 完成 | AIStegDetectionEngine（模型驅動） |
| 批次掃描 | ✅ 完成 | 目錄遞迴掃描 |
| AI 模型訓練 | ✅ 完成 | 自訂訓練資料 |
| 綜合偵測 | ✅ 完成 | EnhancedSteganographyDetector |

> ⚠️ `legacy/` 目錄內的舊版程式碼（`steganography_original.py`）已廢棄，**請勿使用**。

## 架構

```
function_steganography/
├── manager.py          # 主入口（SteganographyManager）
├── models.py           # 資料模型（EnhancedSteganographyDetector）
├── engines/
│   ├── stegx_engine.py       # StegX 嵌入/提取引擎
│   └── ai_steg_engine.py     # AI 隱寫偵測引擎
└── legacy/             # ⛔ 廢棄，勿使用
    └── steganography_original.py
```

## 執行方式

### 直接使用

```python
from services.features.function_steganography.manager import SteganographyManager

manager = SteganographyManager()

# 偵測隱藏資料（主要用途）
result = manager.detect_hidden_data("image.png")

# AI 批次掃描目錄
results = manager.ai_batch_scan("/path/to/images", recursive=True)

# StegX 嵌入資料
manager.stegx_hide_file("cover.png", "secret.txt", "output.png", password="pass")

# StegX 提取資料
manager.stegx_extract_file("stego.png", "extracted.txt", password="pass")
```

## 可調用方法（公開 API）

| 方法 | 說明 |
|------|------|
| `detect_hidden_data(file_path)` | 偵測圖片中的隱藏資料 |
| `embed_data(carrier_file, secret_file, output_file, password)` | 嵌入資料 |
| `extract_data(stego_file, output_file, password)` | 提取資料 |
| `stegx_hide_file(carrier_image, secret_file, output_image, password, compress)` | StegX 嵌入 |
| `stegx_extract_file(stego_image, output_file, password)` | StegX 提取 |
| `stegx_analyze_image(image_path)` | StegX 圖片分析 |
| `stegx_batch_hide(carrier_images, secret_file, output_dir, password)` | 批次嵌入 |
| `ai_detect_steganography(image_path, model_path)` | AI 單圖偵測 |
| `ai_batch_scan(directory, recursive, extensions)` | AI 批次掃描 |
| `ai_train_model(training_data_dir, output_model_path, epochs, batch_size)` | 訓練 AI 模型 |
| `ai_adjust_threshold(new_threshold)` | 調整偵測閾值 |
| `calculate_capacity(carrier_file, method)` | 計算嵌入容量 |

## 待完成工作

- 將 `steganography_detect` / `steganography_extract` 補全至 `CAPABILITY_CONFIGS`
- 刪除 `legacy/` 目錄
