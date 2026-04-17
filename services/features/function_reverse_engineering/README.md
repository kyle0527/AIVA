# function_reverse_engineering - 逆向工程模組

> **版本**: v1.0.0 | **狀態**: ⬜ 框架完成，依賴外部工具 | **語言**: Python

## 模組概述

逆向工程分析模組，提供二進位分析、APK 反編譯、惡意軟體偵測與字串提取功能。由於這類操作高度專業化，本模組主要作為外部開源工具的封裝，所有功能**嚴重依賴外部二進位工具**（radare2、jadx、apktool 等）。

> ⚠️ 此模組牽涉較多人工分析，目前歸類為「需人工操作」模組，主要提供自動化的初步拆解腳本。

### 功能清單

| 功能 | 說明 | 外部依賴 |
|------|------|---------|
| 二進位分析 | 分析執行檔結構 | radare2 / Ghidra |
| APK 分析與反編譯 | 拆解 Android 應用程式 | apktool / jadx |
| 惡意軟體偵測 | 靜態模式與特徵碼比對 | 無 |
| 字串提取 | 提取檔案中的可印字元 | 無 |

## 架構設計

```
function_reverse_engineering/
├── __init__.py     # 模組入口匯出
├── manager.py      # 主入口 (ReverseEngineeringManager)
└── models.py       # 資料模型 (BinaryAnalysisResult 等)
```

## 執行方式

### 作為 Python 模組匯入

可直接實例化 `ReverseEngineeringManager` 來封裝外部工具的呼叫：

```python
from services.features.function_reverse_engineering import ReverseEngineeringManager

manager = ReverseEngineeringManager()

# 分析二進位（需確保系統已安裝 radare2）
result = manager.analyze_binary("/path/to/binary", mode="static")

# 反編譯 APK（需確保系統已安裝 jadx）
result = manager.decompile_apk("/path/to/app.apk", "/tmp/out")
```

## 可調用方法（內部 API）

| 類別 / 方法 | 說明 |
|------|------|
| `ReverseEngineeringManager.analyze_binary(file_path, mode)` | 二進位分析 |
| `ReverseEngineeringManager.analyze_apk(apk_path)` | APK 靜態分析 |
| `ReverseEngineeringManager.decompile_apk(apk_path, output_dir, decompiler)` | APK 反編譯 (預設用 jadx) |
| `ReverseEngineeringManager.detect_malware(file_path)` | 簡易靜態惡意軟體偵測 |
| `ReverseEngineeringManager.extract_strings(file_path, min_length)` | 從二進位中提取連續的可印字元 |

## 外部工具安裝

確保在執行環境中已經安裝對應的工具：

```bash
# radare2
sudo apt install radare2

# apktool
sudo apt install apktool

# jadx (需手動下載解壓並加入 PATH)
```

## 注意事項

- 本模組只是工具包裝層，進階分析仍需依賴分析師的專業知識。
- 無直接的 CLI 入口。
