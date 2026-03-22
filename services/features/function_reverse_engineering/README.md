# function_reverse_engineering - 逆向工程模組

> **版本**: v1.0.0 | **狀態**: ⬜ 框架完成，依賴外部工具 | **語言**: Python | **能力登錄**: ⬜ 待登錄

## 模組概述

逆向工程分析模組，提供二進位分析、APK 反編譯、惡意軟體偵測與字串提取功能。所有功能依賴外部工具（radare2、jadx、apktool 等）。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 二進位分析 | ✅ 框架完成 | 依賴 radare2 / Ghidra CLI |
| APK 分析 | ✅ 框架完成 | 依賴 apktool / jadx |
| APK 反編譯 | ✅ 框架完成 | 支援 jadx / apktool |
| 惡意軟體偵測 | ✅ 框架完成 | 模式比對 |
| 字串提取 | ✅ 框架完成 | 最小長度過濾 |

> ⚠️ `legacy/` 目錄內的舊版程式碼（`reverse_engineering_original.py`）已廢棄，**請勿使用**。
> ⚠️ 此模組所有功能需要安裝對應的外部工具才能正常運作。

## 架構

```
function_reverse_engineering/
├── manager.py      # 全部實作（ReverseEngineeringManager）
├── models.py       # 資料模型
└── legacy/         # ⛔ 廢棄，勿使用
    └── reverse_engineering_original.py
```

## 執行方式

### 直接使用

```python
from services.features.function_reverse_engineering.manager import ReverseEngineeringManager

manager = ReverseEngineeringManager()

# 分析二進位（需安裝 radare2）
result = manager.analyze_binary("/path/to/binary", mode="static")

# 分析 APK（需安裝 apktool/jadx）
result = manager.analyze_apk("/path/to/app.apk")

# 提取字串
strings = manager.extract_strings("/path/to/binary", min_length=8)
```

## 可調用方法（公開 API）

| 方法 | 說明 | 外部依賴 |
|------|------|---------|
| `analyze_binary(file_path, mode)` | 二進位分析 | radare2 / Ghidra |
| `analyze_apk(apk_path)` | APK 靜態分析 | apktool / jadx |
| `decompile_apk(apk_path, output_dir, decompiler)` | APK 反編譯 | jadx 或 apktool |
| `detect_malware(file_path)` | 惡意軟體偵測 | 無（純模式比對） |
| `extract_strings(file_path, min_length)` | 字串提取 | 無 |

## 外部工具安裝

```bash
# radare2
sudo apt install radare2

# jadx
wget https://github.com/skylot/jadx/releases/download/v1.4.7/jadx-1.4.7.zip
unzip jadx-1.4.7.zip -d ~/.jadx

# apktool
sudo apt install apktool
```

## 待完成工作

- 刪除 `legacy/` 目錄
- 接通 `aiva_external_executor.py` 的 CLI 入口
- 評估是否需要進一步開發（優先度低）

## 注意事項

- 僅限授權安全分析使用
- 所有分析在本地執行，不上傳任何檔案
- 複雜度高，短期內維持現有功能範圍
