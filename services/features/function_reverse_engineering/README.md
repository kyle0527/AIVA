# Reverse Engineering Module

## 📑 目錄

- [模組概述](#模組概述)
- [核心能力](#核心能力)
  - [1. Android 逆向](#1-android-逆向)
  - [2. 二進位分析](#2-二進位分析)
  - [3. Malware 分析](#3-malware-分析)
- [工具整合](#工具整合)
- [使用範例](#使用範例)

---


## 模組概述

逆向工程模組，提供二進位分析和 APK 逆向能力。

**風險等級**: L1  
**模組版本**: 1.0.0

## 核心能力

### 1. Android 逆向
- Androguard
- Apk2Gold
- JadX

### 2. 二進位分析
- Disassembly
- Decompilation
- 靜態分析

### 3. Malware 分析
- Behavior Analysis
- API Hooking
- Code Unpacking

## 工具整合

- Androguard
- Apk2Gold
- JadX
- Ghidra (future)

## 使用範例

```python
from services.features.function_reverse_engineering import ReverseEngineeringManager

manager = ReverseEngineeringManager()

# 分析 APK
result = await manager.analyze_apk(
    apk_file="app.apk",
    extract_code=True,
    analyze_permissions=True
)

# 反編譯
decompiled = await manager.decompile(
    input_file="app.apk",
    output_dir="output/",
    tool="jadx"
)
```
