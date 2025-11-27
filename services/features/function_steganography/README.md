# Steganography Module

## 📑 目錄

- [模組概述](#模組概述)
- [核心能力](#核心能力)
  - [1. 圖片隱寫](#1-圖片隱寫)
  - [2. 音訊隱寫](#2-音訊隱寫)
  - [3. 文字隱寫](#3-文字隱寫)
  - [4. 反隱寫術](#4-反隱寫術)
- [工具整合](#工具整合)
- [使用範例](#使用範例)

---


## 模組概述

隱寫術模組，提供資訊隱藏和提取能力。

**風險等級**: L1  
**模組版本**: 1.0.0

## 核心能力

### 1. 圖片隱寫
- LSB 方法
- Steghide
- Metadata 嵌入

### 2. 音訊隱寫
- Audio LSB
- 頻譜分析

### 3. 文字隱寫
- Whitespace
- Unicode 隱藏

### 4. 反隱寫術
- Stegcracker
- 密碼破解
- 自動檢測

## 工具整合

- Steghide
- Stegcracker
- Snow
- Whitespace

## 使用範例

```python
from services.features.function_steganography import SteganographyManager

manager = SteganographyManager()

# 隱藏資訊
result = await manager.embed_message(
    cover_file="image.jpg",
    message="secret message",
    password="password123",
    output_file="stego.jpg"
)

# 提取資訊
message = await manager.extract_message(
    stego_file="stego.jpg",
    password="password123"
)
```
