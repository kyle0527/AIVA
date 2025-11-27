# Wordlist Generator Module

## 📑 目錄

- [模組概述](#模組概述)
- [核心能力](#核心能力)
  - [1. CUPP (Common User Passwords Profiler)](#1-cupp-common-user-passwords-profiler)
  - [2. 組合生成](#2-組合生成)
  - [3. 混合字典](#3-混合字典)
  - [4. 常見密碼列表](#4-常見密碼列表)
- [工具整合](#工具整合)
- [使用範例](#使用範例)

---


## 模組概述

密碼字典生成模組，支援多種生成策略和自訂規則。

**風險等級**: L1  
**模組版本**: 1.0.0

## 核心能力

### 1. CUPP (Common User Passwords Profiler)
- 基於目標資訊生成
- 個性化字典

### 2. 組合生成
- 字符集組合
- 長度範圍
- 規則引擎

### 3. 混合字典
- 合併多個字典
- 去重排序

### 4. 常見密碼列表
- RockYou
- SecLists
- 自訂列表

## 工具整合

- CUPP
- Crunch
- WlCreator
- Goblin WordGenerator

## 使用範例

```python
from services.features.function_wordlist_generator import WordlistGeneratorManager

manager = WordlistGeneratorManager()

# 基於目標資訊生成
result = await manager.generate_cupp_wordlist(
    name="John Doe",
    birthdate="1990-01-01",
    keywords=["company", "hobby"]
)

# 組合生成
result = await manager.generate_combination(
    charset="lowercase,digits",
    min_length=8,
    max_length=12,
    output_file="wordlist.txt"
)
```
