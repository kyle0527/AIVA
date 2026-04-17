# function_wordlist_generator - 字典生成模組

> **版本**: v1.0.0 | **狀態**: ⬜ 輔助工具 | **語言**: Python

## 🎯 模組概述

字典生成工具管理模組，提供產生特定組合密碼本、CUPP 個人化字典生成、多字典合併與字典分析等功能，作為暴力破解前置作業的輔助。

> 📌 **注意**: 此模組優先度較低。在多數情況下可以直接使用 `cewl` 或 `crunch` 等開源工具替代。它被歸類在「需人工操作/定義策略」的輔助模組中。

### 功能清單

| 功能 | 說明 |
|------|------|
| 字元組合生成 | 指定字元集、長度範圍暴力生成組合字串 |
| CUPP 個人化字典 | 根據對象的姓名/生日/關鍵字組合，猜測可能的密碼 |
| 多字典合併 | 合併多個文字檔並去除重複項目與排序 |
| 字典分析 | 統計特定字典檔案的大小與分佈資訊 |

## 📐 架構設計

```
function_wordlist_generator/
├── __init__.py   # 模組入口匯出
├── manager.py    # 主入口 (WordlistGeneratorManager)
├── handler.py    # (過渡期) 舊的 CommandHandler 實作
└── models.py     # 資料模型
```

## 🚀 執行方式

### 作為 Python 模組匯入

```python
from services.features.function_wordlist_generator import WordlistGeneratorManager

manager = WordlistGeneratorManager()

# 字元組合生成
manager.generate_combination(
    charset="abc123",
    min_length=4,
    max_length=6,
    output_file="wordlist.txt"
)

# CUPP 個人化字典
manager.generate_cupp_wordlist(
    name="john",
    birthdate="1990-01-01",
    keywords=["company"],
    output_file="personal.txt"
)
```

## 🔧 內部 API 參考

| 類別 / 方法 | 說明 |
|------|------|
| `WordlistGeneratorManager.generate_combination(...)` | 字元組合字典 |
| `WordlistGeneratorManager.generate_cupp_wordlist(...)` | 個人化字典 |
| `WordlistGeneratorManager.merge_wordlists(...)` | 合併多個字典文字檔 |
| `WordlistGeneratorManager.analyze_wordlist(...)` | 分析字典文字檔統計資訊 |

