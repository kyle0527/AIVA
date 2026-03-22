# function_wordlist_generator - 字典生成模組

> **版本**: v1.0.0 | **狀態**: ⬜ 框架完成，外部工具依賴 | **語言**: Python | **能力登錄**: ⬜ 低優先度

## 模組概述

字典生成工具管理模組，整合 CUPP（個人化字典）、字元組合生成、多字典合併等功能。

> ⚠️ `legacy/` 目錄內的舊版程式碼已廢棄，**請勿使用**。
> 📌 此模組優先度較低。多數功能可由外部工具（cewl、crunch）取代。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| 字元組合生成 | ✅ 完成 | 指定字元集、長度範圍 |
| CUPP 個人化字典 | ✅ 完成 | 姓名/生日/關鍵字組合 |
| 多字典合併 | ✅ 完成 | 去重 + 排序 |
| 字典分析 | ✅ 完成 | 統計資訊 |

## 架構

```
function_wordlist_generator/
├── manager.py    # 主入口（WordlistGeneratorManager）
├── models.py     # 資料模型
├── handler.py    # 指令處理（WordlistGeneratorCommandHandler）
└── legacy/       # ⛔ 廢棄，勿使用
    └── wordlist_generator_original.py
```

## 執行方式

### 直接使用

```python
from services.features.function_wordlist_generator.manager import WordlistGeneratorManager

manager = WordlistGeneratorManager()

# 字元組合生成
manager.generate_combination(
    charset="abcdefghijklmnopqrstuvwxyz0123456789",
    min_length=4,
    max_length=6,
    output_file="wordlist.txt"
)

# CUPP 個人化字典
manager.generate_cupp_wordlist(
    name="john",
    birthdate="1990-01-01",
    keywords=["admin", "company"],
    output_file="personal.txt"
)

# 合併字典
manager.merge_wordlists(
    input_files=["list1.txt", "list2.txt"],
    output_file="merged.txt",
    deduplicate=True,
    sort=True
)
```

## 可調用方法（公開 API）

| 方法 | 說明 |
|------|------|
| `generate_combination(charset, min_length, max_length, output_file)` | 字元組合字典 |
| `generate_cupp_wordlist(name, birthdate, keywords, output_file)` | 個人化字典 |
| `merge_wordlists(input_files, output_file, deduplicate, sort)` | 合併字典 |
| `analyze_wordlist(file_path)` | 分析字典統計資訊 |

## 建議替代方案

若不想維護此模組，可透過 subprocess 呼叫外部工具：

```python
import subprocess

# cewl（網頁爬取字典）
subprocess.run(["cewl", "https://example.com", "-w", "wordlist.txt"])

# crunch（字元組合）
subprocess.run(["crunch", "4", "6", "abc123", "-o", "wordlist.txt"])
```

## 待完成工作

- 決定是否繼續維護或改用 subprocess 呼叫外部工具
- 若繼續：刪除 `legacy/` 目錄，接通 CLI 入口
