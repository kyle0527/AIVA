# AIVA 系統狀態報告

## 📑 目錄

- [✅ 已完成工作](#-已完成工作)
  - [1. 版本重置](#1-版本重置)
  - [2. AI能力詳細說明已添加](#2-ai能力詳細說明已添加)
  - [3. 能力演示腳本](#3-能力演示腳本)
- [📊 當前數據統計（v1版本）](#-當前數據統計v1版本)
  - [能力總覽](#能力總覽)
  - [AI能力分類](#ai能力分類)
  - [模組分布（合併前→合併後）](#模組分布合併前合併後)
- [🔄 版本控制機制](#-版本控制機制)
  - [當前機制](#當前機制)
  - [差異比對機制](#差異比對機制)
  - [後續執行行為](#後續執行行為)
- [📁 文件結構](#-文件結構)
- [🎯 給人看的資料](#-給人看的資料)
- [🤖 給AI看的資料](#-給ai看的資料)
- [⚠️ 重要提醒](#-重要提醒)
- [🚀 下次執行時](#-下次執行時)

---


生成時間：2026-01-09

## ✅ 已完成工作

### 1. 版本重置
- **重置前**：v1-v12（混亂的版本號）
- **重置後**：從 v1 重新開始（當前版本）
- **路徑**：`C:\D\fold7\AIVA-git\services\integration\data\internal_exploration\analysis_history\v1`
- **機制**：下次分析時，檢測到 v1 存在 → 自動創建 v2

### 2. AI能力詳細說明已添加
- **文件**：`classification_summary.md`（v12版本）
- **內容**：3個AI內部能力的完整說明
  - **Flow 51, 464**：內部循環連接器（能力範圍分類器）
  - **Flow 323**：強化學習模型（DQN/PPO網絡）
  - **Flow 464**：內部循環連接器路徑變體

### 3. 能力演示腳本
- **文件**：`demo_ai_standalone.py`
- **功能**：獨立演示3個AI內部能力
- **位置**：`services/core/aiva_core/internal_exploration/python_tools/`

## 📊 當前數據統計（v1版本）

### 能力總覽
- **總數據流**：676條
- **活躍能力**：171條（去重後）
- **已合併流**：505條（重複路徑）

### AI能力分類
| 類型 | 數量 | 占比 |
|------|------|------|
| AI內部能力 | 3 | 1.8% |
| AI對外能力 | 10 | 5.8% |
| 非AI能力 | 158 | 92.4% |

### 模組分布（合併前→合併後）
| 模組 | 合併前 | 合併後 | 有路徑變體 |
|------|--------|--------|------------|
| 服務骨幹模組 | 41 | 41 | 19 |
| 認知核心模組 | 32 | 32 | 22 |
| 核心能力模組 | 22 | 22 | 9 |
| 任務規劃模組 | 21 | 21 | 9 |
| 內探模組 | 20 | 20 | 20 |
| unknown | 18 | 18 | 11 |
| 學習子系統 | 17 | 17 | 13 |

## 🔄 版本控制機制

### 當前機制
✅ **從 v1 開始的增量版本管理**已內建於 `aiva_exploration_pipeline.py`：
```python
def _get_next_version_dir(self):
    """自動計算下一個版本號"""
    existing_dirs = glob.glob(str(HISTORY_DIR / "v*"))
    max_ver = 0
    
    # 遍歷所有 v* 目錄，找出最大版本號
    for d in existing_dirs:
        if name.startswith('v') and name[1:].isdigit():
            ver = int(name[1:])
            if ver > max_ver:
                max_ver = ver  # 找到當前最大版本
    
    next_ver = max_ver + 1  # 自動遞增（v1存在→創建v2）
    new_dir = HISTORY_DIR / f"v{next_ver}"
    return new_dir
```

**工作流程**：
1. 當前僅有 v1
2. 下次執行分析 → 檢測到 v1 → 創建 v2
3. 再次執行 → 檢測到 v2 → 創建 v3
4. 以此類推...

### 差異比對機制
✅ **自動差異比對**：
```python
def _step_generate_diff(self, new_file, report_file):
    """步驟 3: 生成版本差異報告"""
    if self.prev_version_dir:
        old_file = self.prev_version_dir / "classification_data.json"
        # 比對舊版本與新版本的差異
        # 生成 diff_report.md
```

### 後續執行行為
當再次執行分析時：
```bash
python aiva_exploration_pipeline.py --target core --module core
```

**將會發生**：
1. 檢測到 v1 存在（最大版本號 = 1）
2. 創建 v2 新目錄（next_ver = 1 + 1）
3. 執行完整分析（生成新數據）
4. 與 v1 比對差異
5. 生成 v2/diff_report.md

**保留所有版本**：v1, v2, v3, v4, ...（從v1開始遞增）

## 📁 文件結構

```
services/integration/data/internal_exploration/analysis_history/
└── v1/  ← 當前版本（重置後的起點）
    ├── analysis_results.json      # 原始分析數據（104,817行）
    ├── classification_data.json   # 分類結果（35,398行）
    ├── classification_summary.md  # 人類可讀報告（含AI能力詳細說明）
    └── complete_flow_details.md   # 完整流程詳情
    
    下次執行會生成：
    ├── v2/  ← 新版本（自動檢測v1存在後創建）
    │   ├── diff_report.md         # v1 → v2 差異報告
    │   └── ...
    ├── v3/  ← 再次執行時創建
    └── ...
```

## 🎯 給人看的資料

**主要文件**：`v1/classification_summary.md`
- ✅ 包含3個AI內部能力的詳細說明
- ✅ 包含使用方式、應用場景、CLI指令
- ✅ 包含示例輸出和網絡架構說明

## 🤖 給AI看的資料

**主要文件**：`v1/classification_data.json`
- ✅ 結構化JSON格式
- ✅ 包含完整參數定義（parameters, return_type）
- ✅ 包含CLI命令（cli_command）
- ✅ 包含結構化標籤（structured_tags）
- ✅ Schema版本：v3.3（5M AI特化）

## ⚠️ 重要提醒

1. **不要刪除舊版本**：後續diff需要比對
2. **版本從v1開始自動增長**：v1 → v2 → v3 → ...
3. **latest_classification.json** 始終指向最新版本
4. **README不需要詳細能力說明**：避免內容過多
5. **詳細說明保留在 classification_summary.md**：給人和AI參考

## 🚀 下次執行時

```bash
# 執行分析
python aiva_exploration_pipeline.py --target core --module core

# 結果：
# - 檢測到 v1 存在（max_ver = 1）
# - 創建 v2 目錄（next_ver = 2）
# - 生成新的分析結果
# - 與 v1 比對差異
# - 生成 v2/diff_report.md
# - 更新 latest_classification.json → v2
```

---

**維護者**：AIVA Team  
**狀態**：✅ 系統運行正常
