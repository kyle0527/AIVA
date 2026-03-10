# AIVA Internal Exploration 操作手冊

> **版本**: v10.0.0  
> **更新日期**: 2025-12-10  
> **狀態**: ✅ 生產就緒  
> **新增**: Self-Healing 自我修復模組 🌟  
> **代碼品質**: Zero Errors - 所有工具通過 Pylance + SonarLint 檢查

## 📑 目錄

- [系統概述](#系統概述)
- [快速開始](#快速開始)
- [核心組件](#核心組件)
- [使用指南](#使用指南)
- [故障排除](#故障排除)
- [開發指南](#開發指南)

---

## 系統概述

### 架構說明

AIVA Internal Exploration 是一個**三階段自動化管道系統**，用於分析、分類和執行 AIVA 系統的內部能力流程。

```
┌─────────────────────────────────────────────────────────────┐
│                 AIVA Exploration Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│  Stage 1         Stage 2           Stage 3                  │
│  ┌──────────┐   ┌──────────┐     ┌──────────┐             │
│  │ Analyzer │──▶│Classifier│────▶│ Executor │             │
│  └──────────┘   └──────────┘     └──────────┘             │
│       │              │                  │                    │
│   AST 分析      能力分類         互動執行                    │
└─────────────────────────────────────────────────────────────┘
```

### 三階段說明

1. **Stage 1 - Analyzer** (`aiva_flow_analyzer.py`)
   - AST 語法樹分析
   - 函數調用鏈追蹤
   - 數據流路徑提取

2. **Stage 2 - Classifier** (`aiva_flow_classifier.py`)
   - 能力分類標記
   - 模組歸屬分析
   - 多路徑檢測

3. **Stage 3 - Executor** (`aiva_cli_implementation.py`)
   - 互動式選單
   - 能力執行引擎
   - Dry-run 模式

---

## 快速開始

### 方式一：一鍵啟動（推薦）

```batch
# Windows 批次檔
.\啟動Pipeline.bat
```

### 方式二：完整管道執行

```bash
# 1. 執行完整三階段管道
python aiva_exploration_pipeline.py --target core

# 2. 使用生成的分類數據
python aiva_cli_implementation.py --menu
```

### 方式三：單獨執行各階段

```bash
# Stage 1: 分析
python aiva_flow_analyzer.py --target core --depth 5

# Stage 2: 分類
python aiva_flow_classifier.py --input aiva_flow_analysis/flows.json

# Stage 3: 執行
python aiva_cli_implementation.py --menu
```

---

## 核心組件

### 1. ExplorationPipeline (管道控制器)

**檔案**: `aiva_exploration_pipeline.py`

**功能**:
- 版本控制管理 (`analysis_history/v1`, `v2`, ...)
- 自動觸發三階段分析
- 差異報告生成
- `latest_classification.json` 符號連結維護

**使用範例**:

```python
from aiva_exploration_pipeline import ExplorationPipeline

# 初始化管道
pipeline = ExplorationPipeline(target_path="core")

# 執行完整流程
success = pipeline.run()

# 查看版本歷史
# analysis_history/
#   ├── v1/
#   ├── v2/
#   └── latest/ -> v2
```

**命令列參數**:

```bash
# 完整模式（默認）
python aiva_exploration_pipeline.py --target core

# 指定特定模組
python aiva_exploration_pipeline.py --target cognitive_core

# 全系統掃描
python aiva_exploration_pipeline.py --target .
```

---

### 2. AIVAFlowAnalyzer (代碼分析器)

**檔案**: `aiva_flow_analyzer.py`

**核心功能**:
- Python AST 語法樹解析
- 函數調用關係圖構建
- 數據流路徑追蹤
- 入口函數識別

**輸出檔案**:
- `flows.json` - 完整流程定義
- `analysis_summary.txt` - 統計摘要
- `full_call_tree.json` - 調用樹

**使用範例**:

```python
from aiva_flow_analyzer import AIVAFlowAnalyzer

analyzer = AIVAFlowAnalyzer(target_dir="services/core")

# 分析目標目錄
results = analyzer.analyze_directory(
    target="aiva_core",
    depth=5,
    verbose=True
)

# 保存結果
analyzer.save_results(output_dir="output")
```

**關鍵參數**:
- `target`: 目標目錄/模組
- `depth`: 調用鏈最大深度（建議 3-5）
- `verbose`: 詳細輸出模式

---

### 3. AIVAFlowClassifier (能力分類器)

**檔案**: `aiva_flow_classifier.py`

**功能**:
- 六大模組分類（cognitive_core, internal_exploration, ...）
- 多路徑變體識別
- 智能描述生成
- Markdown 報告輸出

**輸出檔案**:
- `classification_data.json` - 分類後數據（Executor 使用）
- `classification_report.md` - 人類可讀報告
- `complete_flow_details.md` - 完整流程列表

**使用範例**:

```python
from aiva_flow_classifier import AIVAFlowClassifier

classifier = AIVAFlowClassifier(
    flows_json="aiva_flow_analysis/flows.json"
)

# 執行分類
classifier.classify_flows()

# 生成報告
classifier.generate_reports()
```

**分類規則**:
- 依據模組路徑自動分類
- 支援自定義關鍵字匹配
- 多路徑聚合（同起點終點視為變體）

---

### 4. FlowExecutor (執行引擎)

**檔案**: `aiva_cli_implementation.py`

**功能**:
- 互動式雙層選單（模組 → 能力）
- Dry-run 模式（查看不執行）
- 動態類別載入
- 錯誤處理與回滾

**使用範例**:

```python
from aiva_cli_implementation import FlowExecutor

executor = FlowExecutor(
    json_path="latest_classification.json"
)

# 互動式選單
executor.interactive_menu()

# 直接執行特定 Flow
executor.execute_flow(flow_id=42, dry_run=False)
```

**選單操作**:

```
===== AIVA 能力瀏覽器 =====
[1] 認知核心模組        | 45 種能力 (120 路徑)
[2] 內探模組            | 32 種能力 (89 路徑)
[3] 任務規劃模組        | 28 種能力 (76 路徑)
...
[0] 離開

> 請選擇模組: 1

===== 認知核心模組 - 能力列表 =====
[1] process_command → execute_task    (3 變體)
[2] analyze_context → plan_strategy   (2 變體)
...

> 輸入指令: 1      # 執行
> 輸入指令: v 1    # 預覽（dry-run）
```

---

## 使用指南

### 場景 1：首次使用

```bash
# 1. 執行完整管道生成數據
cd services/core/aiva_core/internal_exploration
python aiva_exploration_pipeline.py --target core

# 2. 啟動互動選單
python aiva_cli_implementation.py --menu

# 3. 瀏覽並測試能力
# 在選單中選擇模組 → 選擇能力 → 執行或預覽
```

### 場景 2：代碼更新後重新分析

```bash
# 使用管道自動版本控制
python aiva_exploration_pipeline.py --target core

# 會自動：
# 1. 創建新版本目錄（v3, v4, ...）
# 2. 執行完整分析
# 3. 生成差異報告（diff_report.md）
# 4. 更新 latest 符號連結
```

### 場景 3：僅分析特定模組

```bash
# 分析 cognitive_core
python aiva_exploration_pipeline.py --target cognitive_core

# 分析 external_learning
python aiva_exploration_pipeline.py --target external_learning
```

### 場景 4：開發調試模式

```bash
# Stage 1: 詳細輸出
python aiva_flow_analyzer.py --target core --depth 5 --verbose

# Stage 2: 檢查分類結果
python aiva_flow_classifier.py --input aiva_flow_analysis/flows.json
cat classification_report.md  # 查看報告

# Stage 3: Dry-run 測試
python aiva_cli_implementation.py --data classification_data.json
# 在選單中使用 "v <編號>" 進行 dry-run
```

---

## 故障排除

### 問題 1：找不到模組

**錯誤訊息**:
```
ModuleNotFoundError: No module named 'aiva_flow_analyzer'
```

**解決方法**:
```bash
# 設定 PYTHONPATH
$env:PYTHONPATH="C:\D\fold7\AIVA-git\services\core;C:\D\fold7\AIVA-git\services"
python aiva_exploration_pipeline.py --target core
```

### 問題 2：找不到 flows.json

**錯誤訊息**:
```
[Error] 找不到設定檔: flows.json
```

**解決方法**:
```bash
# 先執行 Analyzer 生成數據
python aiva_flow_analyzer.py --target core --depth 5

# 然後執行 Classifier
python aiva_flow_classifier.py --input aiva_flow_analysis/flows.json
```

### 問題 3：執行 Flow 失敗

**錯誤訊息**:
```
[Error] 執行失敗: 找不到類別 XXX
```

**可能原因與解決**:
1. **類別不存在** - 檢查 `flows.json` 中的類別路徑是否正確
2. **導入錯誤** - 確認 PYTHONPATH 包含正確路徑
3. **版本不匹配** - 重新執行管道更新數據

**驗證步驟**:
```bash
# 1. 使用 dry-run 模式
python aiva_cli_implementation.py --menu
# 輸入 "v <編號>" 查看會執行什麼

# 2. 檢查類別是否存在
python -c "from services.core.aiva_core.cognitive_core.XXX import YYY; print('OK')"
```

### 問題 4：分析結果為空

**症狀**: `flows.json` 只有少量或無數據

**解決方法**:
```bash
# 1. 增加分析深度
python aiva_flow_analyzer.py --target core --depth 7

# 2. 檢查目標路徑
python aiva_flow_analyzer.py --target . --depth 5  # 全系統掃描

# 3. 使用詳細模式查看問題
python aiva_flow_analyzer.py --target core --depth 5 --verbose
```

---

## 開發指南

### 添加新的分類規則

編輯 `aiva_flow_classifier.py`:

```python
# 在 MODULES 字典中添加新模組
MODULES = {
    "cognitive_core": "認知核心模組",
    "your_new_module": "新模組名稱",  # 新增
    # ...
}

# 在 SCRIPT_DESCRIPTIONS 中添加描述
SCRIPT_DESCRIPTIONS = {
    "your_script.py": "腳本功能說明",
    # ...
}
```

### 自定義分析器過濾規則

編輯 `aiva_flow_analyzer.py`:

```python
# 修改 _should_skip_file 方法
def _should_skip_file(self, file_path: Path) -> bool:
    skip_patterns = [
        "__pycache__",
        ".pyc",
        "test_",  # 跳過測試文件
        "your_custom_pattern",  # 自定義跳過規則
    ]
    # ...
```

### 整合到 AI Core

在 `internal_loop_connector.py` 中已整合：

```python
from ..internal_exploration.aiva_exploration_pipeline import ExplorationPipeline

# 在 sync_capabilities_to_rag 中自動觸發
async def sync_capabilities_to_rag(self, force_refresh: bool = False):
    if force_refresh:
        pipeline = ExplorationPipeline(target_path="core")
        success = await asyncio.to_thread(pipeline.run)
    # ...
```

### 版本控制說明

目錄結構：
```
analysis_history/
├── v1/
│   ├── aiva_flow_analysis/
│   │   ├── flows.json
│   │   └── analysis_summary.txt
│   └── classification_data.json
├── v2/
│   └── ...
└── latest -> v2  # 符號連結指向最新版本
```

管道自動管理版本：
- 每次運行創建新版本
- `latest` 始終指向最新
- `FlowExecutor` 預設使用 `latest_classification.json`

---

## 檔案清單

| 檔案名稱 | 用途 | 必要性 |
|---------|------|--------|
| `aiva_exploration_pipeline.py` | 管道控制器 | ✅ 核心 |
| `aiva_flow_analyzer.py` | 代碼分析器 | ✅ 核心 |
| `aiva_flow_classifier.py` | 能力分類器 | ✅ 核心 |
| `aiva_cli_implementation.py` | 執行引擎 | ✅ 核心 |
| `__init__.py` | 模組初始化 | ✅ 必須 |
| `README.md` | 模組說明 | 📄 文檔 |
| `OPERATION_MANUAL.md` | 操作手冊 | 📄 文檔 |
| `啟動Pipeline.bat` | 快速啟動 | 🚀 便利 |

---

## 常用命令速查

```bash
# 快速啟動
.\啟動Pipeline.bat

# 完整管道
python aiva_exploration_pipeline.py --target core

# 互動選單
python aiva_cli_implementation.py --menu

# 單獨分析
python aiva_flow_analyzer.py --target core --depth 5

# 單獨分類
python aiva_flow_classifier.py --input aiva_flow_analysis/flows.json

# 直接執行
python aiva_cli_implementation.py --id 42

# Dry-run 預覽
python aiva_cli_implementation.py --id 42 --dry-run
```

---

## 維護建議

1. **定期重新分析** - 代碼更新後運行管道
2. **檢查差異報告** - 查看 `diff_report.md` 了解變更
3. **清理舊版本** - 定期清理 `analysis_history/` 舊版本
4. **測試關鍵流程** - 使用 dry-run 驗證重要能力

---

**最後更新**: 2025-12-10  
**維護者**: AIVA Development Team
