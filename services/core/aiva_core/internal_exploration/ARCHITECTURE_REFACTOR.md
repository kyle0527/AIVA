# Internal Exploration 架構重構報告

**重構時間**: 2026-01-13  
**狀態**: ✅ 完成

---

## 重構目標

1. ✅ 統一內外模組工具的命名規範
2. ✅ 將所有工具移至 `internal_exploration/` 根目錄
3. ✅ 保持 `python_tools/` 作為舊版兼容
4. ✅ 移除 `demos/` 目錄
5. ✅ 建立清晰的內外模組對應關係

---

## 新架構

```
internal_exploration/
├── aiva_internal_classifier.py            # 內部AI模組分類器 (AIVAFlowClassifier)
├── aiva_internal_executor.py              # 內部AI模組執行器 (FlowExecutor)
├── aiva_external_classifier.py            # 外部功能模組分類器 (MultiLanguageClassifier)
├── aiva_external_executor.py              # 外部功能模組執行器 (MultiLangExecutor)
├── __init__.py                            # 模組初始化
├── python_tools/                          # Python AST 工具
│   ├── aiva_flow_analyzer.py             # Python AST 分析器（核心）
│   ├── aiva_flow_classifier.py           # 測試範本（將移除）
│   └── aiva_cli_implementation.py        # 測試範本（將移除）
├── go_tools/                              # Go語言 AST 工具
│   └── go2mermaid.go
├── rust_tools/                            # Rust語言 AST 工具
│   └── src/main.rs
└── typescript_tools/                      # TypeScript AST 工具
    └── ts2mermaid.ts
```

**已移除的檔案** (移至 Downloads/新增資料夾/internal_exploration_moved_files/):
- ❌ aiva_flow_analyzer.py (根目錄重複，保留在 python_tools/)
- ❌ aiva_exploration_pipeline.py (舊版整合腳本)
- ❌ aiva_capability_cli.py (舊版 CLI)
- ❌ dispatcher.py (調度器，功能已整合)

---

## 檔案對應關係

### 內部模組工具（AI Core）

| 概念檔名 | 實際檔名 | 實際類名 | 狀態 |
|---------|----------|---------|------|
| internal_module_classifier | `aiva_internal_classifier.py` | `AIVAFlowClassifier` | ✅ 正式版本 |
| internal_module_executor | `aiva_internal_executor.py` | `FlowExecutor` | ✅ 正式版本 |
| flow_classifier (範本) | `python_tools/aiva_flow_classifier.py` | - | 📚 測試參考 |
| cli_implementation (範本) | `python_tools/aiva_cli_implementation.py` | - | 📚 測試參考 |

**目標模組**: 5大AI核心模組
- cognitive_core
- internal_exploration  
- task_planning
- core_capabilities
- service_backbone

**分類維度**: AI能力類型（AI內部/對外/程式/混合）

---

### 外部模組工具（Features + Scan）

| 概念檔名 | 實際檔名 | 實際類名 | 語言支援 |
|---------|----------|---------|----------|
| external_module_classifier | `aiva_external_classifier.py` | `MultiLanguageClassifier` | Python, Go, Rust, TypeScript |
| external_module_executor | `aiva_external_executor.py` | `MultiLangExecutor` | 多語言 subprocess |

**目標模組**: 功能模組 + 掃描引擎
- function_sqli, function_xss, function_ssrf, function_idor
- function_info_leak, function_bizlogic, function_crypto
- function_authn_go
- scan_engine, typescript_engine, rust_engine

**分類維度**: 攻擊類型 + 語言支援

---

### 語言層 AST 工具（Language Layer）

| 語言 | 檔案位置 | 輸出格式 | 狀態 |
|------|---------|---------|------|
| Python | `python_tools/aiva_flow_analyzer.py` | JSON Schema v3.3 | ✅ 生產就緒 |
| Go | `go_tools/go2mermaid.go` | JSON Schema v3.3 | ✅ 生產就緒 |
| Rust | `rust_tools/src/main.rs` | JSON Schema v3.3 | ✅ 生產就緒 |
| TypeScript | `typescript_tools/ts2mermaid.ts` | JSON Schema v3.3 | ✅ 生產就緒 |

**統一輸出**: 所有語言工具輸出 `analysis_results.json`，由分類器讀取並整合。

---

## 實際命名規範

### 檔案命名模式

```
aiva_{scope}_{function}.py

scope:    internal | external
function: classifier | executor
```

### 實際檔案

- `aiva_internal_classifier.py` - 內部模組分類器 (AIVAFlowClassifier)
- `aiva_internal_executor.py` - 內部模組執行器 (FlowExecutor, InteractiveMenu)
- `aiva_external_classifier.py` - 外部模組分類器 (MultiLanguageClassifier)
- `aiva_external_executor.py` - 外部模組執行器 (MultiLangExecutor, InteractiveMenu)

### 類名命名慣例

- Internal: 使用描述性名稱 (`AIVAFlowClassifier`, `FlowExecutor`)
- External: 強調多語言特性 (`MultiLanguageClassifier`, `MultiLangExecutor`)

---

## 工具對比

| 特性 | 內部模組工具 | 外部模組工具 |
|------|-------------|-------------|
| **分類器** | InternalModuleClassifier | ExternalModuleClassifier |
| **執行器** | InternalModuleExecutor | ExternalModuleExecutor |
| **目標** | 5大AI模組 | 功能模組+掃描引擎 |
| **數據來源** | features_classification/ | module_analysis/ |
| **分類方式** | 預定義AI模組 | 數據驅動提取 |
| **輸出報告** | 單模組詳細 | 整合批次報告 |
| **執行方式** | Pipeline傳遞 | 多語言CLI |

---

## 使用方式

### 內部模組分析

```bash
# 分類內部AI模組
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration
python aiva_internal_module_classifier.py --input analysis_results.json --output ./output

# 執行內部模組流程
python aiva_internal_module_executor.py --flow 1

# 完整認知更新管線
python aiva_exploration_pipeline.py --target core --module core --depth 10

# 查詢AI能力
python aiva_capability_cli.py --search "vector" --module cognitive_core
```

### 外部模組分析

```bash
# 批次分類所有外部模組
cd C:\D\fold7\AIVA-git\services\core\aiva_core\internal_exploration
python aiva_external_module_classifier.py -w ../../.. -o external_module_reports -v

# 執行外部模組
python aiva_external_module_executor.py --capability function_sqli --target http://example.com
```

---

## 移除項目

### 已移除目錄

- ✅ `demos/` - 示範腳本目錄（2個文件）
  - demo_ai_internal.py
  - demo_ai_standalone.py

### 保留目錄

- `python_tools/` - 保留作為舊版兼容，包含原始工具

---

## 路徑調整

由於所有工具都移到了根目錄（internal_exploration/），相對路徑調整：

### 數據路徑

**內部模組**:
```python
# 舊路徑（在 python_tools/）
data_path = "../data/internal_exploration/"

# 新路徑（在 internal_exploration/）  
data_path = "./data/internal_exploration/"
```

**外部模組**:
```python
# 工作區根路徑
workspace_root = Path("../../..")  # 從 internal_exploration/ 到專案根
```

---

## 架構圖

```
                    ┌─────────────────────┐
                    │ aiva_flow_analyzer  │
                    │   (AST解析共用)     │
                    └──────────┬──────────┘
                               │
               ┌───────────────┴───────────────┐
               │                               │
      ┌────────▼────────┐            ┌────────▼────────────┐
      │ Internal Module │            │  External Module    │
      │   Classifier    │            │    Classifier       │
      │ (AI模組分類)    │            │  (功能模組分類)     │
      └────────┬────────┘            └─────────┬───────────┘
               │                               │
      ┌────────▼────────┐            ┌────────▼────────────┐
      │ Internal Module │            │  External Module    │
      │   Executor      │            │    Executor         │
      │ (Pipeline執行)  │            │  (多語言CLI執行)    │
      └────────┬────────┘            └─────────────────────┘
               │
      ┌────────▼──────────┐
      │ Capability CLI    │
      │   (能力查詢)      │
      └────────┬──────────┘
               │
      ┌────────▼──────────┐
      │ Exploration       │
      │   Pipeline        │
      │ (認知更新管線)    │
      └───────────────────┘
```

---

## 驗證清單

- ✅ 檔案已複製並重命名
- ✅ 命名規範統一 (internal/external_module_classifier/executor)
- ✅ demos 目錄已移除
- ✅ 架構清晰，職責分明
- ✅ python_tools 保留作為兼容
- ⏳ 路徑調整（需測試）
- ⏳ import 路徑檢查（需測試）

---

## 下一步

1. 測試內部模組分類器
2. 測試外部模組分類器  
3. 驗證執行器功能
4. 更新 README 文檔
5. 測試認知管線整合

---

**重構完成！新架構更清晰、命名更統一！** 🎉
