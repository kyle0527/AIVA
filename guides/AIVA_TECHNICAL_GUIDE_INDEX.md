# AIVA 技術指南索引

> **版本**: v1.0  
> **日期**: 2026-01-12  
> **用途**: 按照 AI 運作流程的 8 個階段組織所有技術指南

---

## 📖 使用說明

本索引按照 AIVA AI 內部運作的 **8 個階段** 組織技術指南：

```
HTTP Request → [1] → [2] → [3] → [4] → [5] → [6] → [7] → [8] → 學習更新
```

每個階段都有：
- ✅ **已完成的指南**：可直接參考使用
- 🔄 **待討論的問題**：需要逐步討論並建立新指南
- ⚠️ **已知問題**：標記需要修復的技術問題

---

## 🗂️ 階段式技術指南目錄

### [階段 1] 入口層 - 系統啟動與 HTTP 接收

**核心模組**: `app.py` (FastAPI)  
**職責**: 系統唯一入口，HTTP API 端點

| 指南 | 檔案 | 狀態 | 說明 |
|------|------|------|------|
| 啟動流程指南 | `stage1_system_startup.md` | 🔄 待建立 | 系統啟動、後台任務、健康檢查 |
| API 端點設計 | `stage1_api_endpoints.md` | 🔄 待建立 | RESTful API 設計規範 |
| 錯誤處理機制 | `stage1_error_handling.md` | 🔄 待建立 | 統一錯誤處理和回應格式 |

**⚠️ 已知問題**:
- Internal/External loops 未啟用 (import 路徑錯誤)
- `periodic_update` 檔案不存在

---

### [階段 2] 狀態管理層 - 服務協調與狀態管理

**核心模組**: `CoreServiceCoordinator`  
**職責**: 被動的狀態管理器和服務工廠

| 指南 | 檔案 | 狀態 | 說明 |
|------|------|------|------|
| 狀態管理架構 | `stage2_state_management.md` | 🔄 待建立 | 服務生命週期、狀態追蹤 |
| 服務工廠模式 | `stage2_service_factory.md` | 🔄 待建立 | 延遲加載、依賴注入 |
| 上下文管理 | `stage2_context_management.md` | 🔄 待建立 | 會話管理、執行上下文 |

**⚠️ 已知問題**:
- 無明顯問題，運作正常

---

### [階段 3] 命令路由層 - 智能命令分類與路由

**核心模組**: `CommandRouter`  
**職責**: 判斷 AI vs 非AI，複雜度分析

| 指南 | 檔案 | 狀態 | 說明 |
|------|------|------|------|
| 命令路由邏輯 | `stage3_command_routing.md` | 🔄 待建立 | AI/非AI 判斷、複雜度分析 |
| 命令優先級管理 | `stage3_priority_management.md` | 🔄 待建立 | 優先級計算、隊列管理 |

**⚠️ 已知問題**:
- 無明顯問題，運作正常

---

### [階段 4] 執行規劃層 - 任務步驟編排

**核心模組**: `ExecutionPlanner`  
**職責**: 創建執行計劃、步驟編排

| 指南 | 檔案 | 狀態 | 說明 |
|------|------|------|------|
| 執行計劃設計 | `stage4_execution_planning.md` | 🔄 待建立 | 步驟生成、依賴處理 |
| 資源檢查機制 | `stage4_resource_checking.md` | 🔄 待建立 | 資源可用性檢查 |

**⚠️ 已知問題**:
- TODO: 實現文本解析邏輯 (ast_parser.py:219)

---

### [階段 5] AI 決策引擎 - 核心智能決策 ⭐

**核心模組**: `CapabilityOrchestrator`  
**職責**: RAG 查詢、能力選擇、序列生成

| 指南 | 檔案 | 狀態 | 說明 |
|------|------|------|------|
| 決策流程詳解 | `stage5_decision_flow.md` | 🔄 待建立 | plan() 五步驟詳解 |
| RAG 查詢機制 | `stage5_rag_query.md` | 🔄 待建立 | 語義搜索、精確匹配 |
| **能力權重計算** | `stage5_capability_weighting.md` | 🔄 **待討論** | ⭐ **核心問題：權重計算邏輯** |
| 執行序列生成 | `stage5_sequence_generation.md` | 🔄 待建立 | 依賴排序、優先級 |
| CLI 命令轉換 | `stage5_cli_conversion.md` | 🔄 待建立 | 能力→CLI 映射 |

**⚠️ 已知問題**:
- InternalLoopConnector fallback 降低查詢質量
- **權重計算邏輯需要優化** (最重要)

---

### [階段 6] RAG 知識庫層 - 能力查詢與同步

**核心模組**: `InternalLoopConnector`  
**職責**: 內部探索結果→RAG 知識庫

| 指南 | 檔案 | 狀態 | 說明 |
|------|------|------|------|
| RAG 架構設計 | `stage6_rag_architecture.md` | 🔄 待建立 | 向量存儲、知識庫結構 |
| 能力同步機制 | `stage6_capability_sync.md` | 🔄 待建立 | 三階段管道詳解 |
| 能力分類系統 | `stage6_capability_classification.md` | 🔄 待建立 | Scope/Visibility/Access |

**⚠️ 已知問題**:
- ModuleExplorer 未實現 (用 aiva_flow_analyzer 替代)
- CapabilityAnalyzer 未實現 (用 aiva_flow_classifier 替代)
- CapabilityRegistry 未實現 (dual-write disabled)
- CapabilityEncoder 不可用時用 hash embedding

---

### [階段 7] 統一執行層 - CLI 命令執行

**核心模組**: `UnifiedExecutor`, `AsyncProcessManager`  
**職責**: 執行 CLI 命令、收集遙測數據

| 指南 | 檔案 | 狀態 | 說明 |
|------|------|------|------|
| 異步執行機制 | `stage7_async_execution.md` | 🔄 待建立 | AsyncProcessManager 詳解 |
| **遙測數據收集** | `stage7_telemetry_collection.md` | 🔄 **待討論** | ⭐ HTTP 狀態碼、WAF 檢測 |
| 超時與錯誤處理 | `stage7_timeout_error.md` | 🔄 待建立 | 超時機制、重試邏輯 |

**⚠️ 已知問題**:
- **遙測數據未充分用於學習** (待整合)

---

### [階段 8] 學習系統 - 分析與學習整合 ⭐

**核心模組**: `ContinuousLearningEngine`, `ExternalLoopConnector`  
**職責**: 從執行結果學習、權重更新

| 指南 | 檔案 | 狀態 | 說明 |
|------|------|------|------|
| **分析與學習整合** | `stage8_analysis_learning_integration.md` | 🔄 **待討論** | ⭐ **核心問題：外部學習整合** |
| 持續學習引擎 | `stage8_continuous_learning.md` | 🔄 待建立 | 線上/批次學習 |
| 經驗管理機制 | `stage8_experience_management.md` | 🔄 待建立 | 經驗緩衝、採樣策略 |
| 獎勵函數設計 | `stage8_reward_function.md` | 🔄 **待討論** | HTTP 狀態碼→獎勵映射 |
| 權重更新策略 | `stage8_weight_update.md` | 🔄 待建立 | 權重管理器、版本控制 |

**⚠️ 已知問題**:
- **外部學習與 AI 分析未整合** (最重要)
- ExternalLearningListener 未啟用
- PyTorch 依賴缺失時訓練功能 disabled
- 內部閉環更新被禁用

---

## 🎯 通用技術指南 (跨階段適用)

| 指南 | 檔案 | 狀態 | 說明 |
|------|------|------|------|
| 錯誤處理規範 | `common_error_handling.md` | 🔄 待建立 | 統一錯誤處理機制 |
| 日誌記錄規範 | `common_logging.md` | 🔄 待建立 | 日誌級別、格式 |
| 性能監控指南 | `common_monitoring.md` | 🔄 待建立 | 指標收集、追蹤 |
| 測試編寫規範 | `common_testing.md` | 🔄 待建立 | 單元測試、整合測試 |
| 數據模型規範 | `common_data_models.md` | 🔄 待建立 | Pydantic 模型設計 |

---

## 🔥 優先級討論順序 (建議)

基於目前討論和已知問題，建議按此順序逐步建立指南：

### P0 - 核心架構問題
1. ⭐ **[階段 8] 分析與學習整合** (`stage8_analysis_learning_integration.md`)
   - **問題**: 外部學習與 AI 分析分離
   - **討論重點**: 如何將 HTTP 狀態碼等遙測數據同步轉為學習信號
   
2. ⭐ **[階段 5] 能力權重計算** (`stage5_capability_weighting.md`)
   - **問題**: _select_best_capabilities() 權重邏輯待優化
   - **討論重點**: 權重計算公式、歷史表現整合

### P1 - 連接層問題
3. **[階段 1] 啟動流程指南** (`stage1_system_startup.md`)
   - **問題**: Internal/External loops 未啟用
   - **討論重點**: 正確的 import 路徑、periodic_update 實現

4. **[階段 6] 能力同步機制** (`stage6_capability_sync.md`)
   - **問題**: 多個組件未實現（ModuleExplorer, CapabilityAnalyzer）
   - **討論重點**: 確認替代方案是否完整

### P2 - 功能優化
5. **[階段 7] 遙測數據收集** (`stage7_telemetry_collection.md`)
   - **問題**: 數據收集完整但未充分利用
   - **討論重點**: 遙測數據結構設計

6. **[階段 8] 獎勵函數設計** (`stage8_reward_function.md`)
   - **問題**: HTTP 狀態碼→獎勵的映射規則
   - **討論重點**: 200/403/500 的獎勵值設計

---

## 📝 指南建立流程

當我們討論某個階段時，會經歷以下流程：

```
1. 確認問題 → 2. 討論解決方案 → 3. 建立技術指南 → 4. 更新索引
```

每個指南包含：
- **問題描述**: 當前狀態和待解決問題
- **架構設計**: 推薦的技術方案
- **實現細節**: 代碼範例和最佳實踐
- **測試策略**: 如何驗證實現正確
- **相關指南**: 連結到其他階段指南

---

## 📚 現有指南 (guides/ 目錄)

以下指南已經存在，可能需要整合到階段式體系中：

- `DUAL_LOOP_DESIGN_GUIDE.md` - 雙重閉環設計 (與階段 6, 8 相關)
- `DUAL_LOOP_OPERATION_GUIDE.md` - 雙重閉環操作 (與階段 6, 8 相關)
- `INTERNAL_LOOP_EXECUTION_GUIDE.md` - 內部閉環執行 (與階段 6 相關)

---

## 🔄 更新記錄

| 日期 | 版本 | 變更說明 |
|------|------|----------|
| 2026-01-12 | v1.0 | 初始版本，建立 8 階段索引結構 |

---

**下一步**: 請選擇一個階段開始討論，我們會逐步建立對應的技術指南。

**建議從 P0 開始**: `stage8_analysis_learning_integration.md` 或 `stage5_capability_weighting.md`
