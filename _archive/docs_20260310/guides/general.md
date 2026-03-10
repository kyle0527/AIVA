# AIVA 通用文檔 (General Documentation)

> **版本**: 2.0 | **更新日期**: 2026-01-31

## 1. 系統架構 (五大核心模組)

AIVA 系統基於「五大核心模組」架構構建，旨在實現可擴展性、自主性和跨平台能力。

### 1.1 核心模組 (Core Modules)

1.  **認知核心 (Cognitive Core)** (`services/core/aiva_core/cognitive_core`)
    *   **角色**: 系統的大腦。
    *   **功能**: 處理決策、學習和知識檢索。它整合了 5M 神經核心 (Neural Core) 和 RAG 系統。此模組已吸收了舊有的「外部學習 (External Learning)」模組的功能。
    *   **關鍵組件**: `enhanced_decision_agent.py`, `internal_loop_connector.py`。

2.  **任務規劃 (Task Planning)** (`services/core/aiva_core/task_planning`)
    *   **角色**: 策略家。
    *   **功能**: 將高層意圖分解為可執行的計劃。它使用 `PlanningDispatcher` (前身為 `TaskDispatcher`) 來路由命令。
    *   **關鍵組件**: `PlanningDispatcher`, `UnifiedExecutor`, `CommandBuilder`。

3.  **內部探索 (Internal Exploration)** (`services/core/aiva_core/internal_exploration`)
    *   **角色**: 研究員。
    *   **功能**: 探索代碼庫和系統環境，以了解可用的工具和能力。它採用雙層架構，將語言處理工具與業務邏輯執行器分開。
    *   **關鍵組件**: `aiva_internal_executor.py`, `aiva_flow_classifier.py`。

4.  **核心能力 (Core Capabilities)** (`services/core/aiva_core/core_capabilities`)
    *   **角色**: 工具箱。
    *   **功能**: 包含系統使用的基礎能力，例如對話處理和基礎分析工具。
    *   **關鍵組件**: `AIVACommandProcessor` (取代 `AIVADialogAssistant`)。

5.  **服務骨幹 (Service Backbone)** (`services/core/aiva_core/service_backbone`)
    *   **角色**: 基礎設施。
    *   **功能**: 管理訊息傳遞、資源分配和模組間通訊。
    *   **關鍵組件**: 訊息總線 (Messaging bus), 資源管理器 (Resource Manager)。

### 1.2 功能模組 (Feature Modules) (`services/features`)

AIVA 透過位於 `services/features/` 的專用功能模組擴展其核心能力。每個功能模組 (例如 `function_xss`, `function_sqli`) 都是一個獨立單元，擁有自己的 CLI 介面 (`__main__.py`)。

## 2. 使用者介面指南 (User Interface Guide)

AIVA Web UI 設計概念旨在提供一個用於監控系統和手動觸發安全模組的儀表板。

### 2.1 儀表板概覽

*   **狀態面板 (Status Panel)**: 顯示系統健康狀況、活動線程和記憶體使用量。
*   **日誌面板 (Log Panel)**: 顯示系統活動和執行結果的即時日誌。
*   **模組列表 (Module List)**: 列出可用的安全測試模組 (例如：價格操縱、IDOR、XSS 檢測)，這些模組是從配置動態加載的。

### 2.2 執行模組

執行模組的標準流程設計如下：

1.  在列表中找到所需的模組 (例如：**XSS Detection**)。
2.  點擊 **執行 (Execute)** 按鈕 (齒輪圖標)。
3.  系統將顯示配置模態框。
4.  輸入所需的參數：
    *   **目標 URL (Target URL)**: 要測試的 URL (例如 `http://localhost:3000/search`)。
    *   **參數 (Parameters)**: 特定參數，如 `product_id`, `price`, `type` (Reflected/Stored/DOM) 等。
    *   **選項 (Options)**: 用於切換功能的複選框，如「完整掃描 (Full Scan)」。
5.  在模態框中點擊 **執行 (Execute)**。
6.  監控 **日誌面板** 中的執行開始訊息 (`[INFO] 開始執行...`) 和結果。

### 2.3 CLI 使用 (進階)

模組也可以直接從專案根目錄透過 CLI 執行。這對於自動化或無頭操作非常有用。

**通用語法:**
```bash
python3 -m services.features.<function_name> [OPTIONS]
```

**範例:**

*   **XSS 檢測 (XSS Detection):**
    ```bash
    python3 -m services.features.function_xss --url "http://localhost:3000/search" --type reflected --param q
    ```

*   **SQL 注入 (SQL Injection):**
    ```bash
    python3 -m services.features.function_sqli --url "http://localhost:3000/login" --level 3
    ```

## 3. 擴展 AIVA (Extending AIVA)

### 3.1 添加新的功能模組

1.  在 `services/features/` 中創建一個新目錄 (例如 `function_new_test`)。
2.  實現 `__main__.py` 以處理 CLI 參數 (使用 `argparse` 或 `click`)。
3.  確保模組以 JSON 格式輸出結果，以便系統解析。
4.  (可選) 將模組添加到 UI 配置中，以便透過儀表板訪問。

### 3.2 更新 UI 配置

UI 是數據驅動的。要將新模組添加到儀表板：
1.  定位到 UI 的模組配置部分。
2.  向陣列中添加一個新物件：
    ```javascript
    {
        id: 'new_test',
        name: 'New Test (Description)',
        desc: 'Short description of the test',
        params: [
            { name: 'url', label: 'Target URL', type: 'url', default: 'http://...' },
            { name: 'depth', label: 'Scan Depth', type: 'number', default: '1' }
        ]
    }
    ```

## 4. 故障排除 (Troubleshooting)

*   **UI 未加載**:
    *   確保 Web 伺服器正在運行。
    *   檢查 UI 文件是否可訪問。
*   **模組執行失敗**:
    *   檢查瀏覽器控制台 (F12) 是否有 JavaScript 錯誤。
    *   驗證處理請求的後端服務是否處於活動狀態。
    *   確保 Python 環境已安裝所有依賴項 (`pip install -r requirements.txt`)。
*   **日誌未更新**:
    *   刷新頁面。
    *   檢查 WebSocket 連接 (如果適用) 或日誌輪詢機制。
*   **CLI 錯誤**:
    *   確保 `PYTHONPATH` 包含專案根目錄。
    *   從根目錄運行: `export PYTHONPATH=$PYTHONPATH:.`

## 5. 歸檔 (Archive)

引用舊有「六大模組 (Six Modules)」架構或將「外部學習 (External Learning)」作為獨立頂層模組的過時文檔可以在 `docs/_archive/` 中找到。
