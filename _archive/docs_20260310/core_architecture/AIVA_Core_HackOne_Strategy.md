# AIVA Core 架構優化與 HackOne 戰略規劃書：去語意化反射引擎 (De-semanticized Reflex Engine)

版本: v2.0 (Expanded & Comprehensive)

日期: 2026-01-08

目標: 針對 HackOne 黑盒測試的高強度競爭環境，將 AIVA Core
從一個依賴通用語言模型的「AI
助理」徹底轉型為一個「特化型特徵-動作反射系統」。本計畫旨在實現極致的低延遲決策、手術刀般的精確度，以及絕對的人類指揮權，以應對現代
Web 應用程式的複雜防禦機制。

## 1. 核心戰略：去語意化 RAG 架構 (De-semanticized RAG Architecture)

### 1.1 為什麼要這樣做？ (The Strategic Necessity)

在競爭激烈的 Bug Bounty (如 HackOne、Bugcrowd)
領域中，時間窗口極短，防禦機制極其敏感。目前的 AIVA
架構在設計上過度依賴大型語言模型 (LLM) 進行自然語言處理 (NLP)
來「理解」任務。這種「通用型智慧」在實際的黑盒滲透測試中，暴露出了嚴重的效能瓶頸與邏輯缺陷，具體分析如下：

- 算力與時間的無效消耗 (Inefficient Resource Allocation)：\
  讓 AI 去閱讀並理解 \"SQL Injection 是什麼\" 或者
  \"如何構造一個閉合單引號的 Payload\"
  對於成功執行一次注入攻擊毫無幫助，這就像是在戰場上閱讀槍械說明書。通用模型通常需要花費數秒鐘來生成一段解釋性的推理文字，然而在自動化攻擊中，一個經過精心設計、毫秒級響應的正則表達式
  (Regex) 匹配，往往比數億參數的模型推論來得更快速且準確。我們不需要 AI
  當教授，我們需要它當狙擊手。

- 跨語言與維護的災難 (Polyglot Maintenance Nightmare)：\
  我們的工具庫 (features_ready) 是一個異質化的生態系統，包含了 Python
  (function_sqli 邏輯層)、Go (go_tools 高併發掃描)、Rust
  (function_crypto 加密運算)
  等多種語言編寫的模組。若依賴語意理解，我們必須為每種語言、每個工具編寫詳細的
  Prompt 說明書，教導 LLM 如何調用它們。當工具版本更新或參數變更時，這些
  Prompt 也必須同步更新，這造成了極高的維護成本與潛在的「認知裂痕」。

- 不可控的幻覺風險 (Hallucination Risks in Security)：\
  通用 LLM
  的訓練目標是「預測下一個字」並傾向於「取悅」使用者。在資安測試中，這是一個危險的特質。模型可能會在沒有找到漏洞時，為了符合使用者的期待而編造出虛假的漏洞報告（False
  Positive）；或者更糟，它可能會誤解指令，在未經明確授權的情況下執行高風險操作（例如：誤將
  SELECT 語句優化為 DROP
  TABLE，或對生產環境發送過高頻率的請求），這將直接導致違反 HackOne
  的交戰規則 (Rules of Engagement)，甚至引發法律問題。

- 黑盒測試的本質 (The Nature of Black-Box Testing)：\
  黑盒測試本質上不是一個語言學遊戲，而是一個「輸入-輸出」的統計學與訊號處理遊戲。攻擊者的目標是尋找系統行為的異常（Anomaly
  Detection）。當我們輸入一個單引號
  \'，伺服器回應長度改變了，這是一個「數值訊號」，而不是一段「文字故事」。因此，我們需要的不是一個能寫詩的語言模型，而是一個能精準識別「異常訊號」並將其映射到「最佳對應工具」的高維度數學模型。

### 1.2 解決方案：特徵-動作 映射引擎 (Feature-Action Mapping Engine)

為了徹底解決上述問題，我們將系統重構，建立一套類似生物「反射神經」的機制，跳過繁瑣的認知過程，直接連接感知與行動：

- 去語意化 (De-semanticization)：\
  AI
  不再閱讀人類語言，也不嘗試理解業務邏輯的文字描述。系統將目標網站的狀態------包括
  HTTP 狀態碼 (Status Code)、標頭特徵 (Headers)、回應長度 (Body
  Length)、HTML 結構雜湊 (Structure Hash)、WAF 指紋 (WAF
  Signature)------編碼為一組純數值的「環境特徵向量 (Environment Feature
  Vector)」。例如，一個回傳 500 錯誤且包含 MySQL
  關鍵字的頁面，可能會被編碼為 \[1.0, 0.0, 1.0, 0.5, \...\]。

- 向量檢索決策 (RAG-Driven Decision)：\
  AI 透過檢索向量資料庫 (Vector DB)，將當前的「環境特徵向量」作為查詢鍵
  (Query
  Key)，去尋找資料庫中「在相似特徵下，歷史成功率最高」或「專家定義權重最匹配」的工具配置。這是一個純數學的相似度計算過程
  (Cosine Similarity)，完全不涉及語言推論，因此速度極快且結果確定性高。

- 人機分離協議 (Human-in-the-Loop Protocol)：\
  我們將決策權責進行了明確的切割：

  - **機器
    (Machine)**：負責廣度篩選、特徵識別與戰術推薦。它產出的是一份結構化的
    JSON 建議書，告訴你「根據數據，我建議這樣做」。

  - **人類 (Human)**：負責深度決策、風險評估與審核。透過 CLI
    介面，人類可以快速瀏覽機器的建議，並透過簡單的指令調整參數（如攻擊深度、速率限制），確保所有行動都在掌控之中。

## 2. 檔案與程式碼調整詳細清單 (Implementation Blueprint)

### 2.1 核心定義層：建立 capability.json (單一真理來源)

在 features_ready/ 下的每個工具子目錄中，必須新增一個標準化的
capability.json。這個檔案將取代原本分散在 config.py、Python Docstrings
或 YAML Manifest 中的設定，成為連接 RAG 系統（機器視角）、CLI
介面（人類視角）與執行
Worker（系統視角）的唯一橋樑。這確保了系統的一致性，並簡化了新工具的整合流程。

- **檔案路徑範例**: features_ready/function_sqli/capability.json

#### JSON **結構詳解與擴充說明：**

{\
\"meta\": {\
\"id\": \"sqli_error_based_mysql\",\
\"name\": \"MySQL Error-Based Injection\",\
\"description\": \"針對回傳 MySQL
錯誤訊息的頁面進行注入測試，適用於偵錯模式未關閉的目標。\",\
\"version\": \"1.0.0\",\
\"author\": \"User\",\
\"tags\": \[\"sqli\", \"mysql\", \"high-risk\"\]\
},\
\"rag_trigger\": {\
\"desc\": \"AI
決策權重表：系統會將目標網站的特徵與此權重表進行點積運算，決定推薦分數。\",\
\"weights\": {\
\"http_status_500\": 2.5, // 目標回傳 500 時，強烈推薦此工具\
\"keyword_mysql_error\": 5.0, // 出現 MySQL 特定錯誤訊息，極度推薦\
\"keyword_syntax_error\": 3.0, // 出現通用語法錯誤，高度推薦\
\"waf_detected\": -2.0, // 若偵測到 WAF，降低此工具權重（應改用 Tamper
版本）\
\"latency_high\": -0.5, // 高延遲環境下略微降低權重，避免 Timeout\
\"tech_stack_php\": 1.0 // 若偵測到 PHP 技術棧，增加權重\
},\
\"threshold\": 0.7 // 只有當計算出的匹配分數 \> 0.7 時才列入推薦清單\
},\
\"parameters\": {\
\"desc\": \"CLI 互動定義層：用於驗證使用者輸入並生成 Help 訊息。\",\
\"schema\": {\
\"level\": {\
\"type\": \"int\",\
\"default\": 2,\
\"min\": 1,\
\"max\": 5,\
\"help\": \"測試深度 (1=基本測試, 5=包含時間盲注與暴力測試)\"\
},\
\"risk\": {\
\"type\": \"int\",\
\"default\": 1,\
\"min\": 1,\
\"max\": 3,\
\"help\": \"風險等級 (1=無害, 3=可能修改數據)\"\
},\
\"tamper\": {\
\"type\": \"string\",\
\"default\": \"\",\
\"options\": \[\"\", \"space2comment\", \"between\", \"randomcase\",
\"charencode\"\],\
\"help\": \"WAF 繞過腳本名稱，對應 features_ready/function_sqli/tamper/
下的腳本\"\
},\
\"threads\": {\
\"type\": \"int\",\
\"default\": 1,\
\"max\": 10,\
\"help\": \"併發執行緒數量\"\
}\
}\
},\
\"execution\": {\
\"desc\": \"Worker 執行層：定義如何調用底層 Python/Go/Rust 代碼。\",\
\"type\": \"python_module\", // 支援 python_module, binary_exec,
shell_script\
\"module_path\": \"features_ready.function_sqli.worker\",\
\"entry_function\": \"run_scan\",\
\"timeout_seconds\": 300,\
\"resource_limit\": {\
\"cpu\": \"1.0\",\
\"memory\": \"512Mi\"\
},\
\"env_vars\": {\
\"SQLMAP_OUTPUT_PATH\": \"/tmp/aiva_results\"\
}\
}\
}

### 2.2 核心邏輯層：修改 aiva_core

#### A. 能力註冊中心 (core_capabilities/capability_registry.py)

- **現狀問題**: 目前的註冊機制依賴 Python 的 import
  語法，這意味著要新增一個工具，必須修改代碼並處理依賴關係。這導致載入速度慢，且難以統一管理非
  Python 語言編寫的工具。

- **修改內容**: 將整個 Registry 重寫為一個高效的 JSON Loader。

- **新邏輯流程**:

  1.  **啟動掃描**: 系統啟動時，使用
      glob.glob(\"features_ready/\*\*/capability.json\")
      快速遍歷目錄，不需要載入實際的程式碼，只讀取設定檔。

  2.  **雙向索引構建 (Two-Way Indexing)**:

      - **Machine Index (Vector DB)**: 提取每個 capability.json 中的
        rag_trigger.weights，將其正規化為特徵向量，存入 ChromaDB
        或高效的內存索引結構 (如 Faiss) 中。這是 AI 檢索的依據。

      - **Human Index (Metadata Map)**: 提取 meta 與 parameters
        區塊，存入一個全域字典 Dict\[ToolID, ToolInfo\]。這是 CLI
        工具用來顯示說明、驗證參數以及生成幫助文檔的依據。

  3.  **熱重載 (Hot-Reload)**: 實作檔案系統監控 (File
      Watcher)，當開發者編輯了某個 capability.json
      檔案後，系統能即時更新內存中的索引，無需重啟主程式，實現真正的「熱插拔」開發體驗。

#### B. 任務生成器 (task_planning/planner/task_generator.py)

- **現狀問題**: 目前的生成器傾向於直接生成可執行的 AICommand
  對象，缺乏一個中間的「計畫審核層」。這導致一旦生成，就難以修改。

- **新功能**: 生成標準化的 **JSON 攻擊計畫 (Attack Plan Schema)**。

- **運作邏輯**:

  - 接收 RAG 引擎檢索出的 Top-K Tool_ID。

  - 從 Registry 中讀取該 Tool 的 parameters.schema。

  - 使用 Schema 中的 default 值填充參數，並根據目標環境（如
    WAF）自動調整部分參數（例如：若偵測到 WAF，自動選擇 tamper 腳本）。

  - 產出如下的 Plan Draft，狀態標記為 PENDING_APPROVAL，等待人類確認：\
    {\
    \"plan_id\": \"plan_xp92_alpha\",\
    \"target\":
    \"\[https://target.com/vuln.php?id=1\](https://target.com/vuln.php?id=1)\",\
    \"timestamp\": \"2026-01-08T10:00:00Z\",\
    \"context\": {\
    \"detected_signals\": \[\"status_500\", \"mysql_error\",
    \"waf_absent\"\],\
    \"confidence_score\": 0.92\
    },\
    \"proposed_action\": {\
    \"tool_id\": \"sqli_error_based_mysql\",\
    \"reason\": \"Matching Score: 0.92 (High keyword match)\",\
    \"params\": {\
    \"level\": 2,\
    \"risk\": 1,\
    \"tamper\": \"\",\
    \"threads\": 1\
    }\
    },\
    \"status\": \"PENDING_APPROVAL\"\
    }

#### C. RAG 引擎與特徵工程 (cognitive_core/rag/vector_store.py)

- **核心變革**: 這是從 NLP 轉向數值反射的關鍵技術點。

- **移除**: 徹底移除 BERT、OpenAI Embedding
  或其他語言模型的調用。這些模型對於我們的特徵向量來說太重且不必要。

- **新增**: **特徵雜湊 (Feature Hashing) 編碼器**。

  - 定義一個固定的、語意無關的特徵空間 (例如 128 維或 256 維)。

  - 建立映射規則：

    - http_status_500 映射到維度 \[0\]。

    - waf_detected 映射到維度 \[1\]。

    - keyword_syntax 映射到維度 \[2\]。

    - param_id 映射到維度 \[3\]。

  - 這種雜湊映射是確定性的，不需要訓練，運算速度極快。

- **查詢邏輯**:

  - 當 analysis_engine 完成初步掃描，回報狀態為 {\"http_status_500\":
    true, \"waf\": false}。

  - RAG 引擎立即生成查詢向量 \[1, 0, 0, \...\]。

  - 計算此查詢向量與所有 capability.json 中 rag_trigger 向量的餘弦相似度
    (Cosine Similarity)，選出分數最高的工具。

### 2.3 互動介面層：CLI 工具 (aiva-cli)

CLI
將成為您控制整個系統的「駕駛艙」。我們摒棄了自然語言對話的模糊性，所有互動都被限制在嚴格定義的指令協議中，確保效率與準確性。

- **aiva-cli plan view \[target\]**:

  - 觸發掃描、特徵提取、RAG 檢索流程。

  - 以易讀的表格格式 (Table Format) 顯示 AI
    推薦的工具、匹配理由以及預設參數。

  - 顯示目前的計畫 ID 與狀態。

- **aiva-cli plan edit \--set \[path\]=\[value\]**:

  - 這是「人類介入」的關鍵指令。直接修改 JSON 計畫中的節點。

  - 範例：aiva-cli plan edit \--set params.level=5 \--set
    params.tamper=space2comment。

  - 系統會即時根據 capability.json 的 Schema 自動檢查輸入值（例如
    level=5 是否在 min/max 範圍內），防止設定錯誤導致執行失敗。

- **aiva-cli plan execute**:

  - 將最終確認、人類審核過的 JSON 發送給 UnifiedAttackExecutor。

  - 執行器根據 JSON 中的 execution 區塊，精確調用對應的 Worker
    進行攻擊。

## 3. 預計成果 (Expected Outcomes)

### ✅ 極限算力優化 **(Extreme Efficiency)**

- **參數需求降至 \< 100k**: 整個決策系統僅需運作一個輕量的向量檢索系統
  (Vector Search) 和特徵編碼器 (Feature Encoder)。相比於動輒數十億參數的
  LLM，這個架構幾乎不佔用顯存。

- **無需 GPU**: 可以在任何普通的筆記本電腦、低階 VPS
  甚至樹莓派上流暢運行，極大地降低了運營成本。

- **毫秒級決策**: 相比於 LLM 需要數秒生成文字，向量搜索與 JSON 生成可在
  10-50ms
  內完成，這意味著在大規模掃描任務中，決策不會成為瓶頸，大幅提升吞吐量
  (Throughput)。

### ✅ 高度擴展性 (Hot-swappable Extensibility)

- **零代碼適配**: 當您發現一個新的 CVE 或漏洞類型（如 GraphQL
  注入、WebSocket 漏洞），完全不需要修改 AIVA 核心的 Python 代碼。

- **即插即用流程**:

  1.  編寫 worker.py (攻擊邏輯，可以是簡單的 PoC 腳本)。

  2.  編寫 capability.json (定義特徵權重 rag_trigger，告訴 AI
      何時使用)。

  3.  將資料夾丟入 features_ready/。

  4.  系統自動索引，下次遇到符合特徵的目標時，AI
      就會自動推薦這個新工具。這讓您的武器庫可以隨著新的安全威脅即時演進。

### ✅ 精確且安全的人類控制 (Precise **Human Control)**

- 解決了 AI 系統常見的「黑盒不可控」問題。透過 **JSON
  表單審核機制**，您可以在攻擊封包真正發送前，攔截任何可能的高風險操作（例如：阻止
  AI 對生產資料庫執行 DELETE）。

- 確保操作完全符合 HackOne 的 RoE (交戰規則)，防止因 AI
  誤判（例如誤掃描了禁止測試的子網域）導致的違規，保護您的帳號安全。

### **✅ 跨語言無縫整合 (Language Agnostic Integration)**

- AIVA Core 不再在乎底層工具是用什麼語言寫的。無論是 Python
  (features_ready/function_xss), Rust (function_crypto), Go 還是 Shell
  Script。

- 對 AIVA 來說，它們都只是 capability.json 中 execution
  區塊的一行設定指令。這讓您可以整合 GitHub
  上任何優秀的開源資安工具，而不受限於語言生態。

## 4. 下一步行動 (Execution Roadmap)

1.  **Freeze (凍結與準備)**:

    - 鎖定 aiva_core 當前程式碼狀態，停止開發新功能，建立
      refactor/rag-reflex 分支。

    - 備份現有的 features_ready 目錄。

2.  **Define (定義與標準化)**:

    - 優先為 features_ready/function_sqli 編寫 capability.json 作為原型
      (Prototype)，定義 SQL 注入的特徵向量。

    - 接著為 function_xss (XSS 攻擊), function_ssrf (服務端請求偽造)
      建立定義檔，覆蓋主要攻擊面。

3.  **Refactor (核心重構)**:

    - 修改 core_capabilities/capability_registry.py，移除 Python Import
      邏輯，實作 JSON 載入與向量化索引功能。

    - 實作 task_planning 中的 Plan Generator 與 JSON Schema
      驗證邏輯，確保生成的計畫格式正確。

    - 更新 rag_engine 以支援特徵雜湊 (Feature Hashing) 查詢。

4.  **Interface (介面開發)**:

    - 更新 CLI 工具 (aiva_cli.py)，移除舊的對話邏輯，新增支援 plan view,
      plan edit, plan execute 指令集。

5.  **Test (整合測試)**:

    - 在靶場環境 (如 DVWA, PortSwigger Labs 或自行搭建的 Docker 靶機)
      中進行端對端測試。

    - 驗證完整的「特徵觸發 -\> RAG 檢索 -\> 計畫生成 -\> CLI 修改 -\>
      執行 -\> 結果回饋」迴路。

    - 確認系統能否正確「反射」：看到 SQL 錯誤就推薦 SQL 工具，看到 URL
      參數就推薦 XSS 工具。
