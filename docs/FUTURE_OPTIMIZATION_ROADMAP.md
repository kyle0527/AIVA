# AIVA 未來優化與擴展技術藍圖 (Future Optimization Roadmap)

本文件紀錄了在 AIVA v2 階段為了「確保核心系統首發穩定與可用性」，而暫時被移除、凍結或是簡化的進階技術與模組。當未來開發人力充足，且核心系統已在生產環境穩定運作後，可以此藍圖作為再次優化升級的依據。

## 1. 真實多 AI 協同作業 (Advanced Multi-AI Agent Coordination)
在 `services/core/aiva_core/service_backbone/coordination/ai_controller.py` 中，我們目前採用了**循序漸進式的狀態模擬 (Sequential Execution)** 來保證現階段的絕對穩定，取代了原始設計中帶有 TODO 的未完成真實協調邏輯。

**未來優化方向：**
*   **平行任務委派與非同步通訊：** 真正喚醒 `master_ai` (總指揮)、`code_fixer` (研發修復) 與 `detectors` (安全檢測) 的獨立 LLM 實例，各司其職並行處理任務。
*   **RAG 上下文共享 (RAG Context Sharing)：** 讓不同的 AI Agent 可以動態存取同一個向量資料庫 (Vector DB) 中的測試結果，`detectors` 發現漏洞後立即透過統一資料結構交由 `code_fixer` 產生修復建議。
*   **AI 內部辯論糾錯 (Self-Correction/Debate)：** 建立 AI 間的驗證機制，如果有不確定的安全風險，AI 將主動觸發針對同一程式碼的「紅藍隊模擬攻防」對話來確定漏洞的有效性。

## 2. 自動化模型訓練與經驗回饋系統 (AI Experience System)
在 `services/aiva_common/schemas/__init__.py` 之中，我們清除了因為 AI 學習機制造成的循環依賴模型 (諸如 `AITrainingStartPayload`, `ExperienceSample`, `TraceRecord`)。

**未來優化方向：**
*   **AIVA 經驗資料庫 (Experience Database)：** 重新實作 `ExperienceSample` 模型，將每次成功執行滲透、修復成功或產生誤判的歷程，作為結構化的遙測 (Telemetry) 記錄下來。
*   **微調 (Fine-tuning) 與 RAG 動態擴充：** 利用上述經驗資料庫，定期觸發模型權重的 LoRA 微調訓練，或是動態追加至 RAG Knowledge 中。讓 AIVA 能夠學習特定企業或環境特有的架構盲點，並提升攻擊/檢測成功率。

## 3. 已凍結的 5 大實驗性模組 (Suspended Experimental Modules)
為了聚焦於 AIVA 的核心網路安全與程式檢測功能，避免專案戰線過度發散，以下 5 個功能相對特殊且需要大量客製化的初期實驗性模組已在現階段宣告凍結並從主流程排除。

**未來優化方向 (視專案未來定位而定)：**
1.  **社交工程模組 (`function_social_engineering`)：** 原本預計用於釣魚郵件生成與內部網路釣魚演練。未來可高度融合 LLM 強化的精準文本生成能力進行測試。
2.  **鑑識分析模組 (`function_forensic`)：** 伺服器 Log 爬搜、惡意程式足跡追蹤與資安事件爆發後的逆向分析。
3.  **逆向工程模組 (`function_reverse_engineering`)：** 針對編譯後的二進制檔案 (ELF/PE) 或是 APK 進行靜態指令集分析、拆解與破解。
4.  **隱寫術模組 (`function_steganography`)：** 用於檢測圖片、音檔夾帶的加密敏感資料，做為深度資料外洩 (DLP) 的防護演練。
5.  **字典生成器模組 (`function_wordlist_generator`)：** 高度客製化的密碼爆破/目錄掃描字典生成器，規劃利用 AI 依據目標單位的公司名稱、文化、網頁原始碼自動推演出最適用的針對性爆破單字對組合。

## 4. 高效能雙引擎進階分析 (Go/Rust Engine Enhancements)
目前位於 `services/scan/` 中的 Go 與 Rust 引擎已經能發揮關鍵功效 (例如 Go Engine 的 SSRF Bypass payloads 部署，或是 Rust Engine 透過 Pattern Matching 計算的 Risk Scoring)。

**未來優化方向：**
*   **Go Engine 分散式無盲區掃描：** 擴充 `verifier.go` 使其能夠將封包動態派發給多台遠端邊緣節點 (Edge Nodes)，進行百萬級並發的 OOB (Out-of-Band) 盲注測試，並防止被 WAF 單一 IP 阻擋。
*   **Rust Engine 深層語法樹探索 (AST Deep-Dive)：** 在 `main.rs` 的風險計分系統 (Risk Score Algorithm) 中導入正則/字串匹配以外的能力，讓 Rust 可以分析目標專案的 AST，找出 `Regex` 無法察覺的越權存取與深層商業邏輯漏洞 (Business Logic Flaws)。

## 5. 無伺服器與基礎建設完全體 (Infrastructure Full-Deployment)
目前 `aiva_common` 中雖然具備了 1000 多行強大的基礎建設核心 (監控、組態、數據流、服務發現、安全)，但在開發階段我們暫不要求連線真實的雲端中介。

**未來優化方向：**
*   **真實分散式架構：** 以真正的 Redis 實體介入 Message Queue，並啟動 Consul / etcd 取代內部 Memory 註冊表，完成跨機器的主從高可用性 (High Availability) 部署。
*   **全面監控端點：** 啟用 Prometheus 甚至 Grafana 大表板去對接 `observability/monitoring.py` 採集到的 `SystemMetrics` 與日誌，讓開發團隊可以一目了然 AI 正在攻擊的分支與消耗。
