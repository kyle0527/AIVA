# AIVA 專案樹狀結構圖

> **生成時間**: 2025-10-13
> **專案**: AIVA - AI-Powered Intelligent Vulnerability Analysis Platform
> **統計**: 94 個目錄, 237 個程式碼檔案 (已排除文檔檔案)

---

## 📊 專案統計

- **總目錄數**: 94
- **程式碼檔案數**: 237 (已排除 .md/.txt/.mmd/.log/.json)
- **主要語言**: Python (81%), Go, Rust, TypeScript
- **架構類型**: 多語言微服務架構

---

## 🌳 完整程式碼目錄樹狀結構

```plaintext
/workspaces/AIVA
├── __init__.py                           # Python 包初始化
├── _out/                                 # 輸出目錄
│   ├── analysis/                         # 分析報告目錄
│   ├── ext_counts.csv                    # 副檔名統計
│   ├── loc_by_ext.csv                    # 程式碼行數統計
│   └── tree.html                         # HTML 樹狀圖
├── check_status.ps1                      # 狀態檢查腳本
├── demo_bio_neuron_agent.py              # AI 神經網路代理示範
├── demo_ui_panel.py                      # UI 面板示範
├── docker/                               # Docker 容器化
│   ├── Dockerfile.integration            # 整合測試容器
│   ├── docker-compose.production.yml     # 生產環境編排
│   ├── docker-compose.yml                # 開發環境編排
│   ├── entrypoint.integration.sh         # 容器入口腳本
│   └── initdb/                           # 資料庫初始化
│       └── 001_schema.sql                # 資料庫結構定義
├── docs/                                 # 文檔目錄
│   └── diagrams/                         # 架構圖目錄
├── generate_clean_tree.ps1               # 生成乾淨樹狀圖
├── generate_project_report.ps1           # 生成專案報告 (PS)
├── generate_project_report.sh            # 生成專案報告 (Bash)
├── generate_stats.ps1                    # 生成統計資料
├── init_go_deps.ps1                      # Go 依賴初始化
├── mypy.ini                              # MyPy 類型檢查配置
├── pyproject.toml                        # Python 專案配置
├── pyrightconfig.json                    # Pyright 配置
├── ruff.toml                             # Ruff Linter 配置
├── services/                             # 服務模組 🎯
│   ├── __init__.py                       # 服務包初始化
│   ├── aiva_common/                      # 共用組件 📦
│   │   ├── __init__.py
│   │   ├── config.py                     # 全局配置
│   │   ├── enums.py                      # 枚舉定義
│   │   ├── mq.py                         # 訊息佇列客戶端
│   │   ├── py.typed                      # 類型標註
│   │   ├── schemas.py                    # 資料結構定義
│   │   └── utils/                        # 工具函數
│   │       ├── __init__.py
│   │       ├── dedup/                    # 去重工具
│   │       │   ├── __init__.py
│   │       │   └── dedupe.py             # 去重邏輯
│   │       ├── ids.py                    # ID 生成器
│   │       ├── logging.py                # 日誌工具
│   │       └── network/                  # 網路工具
│   │           ├── __init__.py
│   │           ├── backoff.py            # 退避重試
│   │           └── ratelimit.py          # 速率限制
│   ├── core/                             # 核心引擎 🤖
│   │   └── aiva_core/
│   │       ├── __init__.py
│   │       ├── ai_engine/                # AI 引擎
│   │       │   ├── __init__.py
│   │       │   ├── bio_neuron_core.py    # 生物神經網路核心
│   │       │   ├── bio_neuron_core.py.backup
│   │       │   ├── bio_neuron_core_v2.py # V2 版本
│   │       │   ├── knowledge_base.py     # 知識庫
│   │       │   ├── knowledge_base.py.backup
│   │       │   └── tools.py              # AI 工具函數
│   │       ├── ai_model/                 # AI 模型
│   │       │   └── train_classifier.py   # 分類器訓練
│   │       ├── ai_ui_schemas.py          # AI UI 資料結構
│   │       ├── analysis/                 # 分析模組
│   │       │   ├── __init__.py
│   │       │   ├── dynamic_strategy_adjustment.py  # 動態策略調整
│   │       │   ├── initial_surface.py    # 初始攻擊面分析
│   │       │   └── strategy_generator.py # 策略生成器
│   │       ├── app.py                    # 核心應用入口
│   │       ├── execution/                # 執行引擎
│   │       │   ├── __init__.py
│   │       │   ├── execution_status_monitor.py  # 執行狀態監控
│   │       │   ├── task_generator.py     # 任務生成器
│   │       │   └── task_queue_manager.py # 任務佇列管理
│   │       ├── ingestion/                # 資料接收
│   │       │   ├── __init__.py
│   │       │   └── scan_module_interface.py  # 掃描模組介面
│   │       ├── output/                   # 輸出處理
│   │       │   ├── __init__.py
│   │       │   └── to_functions.py       # 輸出到檢測模組
│   │       ├── schemas.py                # 核心資料結構
│   │       ├── state/                    # 狀態管理
│   │       │   ├── __init__.py
│   │       │   └── session_state_manager.py  # 會話狀態管理
│   │       └── ui_panel/                 # UI 面板
│   │           ├── __init__.py
│   │           ├── dashboard.py          # 儀表板
│   │           ├── dashboard.py.backup
│   │           ├── improved_ui.py        # 改進版 UI
│   │           ├── server.py             # UI 服務器
│   │           └── server.py.backup
│   ├── function/                         # 檢測功能模組 🔍
│   │   ├── common/                       # 共用檢測組件
│   │   │   ├── __init__.py
│   │   │   ├── detection_config.py       # 檢測配置
│   │   │   └── unified_smart_detection_manager.py  # 統一智能檢測管理器
│   │   ├── function_authn_go/            # 身份驗證檢測 (Go) 🔷
│   │   │   ├── cmd/
│   │   │   │   └── worker/
│   │   │   │       └── main.go           # Worker 主程式
│   │   │   ├── go.mod                    # Go 模組定義
│   │   │   ├── internal/
│   │   │   │   ├── brute_force/          # 暴力破解
│   │   │   │   │   └── brute_forcer.go
│   │   │   │   ├── token_test/           # Token 測試
│   │   │   │   │   └── token_analyzer.go
│   │   │   │   └── weak_config/          # 弱配置檢測
│   │   │   │       └── config_tester.go
│   │   │   └── pkg/
│   │   │       ├── messaging/            # 訊息處理
│   │   │       │   ├── consumer.go
│   │   │       │   └── publisher.go
│   │   │       └── models/               # 資料模型
│   │   │           └── models.go
│   │   ├── function_cspm_go/             # 雲端安全態勢管理 (Go) 🔷
│   │   │   ├── cmd/
│   │   │   │   └── worker/
│   │   │   │       └── main.go
│   │   │   ├── go.mod
│   │   │   ├── internal/
│   │   │   │   └── scanner/
│   │   │   │       └── cspm_scanner.go   # CSPM 掃描器
│   │   │   └── pkg/
│   │   │       ├── messaging/
│   │   │       │   ├── consumer.go
│   │   │       │   └── publisher.go
│   │   │       └── models/
│   │   │           └── models.go
│   │   ├── function_idor/                # IDOR 檢測 (Python) 🐍
│   │   │   └── aiva_func_idor/
│   │   │       ├── __init__.py
│   │   │       ├── bfla_tester.py        # BFLA 測試器
│   │   │       ├── cross_user_tester.py  # 跨用戶測試
│   │   │       ├── enhanced_worker.py    # 增強型 Worker
│   │   │       ├── mass_assignment_tester.py  # 大量賦值測試
│   │   │       ├── resource_id_extractor.py   # 資源 ID 提取
│   │   │       ├── smart_idor_detector.py     # 智能 IDOR 檢測
│   │   │       ├── vertical_escalation_tester.py  # 垂直提權測試
│   │   │       └── worker.py             # Worker 主程式
│   │   ├── function_sast_rust/           # 靜態程式碼分析 (Rust) 🦀
│   │   │   ├── Cargo.toml                # Rust 專案配置
│   │   │   └── src/
│   │   │       ├── analyzers.rs          # 分析器
│   │   │       ├── main.rs               # 主程式
│   │   │       ├── models.rs             # 資料模型
│   │   │       ├── parsers.rs            # 語法解析器
│   │   │       ├── rules.rs              # 規則引擎
│   │   │       └── worker.rs             # Worker
│   │   ├── function_sca_go/              # 軟體組成分析 (Go) 🔷
│   │   │   ├── cmd/
│   │   │   │   └── worker/
│   │   │   │       └── main.go
│   │   │   ├── go.mod
│   │   │   ├── internal/
│   │   │   │   └── scanner/
│   │   │   │       └── sca_scanner.go    # SCA 掃描器
│   │   │   └── pkg/
│   │   │       ├── messaging/
│   │   │       │   └── publisher.go
│   │   │       └── models/
│   │   │           └── models.go
│   │   ├── function_sqli/                # SQL 注入檢測 (Python) 🐍
│   │   │   ├── __init__.py
│   │   │   └── aiva_func_sqli/
│   │   │       ├── __init__.py
│   │   │       ├── backend_db_fingerprinter.py  # 資料庫指紋識別
│   │   │       ├── config.py
│   │   │       ├── detection_models.py   # 檢測模型
│   │   │       ├── engines/              # 檢測引擎 ⚡
│   │   │       │   ├── __init__.py
│   │   │       │   ├── boolean_detection_engine.py  # 布爾盲注
│   │   │       │   ├── error_detection_engine.py    # 錯誤注入
│   │   │       │   ├── oob_detection_engine.py      # 帶外注入
│   │   │       │   ├── time_detection_engine.py     # 時間盲注
│   │   │       │   └── union_detection_engine.py    # UNION 注入
│   │   │       ├── exceptions.py         # 異常定義
│   │   │       ├── payload_wrapper_encoder.py  # Payload 包裝編碼
│   │   │       ├── result_binder_publisher.py  # 結果綁定發布
│   │   │       ├── schemas.py
│   │   │       ├── smart_detection_manager.py  # 智能檢測管理器
│   │   │       ├── task_queue.py         # 任務佇列
│   │   │       ├── telemetry.py          # 遙測資料
│   │   │       ├── worker.py             # Worker 主程式
│   │   │       └── worker_legacy.py      # 舊版 Worker
│   │   ├── function_ssrf/                # SSRF 檢測 (Python) 🐍
│   │   │   ├── __init__.py
│   │   │   └── aiva_func_ssrf/
│   │   │       ├── __init__.py
│   │   │       ├── enhanced_worker.py    # 增強型 Worker
│   │   │       ├── internal_address_detector.py  # 內網位址檢測
│   │   │       ├── oast_dispatcher.py    # OAST 調度器
│   │   │       ├── param_semantics_analyzer.py   # 參數語義分析
│   │   │       ├── result_publisher.py   # 結果發布器
│   │   │       ├── schemas.py
│   │   │       ├── smart_ssrf_detector.py  # 智能 SSRF 檢測
│   │   │       └── worker.py
│   │   ├── function_ssrf_go/             # SSRF 檢測 (Go) 🔷
│   │   │   ├── cmd/
│   │   │   │   └── worker/
│   │   │   │       └── main.go
│   │   │   ├── go.mod
│   │   │   └── internal/
│   │   │       └── detector/
│   │   │           └── ssrf.go           # SSRF 檢測器
│   │   └── function_xss/                 # XSS 檢測 (Python) 🐍
│   │       ├── __init__.py
│   │       └── aiva_func_xss/
│   │           ├── __init__.py
│   │           ├── blind_xss_listener_validator.py  # 盲 XSS 監聽驗證
│   │           ├── dom_xss_detector.py   # DOM XSS 檢測
│   │           ├── payload_generator.py  # Payload 生成器
│   │           ├── result_publisher.py   # 結果發布器
│   │           ├── schemas.py
│   │           ├── stored_detector.py    # 儲存型 XSS 檢測
│   │           ├── task_queue.py         # 任務佇列
│   │           ├── traditional_detector.py  # 傳統 XSS 檢測
│   │           └── worker.py
│   ├── integration/                      # 整合服務 🔗
│   │   ├── aiva_integration/
│   │   │   ├── __init__.py
│   │   │   ├── analysis/                 # 分析模組
│   │   │   │   ├── __init__.py
│   │   │   │   ├── compliance_policy_checker.py  # 合規性檢查
│   │   │   │   ├── risk_assessment_engine.py     # 風險評估引擎
│   │   │   │   └── vuln_correlation_analyzer.py  # 漏洞關聯分析
│   │   │   ├── app.py                    # 整合服務入口
│   │   │   ├── attack_path_analyzer/     # 攻擊路徑分析
│   │   │   │   ├── __init__.py
│   │   │   │   ├── engine.py             # 分析引擎
│   │   │   │   ├── graph_builder.py      # 圖構建器
│   │   │   │   └── visualizer.py         # 視覺化工具
│   │   │   ├── config_template/          # 配置模板
│   │   │   │   ├── __init__.py
│   │   │   │   └── config_template_manager.py  # 模板管理器
│   │   │   ├── middlewares/              # 中間件
│   │   │   │   ├── __init__.py
│   │   │   │   └── rate_limiter.py       # 速率限制器
│   │   │   ├── observability/            # 可觀測性
│   │   │   │   ├── __init__.py
│   │   │   │   └── metrics.py            # 指標收集
│   │   │   ├── perf_feedback/            # 效能回饋
│   │   │   │   ├── __init__.py
│   │   │   │   ├── improvement_suggestion_generator.py  # 改進建議生成
│   │   │   │   └── scan_metadata_analyzer.py  # 掃描元資料分析
│   │   │   ├── reception/                # 資料接收層
│   │   │   │   ├── __init__.py
│   │   │   │   ├── data_reception_layer.py  # 資料接收層
│   │   │   │   └── sql_result_database.py   # SQL 結果資料庫
│   │   │   ├── reporting/                # 報告生成 📊
│   │   │   │   ├── __init__.py
│   │   │   │   ├── formatter_exporter.py    # 格式化匯出器
│   │   │   │   ├── report_content_generator.py  # 報告內容生成
│   │   │   │   └── report_template_selector.py  # 報告模板選擇器
│   │   │   ├── security/                 # 安全模組
│   │   │   │   ├── __init__.py
│   │   │   │   └── auth.py               # 身份驗證
│   │   │   ├── settings.py               # 設定檔
│   │   │   └── threat_intel/             # 威脅情報
│   │   │       └── __init__.py
│   │   ├── alembic/                      # 資料庫遷移
│   │   │   ├── env.py                    # Alembic 環境配置
│   │   │   └── versions/                 # 遷移版本
│   │   │       └── 001_initial_schema.py # 初始資料庫結構
│   │   ├── alembic.ini                   # Alembic 配置
│   │   └── api_gateway/                  # API 閘道
│   │       └── api_gateway/
│   │           └── app.py                # 閘道應用
│   └── scan/                             # 掃描引擎 🌐
│       ├── aiva_scan/                    # Python 掃描引擎 🐍
│       │   ├── __init__.py
│       │   ├── authentication_manager.py # 身份驗證管理器
│       │   ├── config_control_center.py  # 配置控制中心
│       │   ├── core_crawling_engine/     # 核心爬蟲引擎
│       │   │   ├── __init__.py
│       │   │   ├── http_client_hi.py     # HTTP 客戶端
│       │   │   ├── static_content_parser.py  # 靜態內容解析器
│       │   │   └── url_queue_manager.py  # URL 佇列管理器
│       │   ├── dynamic_engine/           # 動態內容引擎
│       │   │   ├── __init__.py
│       │   │   ├── dynamic_content_extractor.py  # 動態內容提取器
│       │   │   ├── example_browser_pool.py       # 瀏覽器池範例
│       │   │   ├── example_extractor.py
│       │   │   ├── example_usage.py
│       │   │   ├── example_usage.py.backup
│       │   │   ├── headless_browser_pool.py      # 無頭瀏覽器池
│       │   │   └── js_interaction_simulator.py   # JS 互動模擬器
│       │   ├── fingerprint_manager.py    # 指紋管理器
│       │   ├── header_configuration.py   # 標頭配置
│       │   ├── info_gatherer/            # 資訊收集器
│       │   │   ├── __init__.py
│       │   │   ├── javascript_source_analyzer.py  # JS 源碼分析
│       │   │   ├── passive_fingerprinter.py       # 被動指紋識別
│       │   │   └── sensitive_info_detector.py     # 敏感資訊檢測
│       │   ├── scan_context.py           # 掃描上下文
│       │   ├── scan_orchestrator.py      # 掃描協調器
│       │   ├── scan_orchestrator_new.py  # 新版協調器
│       │   ├── scan_orchestrator_old.py  # 舊版協調器
│       │   ├── schemas.py                # 資料結構
│       │   ├── scope_manager.py          # 範圍管理器
│       │   ├── strategy_controller.py    # 策略控制器
│       │   └── worker.py                 # Worker 主程式
│       ├── aiva_scan_node/               # TypeScript 掃描引擎 📘
│       │   └── src/
│       │       ├── index.ts              # 入口檔案
│       │       ├── services/
│       │       │   └── scan-service.ts   # 掃描服務
│       │       └── utils/
│       │           └── logger.ts         # 日誌工具
│       └── info_gatherer_rust/           # Rust 資訊收集器 🦀
│           ├── Cargo.toml                # Rust 專案配置
│           └── src/
│               ├── git_history_scanner.rs  # Git 歷史掃描
│               ├── main.rs               # 主程式
│               ├── scanner.rs            # 掃描器
│               └── secret_detector.rs    # 秘密檢測器
├── setup_env.bat                         # 環境設置 (Windows)
├── setup_multilang.ps1                   # 多語言環境設置
├── start_all.ps1                         # 啟動所有服務
├── start_all_multilang.ps1               # 啟動多語言服務
├── start_dev.bat                         # 開發環境啟動
├── stop_all.ps1                          # 停止所有服務
├── stop_all_multilang.ps1                # 停止多語言服務
├── test_scan.ps1                         # 掃描測試腳本
└── tools/                                # 開發工具 🛠️
    ├── analyze_codebase.py               # 程式碼分析工具
    ├── find_non_cp950_filtered.py        # 非 CP950 字元查找
    ├── generate_mermaid_diagrams.py      # Mermaid 圖表生成器
    ├── markdown_check.py                 # Markdown 檢查工具
    ├── py2mermaid.py                     # Python 轉 Mermaid
    ├── replace_emoji.py                  # Emoji 替換工具
    ├── replace_non_cp950.py              # 字元替換工具
    ├── test_tools.py                     # 工具測試
    └── update_imports.py                 # 匯入更新工具

94 directories, 237 files
```

---

## 📁 主要模組說明

### 🤖 核心引擎 (services/core/aiva_core)

**AI 引擎** - 生物神經網路啟發的智能決策系統

- `bio_neuron_core.py` - 模擬神經元的漏洞檢測核心
- `knowledge_base.py` - 漏洞知識庫和規則引擎
- `train_classifier.py` - 機器學習分類器訓練

**分析模組** - 智能策略生成

- `dynamic_strategy_adjustment.py` - 根據掃描結果動態調整檢測策略
- `initial_surface.py` - 初始攻擊面分析
- `strategy_generator.py` - 自動化策略生成

**執行引擎** - 任務編排和執行

- `task_generator.py` - 自動生成檢測任務
- `task_queue_manager.py` - 分散式任務佇列管理
- `execution_status_monitor.py` - 即時執行狀態監控

### 🔍 檢測功能模組 (services/function)

#### Python 實現 🐍

| 模組 | 功能 | 特色 |
|------|------|------|
| **SQLi** | SQL 注入檢測 | 5 種引擎：布爾/錯誤/時間/UNION/帶外 |
| **XSS** | 跨站腳本攻擊 | 反射型/儲存型/DOM 型全覆蓋 |
| **SSRF** | 伺服器端請求偽造 | 內網探測 + OAST 平台整合 |
| **IDOR** | 不安全的直接對象引用 | BFLA/垂直提權/跨用戶測試 |

#### Go 實現 🔷 (高效能)

| 模組 | 功能 | 優勢 |
|------|------|------|
| **AuthN** | 身份驗證漏洞 | 暴力破解/Token 分析/弱配置 |
| **CSPM** | 雲端安全態勢管理 | AWS/Azure/GCP 配置檢查 |
| **SCA** | 軟體組成分析 | 依賴套件漏洞掃描 |
| **SSRF** | SSRF 高效能版本 | 並發處理 10000+ 請求 |

#### Rust 實現 🦀 (記憶體安全)

| 模組 | 功能 | 特點 |
|------|------|------|
| **SAST** | 靜態程式碼分析 | 語法樹解析 + 規則匹配 |
| **Secret Scanner** | 秘密掃描 | Git 歷史 + 檔案內容掃描 |

### 🌐 掃描引擎 (services/scan)

#### Python 爬蟲引擎 🐍

- **靜態爬蟲** - 高效能 HTTP 客戶端 + HTML/CSS 選擇器解析
- **動態爬蟲** - Playwright 無頭瀏覽器自動化
- **資訊收集** - JS 分析 / 指紋識別 / 敏感資訊檢測

#### TypeScript 引擎 📘

- **瀏覽器自動化** - 基於 Node.js 的動態內容提取
- **異步處理** - Promise/async-await 並發爬取

#### Rust 秘密掃描 🦀

- **Git 歷史掃描** - 檢測歷史提交中的敏感資訊
- **高速掃描** - 正則表達式引擎優化

### 🔗 整合服務 (services/integration)

| 模組 | 功能 |
|------|------|
| **攻擊路徑分析** | 漏洞關聯 + 攻擊鏈構建 + 圖形化展示 |
| **風險評估** | CVSS 評分 + 業務影響評估 + 優先級排序 |
| **合規性檢查** | OWASP Top 10 / PCI DSS / GDPR 對照 |
| **報告生成** | PDF/HTML/JSON/CSV 多格式匯出 |

---

## 🏗️ 技術架構特點

### 多語言混合架構

```
Python 81%  ████████████████████████████████████████
Go     12%  ██████
Rust    4%  ██
TS/JS   3%  █
```

#### 語言選擇原則

| 語言 | 使用場景 | 優勢 | 典型模組 |
|------|---------|------|---------|
| **Python 🐍** | 核心邏輯、Web API、複雜檢測 | 豐富生態、快速開發 | SQLi, XSS, 核心引擎 |
| **Go 🔷** | 高併發、網路密集型 | 高效能、原生並發 | AuthN, CSPM, SCA |
| **Rust 🦀** | 靜態分析、系統級工具 | 記憶體安全、極致效能 | SAST, 秘密掃描 |
| **TypeScript 📘** | 動態爬蟲、瀏覽器自動化 | 異步處理、豐富 Web 工具 | 動態掃描引擎 |

### 微服務設計

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  掃描引擎    │───▶│  訊息佇列     │◀───│  檢測模組    │
│  (Scan)     │    │  (RabbitMQ)  │    │  (Function) │
└─────────────┘    └──────────────┘    └─────────────┘
       │                   │                    │
       └───────────────────┼────────────────────┘
                          │
                  ┌───────▼────────┐
                  │   核心引擎      │
                  │   (Core AI)    │
                  └────────────────┘
                          │
                  ┌───────▼────────┐
                  │   整合服務      │
                  │  (Integration) │
                  └────────────────┘
```

**特點**:

- ✅ 各檢測模組獨立部署
- ✅ RabbitMQ 訊息佇列解耦
- ✅ PostgreSQL 集中式結果儲存
- ✅ 支援水平擴展

### AI 驅動智能檢測

- 🧠 **生物神經網路**啟發的決策引擎
- 🎯 **動態策略調整** - 根據掃描結果自適應
- 📊 **知識庫驅動** - 規則引擎 + 專家系統
- 🔄 **持續學習** - 從檢測結果中學習優化

---

## 🚀 快速開始

### 安裝依賴

```bash
# Python 依賴
pip install -e .[dev]

# Go 依賴
./init_go_deps.ps1

# Node.js 依賴
cd services/scan/aiva_scan_node && npm install

# Rust 依賴
cd services/function/function_sast_rust && cargo build --release
cd services/scan/info_gatherer_rust && cargo build --release
```

### 啟動服務

```bash
# 啟動所有多語言服務
./start_all_multilang.ps1

# 或單獨啟動
python services/core/aiva_core/app.py                          # Python 核心
cd services/function/function_authn_go && go run cmd/worker/main.go  # Go Worker
cd services/function/function_sast_rust && cargo run            # Rust Worker
cd services/scan/aiva_scan_node && npm start                    # TypeScript 掃描
```

---

## 📝 相關文檔

- [COMPREHENSIVE_PROJECT_ANALYSIS.md](./COMPREHENSIVE_PROJECT_ANALYSIS.md) - 完整專案分析
- [ARCHITECTURE_REPORT.md](./ARCHITECTURE_REPORT.md) - 架構設計報告
- [QUICK_START.md](./QUICK_START.md) - 快速開始指南
- [MULTI_LANGUAGE_ARCHITECTURE_PROPOSAL.md](./MULTI_LANGUAGE_ARCHITECTURE_PROPOSAL.md) - 多語言架構提案

---

## 📊 圖示說明

| 圖示 | 說明 | 圖示 | 說明 |
|-----|------|-----|------|
| 🐍 | Python 實現 | 🔷 | Go 實現 |
| 🦀 | Rust 實現 | 📘 | TypeScript 實現 |
| 🤖 | AI/機器學習 | 🔍 | 掃描/檢測 |
| 🔗 | 整合/API | 🗄️ | 資料庫/儲存 |
| 📊 | 分析/報告 | 🎯 | 核心服務 |
| 📦 | 共用組件 | 🛠️ | 開發工具 |
| ⚡ | 高效能模組 | 🌐 | 網路服務 |

---

**生成時間**: 2025-10-13
**專案版本**: v1.0
**文檔版本**: v1.0
**樹狀圖來源**: `_out/tree_clean.txt` (已過濾文檔檔案)
