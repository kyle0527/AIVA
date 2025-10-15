# AIVA 專案架構圖（含說明）

生成時間: 2025-10-14
總目錄數: 2,435
總文件數: 4,013
程式碼總行數: 28,959

---

## 📁 根目錄腳本

```
AIVA/
├── analyze_crypto_security_v2.ps1      # 密碼學安全分析腳本（改進版，減少誤報）
├── analyze_crypto_security.ps1         # 密碼學漏洞檢測和分析
├── check_status.ps1                    # AIVA 系統狀態檢查腳本
├── deploy_services.ps1                 # 自動部署和測試腳本
├── enhance_cspm_service.ps1            # Go CSPM 服務增強（AWS/Azure/GCP 雲端安全規則）
├── enhance_sca_service.ps1             # Go SCA 服務增強（多語言支援和深度依賴分析）
├── fix_authn_models.ps1                # 修復 function_authn_go 的 models 依賴問題
├── fix_authn_schemas.ps1               # 修正 function_authn_go 程式碼結構
├── generate_clean_tree.ps1             # 生成乾淨的專案樹狀圖（排除虛擬環境）
├── generate_project_report.ps1         # AIVA 專案完整報告生成器
├── generate_stats.ps1                  # 專案統計生成（文件統計和程式碼行數）
├── implement_error_tracking.ps1        # 跨服務錯誤追蹤實作（Go 和 Rust 服務）
├── init_go_common.ps1                  # AIVA Go 共用模組初始化
├── init_go_deps.ps1                    # Go 模組依賴初始化
├── migrate_sca_service.ps1             # SCA 服務遷移到共用模組
├── optimize_ssrf_service.ps1           # Go SSRF 偵測優化（降低誤報率）
├── setup_monitoring.ps1                # Go 服務效能監控增強
├── setup_multilang.ps1                 # AIVA 多語言環境設置
├── start_all_multilang.ps1             # 啟動所有多語言模組（Python/Node.js/Go/Rust）
├── start_all.ps1                       # 一鍵啟動所有 Python 模組
├── start_ui_auto.ps1                   # AIVA UI 面板自動啟動（自動選擇端口）
├── stop_all_multilang.ps1              # 停止所有多語言模組
├── stop_all.ps1                        # 停止所有服務
├── test_enhanced_sca.ps1               # SCA 服務增強測試
├── test_scan.ps1                       # 發送測試掃描任務
├── demo_bio_neuron_agent.py            # 生物神經網路代理演示
├── demo_ui_panel.py                    # UI 面板演示
├── enhance_python_types.py             # Python 類型增強工具
├── start_ui_auto.py                    # UI 自動啟動 Python 版本
├── test_ai_integration.py              # AI 整合測試
├── setup_env.bat                       # 環境設置批次檔
├── start_dev.bat                       # 開發環境啟動批次檔
├── mypy.ini                            # MyPy 靜態類型檢查配置
├── pyproject.toml                      # Python 專案配置（Poetry/工具設定）
├── pyrightconfig.json                  # Pyright 類型檢查配置
├── ruff.toml                           # Ruff Linter 配置
└── README.md                           # 專案說明文件
```

## 📁 主要目錄結構

### 1. 核心服務 (services/)

```
services/
│
├── aiva_common/                        # 共用模組（Python）
│   ├── utils/                          # 工具函數
│   │   ├── dedup/                      # 去重複功能
│   │   └── network/                    # 網路工具（退避、限流）
│   ├── config.py                       # 配置管理
│   ├── enums.py                        # 列舉定義
│   ├── mq.py                           # 消息隊列客戶端
│   └── schemas.py                      # 數據結構定義
│
├── core/                               # 核心模組
│   └── aiva_core/
│       ├── ai_engine/                  # AI 引擎
│       │   ├── bio_neuron_core_v2.py   # 生物神經網路核心 V2
│       │   ├── bio_neuron_core.py      # 生物神經網路核心 V1
│       │   ├── knowledge_base.py       # RAG 知識庫
│       │   └── tools.py                # AI 工具集
│       ├── analysis/                   # 分析引擎
│       │   ├── initial_surface.py      # 初始攻擊面分析
│       │   ├── risk_assessment_engine.py # 風險評估引擎
│       │   └── strategy_generator.py   # 策略生成器
│       ├── execution/                  # 執行引擎
│       │   ├── task_generator.py       # 任務生成器
│       │   ├── task_queue_manager.py   # 任務佇列管理
│       │   └── execution_status_monitor.py # 執行狀態監控
│       ├── authz/                      # 授權管理
│       │   ├── authz_mapper.py         # 授權映射器
│       │   └── permission_matrix.py    # 權限矩陣
│       ├── ui_panel/                   # UI 面板
│       │   ├── dashboard.py            # 儀表板
│       │   ├── server.py               # UI 服務器
│       │   └── auto_server.py          # 自動服務器
│       ├── app.py                      # FastAPI 主應用
│       ├── ai_controller.py            # AI 控制器
│       ├── multilang_coordinator.py    # 多語言協調器
│       └── schemas.py                  # 核心數據結構
│
└── function/                           # 功能模組
    │
    ├── common/                         # 功能共用模組
    │   ├── go/aiva_common_go/          # Go 共用庫
    │   │   ├── config/                 # 配置管理
    │   │   ├── logger/                 # 日誌系統
    │   │   ├── mq/                     # 消息隊列客戶端
    │   │   └── schemas/                # 數據結構
    │   ├── detection_config.py         # 檢測配置
    │   └── unified_smart_detection_manager.py # 統一智能檢測管理器
    │
    ├── function_sca_go/                # 軟體組成分析（Go）
    │   ├── cmd/worker/                 # Worker 入口點
    │   ├── internal/
    │   │   ├── analyzer/               # 分析器
    │   │   │   ├── dependency_analyzer.go    # 依賴分析器（8種語言）
    │   │   │   └── enhanced_analyzer.go      # 增強分析器（並發處理）
    │   │   ├── scanner/                # 掃描器
    │   │   │   └── sca_scanner.go            # SCA 掃描器（OSV 集成）
    │   │   └── vulndb/                 # 漏洞資料庫
    │   │       └── osv.go                    # OSV API 客戶端
    │   ├── pkg/
    │   │   ├── models/                 # 業務模型
    │   │   └── schemas/                # 統一數據結構
    │   ├── .golangci.yml               # Linting 配置
    │   ├── go.mod                      # Go 模組定義
    │   ├── GO_SCA_OPTIMIZATION_REPORT.md    # 優化分析報告
    │   ├── MIGRATION_REPORT.md         # 遷移報告
    │   └── ARCHITECTURE_TREE.txt       # 架構樹狀圖
    │
    ├── function_authn_go/              # 身份驗證檢測（Go）
    │   ├── cmd/worker/                 # Worker 入口點
    │   ├── internal/
    │   │   ├── brute_force/            # 暴力破解檢測
    │   │   ├── cache/                  # Redis 快取
    │   │   ├── metrics/                # 性能指標
    │   │   └── token_test/             # Token 分析
    │   └── go.mod                      # Go 模組定義
    │
    ├── function_crypto_go/             # 密碼學檢測（Go）
    │   ├── cmd/worker/                 # Worker 入口點
    │   ├── internal/analyzer/          # 密碼分析器
    │   └── go.mod                      # Go 模組定義
    │
    ├── function_cspm_go/               # 雲端安全態勢管理（Go）
    │   ├── cmd/worker/                 # Worker 入口點
    │   ├── internal/scanner/           # CSPM 掃描器
    │   └── go.mod                      # Go 模組定義
    │
    ├── function_sast_rust/             # 靜態應用安全測試（Rust）
    │   ├── src/
    │   │   ├── main.rs                 # 主程式
    │   │   ├── analyzers.rs            # 分析器
    │   │   ├── parsers.rs              # 解析器
    │   │   ├── rules.rs                # 規則引擎
    │   │   └── worker.rs               # Worker 實現
    │   └── Cargo.toml                  # Rust 專案配置
    │
    ├── function_sqli/                  # SQL 注入檢測（Python）
    │   └── aiva_func_sqli/
    │       ├── engines/                # 檢測引擎
    │       │   ├── boolean_detection_engine.py    # 布林檢測
    │       │   ├── error_detection_engine.py      # 錯誤檢測
    │       │   ├── time_detection_engine.py       # 時間盲注檢測
    │       │   └── union_detection_engine.py      # UNION 查詢檢測
    │       ├── smart_detection_manager.py         # 智能檢測管理器
    │       └── worker.py               # Worker 實現
    │
    ├── function_ssrf/                  # SSRF 檢測（Python）
    │   └── aiva_func_ssrf/
    │       ├── engines/                # 檢測引擎
    │       ├── smart_ssrf_detector.py  # 智能 SSRF 檢測器
    │       └── worker.py               # Worker 實現
    │
    ├── function_xss/                   # XSS 檢測（Python）
    │   └── aiva_func_xss/
    │       ├── engines/                # 檢測引擎
    │       ├── context_analyzer.py     # 上下文分析器
    │       └── worker.py               # Worker 實現
    │
    ├── function_idor/                  # IDOR 檢測（Python）
    │   └── aiva_func_idor/
    │       ├── smart_idor_detector.py  # 智能 IDOR 檢測器
    │       └── worker.py               # Worker 實現
    │
    └── function_postex/                # 後滲透檢測（Python）
        ├── lateral_movement.py         # 橫向移動檢測
        ├── persistence_checker.py      # 持久化檢查
        └── privilege_escalator.py      # 權限提升檢測
```

### 2. Docker 配置

```
docker/
├── docker-compose.yml                  # Docker Compose 開發環境配置
├── docker-compose.production.yml       # Docker Compose 生產環境配置
├── Dockerfile.integration              # 整合測試 Dockerfile
├── entrypoint.integration.sh           # 整合測試入口腳本
└── initdb/                             # 資料庫初始化腳本
    ├── 001_schema.sql                  # 基礎資料庫架構
    └── 002_enhanced_schema.sql         # 增強資料庫架構
```

### 3. 文檔目錄

```
docs/
└── ARCHITECTURE_MULTILANG.md           # 多語言架構文檔
```

### 4. 輸出目錄 (_out/)

```
_out/
├── analysis/                           # 分析報告
│   ├── analysis_report_*.json          # 程式碼分析報告（JSON）
│   ├── analysis_report_*.txt           # 程式碼分析報告（文本）
│   ├── multilang_analysis_*.json       # 多語言分析報告（JSON）
│   └── multilang_analysis_*.txt        # 多語言分析報告（文本）
├── type_analysis/                      # 類型分析
│   ├── enhancement_suggestions.json    # 類型增強建議
│   └── missing_types_analysis.json     # 缺失類型分析
├── ARCHITECTURE_DIAGRAMS.md            # 架構圖文檔
├── architecture_recovery_report.md     # 架構恢復報告
├── core_module_comprehensive_analysis.md # 核心模組深度分析
├── core_optimization_*.md              # 核心優化相關報告
├── crypto_security_analysis*.json      # 密碼學安全分析
├── ext_counts.csv                      # 副檔名統計
├── loc_by_ext.csv                      # 程式碼行數統計
├── tree_clean.txt                      # 乾淨的樹狀圖（450KB）
└── tree.html                           # HTML 可視化樹狀圖
```

### 5. 工具目錄 (tools/)

```
tools/
├── analyze_codebase.py                 # 程式碼庫分析工具
├── generate_mermaid_diagrams.py        # Mermaid 圖表生成器
├── py2mermaid.py                       # Python 轉 Mermaid 圖表
├── markdown_check.py                   # Markdown 檢查工具
├── replace_emoji.py                    # Emoji 替換工具
├── find_non_cp950_filtered.py          # 非 CP950 字符查找
└── README.md                           # 工具說明文檔
```

---

## 📊 統計摘要

### 檔案類型分布（Top 10）

| 副檔名 | 文件數 | 說明 |
|--------|--------|------|
| .json | 718 | 配置和數據文件 |
| .py | 169 | Python 源代碼 |
| .no_ext | 59 | 無副檔名文件 |
| .mmd | 24 | Mermaid 圖表文件 |
| .md | 8 | Markdown 文檔 |
| .txt | 6 | 文本文件 |
| .backup | 5 | 備份文件 |
| .ps1 | 3 | PowerShell 腳本 |
| .toml | 2 | TOML 配置文件 |
| .yml | 2 | YAML 配置文件 |

### 程式碼行數分布（Top 10）

| 副檔名 | 總行數 | 文件數 | 平均行數/文件 |
|--------|--------|--------|---------------|
| .py | 24,063 | 169 | 142.4 |
| .md | 3,180 | 8 | 397.5 |
| .ps1 | 518 | 3 | 172.7 |
| .txt | 498 | 6 | 83.0 |
| .yml | 216 | 2 | 108.0 |
| .sql | 178 | 1 | 178.0 |
| .toml | 130 | 2 | 65.0 |
| .json | 77 | 3 | 25.7 |
| .sh | 65 | 1 | 65.0 |
| .yaml | 49 | 1 | 49.0 |

**總計**: 28,959 行程式碼

---

## 🎯 技術棧

### 程式語言
- **Python** (169 個文件, 24,063 行) - 核心邏輯、AI 引擎、大部分功能模組
- **Go** (多個服務) - 高性能功能模組（SCA, AUTHN, CRYPTO, CSPM）
- **Rust** (1 個服務) - SAST 靜態分析
- **PowerShell** (3 個腳本) - 自動化和部署腳本
- **JavaScript/TypeScript** - 前端 UI

### 框架和工具
- **FastAPI** - Python Web 框架
- **RabbitMQ** - 消息隊列
- **Docker** - 容器化
- **Poetry** - Python 依賴管理
- **Cargo** - Rust 包管理器
- **Go Modules** - Go 依賴管理

### AI/ML 工具
- **OpenAI API** - AI 模型集成
- **LangChain** - RAG 知識庫
- **生物神經網路** - 自定義 AI 引擎

---

## 📝 重要說明文件

1. **README.md** - 專案主說明文件
2. **GO_SCA_OPTIMIZATION_REPORT.md** - Go SCA 服務優化分析（本次工作產出）
3. **ARCHITECTURE_TREE.txt** - Go SCA 服務架構樹狀圖
4. **ARCHITECTURE_MULTILANG.md** - 多語言架構設計文檔
5. **MIGRATION_REPORT.md** - 服務遷移報告
6. **核心優化報告** - 多份核心模組優化分析

---

## 🔧 快速啟動指令

```powershell
# 設置環境
.\setup_env.bat
.\setup_multilang.ps1

# 啟動所有服務
.\start_all_multilang.ps1

# 啟動 UI
.\start_ui_auto.ps1

# 停止服務
.\stop_all_multilang.ps1

# 檢查狀態
.\check_status.ps1

# 生成統計報告
.\generate_stats.ps1
.\generate_project_report.ps1
```

---

**生成工具**: generate_clean_tree.ps1, generate_stats.ps1
**更新日期**: 2025-10-14
**維護者**: AIVA 開發團隊
