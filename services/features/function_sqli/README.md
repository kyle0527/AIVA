# function_sqli - SQL 注入檢測模組

> **版本**: v3.0.0 | **狀態**: ✅ 完成 | **語言**: Python

## 模組概述

AIVA 平台的 SQL 注入綜合檢測模組，涵蓋從基礎 Payload 測試到進階智慧掃描的完整能力。此模組已經全面拋棄舊的 CommandHandler 架構，支援 CLI 驅動或直接被外部模組呼叫。

### 功能完成狀態

| 功能 | 說明 |
|------|------|
| Boolean-based SQLi | 模糊邏輯 + 回應差異分析 |
| Error-based SQLi | 20+ 資料庫錯誤模式 |
| Time-based Blind SQLi | 動態延遲閾值 |
| Union-based SQLi | 欄位數自動偵測 |
| Out-of-Band (OOB) | DNS/HTTP 外帶回調 (依據 OOB_INTERACTSH_PLAN) |
| HackingTool 整合 | sqlmap 等外部工具管理 |
| WAF 繞過 | 4 級混淆（0=無，3=最強） |
| NoSQL 注入 | MongoDB 等 NoSQL 目標 |
| 資料庫指紋識別 | MySQL/PostgreSQL/Oracle/MSSQL |

## 架構

```
function_sqli/
├── hackingtool_sql_cli.py      # 工具管理 CLI
├── smart_detection_manager.py  # 主入口與編排器（SmartDetectionManager）
├── config/
│   └── sqli_config.py          # 集中配置（SqliConfig）
├── payload_wrapper_encoder.py  # Payload 編碼與 Tamper 邏輯
├── detection_models.py         # 共享資料模型（DetectionResult）
├── backend_db_fingerprinter.py # 資料庫指紋識別
├── hackingtool_manager.py      # 外部工具管理（HackingToolSQLManager）
├── hackingtool_config.py       # 外部工具配置
├── telemetry.py                # 執行遙測（SqliExecutionTelemetry）
├── engines/                    # 偵測引擎
│   ├── base_detector.py
│   ├── boolean_detection_engine.py
│   ├── error_detection_engine.py
│   ├── time_detection_engine.py
│   ├── union_detection_engine.py
│   ├── oob_detection_engine.py
│   └── hackingtool_engine.py   # 跨語言工具引擎（CrossLanguageSQLEngine）
└── integration_tools/
    └── sql_tools.py            # HackingToolSQLIntegrator（外部工具整合入口）
```

## 執行方式

### 工具管理 CLI

```bash
# 查看 SQL 工具安裝狀態
python services/features/function_sqli/hackingtool_sql_cli.py status

# 測試特定目標
python services/features/function_sqli/hackingtool_sql_cli.py test sqlmap https://example.com
```

### 直接作為 Python 模組使用

```python
from services.features.function_sqli.smart_detection_manager import SmartDetectionManager
from services.features.function_sqli.config.sqli_config import SqliConfig

config = SqliConfig(waf_evasion_level=1)
manager = SmartDetectionManager(config)

# 執行目標檢測
task = {"url": "https://example.com/api/users?id=1"}
result = manager.scan_target(task)
```

## 配置說明（SqliConfig）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `waf_evasion_level` | 0 | 0=無，1=低（隨機大小寫），2=中（Space2Comment），3=高（雙重 URL 編碼） |
| `stability_threshold` | 0.85 | 頁面穩定性最低相似度（0.0–1.0） |
| `fuzzy_similarity_threshold` | 0.1 | Boolean 誤報過濾閾值（0.0–1.0） |

## 掃描流程

1. **穩定性檢查** — 多次請求建立基線相似度，不穩定時調整閾值
2. **資料庫指紋** — 識別後端資料庫（MySQL/PostgreSQL/Oracle 等）以最佳化 Payload
3. **引擎偵測** — 依序執行 Boolean / Error / Time / Union / OOB 引擎
4. **外部工具** — 依配置整合 sqlmap 等外部工具

## 注意事項

- 僅限授權滲透測試使用
- 預設安全模式：不執行破壞性操作
- 外部工具（sqlmap）需另行安裝
