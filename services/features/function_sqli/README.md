# function_sqli - SQL 注入檢測模組

> **版本**: v2.0.0 | **狀態**: ✅ 完成 | **語言**: Python | **能力登錄**: ✅ 已登錄 (`sqli_multi_engine`)

## 模組概述

AIVA 平台的 SQL 注入綜合檢測模組，涵蓋從基礎 Payload 測試到進階智慧掃描的完整能力。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| Boolean-based SQLi | ✅ 完成 | 模糊邏輯 + 回應差異分析 |
| Error-based SQLi | ✅ 完成 | 20+ 資料庫錯誤模式 |
| Time-based Blind SQLi | ✅ 完成 | 動態延遲閾值 |
| Union-based SQLi | ✅ 完成 | 欄位數自動偵測 |
| Out-of-Band (OOB) | ✅ 完成 | DNS/HTTP 外帶回調 |
| HackingTool 整合 | ✅ 完成 | sqlmap 等外部工具管理 |
| WAF 繞過 | ✅ 完成 | 4 級混淆（0=無，3=最強） |
| NoSQL 注入 | ✅ 完成 | MongoDB 等 NoSQL 目標 |
| 資料庫指紋識別 | ✅ 完成 | MySQL/PostgreSQL/Oracle/MSSQL |

## 架構

```
function_sqli/
├── smart_detection_manager.py  # 主入口與編排器（SmartDetectionManager）
├── config.py                   # 集中配置（SqliConfig）
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
    └── sql_tools.py            # SQLInjectionManager（綜合入口）
```

## 執行方式

### 透過 AIVA 執行器（推薦）

```bash
# 智慧偵測（自動選擇最佳引擎）
python services/core/aiva_core/internal_exploration/aiva_external_executor.py \
    --lang python --func SmartDetectionManager.scan_target --target https://example.com

# 綜合掃描（整合所有工具）
python services/core/aiva_core/internal_exploration/aiva_external_executor.py \
    --lang python --func SQLInjectionManager.comprehensive_scan --target https://example.com
```

### 直接使用

```python
from services.features.function_sqli.smart_detection_manager import SmartDetectionManager
from services.features.function_sqli.config import SqliConfig

config = SqliConfig(waf_evasion_level=1)
manager = SmartDetectionManager(config)
```

## 配置說明（SqliConfig）

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `waf_evasion_level` | 0 | 0=無，1=低（隨機大小寫），2=中（Space2Comment），3=高（雙重 URL 編碼） |
| `stability_threshold` | 0.85 | 頁面穩定性最低相似度（0.0–1.0） |
| `fuzzy_similarity_threshold` | 0.1 | Boolean 誤報過濾閾值（0.0–1.0） |

## 可調用方法（公開 API）

| 類別 | 方法 | 說明 |
|------|------|------|
| `SmartDetectionManager` | `scan_target(task)` | 主要掃描入口 |
| `SmartDetectionManager` | `start_detection(target, config)` | 開始偵測 session |
| `SmartDetectionManager` | `get_detection_status(session_id)` | 查詢狀態 |
| `SmartDetectionManager` | `stop_detection(session_id)` | 停止偵測 |
| `SQLInjectionManager` | `comprehensive_scan(target_url, options)` | 整合所有工具的綜合掃描 |
| `HackingToolSQLManager` | `get_tool_recommendations(target_type)` | 外部工具推薦 |
| `HackingToolSQLManager` | `install_all_tools()` | 批次安裝外部工具 |
| `BackendDbFingerprinter` | `fingerprint(response)` | 資料庫指紋識別 |

## 掃描流程

1. **穩定性檢查** — 多次請求建立基線相似度，不穩定時調整閾值
2. **資料庫指紋** — 識別後端資料庫（MySQL/PostgreSQL/Oracle 等）以最佳化 Payload
3. **引擎偵測** — 依序執行 Boolean / Error / Time / Union / OOB 引擎
4. **外部工具** — 依配置整合 sqlmap 等外部工具

## 注意事項

- 僅限授權滲透測試使用
- 預設安全模式：不執行破壞性操作
- 外部工具（sqlmap）需另行安裝：`HackingToolSQLManager.install_all_tools()`
