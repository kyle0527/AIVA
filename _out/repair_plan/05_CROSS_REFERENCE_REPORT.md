# 修復計畫 vs 架構樹 交叉比對報告

> 生成日期: 2026-03-11
> 對比基準: `_out/tree_ultimate_chinese_20260311_033329.txt` (673 檔案, 200,362 行)
> 修復計畫: `01_PHASE1` ~ `04_GIT_CLEANUP`

---

## 結論摘要

修復計畫 Phase 1~4 **聚焦在 P0 關鍵阻塞鏈上是正確的**，但從架構樹對照來看，
有 **7 大模組區塊未在修復計畫中被提及或分析**。
這些未涵蓋的區塊目前不阻塞 `/scan` 端點運作，但對系統完整性有影響。

| 類別 | 覆蓋狀態 | 說明 |
|------|----------|------|
| 掃描關鍵鏈 (P0) | ✅ 完整覆蓋 | attack_coordinator → sqli_detector → commander → app.py |
| 部署 (P1) | ✅ 完整覆蓋 | Docker + DB + entrypoint |
| 功能模組 (P2) | ⚠️ 部分覆蓋 | 12 模組列出但未深入分析每個模組內部 |
| Git 清理 (P3) | ✅ 完整覆蓋 | 80+ 變更分類 |
| **以下為差距** | | |
| `services/aiva_common/` 共用庫 | ❌ 未涵蓋 | 20+ 子套件，是所有模組的基礎 |
| `services/scan/` 多引擎掃描 | ❌ 未涵蓋 | Go/Rust/TS/Python 四語言引擎 |
| `services/integration/` 整合層 | ❌ 未涵蓋 | 能力註冊、coordinator、資料管理 |
| `services/dashboard/` 儀表板 | ❌ 未涵蓋 | Streamlit 4 頁面 |
| `cognitive_core` 子模組 | ⚠️ 僅涵蓋 neural | 學習系統、RAG、嵌入知識、反幻覺 |
| `core_capabilities/` | ❌ 未涵蓋 | 分析/攻擊/CLI/對話/協調 |
| `training/` 訓練基礎建設 | ❌ 未涵蓋 | 模型蒸餾、詞彙建構、訓練腳本 |

---

## 逐區塊詳細對比

### 1. `_archive/` — ✅ 已確認

修復計畫已確認以下刪除為**有意的歸檔**：

| 歸檔目錄 | 重要檔案 | 修復計畫引用 |
|----------|---------|-------------|
| `09_integration_archive/alembic/` | `env.py`, `001_initial_schema.py` | Phase 2.3 提及 |
| `09_integration_archive/deprecated_managers/` | `authn_manager.py`, `postex_manager.py`, `scanner_manager.py` | Phase 4 提及 |
| `base_feature_infrastructure/base/` | `feature_registry.py`, `integration_helper.py`, `result_schema.py` | Phase 4 提及 |

**結論**: 無差距。

---

### 2. `_dev_tools/` — ⚠️ 未涵蓋（低風險）

架構樹顯示 37 個開發工具檔案：

```
_dev_tools/
├── common/development/ — 15 個分析/分類/轉換工具
├── converters/         — 5 個轉換器 (docx→md, sarif, schema codegen)
├── integration/        — 4 個語言插件 (contracts, enums, go, schemas)
├── aiva_5M_replacement_evaluation.py
├── aiva_capability_orchestrator.py
├── aiva_model_manager.py
├── debug_rust_call.py
├── real_ai_core.py
└── scan_real_targets.py
```

**影響**: 這些是開發/維護工具，不影響運行時功能。
**建議**: 無需納入修復計畫，但 `scan_real_targets.py` 可在 Phase 3 整合測試時參考。

---

### 3. `services/aiva_common/` — ❌ 未涵蓋（中風險）

架構樹顯示 **20 個子套件、~80 個檔案**，這是整個專案的共用基礎庫：

```
services/aiva_common/
├── ai/              — AI 介面、效能設定、註冊表 (4 檔)
├── async_utils/     — 非同步工具 (1 檔)
├── cli/             — CLI 框架 (1 檔)
├── config/          — 統一設定管理 (6 檔: config_manager, defaults, paths, settings, unified_config)
├── core/            — 核心工具 (3 檔: command_center, error_handling)
├── cross_language/  — 跨語言橋接 (5 檔: go_adapter, rust_adapter, core, errors)
├── detection/       — 偵測基礎 (5 檔: metrics_collector, rate_limiter, smart_detection_manager, timeout_manager)
├── enums/           — 列舉定義 (15 檔: 涵蓋 academic, ai, business, capabilities, security 等)
├── messaging/       — MQ 訊息 (5 檔: compatibility_layer, mq, retry_handler, unified_topic_manager)
├── observability/   — 可觀測性 (4 檔: metrics, monitoring, monitoring_log_handler)
├── pipeline/        — 資料管線 (3 檔: data_pipeline, stream_processor)
├── plugins/         — 插件系統 (1 檔)
├── protocols/       — gRPC protobuf (7 檔: aiva_enums, aiva_errors, aiva_services + _grpc)
├── schemas/         — 結構化 schema (40+ 檔，含 8 個子目錄: _base, analysis, generated, infrastructure, interfaces, risk, security, testing)
├── security/        — 安全中間件 (4 檔: security_config, security_middleware, security)
├── services/        — 服務發現 (3 檔: service_discovery + TypeScript schemas)
├── tools/           — 共用工具 (4 檔: module_connectivity_checker, schema_codegen_tool, schema_validator, statistics)
├── utils/           — 通用工具 (8 檔: dedup, network/backoff, network/ratelimit, ids, logging, retry)
└── version.py
```

**為什麼重要**:
- 所有 `services/core/` 和 `services/features/` 都依賴這個共用庫
- 如果 `aiva_common` 有 import 問題，會連帶影響所有模組
- `detection/smart_detection_manager.py` 與 `function_sqli/smart_detection_manager.py` 可能有命名衝突
- `security/security_middleware.py` 是 API 安全層的基礎

**建議**: 新增至修復計畫 Phase 1 的前置檢查 — 確認 `import services.aiva_common` 成功，
以及各子套件沒有循環引用問題。

---

### 4. `services/core/aiva_core/cognitive_core/` — ⚠️ 僅部分涵蓋

修復計畫只涵蓋了 `neural/real_neural_core.py`（Phase 1.4 的 `decide()` 介面修復）。
但架構樹顯示 cognitive_core 下有 **7 個子模組、~45 個檔案**：

| 子模組 | 檔案數 | 修復計畫覆蓋 | 說明 |
|--------|--------|------------|------|
| `anti_hallucination/` | 2 | ❌ | 反幻覺模組 |
| `decision/` | 7+3 tracing+2 training | ❌ (除 decide() 介面) | 決策代理、策略、經驗管理 |
| `embedded_knowledge/` | 7 | ❌ | 內嵌安全知識 (CVE、WAF bypass、Web 架構) |
| `learning_system/` | 12 | ❌ | 持續學習、RL 模型、訓練器 |
| `neural/` | 7 | ✅ 部分 | 只覆蓋 real_neural_core.py |
| `rag/` | 7 | ❌ | RAG 引擎、向量存儲、知識庫同步 |
| 根目錄 | 8 | ❌ | dispatcher, connectors, capability query/encoder/orchestrator |

**未涵蓋的重要檔案**:
- `bounty_strategy_agent.py` — 新增的 (untracked `??`)，Bug Bounty 策略代理
- `knowledge_decision_mixin.py` — 新增的 (untracked `??`)，知識決策混入
- `rag/rag_engine.py` — RAG 引擎核心
- `learning_system/learning/continuous_learning.py` — 持續學習
- `learning_system/learning/rl_models.py` — 強化學習模型

**影響**: 這些子模組在 `/scan` 13 步驟流程的「AI 決策」階段會被調用。
Phase 1 修復 `decide()` 介面後，下一層的斷裂可能在這些子模組顯現。

**建議**: 在 Phase 1.5 端到端測試中增加對這些子模組的 import 驗證。

---

### 5. `services/core/aiva_core/core_capabilities/` — ❌ 未涵蓋

架構樹顯示 **8 個子目錄、19 個檔案**：

```
core_capabilities/
├── analysis/          — analysis_engine, bizlogic_scanner, initial_surface
├── attack/            — attack_chain, custom_exploits_example, exploit_orchestrator
├── cli/               — aiva_cli.py (主 CLI 入口)
├── dialog/            — ai_menu, assistant (對話式介面)
├── ingestion/         — scan_module_interface (掃描模組介面)
├── orchestration/     — two_phase_scan_orchestrator (兩階段掃描)
├── output/            — to_functions (結果轉換)
├── processing/        — scan_result_processor (結果處理)
├── capability_registry.py
├── multilang_coordinator.py
├── risk_policy_manager.py
└── task_context.py
```

**為什麼重要**:
- `two_phase_scan_orchestrator.py` — 這是 13 步驟掃描流程的**核心協調器**
- `scan_module_interface.py` — 掃描模組的統一介面
- `aiva_cli.py` — 如果要支援 CLI 操作模式
- `capability_registry.py` — 功能註冊中心

**影響**: Phase 1 修復 attack_coordinator 後，掃描流程的下一站就是 `two_phase_scan_orchestrator`。
如果此模組有問題，端到端測試仍會失敗。

**建議**: 在 Phase 1.5 中加入 `two_phase_scan_orchestrator` 的 import 和基本初始化測試。

---

### 6. `services/core/aiva_core/service_backbone/` — ⚠️ 僅涵蓋 app.py

修復計畫只涵蓋 `api/app.py`（啟動/路由），但 service_backbone 有 **11 個子目錄、30+ 個檔案**：

| 子目錄 | 檔案 | 功能 | 修復計畫覆蓋 |
|--------|------|------|------------|
| `adapters/` | protocol_adapter | 協議適配 | ❌ |
| `api/` | ai_service, app, enhanced_unified_caller, scan_endpoints, sse, unified_function_caller | API 層 | ✅ `app.py` only |
| `authz/` | authz_mapper, matrix_visualizer, permission_matrix | 授權 | ❌ |
| `coordination/` | ai_controller, ai_manager, core_service_coordinator | 協調 | ❌ |
| `messaging/` | message_broker, result_collector, task_dispatcher | 訊息 | ❌ |
| `performance/` | diagnose, health_check, monitoring, parallel_processor, unified_memory_manager | 效能 | ❌ |
| `state/` | session_state_manager | 狀態 | ❌ |
| `storage/` | backends, command_repository, config, db_helper, models, storage_manager | 儲存 | ❌ |
| `utils/` | logging_formatter, repair_tool | 工具 | ❌ |
| 根目錄 | context_manager, dispatcher_base | 基礎 | ❌ |

**關鍵發現**:
- `scan_endpoints.py` — 可能是 `/scan` 路由的實際處理邏輯（與 app.py 分離）
- `unified_function_caller.py` — Phase 1.1 提到此檔案引用已刪除的 sqli_detector
- `enhanced_unified_caller.py` — 可能也有同樣問題
- `storage/` — DB 層的 ORM 實作，Phase 2.3 的 DB 初始化會用到

**建議**: `scan_endpoints.py` 和 `enhanced_unified_caller.py` 應加入 Phase 1 的 import 檢查清單。

---

### 7. `services/core/aiva_core/task_planning/` — ⚠️ 僅涵蓋 commander 的 2 個檔案

修復計畫覆蓋了 `commander/__init__.py` 和 `attack_coordinator.py`，
但 task_planning 有 **4 個子目錄、24 個檔案**：

```
task_planning/
├── commander/
│   ├── __init__.py              ✅ 已覆蓋
│   ├── attack_coordinator.py    ✅ 已覆蓋
│   ├── capability_manager.py    ❌
│   ├── capability_matcher.py    ❌
│   ├── learning_adapter.py      ❌
│   ├── plan_builder.py          ❌
│   ├── policy_manager.py        ❌
│   ├── strategy_engine.py       ❌
│   └── types.py                 ❌
├── executor/
│   ├── attack_plan_mapper.py    ❌
│   ├── execution_status_monitor.py ❌
│   ├── plan_executor.py         ❌
│   ├── task_executor.py         ❌
│   └── task_queue_manager.py    ❌
├── planner/
│   ├── ast_parser.py            ❌
│   ├── plan_comparator.py       ❌
│   ├── task_converter.py        ❌
│   ├── task_execution_planner.py ❌
│   ├── task_generator.py        ❌
│   └── tool_selector.py         ❌
├── command_builder.py           ❌
├── command_router.py            ❌
├── dispatcher.py                ❌
├── mode_manager.py              ❌
├── strategy_profiles.py         ❌
└── unified_executor.py          ✅ 測試中有提及
```

**影響**: `CommanderCoordinator` 會使用 `@property` 延遲載入這些子模組。
即使 Phase 1 修好 attack_coordinator，當 commander 調用 `plan_builder.py` 
或 `strategy_engine.py` 時，可能觸發新的 import 錯誤。

**建議**: 在 Phase 1.5 中加入 commander 全子模組的 import 掃描。

---

### 8. `services/core/aiva_core/internal_exploration/` — ❌ 未涵蓋

架構樹顯示 **5 個子目錄、16 個檔案** 的自我探索/自我修復系統：

```
internal_exploration/
├── classification_results/  — analyze_same_file, check_connections
├── go_tools/               — go2mermaid, paths_config (Go)
├── python_tools/           — aiva_flow_analyzer
├── rust_tools/src/         — main.rs, paths_config.rs (Rust)
├── self_healing/           — 7 個自我修復分析器
├── typescript_tools/       — paths.config.ts, ts2mermaid.ts
├── utils/                  — standalone_cli_validator
├── aiva_external_classifier/executor
├── aiva_internal_classifier/executor
└── system_self_explorer.py
```

**影響**: 此模組是「內部迴路」的一部分，用於 AIVA 自我分析和修復。
不影響 `/scan` 的核心流程，但是 dual-loop 架構的關鍵組件。

**建議**: 可延後到 Phase 3 處理。但注意 `self_healing/` 下的分析器可能在啟動時被調用。

---

### 9. `services/scan/` — ❌ 未涵蓋（中高風險）

架構樹顯示 **四語言掃描引擎**，這是實際執行掃描的核心：

```
services/scan/
├── coordinators/
│   └── multi_engine_coordinator.py    ← 多引擎協調器
├── go_engine/
│   ├── cmd/ — cspm-scanner, sca-scanner, ssrf-scanner (3 個 Go 執行檔)
│   ├── internal/ — common, cspm, detectors, fuzzer, sca, ssrf (12 個 .go 檔)
│   ├── pkg/models/ — models.go
│   └── tools/ — analyze_call_chain.go
├── python_engine/
│   ├── deserialization_detector.py + v2
│   ├── passive_analyzer.py
│   └── xxe_detector.py
├── rust_engine/
│   └── src/ — 12 個 .rs 檔 (attack_surface, auth_brute, endpoint_discovery, js_analyzer, etc.)
├── typescript_engine/
│   ├── src/ — 10 個 .ts 檔 (dom-security, spa-route, websocket, services/*)
│   └── phase-i-integration.service.ts
└── __init__.py
```

**為什麼重要**:
- `multi_engine_coordinator.py` 被 `CommanderCoordinator` 引用
- Phase 1.3 提到需確認 `services/scan/` 路徑存在 — 它確實存在
- Go/Rust/TypeScript 引擎需要各自的工具鏈才能編譯
- `python_engine/` 的 4 個偵測器（反序列化、XXE、被動分析）不在功能模組表中

**影響**: 即使 Python 核心修好了，Go/Rust/TS 引擎可能無法編譯或執行。

**建議**: 
- Phase 1.3 已部分覆蓋（確認路徑），但需深入檢查 `multi_engine_coordinator.py` 的 import
- Python engine 的 4 個偵測器應加入 Phase 3 模組表

---

### 10. `services/features/` — ⚠️ 額外發現

修復計畫 Phase 3.1 列出了 12 個模組的表格，但架構樹揭示了幾個額外問題：

#### 10a. 重複/衝突目錄
| 檔案路徑 | 問題 |
|----------|------|
| `function_postex/detector/postex_detector.py` | ❗ 同時存在 `detector/` 和 `detectors/` 兩個目錄 |
| `function_postex/detectors/postex_detector.py` | 同名檔案在兩個目錄，可能 import 混亂 |
| `function_info_leak/` | ❗ 同時存在 `function_info_leak/` 和 `function_infoleak/`（空目錄？）|

#### 10b. 根層級被遺漏的檔案
| 路徑 | 說明 |
|------|------|
| `services/features/feature_step_executor.py` | 功能步驟執行器（可能被 commander 調用）|
| `services/features/smart_detection_manager.py` | 根層級的偵測管理器（與 function_sqli 內的同名？）|
| `services/features/validate_handlers.py` | Handler 驗證工具 |

#### 10c. 未列出的模組
| 模組 | Phase 3 是否列出 | 實際狀態 |
|------|-----------------|---------|
| `function_forensic/` | ❌ 未列出 | 有 manager + models + legacy |
| `function_reverse_engineering/` | ❌ 未列出 | 有 manager + models + legacy |
| `function_social_engineering/` | ❌ 未列出 | 有 manager + models + legacy |
| `function_steganography/` | ❌ 未列出 | 有 engines (ai_steg, stegx) + manager + models |
| `function_wordlist_generator/` | ❌ 未列出 | 有 handler + manager + models |
| `function_info_leak/` | ❌ 未列出 | 有 sensitive_info_detector |

**建議**: Phase 3.1 的模組表應從 12 個擴充到 **18 個**（加入上述 6 個）。

---

### 11. `services/integration/` — ❌ 未涵蓋

架構樹顯示完整的整合層：

```
services/integration/
├── capability/
│   ├── adapters/hackingtool_adapter    — HackingTool 適配器
│   ├── capabilities/sqli, ssrf, xss   — 各漏洞類型能力定義
│   ├── bug_bounty_reporting            — Bug Bounty 報告
│   ├── lifecycle, lifecycle_cli         — 能力生命週期
│   ├── registry                        — 能力註冊中心
│   ├── forensic_tools, reverse_engineering_tools, steganography_tools — 工具整合
│   └── 其他 (config, models, toolkit, sync_from_analysis, etc.)
├── coordinators/
│   ├── base_coordinator
│   └── xss_coordinator
├── data/ — 大型資料目錄 (attack_paths, backups, experiences, internal_exploration, training_datasets)
├── scripts/ — backup, cleanup
├── tools/ — sop_compliance_checker
├── models.py
├── search_command_handler.py
└── simple_data_manager.py
```

**影響**: `capability/registry.py` 是功能模組註冊到核心的樞紐。
如果 Phase 1 修復 import 但 capability registry 有問題，功能模組仍無法被 commander 調用。

**建議**: Phase 1.5 端到端測試中應加入 `from services.integration.capability.registry import ...` 驗證。

---

### 12. `services/dashboard/` — ❌ 未涵蓋（低風險）

Streamlit 儀表板，4 個頁面：
- `1_🎯_掃描控制台.py` — 掃描控制
- `2_📊_即時監控.py` — 即時監控
- `3_🔍_結果分析.py` — 結果分析
- `4_📜_歷史記錄.py` — 歷史記錄
- `api_client.py` — API 客戶端
- `streamlit_app.py` — 主入口

**影響**: 不影響核心 API，但是用戶友好的操作介面。
**建議**: 可延後到 Phase 3 之後處理。確認 `api_client.py` 的 API 端點與 `app.py` 一致。

---

### 13. `services/core/aiva_core/` 根目錄檔案 — ⚠️ 未涵蓋

| 檔案 | 說明 | 影響 |
|------|------|------|
| `main.py` | 服務入口點 (port 9000 安全閘道) | ⚠️ Phase 2.2 的 entrypoint 應引用此檔 |
| `ai_models.py` | AI 模型定義 | ⚠️ 可能被 neural core 引用 |
| `models.py` | 資料模型定義 | ⚠️ 可能被 storage 引用 |
| `session_state_manager.py` | 會話狀態管理 | 低 |
| `startup_guide.py` | 啟動導引 | 低 |
| `_fix_all_readmes.py` | 維護腳本 | 無 |

**建議**: `main.py` 應加入 Phase 2 的部署分析，確認 port 9000 閘道 vs port 8000 API 的關係。

---

### 14. `training/` — ❌ 未涵蓋（低風險）

ML 訓練基礎建設：
```
training/
├── data/distillation_dataset/
├── data/security_vocabulary/
├── data/data_converter.py
├── scripts/build_security_vocabulary.py
├── scripts/data_converter.py
├── scripts/generate_distillation_dataset.py
├── scripts/train_all.py
└── scripts/train_student_model.py
```

**影響**: 不影響運行時。這些是離線訓練工具。
**建議**: 不需納入修復計畫。但確認 `train_all.py` 能正確載入 neural core 模型。

---

### 15. `tests/` — ✅ 已涵蓋

修復計畫 Phase 3.3 已列出現有 6 個測試檔案和補寫計畫。
架構樹確認一致：
```
tests/
├── test_attack_coordinator_simple.py
├── test_cli_architecture.py
├── test_direct_import.py
├── verify_attack_coordinator.py
├── verify_dispatcher.py
└── verify_internal_loop.py
```

---

## 修復計畫補充建議

基於以上交叉比對，建議對現有修復計畫做以下補充：

### Phase 1 補充（加入前置驗證）

新增 **Phase 1.0: 基礎 import 健康檢查**，在修復 sqli_detector 之前先確認共用庫無問題：

```powershell
# 建議的 import 健康檢查腳本
python -c "
import sys; sys.path.insert(0,'.')
modules = [
    'services.aiva_common',
    'services.aiva_common.config',
    'services.aiva_common.schemas',
    'services.aiva_common.security',
    'services.aiva_common.detection',
    'services.aiva_common.enums',
    'services.core.aiva_core.cognitive_core',
    'services.core.aiva_core.core_capabilities',
    'services.core.aiva_core.service_backbone',
    'services.core.aiva_core.task_planning',
    'services.scan',
    'services.integration',
    'services.features',
]
for m in modules:
    try:
        __import__(m)
        print(f'  ✅ {m}')
    except Exception as e:
        print(f'  ❌ {m}: {e}')
"
```

### Phase 1.5 補充（擴大端到端測試範圍）

除了測試 `POST /scan` 外，還應測試：
1. `two_phase_scan_orchestrator` 是否能初始化
2. `multi_engine_coordinator` 是否能初始化
3. `capability_registry` 是否能載入所有已註冊能力
4. `scan_endpoints.py` 和 `enhanced_unified_caller.py` import 是否正常

### Phase 3.1 補充（擴充模組表）

在現有 12 個模組基礎上加入 6 個遺漏模組：

| 模組 | `__main__.py` | 狀態 | 備註 |
|------|---------------|------|------|
| `function_forensic` | ❌ | 骨架 | legacy + manager/models |
| `function_reverse_engineering` | ❌ | 骨架 | legacy + manager/models |
| `function_social_engineering` | ❌ | 骨架 | legacy + manager/models |
| `function_steganography` | ❌ | 骨架 | 有 AI 引擎 + stegx |
| `function_wordlist_generator` | ❌ | 骨架 | 有 handler |
| `function_info_leak` | ❌ | 骨架 | 有 sensitive_info_detector |

### Phase 3 補充（清理重複結構）

需要解決：
- `function_postex/detector/` vs `function_postex/detectors/` — 合併或刪除一方
- `function_info_leak/` vs `function_infoleak/` — 確認 infoleak 是否為空目錄
- `services/features/smart_detection_manager.py` vs `services/features/function_sqli/smart_detection_manager.py` — 確認關係

### Phase 2 補充（入口點確認）

`services/core/aiva_core/main.py` (port 9000 安全閘道) 與 
`services/core/aiva_core/service_backbone/api/app.py` (port 8000 API) 的關係需在 Phase 2 中釐清。
Docker entrypoint 應啟動哪一個？

---

## 最終覆蓋度評估

| 區塊 | 檔案數 (估) | 修復計畫覆蓋 | 需補充 |
|------|------------|------------|--------|
| `_archive/` | ~50 | ✅ 已確認 | — |
| `_dev_tools/` | ~37 | ⚠️ 不需修復 | — |
| `services/aiva_common/` | ~80 | ❌ 未涵蓋 | 加入 Phase 1.0 |
| `services/core/cognitive_core/` | ~45 | ⚠️ 1/45 | 加入 Phase 1.5 |
| `services/core/core_capabilities/` | ~19 | ❌ 未涵蓋 | 加入 Phase 1.5 |
| `services/core/service_backbone/` | ~30 | ⚠️ 1/30 | 加入 Phase 1.5 |
| `services/core/task_planning/` | ~24 | ⚠️ 2/24 | 加入 Phase 1.5 |
| `services/core/internal_exploration/` | ~16 | ❌ 未涵蓋 | 延後 Phase 3 |
| `services/features/` | ~120 | ⚠️ 表格涵蓋 | 補齊 6 模組 |
| `services/scan/` | ~35 | ❌ 未涵蓋 | 加入 Phase 1.5 |
| `services/integration/` | ~25 | ❌ 未涵蓋 | 加入 Phase 1.5 |
| `services/dashboard/` | ~8 | ❌ 未涵蓋 | 延後 |
| `training/` | ~8 | ❌ 不需修復 | — |
| `tests/` | 6 | ✅ 已涵蓋 | — |
| Docker/Config | ~15 | ✅ 已涵蓋 | — |
| **總計** | **~673** | **修復計畫直接涉及 ~10 個檔案** | **需增加 import 健康檢查覆蓋約 200 個模組** |

---

## 結論

1. **修復計畫 Phase 1~4 的修復「動作」是正確的** — 找到了最關鍵的阻塞點 (sqli_detector import)，修復路徑正確
2. **覆蓋範圍確實不夠廣** — 673 個檔案中，修復計畫直接提及的只有 ~10 個
3. **最大風險是「修好 A，B 又壞」** — 因為沒有對 aiva_common、scan、integration 做 import 掃描
4. **建議**: 在 Phase 1 最前面加入自動化 import 健康檢查，一次掃描所有 400+ Python 模組的 import 狀態，
   可以在 5 分鐘內完成，並一次性暴露所有斷裂的 import 鏈
