# AIVA 五大模組內部架構模組化計畫

## 📑 目錄

- [📋 總覽](#總覽)
- [🏗️ 一、Core 模組（已有六大子模組，作為參考範例）](#一core-模組已有六大子模組作為參考範例)
  - [現有架構 ✅](#現有架構)
  - [架構原則（供參考）](#架構原則供參考)
- [🔧 二、Common 模組內部模組化設計（根據實際需求）](#二common-模組內部模組化設計根據實際需求)
  - [現狀分析](#現狀分析)
  - [分析：Common 模組的實際需求](#分析common-模組的實際需求)
  - [建議模組化架構（四層架構）](#建議模組化架構四層架構)
  - [遷移計畫](#遷移計畫)
- [🎯 三、Features 模組內部模組化設計（根據實際需求）](#三features-模組內部模組化設計根據實際需求)
  - [分析：Features 模組的實際需求](#分析features-模組的實際需求)
  - [現狀分析](#現狀分析-1)
  - [建議模組化架構（三層分類 + 支援層）](#建議模組化架構三層分類-支援層)
  - [遷移計畫](#遷移計畫-1)
- [🔗 四、Integration 模組內部模組化設計（根據實際需求）](#四integration-模組內部模組化設計根據實際需求)
  - [分析：Integration 模組的實際需求](#分析integration-模組的實際需求)
  - [現狀分析](#現狀分析-1)
  - [建議模組化架構（按數據流設計）](#建議模組化架構按數據流設計)
  - [遷移計畫](#遷移計畫-1)
- [🔬 五、Scan 模組內部模組化設計（根據實際需求）](#五scan-模組內部模組化設計根據實際需求)
  - [分析：Scan 模組的實際需求](#分析scan-模組的實際需求)
  - [現狀分析](#現狀分析-1)
  - [建議模組化架構（三層架構）](#建議模組化架構三層架構)
  - [遷移計畫](#遷移計畫-1)
- [📊 六、架構對比與優勢](#六架構對比與優勢)
  - [模組化前後對比](#模組化前後對比)
  - [設計優勢](#設計優勢)
- [🚀 七、實施步驟](#七實施步驟)
  - [Phase 1: 規劃階段（1-2 天）](#phase-1-規劃階段12-天)
  - [Phase 2: Common 模組重構（3-4 天）](#phase-2-common-模組重構34-天)
  - [Phase 3: Features 模組重構（4-5 天）](#phase-3-features-模組重構45-天)
  - [Phase 4: Integration 模組重構（3-4 天）](#phase-4-integration-模組重構34-天)
  - [Phase 5: Scan 模組重構（2-3 天）](#phase-5-scan-模組重構23-天)
  - [Phase 6: 文檔與驗證（2-3 天）](#phase-6-文檔與驗證23-天)
- [✅ 驗證清單](#驗證清單)
  - [架構合理性](#架構合理性)
  - [功能完整性](#功能完整性)
  - [測試覆蓋](#測試覆蓋)
  - [文檔完整性](#文檔完整性)
- [📝 注意事項](#注意事項)
  - [🔴 遷移風險](#遷移風險)
  - [🟡 向後兼容](#向後兼容)
- [🎯 預期成果](#預期成果)
  - [設計原則總結](#設計原則總結)
  - [短期成果（1 個月內）](#短期成果1-個月內)
  - [長期成果（3-6 個月）](#長期成果36-個月)
- [📞 後續步驟](#後續步驟)

---
---
---
## 📋 總覽

根據每個模組的**實際職責與需求**，設計適合的內部模組化結構。不強求統一數量，以實用性為主。

---

## 🏗️ 一、Core 模組（已有六大子模組，作為參考範例）

### 現有架構 ✅
```
services/core/aiva_core/
├── cognitive_core/           # 1️⃣ 認知核心（AI 大腦）
│   ├── anti_hallucination/
│   ├── decision/
│   ├── neural/
│   ├── rag/
│   └── nlg_system.py
├── external_learning/        # 2️⃣ 對外學習
│   ├── ai_model/
│   ├── analysis/
│   ├── learning/
│   ├── tracing/
│   └── training/
├── internal_exploration/     # 3️⃣ 內部探索
│   ├── capability_analyzer.py
│   ├── language_extractors.py
│   └── module_explorer.py
├── task_planning/            # 4️⃣ 任務規劃
│   ├── executor/
│   ├── planner/
│   ├── ai_commander.py
│   └── command_router.py
├── service_backbone/         # 5️⃣ 服務骨幹
│   ├── adapters/
│   ├── api/
│   ├── authz/
│   ├── coordination/
│   ├── messaging/
│   ├── performance/
│   ├── state/
│   └── storage/
└── ui_panel/                 # 6️⃣ UI 面板
    ├── dashboard.py
    ├── rich_cli.py
    └── server.py
```

### 架構原則（供參考）
- ✅ **職責清晰**：每個子模組有明確的功能定位
- ✅ **層次分明**：子模組內部進一步分層
- ✅ **低耦合**：子模組間通過接口通信
- ✅ **高內聚**：相關功能集中在同一子模組

**重要**: 其他模組不需要強制套用六大模組結構，應根據實際需求設計。

---

## 🔧 二、Common 模組內部模組化設計（根據實際需求）

### 現狀分析
```
services/aiva_common/
├── ai/                      # AI 相關
├── async_utils/             # 異步工具
├── cli/                     # CLI 工具
├── config/                  # 配置管理
├── cross_language/          # 跨語言
├── enums/                   # 枚舉定義
├── messaging/               # 消息傳遞
├── observability/           # 可觀測性
├── plugins/                 # 插件系統
├── protocols/               # gRPC 協議
├── schemas/                 # 數據模型
├── utils/                   # 工具函數
└── v2_client/               # 客戶端
```

### 分析：Common 模組的實際需求

Common 模組是**共享基礎設施**，主要提供：
1. 數據契約定義（schemas, enums）
2. 跨服務通信（messaging, protocols）
3. 基礎工具函數（utils, async_utils）
4. 配置與安全（config, security）

### 建議模組化架構（四層架構）

```
services/aiva_common/
├── contracts/               # 📜 契約定義（Contracts）
│   ├── schemas/             # 數據模型定義
│   ├── enums/               # 枚舉類型
│   └── protocols/           # gRPC 協議定義
│
├── communication/           # 📡 通信基礎（Communication）
│   ├── messaging/           # 消息隊列封裝
│   ├── cross_language/      # 跨語言橋接
│   └── v2_client/           # 客戶端 SDK
│
├── infrastructure/          # 🏗️ 基礎設施（Infrastructure）
│   ├── config/              # 配置管理
│   ├── security/            # 安全機制（security.py, security_*.py）
│   ├── monitoring/          # 監控系統（monitoring*.py, metrics.py）
│   ├── observability/       # 可觀測性
│   └── service_discovery/   # 服務發現
│
└── utilities/               # 🛠️ 工具集（Utilities）
    ├── async_utils/         # 異步工具
    ├── cli/                 # CLI 工具
    ├── utils/               # 通用工具函數
    ├── plugins/             # 插件系統
    ├── ai/                  # AI 相關工具
    └── tools/               # 開發工具（codegen, validator）
```

**設計理由**：
- **四層足夠**：Common 的職責明確，不需要過度細分
- **契約層獨立**：schemas/enums 是最核心的定義，單獨一層便於管理
- **通信層專注**：messaging 和跨語言是關鍵功能，獨立管理
- **工具層整合**：小工具集中管理，避免碎片化

### 遷移計畫

| 目前位置 | 新位置 | 模組 | 說明 |
|---------|--------|------|------|
| `schemas/` | `1_contracts/schemas/` | 合約層 | 數據契約 |
| `enums/` | `1_contracts/enums/` | 合約層 | 枚舉定義 |
| `protocols/` | `1_contracts/protocols/` | 合約層 | gRPC 定義 |
| `messaging/` | `2_communication/messaging/` | 通信層 | 消息傳遞 |
| `cross_language/` | `2_communication/cross_language/` | 通信層 | 跨語言 |
| `v2_client/` | `2_communication/v2_client/` | 通信層 | 客戶端 |
| `config/` | `3_infrastructure/config/` | 基礎設施 | 配置 |
| `security*` | `3_infrastructure/security/` | 基礎設施 | 安全 |
| `monitoring*` | `3_infrastructure/monitoring/` | 基礎設施 | 監控 |
| `observability/` | `3_infrastructure/observability/` | 基礎設施 | 可觀測 |
| `async_utils/` | `4_utilities/async_utils/` | 工具層 | 異步 |
| `cli/` | `4_utilities/cli/` | 工具層 | CLI |
| `utils/` | `4_utilities/utils/` | 工具層 | 通用 |
| `plugins/` | `5_extensions/plugins/` | 擴展層 | 插件 |
| `ai/` | `5_extensions/ai/` | 擴展層 | AI |
| `tools/` | `6_tools/` | 開發工具 | 工具 |

---

## 🎯 三、Features 模組內部模組化設計（根據實際需求）

### 分析：Features 模組的實際需求

Features 模組包含**各種檢測功能**，每個功能相對獨立。關鍵問題是：
- 20+ 功能模組平鋪，難以分類管理
- 需要按照**攻擊類型**或**檢測階段**分類
- 每個功能模組內部結構已經完善（config, detector, engine, worker）

### 現狀分析
```
services/features/
├── base/                    # 基礎類
├── common/                  # 公共組件
├── docs/                    # 文檔
├── function_authn_go/       # 認證功能 (Go)
├── function_bizlogic/       # 業務邏輯
├── function_crypto/         # 加密
├── function_ddos/           # DDoS
├── function_exploit_framework/ # 漏洞利用
├── function_forensic/       # 取證
├── function_idor/           # IDOR
├── function_payload_generator/ # Payload 生成
├── function_postex/         # 後滲透
├── function_reverse_engineering/ # 逆向工程
├── function_social_engineering/ # 社工
├── function_sqli/           # SQL 注入
├── function_ssrf/           # SSRF
├── function_steganography/ # 隱寫術
├── function_web_scanner/   # Web 掃描
├── function_wordlist_generator/ # 字典生成
└── function_xss/            # XSS
```

### 建議模組化架構（三層分類 + 支援層）

```
services/features/
├── vulnerability_detection/  # 🔍 漏洞檢測（Web 安全）
│   ├── function_sqli/        # SQL 注入
│   ├── function_xss/         # XSS 跨站腳本
│   ├── function_ssrf/        # SSRF 服務端請求偽造
│   ├── function_idor/        # IDOR 不安全直接對象引用
│   ├── function_authn_go/    # 認證繞過 (Go)
│   ├── function_bizlogic/    # 業務邏輯漏洞
│   └── function_crypto/      # 密碼學弱點
│
├── attack_utilities/         # ⚔️ 攻擊工具（滲透輔助）
│   ├── function_exploit_framework/  # 漏洞利用框架
│   ├── function_payload_generator/  # Payload 生成
│   ├── function_postex/      # 後滲透
│   ├── function_forensic/    # 數字取證
│   ├── function_wordlist_generator/ # 字典生成
│   └── function_reverse_engineering/ # 逆向工程
│
├── specialized_tools/        # 🛠️ 專用工具（特殊場景）
│   ├── function_web_scanner/ # Web 掃描
│   ├── function_ddos/        # DDoS 測試
│   ├── function_social_engineering/ # 社工模擬
│   └── function_steganography/      # 隱寫術分析
│
└── shared/                   # 📦 共享資源
    ├── base/                 # 基礎類和抽象接口
    ├── common/               # 公共組件（Go 共享庫等）
    └── docs/                 # 文檔
```

**設計理由**：
- **三層分類更合理**：按照功能性質分類（檢測 vs 工具 vs 專用）
- **保持模組獨立性**：不破壞現有模組內部結構
- **便於擴展**：新功能容易歸類
- **符合實際使用場景**：開發者能快速找到需要的功能

### 遷移計畫

| 功能模組 | 新分類 | 模組 | 檢測目標 |
|---------|--------|------|---------|
| `function_sqli/` | `1_detection_modules/sqli/` | 檢測 | SQL 注入 |
| `function_xss/` | `1_detection_modules/xss/` | 檢測 | XSS |
| `function_ssrf/` | `1_detection_modules/ssrf/` | 檢測 | SSRF |
| `function_idor/` | `1_detection_modules/idor/` | 檢測 | IDOR |
| `function_crypto/` | `1_detection_modules/crypto/` | 檢測 | 密碼弱點 |
| `function_authn_go/` | `2_authentication_modules/authn/` | 認證 | 認證繞過 |
| `function_bizlogic/` | `2_authentication_modules/bizlogic/` | 認證 | 業務邏輯 |
| `function_exploit_framework/` | `3_exploitation_modules/exploit_framework/` | 利用 | 漏洞利用 |
| `function_payload_generator/` | `3_exploitation_modules/payload_generator/` | 利用 | Payload |
| `function_postex/` | `3_exploitation_modules/postex/` | 利用 | 後滲透 |
| `function_web_scanner/` | `4_reconnaissance_modules/web_scanner/` | 偵察 | Web 掃描 |
| `function_forensic/` | `4_reconnaissance_modules/forensic/` | 偵察 | 取證 |
| `function_wordlist_generator/` | `4_reconnaissance_modules/wordlist_generator/` | 偵察 | 字典 |
| `function_ddos/` | `5_attack_simulation_modules/ddos/` | 模擬 | DDoS |
| `function_social_engineering/` | `5_attack_simulation_modules/social_engineering/` | 模擬 | 社工 |
| `function_reverse_engineering/` | `6_support_modules/reverse_engineering/` | 支援 | 逆向 |
| `function_steganography/` | `6_support_modules/steganography/` | 支援 | 隱寫 |

---

## 🔗 四、Integration 模組內部模組化設計（根據實際需求）

### 分析：Integration 模組的實際需求

Integration 模組負責**數據整合與分析**，是數據流轉的中樞：
- 接收來自 Scan 和 Features 的數據
- 進行風險分析、威脅情報關聯
- 生成報告和修復建議
- 提供 API 網關

**數據流**：接收 → 分析 → 輸出

### 現狀分析
```
services/integration/
├── aiva_integration/
│   ├── analysis/            # 分析引擎
│   ├── attack_path_analyzer/ # 攻擊路徑
│   ├── config_template/     # 配置模板
│   ├── examples/            # 示例
│   ├── middlewares/         # 中間件
│   ├── observability/       # 可觀測性
│   ├── perf_feedback/       # 性能反饋
│   ├── reception/           # 數據接收
│   ├── remediation/         # 修復建議
│   ├── reporting/           # 報告生成
│   ├── security/            # 安全
│   └── threat_intel/        # 威脅情報
├── alembic/                 # 數據庫遷移
├── api_gateway/             # API 網關
├── capability/              # 能力註冊
├── coordinators/            # 協調器
├── docs/                    # 文檔
├── scripts/                 # 腳本
└── tools/                   # 工具
```

### 建議模組化架構（按數據流設計）

```
services/integration/
├── ingestion/               # 📥 數據接收（Input）
│   ├── reception/           # 數據接收與存儲
│   └── capability/          # 能力註冊與管理
│
├── processing/              # 🧠 數據處理（Processing）
│   ├── analysis/            # 風險分析引擎
│   ├── attack_path_analyzer/ # 攻擊路徑分析
│   ├── threat_intel/        # 威脅情報關聯
│   ├── perf_feedback/       # 性能反饋分析
│   └── coordinators/        # 功能協調器
│
├── delivery/                # 📤 數據輸出（Output）
│   ├── reporting/           # 報告生成
│   ├── remediation/         # 修復建議
│   └── api_gateway/         # API 網關
│
└── infrastructure/          # 🏗️ 基礎設施（Infrastructure）
    ├── aiva_integration/    # 核心模組入口
    ├── alembic/             # 數據庫遷移
    ├── middlewares/         # 中間件
    ├── observability/       # 可觀測性
    ├── security/            # 安全機制
    ├── config_template/     # 配置模板
    ├── scripts/             # 腳本工具
    ├── tools/               # 開發工具
    ├── examples/            # 示例代碼
    └── docs/                # 文檔
```

**設計理由**：
- **遵循數據流**：接收 → 處理 → 輸出，三層清晰
- **processing 層集中**：所有分析引擎放在一起，便於協調
- **infrastructure 整合**：基礎設施和工具集中管理
- **簡化架構**：從 6 層簡化為 4 層，符合實際業務邏輯

### 遷移計畫

| 目前位置 | 新位置 | 模組 | 說明 |
|---------|--------|------|------|
| `reception/` | `1_data_ingestion/reception/` | 數據接收 | 接收層 |
| `capability/` | `2_capability_management/capability/` | 能力管理 | 註冊 |
| `coordinators/` | `2_capability_management/coordinators/` | 能力管理 | 協調 |
| `analysis/` | `3_analysis_engine/analysis/` | 分析引擎 | 分析 |
| `attack_path_analyzer/` | `3_analysis_engine/attack_path_analyzer/` | 分析引擎 | 攻擊路徑 |
| `threat_intel/` | `3_analysis_engine/threat_intel/` | 分析引擎 | 威脅情報 |
| `perf_feedback/` | `3_analysis_engine/perf_feedback/` | 分析引擎 | 性能 |
| `reporting/` | `4_output_delivery/reporting/` | 輸出交付 | 報告 |
| `remediation/` | `4_output_delivery/remediation/` | 輸出交付 | 修復 |
| `api_gateway/` | `5_infrastructure/api_gateway/` | 基礎設施 | 網關 |
| `middlewares/` | `5_infrastructure/middlewares/` | 基礎設施 | 中間件 |
| `observability/` | `5_infrastructure/observability/` | 基礎設施 | 監控 |
| `security/` | `5_infrastructure/security/` | 基礎設施 | 安全 |
| `alembic/` | `5_infrastructure/alembic/` | 基礎設施 | 數據庫 |
| `config_template/` | `6_support/config_template/` | 支援 | 配置 |
| `examples/` | `6_support/examples/` | 支援 | 示例 |
| `scripts/` | `6_support/scripts/` | 支援 | 腳本 |
| `tools/` | `6_support/tools/` | 支援 | 工具 |
| `docs/` | `6_support/docs/` | 支援 | 文檔 |

---

## 🔬 五、Scan 模組內部模組化設計（根據實際需求）

### 分析：Scan 模組的實際需求

Scan 模組負責**多引擎協調掃描**：
- 4 個不同語言的掃描引擎（Python, Go, Rust, TypeScript）
- 引擎協調和任務分發
- 目標管理和範圍控制

**核心問題**：如何協調多個異構引擎？

### 現狀分析
```
services/scan/
├── archived_docs/           # 歸檔文檔
├── coordinators/            # 協調器
│   ├── engines/             # 引擎適配器
│   ├── target_generators/   # 目標生成器
│   └── multi_engine_coordinator.py
├── engines/                 # 掃描引擎
│   ├── go_engine/           # Go 引擎
│   ├── python_engine/       # Python 引擎
│   ├── rust_engine/         # Rust 引擎
│   └── typescript_engine/   # TypeScript 引擎
├── image/                   # 文檔圖片
└── command_handler.py
```

### 建議模組化架構（三層架構）

```
services/scan/
├── engines/                 # 🚀 掃描引擎（Scan Engines）
│   ├── python_engine/       # Python 爬蟲與掃描引擎
│   ├── go_engine/           # Go 高性能掃描引擎
│   ├── rust_engine/         # Rust 安全掃描引擎
│   └── typescript_engine/   # TypeScript 動態掃描引擎
│
├── coordination/            # 🎯 引擎協調（Coordination）
│   ├── multi_engine_coordinator/ # 多引擎協調器
│   ├── adapters/            # 引擎適配器（go, python, rust, ts）
│   ├── target_generators/   # 目標生成器
│   └── scan_models/         # 掃描數據模型
│
└── infrastructure/          # 🏗️ 基礎設施（Infrastructure）
    ├── command_handler/     # 命令處理
    ├── archived_docs/       # 歸檔文檔
    └── image/               # 圖片與圖表
```

**設計理由**：
- **三層足夠**：引擎層 + 協調層 + 基礎設施
- **引擎獨立**：每個引擎保持完整目錄結構
- **協調層集中**：所有協調邏輯統一管理
- **避免過度設計**：Scan 模組職責單一，不需要複雜分層

### 遷移計畫

| 目前位置 | 新位置 | 模組 | 說明 |
|---------|--------|------|------|
| `engines/python_engine/` | `1_scan_engines/python_engine/` | 掃描引擎 | Python |
| `engines/go_engine/` | `1_scan_engines/go_engine/` | 掃描引擎 | Go |
| `engines/rust_engine/` | `1_scan_engines/rust_engine/` | 掃描引擎 | Rust |
| `engines/typescript_engine/` | `1_scan_engines/typescript_engine/` | 掃描引擎 | TypeScript |
| `coordinators/multi_engine_coordinator.py` | `2_engine_coordination/multi_engine_coordinator/` | 協調層 | 協調器 |
| `coordinators/engines/` | `2_engine_coordination/adapters/` | 協調層 | 適配器 |
| `coordinators/target_generators/` | `3_target_management/target_generators/` | 目標管理 | 生成器 |
| `command_handler.py` | `5_infrastructure/command_handler/` | 基礎設施 | 命令處理 |
| `archived_docs/` | `6_support/archived_docs/` | 支援 | 歸檔 |
| `image/` | `6_support/image/` | 支援 | 圖片 |

---

## 📊 六、架構對比與優勢

### 模組化前後對比

| 模組 | 模組化前 | 模組化後 | 層數 | 改善 |
|-----|---------|---------|-----|------|
| **Common** | 13 個平級目錄 | 4 層架構 | 4 | ✅ 職責分層清晰 |
| **Features** | 20+ 功能模組平鋪 | 3+1 分類 | 4 | ✅ 功能聚類明確 |
| **Integration** | 12 個功能目錄 | 4 層結構 | 4 | ✅ 數據流清晰 |
| **Scan** | 3 個主目錄 | 3 層架構 | 3 | ✅ 簡潔實用 |

### 設計優勢

1. **🎯 因地制宜**
   - 根據每個模組的實際需求設計
   - 不強求統一層數，實用為主

2. **📦 職責清晰**
   - 每層職責明確，邊界清楚
   - 便於獨立開發和測試

3. **🔗 易於理解**
   - 符合業務邏輯和數據流
   - 降低學習曲線

4. **🛠️ 易於維護**
   - 問題定位快速
   - 擴展時歸類明確

---

## 🚀 七、實施步驟

### Phase 1: 規劃階段（1-2 天）
1. ✅ 完成架構設計（本文檔）
2. ⏳ 審查並確認架構方案
3. ⏳ 制定詳細遷移計畫

### Phase 2: Common 模組重構（3-4 天）
1. 創建 4 層目錄結構（contracts, communication, infrastructure, utilities）
2. 移動檔案到對應層
3. 更新 import 路徑
4. 執行測試驗證

### Phase 3: Features 模組重構（4-5 天）
1. 創建 3+1 分類目錄（vulnerability_detection, attack_utilities, specialized_tools, shared）
2. 移動功能模組到對應分類
3. 更新模組註冊
4. 執行功能測試

### Phase 4: Integration 模組重構（3-4 天）
1. 創建 4 層結構（ingestion, processing, delivery, infrastructure）
2. 按數據流重組組件
3. 更新 API 路由
4. 驗證集成

### Phase 5: Scan 模組重構（2-3 天）
1. 創建 3 層架構（engines, coordination, infrastructure）
2. 重組引擎和協調器
3. 更新掃描流程
4. 執行掃描測試

### Phase 6: 文檔與驗證（2-3 天）
1. 更新架構文檔
2. 更新開發指南
3. 執行完整回歸測試
4. 性能基準測試

---

## ✅ 驗證清單

### 架構合理性
- [ ] 每個模組的層級數符合實際需求
- [ ] 子模組命名符合業務邏輯
- [ ] 職責劃分清晰明確

### 功能完整性
- [ ] 所有原有功能正常運作
- [ ] Import 路徑全部更新
- [ ] 配置檔案正確引用

### 測試覆蓋
- [ ] 單元測試通過
- [ ] 集成測試通過
- [ ] 端到端測試通過

### 文檔完整性
- [ ] README 更新
- [ ] API 文檔更新
- [ ] 架構圖更新

---

## 📝 注意事項

### 🔴 遷移風險

1. **Import 路徑變更**
   - 影響範圍：所有引用模組的文件
   - 緩解措施：使用自動化腳本批量更新

2. **配置檔案更新**
   - 影響範圍：capability_registry.yaml 等
   - 緩解措施：保留向後兼容性

3. **測試依賴**
   - 影響範圍：所有測試文件
   - 緩解措施：逐模組測試驗證

### 🟡 向後兼容

建議在根目錄保留軟連結或重定向：
```python
# services/aiva_common/schemas/ → services/aiva_common/contracts/schemas/
# 使用 __init__.py 重新導出保持向後兼容
```

---

## 🎯 預期成果

### 設計原則總結

1. **Common 模組**：4 層架構（契約、通信、基礎設施、工具）
2. **Features 模組**：3+1 分類（檢測、工具、專用、共享）
3. **Integration 模組**：4 層結構（接收、處理、輸出、基礎設施）
4. **Scan 模組**：3 層架構（引擎、協調、基礎設施）
5. **Core 模組**：維持現有 6 層結構（已經很好）

### 短期成果（1 個月內）
- ✅ 架構清晰度提升 50%
- ✅ 新功能開發效率提升 30%
- ✅ 問題定位時間減少 40%

### 長期成果（3-6 個月）
- ✅ 維護成本降低 30%
- ✅ 新人上手時間縮短 50%
- ✅ 架構擴展性提升 200%

---

## 📞 後續步驟

1. **審查本方案** - 確認架構設計是否符合需求
2. **制定時間表** - 安排具體實施時間
3. **開始 Phase 1** - 創建目錄結構和遷移腳本
4. **逐模組執行** - 按照 Phase 2-5 順序實施
5. **驗證與文檔** - 完成 Phase 6 驗證工作

---

**文檔版本**: v1.0  
**創建日期**: 2025-11-25  
**狀態**: ✅ 待審查
