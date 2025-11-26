# AIVA Testing Framework

這是AIVA項目的統一測試目錄，按照五大模組架構組織所有測試相關文件和腳本。

---

## 📑 目錄索引

- [目錄結構](#目錄結構)
- [測試分類](#測試分類)
- [使用方式](#使用方式)
- [測試統計](#測試統計)
- [相關文檔](#相關文檔)

---

## 目錄結構

按照 AIVA 五大模組組織測試：

```
testing/
├── core/                # Core 模組測試 (6個)
│   ├── test_ai_components.py
│   ├── test_ai_analysis_system.py
│   ├── test_intelligent_logic.py
│   ├── ai_integration_test.py
│   ├── ai_security_test.py
│   └── ai_autonomous_testing_loop.py
│
├── scan/                # Scan 模組測試 (4個)
│   ├── test_python_engine_direct.py
│   ├── test_scan_integration.py
│   ├── test_verification.py
│   └── comprehensive_test.py
│
├── features/            # Features 模組測試
│   ├── test_exploit_functionality.py
│   └── detectors/       # 檢測器
│       └── test_detector.py
│
├── integration/         # Integration 模組測試 (5個)
│   ├── test_cross_language_validation.py
│   ├── test_api.py
│   ├── test_basic.py
│   ├── test_web_attack.py
│   └── comprehensive_integration_test_suite.py
│
├── common/              # Common 模組測試 (5個)
│   ├── test_vector_storage.py
│   ├── test_unified_storage_config.py
│   ├── test_schema_codegen_converters.py
│   ├── security_test.py
│   └── api_testing.py
│
├── performance/         # 性能測試 (2個)
│   ├── aiva_performance_benchmark_suite.py
│   └── full_validation_test.py
│
└── _archive/            # 已歸檔測試 (30+個)
    └── legacy/          # 老舊測試
        ├── integration/ # 舊整合測試 (23個)
        ├── scan/        # 舊掃描測試 (3個)
        ├── core/        # 舊核心測試 (1個)
        ├── features/    # 舊功能測試 (1個)
        └── performance/ # 舊性能測試 (2個)
```

---

## 測試分類

### Core 模組測試 (`testing/core/`)

測試 AI 核心功能、分析系統和智能邏輯：

| 測試文件 | 用途 |
|---------|------|
| `test_ai_components.py` | AI 組件測試 |
| `test_ai_analysis_system.py` | AI 分析系統測試 |
| `test_intelligent_logic.py` | 智能邏輯測試 |
| `ai_integration_test.py` | AI 整合測試 |
| `ai_security_test.py` | AI 安全測試 |
| `ai_autonomous_testing_loop.py` | AI 自主測試循環 |

> **注意**: `ai_system_connectivity_check` 已遷移至 `services/core/tools/system_connectivity_checker.py` 作為實用工具。

### Scan 模組測試 (`testing/scan/`)

測試掃描引擎和掃描整合功能：

| 測試文件 | 用途 |
|---------|------|
| `test_python_engine_direct.py` | Python 引擎直接測試 |
| `test_scan_integration.py` | 掃描整合測試 |
| `test_verification.py` | 驗證測試 |
| `comprehensive_test.py` | 綜合測試 |

### Features 模組測試 (`testing/features/`)

測試各種漏洞檢測功能：

- **主測試**: `test_exploit_functionality.py`
- **檢測器**: `detectors/` - 漏洞檢測測試

> **注意**: `testers/` (垂直權限提升、跨用戶測試器) 已遷移至 `services/features/common/testers/` 作為實用工具。

### Integration 模組測試 (`testing/integration/`)

測試跨模組整合和系統連接：

| 測試文件 | 用途 |
|---------|------|
| `test_cross_language_validation.py` | 跨語言驗證測試 |
| `test_api.py` | API 測試 |
| `test_basic.py` | 基礎測試 |
| `test_web_attack.py` | Web 攻擊測試 |
| `comprehensive_integration_test_suite.py` | 綜合整合測試 |

> **注意**: `aiva_system_connectivity_sop_check` 已遷移至 `services/integration/tools/sop_compliance_checker.py` 作為實用工具。

### Common 模組測試 (`testing/common/`)

測試通用基礎設施和工具：

| 測試文件 | 用途 |
|---------|------|
| `test_vector_storage.py` | 向量存儲測試 |
| `test_unified_storage_config.py` | 統一存儲配置測試 |
| `test_schema_codegen_converters.py` | Schema 代碼生成測試 |
| `security_test.py` | 安全測試 |
| `api_testing.py` | API 測試工具 |

> **注意**: `module_connectivity_tester` 已遷移至 `services/aiva_common/tools/module_connectivity_checker.py` 作為實用工具。

### Performance 測試 (`testing/performance/`)

測試系統性能和基準：

| 測試文件 | 用途 |
|---------|------|
| `aiva_performance_benchmark_suite.py` | 性能基準測試套件 |
| `full_validation_test.py` | 完整驗證測試 |

---

## 使用方式

### 運行單個模組的測試

```bash
# Core 模組測試
cd testing/core
python test_ai_components.py

# Scan 模組測試
cd testing/scan
python test_scan_integration.py

# Features 模組測試
cd testing/features
python test_exploit_functionality.py
```

### 運行整合測試

```bash
cd testing/integration
python comprehensive_integration_test_suite.py
```

### 運行性能測試

```bash
cd testing/performance
python aiva_performance_benchmark_suite.py
```

### 使用日常測試工具

對於日常開發，建議使用根目錄的測試工具：

```bash
# 快速驗證
python quick_test.py

# 系統診斷
python diagnose.py

# 完整測試套件
python aiva_test.py full
```

---

## 測試統計

### 活躍測試

```
Core 模組:        6個測試
Scan 模組:        4個測試
Features 模組:    2個測試 (含子目錄)
Integration 模組: 5個測試
Common 模組:      5個測試
Performance 測試: 2個測試
────────────────────────────────
總計活躍:        24個測試

已遷移至 services/: 5個實用工具
```

### 歸檔測試

```
舊整合測試:      23個
舊掃描測試:      3個
舊核心測試:      1個
舊功能測試:      1個
舊性能測試:      2個
────────────────────────────────
總計歸檔:        30個測試
```

**歸檔原因**: 老舊過時、實驗性質、一次性驗證、基礎設施已改變

詳見: `_archive/README.md`

---

## 相關文檔

### 測試文檔
- [測試工具使用指南](../TESTING.md) - 日常測試工具說明
- [整合測試套件](../tests/integration/README.md) - 高價值整合測試
- [測試腳本整合總結](../TESTING_CONSOLIDATION.md) - 測試整合歷史

### 遷移文檔
- [實用工具遷移文檔](../UTILITY_TOOLS_MIGRATION.md) - 已遷移工具詳情和使用指南
- [工具遷移總結](../TOOLS_MIGRATION_SUMMARY.md) - 工具遷移總結報告

### 歸檔文檔
- [歸檔測試說明](./_archive/README.md) - 已歸檔測試詳情
- [Testing & Scripts 重組計劃](../TESTING_SCRIPTS_REORGANIZATION.md) - 重組計劃書

### 開發文檔
- [Scripts 總覽](../scripts/README.md) - 工具腳本目錄
- [五大模組架構](../README.md) - AIVA 架構說明

---

## 整合歷史

- **2025-11-16**: 原始 testing/ 目錄建立結構化測試框架
- **2025-11-22**: 
  - 按五大模組重組測試目錄
  - 歸檔 30+ 個老舊測試到 `_archive/legacy/`
  - 創建完整的測試文檔體系
  - 與根目錄測試工具整合
  - **將 5 個實用工具遷移至 services/ 對應模組**:
    - `ai_system_connectivity_check.py` → `services/core/tools/`
    - `module_connectivity_tester.py` → `services/aiva_common/tools/`
    - `aiva_system_connectivity_sop_check.py` → `services/integration/tools/`
    - `testers/` (2個) → `services/features/common/testers/`

---

最後更新: 2025-11-22
