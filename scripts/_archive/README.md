# 📦 Scripts Archive - 已歸檔腳本

## 目錄

- [說明](#說明)
- [歸檔日期](#歸檔日期)
- [歸檔內容](#歸檔內容)
- [歸檔統計](#歸檔統計)
- [查找功能](#查找功能)
- [當前活躍的腳本目錄](#當前活躍的腳本目錄)
- [使用替代方案](#使用替代方案)
- [恢復警告](#恢復警告)
- [恢復步驟](#恢復步驟)
- [相關文檔](#相關文檔)

---

## 說明

此目錄包含從 `scripts/` 目錄歸檔的老舊、過時或功能已整合的工具腳本。

**歸檔原因**:
- 功能已整合到其他工具
- 環境問題已解決，不再需要
- 一次性修復或優化腳本
- 與當前架構重複或衝突

**注意**: 這些腳本可能無法在當前環境下正常運行。

---

## 歸檔日期
2025-11-22

---

## 歸檔內容

### testing/ (3個)
**原始位置**: `scripts/testing/`

- **`test_ai_self_exploration.py`**
  - AI 自我探索測試
  - **歸檔原因**: 功能已整合到 `testing/` 目錄

- **`verify_aiva_system.py`**
  - AIVA 系統驗證
  - **歸檔原因**: 已有根目錄的 `diagnose.py` 和 `quick_test.py`

- **`v3_improvements_preview.py`**
  - v3 改進預覽
  - **歸檔原因**: 臨時預覽腳本，已過時

**替代方案**: 使用根目錄的測試工具
```bash
python quick_test.py      # 快速驗證
python diagnose.py        # 系統診斷
python aiva_test.py full  # 完整測試
```

### utilities/ (6個)
**原始位置**: `scripts/utilities/`

- **`aiva_package_validator.py`**
  - 套件驗證器
  - **歸檔原因**: 已移動到 `scripts/common/validation/`

- **`apply_performance_optimizations.py`**
  - 應用性能優化
  - **歸檔原因**: 一次性優化已完成

- **`fix_offline_dependencies.py`**
  - 修復離線依賴
  - **歸檔原因**: 環境問題已解決

- **`fix_environment_dependencies.py`**
  - 修復環境依賴
  - **歸檔原因**: 環境問題已解決

- **`launch_offline_mode.py`**
  - 啟動離線模式
  - **歸檔原因**: 離線模式已棄用

- **`restore_features_smart.py`**
  - 智能恢復功能
  - **歸檔原因**: 一次性修復已完成

### misc/ (7個)
**原始位置**: `scripts/misc/`

- **`sqli_scanner.py`**
  - SQL 注入掃描器
  - **歸檔原因**: 功能已整合到 `services/features/function_sqli/`

- **`xss_scanner.py`**
  - XSS 掃描器
  - **歸檔原因**: 功能已整合到 `services/features/function_xss/`

- **`system_explorer.py`**
  - 系統探索器
  - **歸檔原因**: 功能已整合到 `scripts/analysis/`

- **`final_validation.py`**
  - 最終驗證
  - **歸檔原因**: 一次性驗證已完成

- **`features_ai_cli.py`**
  - Features AI CLI
  - **歸檔原因**: CLI 已重構

- **`core_scan_integration_cli.py`**
  - Core Scan 整合 CLI
  - **歸檔原因**: CLI 已重構

- **`aiva_ai_coordinator.py`**
  - AIVA AI 協調器
  - **歸檔原因**: 功能已整合到 Core 模組

---

## 📊 歸檔統計

```
測試腳本:     3個 (功能已整合到 testing/)
工具腳本:     6個 (環境修復、一次性優化)
雜項腳本:     7個 (功能已整合到對應模組)
────────────────────────────────────
總計歸檔:     16個腳本
```

---

## 🔍 查找功能

### 如果需要找回某個功能

| 歸檔腳本 | 新位置/替代方案 |
|---------|----------------|
| `test_ai_self_exploration.py` | `testing/core/` 或 `quick_test.py` |
| `verify_aiva_system.py` | `diagnose.py`, `quick_test.py` |
| `aiva_package_validator.py` | `scripts/common/validation/` |
| `sqli_scanner.py` | `services/features/function_sqli/` |
| `xss_scanner.py` | `services/features/function_xss/` |
| `system_explorer.py` | `scripts/analysis/ultimate_organization_discovery_v2.py` |
| `aiva_ai_coordinator.py` | `services/core/aiva_core/` |

---

## 📚 當前活躍的腳本目錄

### 按服務分類

```
scripts/
├── core/                # Core 服務腳本 (10個)
│   ├── ai_analysis/     # AI 分析工具
│   └── update_self_awareness.py
├── scan/                # Scan 服務腳本 (2個)
│   ├── docker/
│   └── reporting/
├── features/            # Features 服務腳本 (1個)
├── integration/         # Integration 服務腳本 (4個)
└── common/              # Common 服務腳本 (6個)
    ├── launcher/
    ├── maintenance/
    ├── setup/
    └── validation/
```

### 工具和分析

```
scripts/
├── utilities/           # 工具腳本 (13個)
│   ├── health_check.py
│   ├── debug_fixer.py
│   ├── import_fixer.py
│   └── generate_*.py
└── analysis/            # 分析工具 (7個)
    ├── duplication_fix_tool.py
    ├── scanner_statistics.py
    └── ultimate_organization_discovery_v2.py
```

---

## 🎯 使用替代方案

### 系統驗證和測試

```bash
# 快速驗證 (替代 verify_aiva_system.py)
python quick_test.py

# 系統診斷 (替代 verify_aiva_system.py)
python diagnose.py

# 完整測試 (替代 test_ai_self_exploration.py)
python aiva_test.py full
```

### 掃描功能

```bash
# SQLi 掃描 (替代 sqli_scanner.py)
# 使用 services/features/function_sqli/ 模組

# XSS 掃描 (替代 xss_scanner.py)
# 使用 services/features/function_xss/ 模組
```

### 系統分析

```bash
# 系統探索 (替代 system_explorer.py)
cd scripts/analysis
python ultimate_organization_discovery_v2.py
```

### 套件驗證

```bash
# 套件驗證 (替代舊的 aiva_package_validator.py)
cd scripts/common/validation
python aiva_package_validator.py
```

---

## ⚠️ 恢復警告

### 不建議恢復的腳本

以下腳本的功能已完全整合或問題已解決，**不建議恢復**：

- ❌ `apply_performance_optimizations.py` - 一次性優化已完成
- ❌ `fix_offline_dependencies.py` - 環境問題已解決
- ❌ `fix_environment_dependencies.py` - 環境問題已解決
- ❌ `launch_offline_mode.py` - 離線模式已棄用
- ❌ `restore_features_smart.py` - 一次性修復已完成
- ❌ `final_validation.py` - 一次性驗證已完成

### 可能需要適配的腳本

如果確實需要恢復以下腳本的某些功能：

- ⚠️  `features_ai_cli.py` - CLI 已重構，需適配新 API
- ⚠️  `core_scan_integration_cli.py` - CLI 已重構，需適配新 API

**建議**: 先檢查當前 CLI 工具是否已包含所需功能。

---

## 🔄 恢復步驟

如果確實需要恢復某個歸檔的腳本：

1. **確認必要性**: 
   - 功能是否已有替代方案？
   - 是否真的需要這個特定腳本？

2. **檢查依賴**: 
   - 腳本依賴的模組是否還存在？
   - API 是否已改變？

3. **更新代碼**: 
   - 更新導入路徑
   - 適配新 API
   - 符合當前編碼規範

4. **測試驗證**: 
   - 在測試環境充分測試
   - 確保不影響現有功能

5. **文檔化**: 
   - 添加清晰的使用說明
   - 說明與其他工具的區別

---

## 📞 相關文檔

- [Scripts 總覽](../README.md)
- [測試工具使用指南](../../TESTING.md)
- [Testing & Scripts 重組計劃](../../TESTING_SCRIPTS_REORGANIZATION.md)
- [Scripts 重組詳細計劃](../REORGANIZATION_PLAN.md)

---

最後更新: 2025-11-22
