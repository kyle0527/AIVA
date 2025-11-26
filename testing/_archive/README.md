# 📦 Testing Archive - 已歸檔測試

## 目錄

- [說明](#說明)
- [歸檔日期](#歸檔日期)
- [歸檔內容](#歸檔內容)
- [歸檔統計](#歸檔統計)
- [查找測試](#查找測試)
- [使用警告](#使用警告)
- [新測試位置](#新測試位置)
- [恢復建議](#恢復建議)
- [相關文檔](#相關文檔)

---

## 說明

此目錄包含從 `testing/` 目錄歸檔的老舊、實驗性質或一次性驗證的測試腳本。

**歸檔原因**:
- 老舊過時的測試代碼
- 實驗性質的測試（如真實攻擊測試）
- 一次性驗證測試（如 P0 修復驗證）
- 基礎設施已改變的測試（如 MQ 測試）

**注意**: 這些測試可能無法在當前架構下正常運行。

---

## 歸檔日期
2025-11-22

---

## 歸檔內容

### legacy/integration/ (20+個)
**原始位置**: `testing/integration/legacy_tests/`

包含早期的整合測試，大部分已被新的測試框架替代：
- `test_real_ai_core.py` - 早期 AI 核心測試
- `test_attack_plan_mapper.py` - 攻擊計劃映射測試
- 等 20+ 個舊整合測試

**歸檔原因**: 測試框架已重構，這些測試使用舊的 API

### legacy/integration/ (單文件)

- **`aiva_module_status_checker.py`**
  - 模組狀態檢查器
  - **歸檔原因**: 功能已整合到新的診斷工具

- **`aiva_full_worker_live_test.py`**
  - 完整 Worker 實時測試
  - **歸檔原因**: Worker 架構已重構

- **`message_queue_test.py`**
  - 消息隊列測試
  - **歸檔原因**: MQ 基礎設施已改變

- **`p0_fixes_validation_test.py`**
  - P0 修復驗證測試
  - **歸檔原因**: 一次性驗證已完成

### legacy/scan/ (3個)

- **`juice_shop_real_attack_test.py`**
  - Juice Shop 真實攻擊測試
  - **歸檔原因**: 實驗性質，包含危險測試代碼

- **`live_pentest_runner.py`**
  - 實時滲透測試運行器
  - **歸檔原因**: 實驗性質

- **`fixed_pentest_runner.py`**
  - 修復版滲透測試運行器
  - **歸檔原因**: 實驗性質

### legacy/core/ (1個)

- **`enhanced_real_ai_attack_system.py`**
  - 增強型真實 AI 攻擊系統
  - **歸檔原因**: 實驗性質，包含真實攻擊代碼

### legacy/features/ (1個)

- **`real_attack_executor.py`**
  - 真實攻擊執行器
  - **歸檔原因**: 危險測試，僅用於研究

### legacy/performance/ (2個)

- **`comprehensive_schema_test.py`**
  - 綜合 Schema 測試
  - **歸檔原因**: 一次性驗證已完成

- **`comprehensive_pentest_runner.py`**
  - 綜合滲透測試運行器
  - **歸檔原因**: 實驗性質

---

## 📊 歸檔統計

```
整合測試:     23個 (legacy_tests/ + 3個單文件)
掃描測試:     3個 (實驗性滲透測試)
核心測試:     1個 (實驗性攻擊系統)
功能測試:     1個 (危險攻擊執行器)
性能測試:     2個 (一次性驗證)
────────────────────────────────────
總計歸檔:     30個測試
```

---

## 🔍 查找測試

### 如果需要找回某個功能的測試

1. **檢查現有測試**: 功能可能已有新的測試實現
   ```bash
   # 在當前 testing/ 目錄搜索
   grep -r "function_name" testing/
   ```

2. **查看歸檔測試**: 作為參考實現
   ```bash
   # 在歸檔目錄搜索
   grep -r "function_name" testing/_archive/legacy/
   ```

3. **適配新架構**: 如需使用歸檔測試
   - 更新導入路徑
   - 適配新的 API
   - 符合當前測試框架規範

---

## ⚠️ 使用警告

### 危險測試

以下測試包含真實攻擊代碼，**僅用於安全研究**：

- ❌ `enhanced_real_ai_attack_system.py`
- ❌ `real_attack_executor.py`
- ❌ `juice_shop_real_attack_test.py`
- ❌ `live_pentest_runner.py`

**警告**: 
- 不要在生產環境運行
- 僅在隔離的測試環境使用
- 需要明確授權才能執行

### 過時測試

以下測試使用舊的 API，需要適配：

- ⚠️  `legacy_tests/` 所有測試
- ⚠️  `aiva_full_worker_live_test.py`
- ⚠️  `message_queue_test.py`

---

## 📚 新測試位置

### 當前活躍的測試目錄

```
testing/
├── core/                # Core 模組測試 (7個)
├── scan/                # Scan 模組測試 (4個)
├── features/            # Features 模組測試
├── integration/         # Integration 模組測試 (6個)
├── common/              # Common 模組測試 (6個)
└── performance/         # 性能測試 (2個)
```

### 整合測試目錄

```
tests/integration/       # 高價值整合測試 (4個)
├── test_ai_command_scan.py
├── test_dual_loop_juice_shop.py
├── test_two_phase_scan.py
└── test_multi_language_analysis.py
```

### 日常測試工具

```
根目錄/
├── quick_test.py        # 快速驗證
├── diagnose.py          # 系統診斷
└── aiva_test.py         # 完整測試套件
```

---

## 🔄 恢復建議

如果確實需要恢復某個歸檔的測試：

1. **評估必要性**: 是否真的需要？現有測試是否已覆蓋？

2. **檢查依賴**: 測試依賴的模組是否還存在？

3. **更新代碼**: 
   - 更新導入路徑
   - 適配新 API
   - 符合當前編碼規範

4. **重新驗證**: 在測試環境充分測試

5. **文檔化**: 添加清晰的使用說明和警告

---

## 📞 相關文檔

- [測試工具使用指南](../../TESTING.md)
- [整合測試說明](../../tests/integration/README.md)
- [測試腳本整合總結](../../TESTING_CONSOLIDATION.md)
- [Testing & Scripts 重組計劃](../../TESTING_SCRIPTS_REORGANIZATION.md)

---

最後更新: 2025-11-22
