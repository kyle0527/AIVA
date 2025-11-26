# 舊測試腳本存檔

## 說明

此目錄包含已被整合或廢棄的舊測試腳本。

**重要**: 4 個高價值的整合測試已移至 `tests/integration/`：
- ✅ `test_ai_command_scan.py` → AI 命令中心整合測試
- ✅ `test_dual_loop_juice_shop.py` → 雙閉環系統測試
- ✅ `test_two_phase_scan.py` → 兩階段掃描編排測試
- ✅ `test_multi_language_analysis.py` → 多語言能力分析測試

所有其他功能已遷移到項目根目錄的三個主要測試工具：

- **`aiva_test.py`** - 完整測試套件
- **`quick_test.py`** - 快速驗證工具
- **`diagnose.py`** - 系統診斷工具

## 歸檔原因

這些腳本在項目早期創建，用於測試特定功能或排查問題。隨著項目成熟，測試腳本過多導致維護困難，因此進行了整合。

## 歸檔日期

2025-11-22

## 歸檔內容

### 已移至整合測試目錄 (4 個)
- `test_ai_command_scan.py` → `tests/integration/` (AI 命令中心測試)
- `test_dual_loop_juice_shop.py` → `tests/integration/` (雙閉環系統測試)
- `test_two_phase_scan.py` → `tests/integration/` (兩階段掃描測試)
- `test_multi_language_analysis.py` → `tests/integration/` (多語言分析測試)

### 引擎測試 (1 個)
- `test_engine_availability.py` → 整合到 `diagnose.py engines`

### HTTP 測試
- `diagnose_http.py` → 整合到 `diagnose.py http`
- `debug_http_client.py` → 整合到 `aiva_test.py http`

### 掃描測試
- `test_all_targets.py` → 整合到 `aiva_test.py all-targets`
- `test_dynamic_scan.py` → 整合到 `aiva_test.py dynamic`
- `test_dynamic_localhost.py` → 整合到 `aiva_test.py dynamic`
- `test_static_parser.py` → 功能已驗證，不再需要
- `test_static_parser_debug.py` → 功能已驗證，不再需要
- `test_targets_detailed.py` → 整合到 `aiva_test.py all-targets`

### 整合測試
- `test_multi_engine_scan.py` → 整合到 `aiva_test.py scan` (功能重複)
- `test_command_handler_quick.py` → 整合到 `quick_test.py` (功能重複)

### 特殊用途測試
- `test_dual_loop_juice_shop.py` → `tests/integration/` ✅ **已保留**
- `test_example_links.py` → 功能已整合
- `test_scan_orchestrator_http.py` → 功能已整合

## 如何使用新工具

```bash
# 快速驗證系統
python quick_test.py

# 系統診斷
python diagnose.py

# 詳細測試
python aiva_test.py full
```

更多信息請參考根目錄的 `TESTING.md`。

## 保留原因

這些腳本被保留而非刪除，原因包括：
1. 保留歷史記錄
2. 可能需要參考特定測試邏輯
3. 作為文檔記錄項目演進

如果這些腳本不再需要，可以安全刪除整個目錄。
