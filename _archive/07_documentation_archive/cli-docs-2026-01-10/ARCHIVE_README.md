# CLI 文檔歸檔說明

**歸檔日期**: 2026-01-10  
**原因**: 數據不準確，基於錯誤的 Flow 數量假設

---

## 歸檔文件

### 1. CLI_USAGE_GUIDE_archived.md (如存在)
- **原位置**: `docs/01_user_documentation/CLI_USAGE_GUIDE.md`
- **問題**: 全文基於 **840 個 Flows** 的錯誤假設
- **實際情況**: latest_classification.json 僅包含 **276 個 Flows** (2026-01-10 驗證)
- **歸檔原因**: 幾乎全文內容都是基於錯誤數據，沒有獨立參考價值

**主要錯誤內容**:
- ✗ "列出所有 840 個 flows"
- ✗ "支援 840 個 flows"
- ✗ "按終點模組分佈（840 個 flows）"
- ✗ 模組分佈統計全部錯誤
- ✗ 所有示例和命令假設都基於 840

---

## 保留並標註的文件

### 1. CLI_DYNAMIC_FLOW_COMMANDS_IMPLEMENTATION.md
- **位置**: `docs/05_implementation_guides/`
- **處理方式**: 在關鍵章節標註 ⚠️ 警告
- **保留原因**: 技術實現方案本身有參考價值
- **標註章節**:
  - 文檔開頭
  - 核心原理章節（range(840) 示例）
  - 優勢章節

### 2. CLI_IMPLEMENTATION_SUMMARY.md  
- **位置**: `docs/05_implementation_guides/`
- **處理方式**: 在關鍵章節標註 ⚠️ 警告
- **保留原因**: 歷史實施記錄有參考價值
- **標註章節**:
  - 文檔開頭
  - 動態命令生成
  - 測試結果
  - 性能指標
  - 關鍵成就

---

## 準確的文檔

### ✅ AIVA_CLI_UNIFIED_GUIDE.md (推薦使用)
- **位置**: `docs/01_user_documentation/user-guides/`
- **狀態**: ✅ 已驗證準確 (2026-01-10)
- **Flow 數量**: **276** (正確)
- **內容**: 兩套 CLI 系統完整說明

---

## 數據準確性對比

| 文檔 | Flow 數量 | 狀態 |
|------|----------|------|
| CLI_USAGE_GUIDE.md (已歸檔) | 840 | ❌ 錯誤 |
| CLI_DYNAMIC_*_IMPLEMENTATION.md (已標註) | 840 | ⚠️ 已標註警告 |
| CLI_IMPLEMENTATION_SUMMARY.md (已標註) | 840 | ⚠️ 已標註警告 |
| **AIVA_CLI_UNIFIED_GUIDE.md** | **276** | ✅ **正確** |
| **實際 latest_classification.json** | **276** | ✅ **驗證來源** |

---

## 建議

1. **用戶使用**: 優先參考 `AIVA_CLI_UNIFIED_GUIDE.md`
2. **技術實現**: 參考已標註警告的 implementation guides
3. **數據查詢**: 以 `latest_classification.json` 為準

---

**處理者**: GitHub Copilot  
**驗證**: 2026-01-10
