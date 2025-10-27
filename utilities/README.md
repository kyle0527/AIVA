# 🔧 AIVA Utilities - 按五大模組分類

## 🔧 修復原則

**保留未使用函數原則**: 在程式碼修復過程中，若發現有定義但尚未使用的函數或方法，只要不影響程式正常運作，建議予以保留。這些函數可能是：
- 預留的 API 端點或介面
- 未來功能的基礎架構
- 測試或除錯用途的輔助函數
- 向下相容性考量的舊版介面

說不定未來會用到，保持程式碼的擴展性和靈活性。

## 🎯 工具分類架構

### 🏠 **common/** - 通用系統工具
- **monitoring/** - 系統監控工具
- **automation/** - 自動化腳本
- **diagnostics/** - 診斷工具

### 🧠 **core/** - 核心模組工具
- **ai_performance_monitor.py** - AI性能監控 (待開發)
- **decision_analytics.py** - 決策分析工具 (待開發)

### 🔍 **scan/** - 掃描模組工具
- **scan_result_analyzer.py** - 掃描結果分析 (待開發)
- **vulnerability_tracker.py** - 漏洞追蹤工具 (待開發)

### 🔗 **integration/** - 整合模組工具
- **api_monitor.py** - API監控工具 (待開發)
- **service_health_checker.py** - 服務健康檢查 (待開發)

### ⚙️ **features/** - 功能模組工具
- **attack_logger.py** - 攻擊日誌工具 (待開發)
- **exploit_tracker.py** - 漏洞利用追蹤 (待開發)

## 🔄 開發計劃

這些工具目錄已經建立，等待後續開發和填充具體的監控、自動化和診斷工具。

### 優先級
1. **common/diagnostics/** - 系統診斷工具
2. **common/monitoring/** - 系統監控工具  
3. **core/ai_performance_monitor** - AI性能監控
4. **scan/scan_result_analyzer** - 掃描結果分析
5. **integration/service_health_checker** - 服務健康檢查

---

**架構建立**: 2025-10-24  
**狀態**: 準備開發