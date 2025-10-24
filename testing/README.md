# 🧪 AIVA Testing - 按五大模組重組

## 🎯 新測試架構

### 🏠 **common/** - 通用測試
- **complete_system_check.py** - 完整系統檢查
- **improvements_check.py** - 改進驗證
- **README.md** - 測試框架指南

### 🧠 **core/** - 核心模組測試
- **ai_working_check.py** - AI工作狀態檢查
- **ai_system_connectivity_check.py** - AI系統連接測試
- **enhanced_real_ai_attack_system.py** - 增強AI攻擊系統測試

### 🔍 **scan/** - 掃描模組測試
- **comprehensive_test.py** - 全面掃描測試
- **juice_shop_real_attack_test.py** - Juice Shop攻擊測試
- **test_scan.ps1** - 掃描功能測試

### 🔗 **integration/** - 整合模組測試
- **aiva_full_worker_live_test.py** - 全功能工作者實時測試
- **aiva_module_status_checker.py** - 模組狀態檢查器
- **aiva_system_connectivity_sop_check.py** - 系統連接SOP檢查

### ⚙️ **features/** - 功能模組測試
- **real_attack_executor.py** - 真實攻擊執行器測試

## 🚀 執行測試

### 按模組執行
```bash
# 核心模組測試
cd testing/core
python ai_working_check.py

# 掃描模組測試
cd testing/scan  
python comprehensive_test.py

# 整合模組測試
cd testing/integration
python aiva_module_status_checker.py

# 功能模組測試
cd testing/features
python real_attack_executor.py
```

### 全系統測試
```bash
# 通用系統測試
cd testing/common
python complete_system_check.py
```

---

**重組完成**: 2025-10-24  
**測試覆蓋**: 五大模組全覆蓋