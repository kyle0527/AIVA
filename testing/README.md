# AIVA Testing Framework

這是AIVA項目的統一測試目錄，整合了所有測試相關文件和腳本。

## 目錄結構

- **common/**: 通用測試工具和實用程式
- **core/**: 核心功能測試
- **features/**: 功能特性測試
- **integration/**: 集成測試
  - **legacy_tests/**: 從舊tests/目錄整合的測試文件
- **performance/**: 性能測試
- **scan/**: 掃描和分析測試

## 使用方式

各個子目錄包含針對不同測試類型的專門測試腳本和配置文件。
請參考各子目錄中的README文件獲取具體使用說明。

## 整合歷史

- 原始 testing/ 目錄: 結構化的測試框架
- 原始 tests/ 目錄: 集成測試文件 → 整合到 integration/legacy_tests/
