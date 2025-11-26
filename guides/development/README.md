# Development Guides README

> **🎯 AIVA v2.0 架構**: 開發指南更新，專精數據合約驅動開發  
> **✅ 系統狀態**: 五大程式模組 + 六大核心服務完整就緒  
> **🔄 最後更新**: 2025年11月22日

## 🛠️ AIVA v2.0 Development Hub

開發指南強調 AIVA v2.0 數據合約驅動的開發模式和最佳實踐。所有開發流程應優先使用統一的 Schema 系統，確保最佳性能和跨語言兼容性，並專注於安全測試和動態檢測能力。

**開發哲學**: 使用 `aiva_common.schemas` 作為所有開發的基礎，確保高效性能和無縫的多語言整合。

## 📊 AIVA v2.0 架構狀態 (2025-11-22)

- **架構版本**: v2.0 (數據合約驅動架構) ✅
- **核心模組**: 五大程式模組 + 六大核心服務 ✅
- **多語言編譯**: Python ✅, Go ✅, TypeScript ✅
- **跨語言覆蓋**: Python ✅, Go ✅, TypeScript ✅
- **測試框架**: 完整測試系統 ✅

## 📖 Development Guides

### 🚀 **Quick Start & Environment**

- **[Development Quick Start Guide](./DEVELOPMENT_QUICK_START_GUIDE.md)** - 快速環境設置
  - *Schema 整合*: 包含 aiva_common 設置與驗證
  
- **[Development Tasks Guide](./DEVELOPMENT_TASKS_GUIDE.md)** - 日常開發工作流程
  - *Schema 整合*: 開發任務中的標準 Schema 使用模式

### 📦 **Dependencies & Configuration**

- **[Dependency Management Guide](./DEPENDENCY_MANAGEMENT_GUIDE.md)** - 深度依賴管理 + ML 依賴混合狀態
  - *Schema 整合*: Schema 相關的依賴解析
  
- **[Multi-Language Environment Standard](./MULTI_LANGUAGE_ENVIRONMENT_STANDARD.md)** - Python/TS/Go 統一配置
  - *Schema 整合*: 跨語言 Schema 驗證設置

### 🔐 **API & Services**

- **[API Verification Guide](./API_VERIFICATION_GUIDE.md)** - API 金鑰驗證配置
  - *Schema 整合*: 使用認證 Schema 進行 API 驗證
  
- **[AI Services User Guide](./AI_SERVICES_USER_GUIDE.md)** - AI 功能實務使用
  - *Schema 整合*: AI 服務 Schema 和訊息格式

### 📝 **Schema & Standards**

- **[Schema Import Guide](./SCHEMA_IMPORT_GUIDE.md)** - Schema 使用標準 ⭐ **必讀**
  - *Schema 整合*: 正確使用 Schema 的核心指南
  
- **[Language Conversion Guide](./LANGUAGE_CONVERSION_GUIDE.md)** - 跨語言程式碼轉換完整指南
  - *Schema 整合*: Schema 保留的語言轉換模式

### ⚡ **Performance & Optimization**

- **[Token Optimization Guide](./TOKEN_OPTIMIZATION_GUIDE.md)** - 開發效率優化
  - *Schema 整合*: Schema 使用優化以提升性能
  
- **[VS Code Configuration Optimization](./VSCODE_CONFIGURATION_OPTIMIZATION.md)** - IDE 性能優化細節
  - *Schema 整合*: IDE 設置以優化 Schema 開發
  
- **[Language Server Optimization Guide](./LANGUAGE_SERVER_OPTIMIZATION_GUIDE.md)** - IDE 性能優化配置
  - *Schema 整合*: 語言伺服器設置以支援 Schema 驗證

### 💾 **Data & Storage**

- **[Data Storage Guide](./DATA_STORAGE_GUIDE.md)** - 數據存儲架構
  - *Schema 整合*: 符合 Schema 的數據持久化模式
  
- **[Metrics Usage Guide](./METRICS_USAGE_GUIDE.md)** - 系統監控和統計
  - *Schema 整合*: Schema 健康指標與監控

### 🖥️ **UI & Extensions**

- **[UI Launch Guide](./UI_LAUNCH_GUIDE.md)** - 介面管理
  - *Schema 整合*: UI 元件使用標準 API Schema
  
- **[Extensions Install Guide](./EXTENSIONS_INSTALL_GUIDE.md)** - 開發工具配置
  - *Schema 整合*: 支援 Schema 開發的擴充功能

### 🔒 **Version Control**

- **[Git Push Guidelines](./GIT_PUSH_GUIDELINES.md)** - 安全代碼推送標準
  - *Schema 整合*: CI/CD 流程中的 Schema 驗證

## 📋 Schema-First Development Checklist

開發新功能時，請確保：

- [ ] 所有新模組使用 `aiva_common.schemas` 中的標準 Schema
- [ ] 性能符合或超越系統基準
- [ ] 透過 Schema 合規性驗證跨語言兼容性
- [ ] API 端點使用標準 APIResponse 和錯誤 Schema
- [ ] 數據模型繼承適當的基礎 Schema (SecurityContract, TaskContract 等)

## 🔗 Essential Resources

### 📚 **Master Documentation**
- **[AIVA v2.0 系統架構文檔](../../README.md)** - v2.0 數據合約驅動架構
- **[Services 架構總覽](../../services/README.md)** - 六大核心服務架構
- **[跨語言最佳實踐](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)** - 實現標準

### 🛠️ **Development Tools**
- **[測試框架手冊](../../testing/README.md)** - 測試策略與實踐
- **[工具集使用手冊](../../tools/README.md)** - 專業工具操作
- **[VS Code 擴充功能清單](../../_out/VSCODE_EXTENSIONS_INVENTORY.md)** - 88 個開發工具插件

### 🏗️ **Architecture Resources**
- **[架構指南](../architecture/README.md)** - v2.0 數據合約架構文檔
- **[跨語言兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)** - 多語言整合

## 🎯 Development Workflow Integration

### For New Features
1. **Design Phase**: 從 `services/aiva_common` 中的 Schema 定義開始
2. **Implementation**: 使用已建立的數據合約模式和基礎類
3. **Validation**: 執行測試並驗證合規性
4. **Testing**: 若需要，驗證跨語言兼容性

### For Legacy Code Migration
1. **Assessment**: 評估當前代碼狀態
2. **Planning**: 參考遷移指南進行系統化計劃
3. **Implementation**: 遵循 Schema 優先的重構模式
4. **Validation**: 確保性能維持或改善

### For Performance Optimization
1. **Baseline**: 建立當前性能指標
2. **Analysis**: 使用工具識別瓶頸
3. **Optimization**: 應用優化技術
4. **Validation**: 確認改善效果

## 📊 Success Metrics

- **模組完成率**: 持續提升功能模組完成度
- **性能維護**: 保持高效性能
- **跨語言支援**: Python/TypeScript/Go 無縫協作
- **開發者生產力**: 通過標準化 Schema 減少開發時間

---

**Maintained by**: AIVA Development Team  
**Last Updated**: 2025-11-22  
**Architecture Version**: v2.0  
**Schema Integration**: High priority for all development activities