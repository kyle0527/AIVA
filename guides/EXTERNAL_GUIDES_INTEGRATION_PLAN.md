# 外部指南整合計劃

## 📊 整合方案總覽

### 🟢 建議移入 guides/ 的指南 (高價值，常用)

#### → guides/development/ 
1. **AIVA_TOKEN_OPTIMIZATION_GUIDE.md** (從 docs/guides/)
   - 理由：開發最佳實踐，使用頻率高
   - 重命名：TOKEN_OPTIMIZATION_GUIDE.md

2. **METRICS_USAGE_GUIDE.md** (從 docs/guides/)
   - 理由：開發必需的統計收集系統指南
   - 重命名：保持原名

3. **DATA_STORAGE_GUIDE.md** (從 docs/DEVELOPMENT/)
   - 理由：核心開發功能，需要集中管理

4. **UI_LAUNCH_GUIDE.md** (從 docs/DEVELOPMENT/)
   - 理由：常用的 UI 啟動指南

#### → guides/architecture/
1. **SCHEMA_GUIDE.md** (從 docs/DEVELOPMENT/)
   - 理由：架構相關，與現有 SCHEMA_GENERATION_GUIDE.md 互補

2. **SCHEMA_COMPLIANCE_GUIDE.md** (從 docs/DEVELOPMENT/)
   - 理由：Schema 規範，屬於架構設計

3. **CROSS_LANGUAGE_SCHEMA_GUIDE.md** (從 reports/schema/)
   - 理由：跨語言架構設計，與現有指南整合

#### → guides/deployment/
1. **ENVIRONMENT_CONFIG_GUIDE.md** (從 reports/documentation/)
   - 理由：部署配置相關

2. **DOCKER_KUBERNETES_GUIDE.md** (從 reports/documentation/)
   - 理由：與現有 DOCKER_GUIDE.md 整合

#### → guides/troubleshooting/
1. **AIVA_TESTING_REPRODUCTION_GUIDE.md** (從 reports/testing/)
   - 理由：測試和故障排除相關

#### → guides/development/
1. **EXTENSIONS_INSTALL_GUIDE.md** (從 tools/common/)
   - 理由：開發環境設置相關

### 🟡 建議整合到現有指南 (內容重疊)

1. **DEVELOPER_GUIDE.md** (從 reports/documentation/)
   - 整合到：guides/development/DEVELOPMENT_QUICK_START_GUIDE.md
   - 理由：內容重疊，避免重複

2. **AIVA_COMPREHENSIVE_GUIDE.md** (從 reports/documentation/)
   - 整合到：主 README.md 或創建總覽指南
   - 理由：綜合性文檔，避免與分類指南重複

### 🔴 建議保留原位置或刪除

1. **FILE_ORGANIZATION_MAINTENANCE_GUIDE.md**
   - 建議：保留在 reports/documentation/
   - 理由：屬於專案維護工具，不是日常開發指南

2. **PROJECT_STRUCTURE_GUIDE.md**
   - 建議：保留在 _out/
   - 理由：自動生成的輸出文件

3. **V3_QUICK_REFERENCE_GUIDE.md**
   - 建議：保留在 archive/
   - 理由：已歸檔的歷史版本

## 🔄 整合執行步驟

### Phase 1: 移動高價值指南 (優先)
1. 移動開發相關指南到 guides/development/
2. 移動架構相關指南到 guides/architecture/
3. 移動部署相關指南到 guides/deployment/
4. 移動故障排除指南到 guides/troubleshooting/

### Phase 2: 內容整合和去重
1. 將重複內容整合到現有指南
2. 更新指南間的交叉引用
3. 統一命名規範 (*_GUIDE.md)

### Phase 3: 更新索引和導航
1. 更新 guides/README.md 索引
2. 更新主 README.md 引用
3. 添加目錄到所有移動的指南

### Phase 4: 清理和驗證
1. 刪除原位置的文件
2. 更新所有文檔中的路徑引用
3. 驗證所有連結正常工作

## 📈 預期效果

- **集中化管理**: 所有常用指南在 guides/ 目錄
- **減少重複**: 整合重疊內容
- **提升可發現性**: 統一的導航和索引
- **標準化命名**: 一致的 *_GUIDE.md 格式
- **改善維護性**: 減少分散的文檔位置