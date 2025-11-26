# AIVA 指南文件整理報告 (2025-11-25)

## 📋 執行摘要

本次整理工作將兩份生產環境相關文檔從 `docs/` 目錄移至 `guides/` 目錄,並進行了以下優化:
1. 添加完整目錄結構
2. 更名為「指南」
3. 建立與現有指南的連結
4. 移除重複的報告文件

---

## ✅ 完成項目

### 1. 新增生產環境指南

#### 系統安裝指南
**位置**: `guides/deployment/SYSTEM_INSTALLATION_GUIDE.md`  
**來源**: `docs/SYSTEM_INSTALLATION_REQUIREMENTS.md` (已移除)

**內容涵蓋**:
- 📦 完整目錄結構 (21 個章節)
- 🔧 核心運行時環境
  - Python 3.13+ (必需)
  - Node.js 20+ (必需,TypeScript Engine)
  - Go 1.21+ (重要,Feature 模組)
  - Rust 1.70+ (可選,SAST)
- 🗄️ 資料庫與中介軟體
  - PostgreSQL 15+ (必需,Integration 模組)
  - Redis 7+ (重要,快取)
  - RabbitMQ 3.12+ (可選,已部分棄用)
- 🔨 編譯工具與系統依賴
- 🚀 Windows & Linux 完整安裝流程
- ✅ 完整驗證清單

**用途**: 生產環境部署、新環境設置、系統遷移

---

#### 生產環境故障排除指南
**位置**: `guides/troubleshooting/PRODUCTION_TROUBLESHOOTING_GUIDE.md`  
**來源**: `docs/PRODUCTION_TROUBLESHOOTING.md` (已移除)

**內容涵蓋**:
- 📦 完整目錄結構 (15 個章節)
- 🚨 30 秒健康檢查命令
- 🔧 9 個常見問題與解決方案
  1. 資料庫連線失敗 ⭐⭐⭐ (高頻)
  2. Playwright 瀏覽器啟動失敗 ⭐⭐
  3. 記憶體不足 (OOM) ⭐⭐
  4. TypeScript Engine 無法啟動 ⭐⭐
  5. Go 模組執行失敗 ⭐
  6. 背景任務無聲停止 ⭐⭐ (隱蔽,P0-2 問題)
  7. SQLite 資料庫鎖定 ⭐ (P2-3 問題)
  8. Python 套件衝突 ⭐
  9. Redis 連線逾時 ⭐
- 🔍 日誌分析與錯誤模式
- 📊 效能監控與優化建議
- 📋 健康檢查清單 (每日/每週/每月)

**用途**: 系統已部署,準備實際運行時遇到的問題

---

### 2. 文檔關係釐清

#### 保留的現有指南
**位置**: `guides/deployment/INSTALLATION_GUIDE.md`  
**用途**: **開發環境** Python 全域環境安裝

**與新指南的區別**:
- **INSTALLATION_GUIDE.md** (舊,保留): 
  - 針對開發環境
  - Python 全域環境 + 可編輯安裝 (`pip install -e .`)
  - 包含 pyproject.toml、測試執行、IDE 配置
  
- **SYSTEM_INSTALLATION_GUIDE.md** (新): 
  - 針對生產環境
  - 完整技術堆疊 (Python/Node/Go/Rust/PostgreSQL/Redis)
  - 不涉及可編輯安裝,專注於生產部署

**結論**: 兩者用途不同,應該保留兩者

---

### 3. 移除的文件

#### 已移除
1. `docs/SYSTEM_INSTALLATION_REQUIREMENTS.md` → 移至 `guides/deployment/SYSTEM_INSTALLATION_GUIDE.md`
2. `docs/PRODUCTION_TROUBLESHOOTING.md` → 移至 `guides/troubleshooting/PRODUCTION_TROUBLESHOOTING_GUIDE.md`
3. `guides/deployment/INSTALLATION_REPORT.md` → 刪除 (安裝報告,非指南)

---

### 4. 更新的連結

#### DEPLOYMENT_CHECKLIST.md
**位置**: `docs/DEPLOYMENT_CHECKLIST.md`

**更新內容**:
```markdown
### 安裝與部署指南
- [系統安裝指南](../guides/deployment/SYSTEM_INSTALLATION_GUIDE.md) - 完整生產環境安裝
- [生產環境故障排除指南](../guides/troubleshooting/PRODUCTION_TROUBLESHOOTING_GUIDE.md) - 運行時問題解決
- [BUILD_GUIDE.md](../guides/deployment/BUILD_GUIDE.md) - 構建流程
- [DOCKER_KUBERNETES_GUIDE.md](../guides/deployment/DOCKER_KUBERNETES_GUIDE.md) - 容器化部署

### 開發環境指南
- [INSTALLATION_GUIDE.md](../guides/deployment/INSTALLATION_GUIDE.md) - Python 開發環境安裝
```

---

#### BUILD_GUIDE.md
**位置**: `guides/deployment/BUILD_GUIDE.md`

**更新內容**:
```markdown
### 安裝與部署
- 📖 [系統安裝指南](./SYSTEM_INSTALLATION_GUIDE.md) - 完整生產環境安裝
- 📖 [生產環境故障排除指南](../troubleshooting/PRODUCTION_TROUBLESHOOTING_GUIDE.md) - 運行時問題解決
- 📖 [Docker/K8s 指南](./DOCKER_KUBERNETES_GUIDE.md) - 容器化部署
- 📖 [部署檢查清單](../../docs/DEPLOYMENT_CHECKLIST.md) - 部署前檢查

### 開發環境
- 📖 [Python 開發環境安裝](./INSTALLATION_GUIDE.md) - 全域環境與可編輯安裝
```

---

#### INSTALLATION_GUIDE.md
**位置**: `guides/deployment/INSTALLATION_GUIDE.md`

**更新內容**:
```markdown
### 生產環境部署
- 📖 [系統安裝指南](./SYSTEM_INSTALLATION_GUIDE.md) - **生產環境完整安裝**
- 📖 [生產環境故障排除指南](../troubleshooting/PRODUCTION_TROUBLESHOOTING_GUIDE.md) - 運行時問題解決
- 📖 [部署檢查清單](../../docs/DEPLOYMENT_CHECKLIST.md) - 發布前修復項目

### 開發環境
- 📖 [當前文件](./INSTALLATION_GUIDE.md) - **開發環境** Python 全域環境安裝
```

---

## 🔍 實際程式碼驗證

### PostgreSQL 使用確認
**檢查位置**: `services/integration/aiva_integration/`

**驗證結果**:
- ✅ `reception/finding_repository.py` - FindingRecord 模型使用 SQLAlchemy + PostgreSQL
- ✅ `unified_data_manager.py` - UnifiedDataManager 支援 PostgreSQL 連接
- ✅ `alembic/env.py` - Alembic 配置使用 PostgreSQL 連線字串
- ✅ `app.py` - 實際硬編碼 `postgresql://postgres:postgres@localhost:5432/aiva_db`

**結論**: 系統安裝指南中 PostgreSQL 要求完全正確

---

### TypeScript Engine 依賴確認
**檢查位置**: `services/scan/engines/typescript_engine/`

**驗證結果**:
- ✅ `package.json` - `playwright@^1.41.0`
- ✅ `package.json` - `amqplib@^0.10.3`
- ✅ `package.json` - `pino@8.21.0`
- ✅ `package-lock.json` - playwright@1.56.1 (實際安裝版本)

**結論**: 系統安裝指南中 Node.js & Playwright 要求完全正確

---

### Go 模組確認
**檢查位置**: `services/features/function_*_go/`

**驗證結果**:
- ✅ `function_authn_go/cmd/worker/main.go` - package main 存在
- ✅ Go 模組需要編譯為 worker.exe (Windows) 或 worker (Linux)

**結論**: 系統安裝指南中 Go 編譯要求完全正確

---

## 📊 文檔架構

### 安裝指南層次

```
生產環境
├── SYSTEM_INSTALLATION_GUIDE.md ⭐ 新增
│   ├── Python 3.13+
│   ├── Node.js 20+
│   ├── Go 1.21+
│   ├── Rust 1.70+
│   ├── PostgreSQL 15+
│   ├── Redis 7+
│   └── 編譯工具
│
└── 故障排除
    └── PRODUCTION_TROUBLESHOOTING_GUIDE.md ⭐ 新增
        ├── 30 秒健康檢查
        ├── 9 個常見問題
        ├── 日誌分析
        └── 效能監控

開發環境
└── INSTALLATION_GUIDE.md (保留)
    ├── Python 全域環境
    ├── pip install -e .
    ├── pyproject.toml
    └── 測試執行
```

---

## 🎯 使用指引

### 新用戶

**開發者 (本機開發)**:
1. 閱讀 `guides/deployment/INSTALLATION_GUIDE.md`
2. 確認 Python 全域環境
3. 執行 `pip install -e .`

**部署團隊 (生產環境)**:
1. 閱讀 `guides/deployment/SYSTEM_INSTALLATION_GUIDE.md`
2. 安裝完整技術堆疊
3. 驗證所有依賴
4. 參考 `guides/troubleshooting/PRODUCTION_TROUBLESHOOTING_GUIDE.md` 解決問題

---

## 📝 後續建議

### 立即行動
- [ ] 在 README.md 中更新安裝指南連結
- [ ] 通知團隊成員新指南位置

### 未來優化
- [ ] 考慮將 INSTALLATION_GUIDE.md 移至 `guides/development/`
- [ ] 統一所有指南的目錄格式
- [ ] 建立指南索引頁面

---

## 🔗 相關文檔

- [DEPLOYMENT_CHECKLIST.md](../docs/DEPLOYMENT_CHECKLIST.md) - 發布前檢查清單
- [ARCHITECTURE_COMPLETE_DESIGN.md](../docs/ARCHITECTURE_COMPLETE_DESIGN.md) - 系統架構
- [BUILD_GUIDE.md](./deployment/BUILD_GUIDE.md) - 構建流程

---

**報告日期**: 2025-11-25  
**執行者**: GitHub Copilot  
**版本**: 1.0.0
