# AIVA 現代化腳本使用指南

> 📅 更新日期: 2025-11-25  
> 🎯 版本: 全域環境版本  
> 📝 狀態: 已優化並現代化

## 🆕 新增的現代化腳本

本次更新將棄用的腳本改造為現代化、優化的版本，所有腳本都已適配全域 Python 環境。

### 📍 腳本位置

```
scripts/common/
├── setup/
│   └── setup_environment.ps1          # ⭐ 多語言環境設置
├── launcher/
│   ├── start_system.ps1               # ⭐ 系統啟動
│   └── stop_system.ps1                # ⭐ 系統停止
└── validation/
    ├── diagnose_system.ps1            # ⭐ 系統診斷
    └── health_check.ps1               # ⭐ 健康檢查
```

---

## 🚀 快速開始指南

### 第一次使用 AIVA？

```powershell
# 步驟 1: 設置環境
cd C:\D\fold7\AIVA-git
.\scripts\common\setup\setup_environment.ps1

# 步驟 2: 診斷系統
.\scripts\common\validation\diagnose_system.ps1

# 步驟 3: 啟動系統
.\scripts\common\launcher\start_system.ps1
```

### 日常使用工作流

```powershell
# 早上開始工作
.\scripts\common\launcher\start_system.ps1 -Mode standard

# 檢查服務狀態
.\scripts\common\validation\health_check.ps1

# 晚上結束工作
.\scripts\common\launcher\stop_system.ps1
```

---

## 📚 詳細使用說明

### 1️⃣ setup_environment.ps1 - 環境設置

**用途**: 一鍵設置所有開發環境（Python、Node.js、Go、Rust）

**基本用法**:
```powershell
# 設置所有環境
.\scripts\common\setup\setup_environment.ps1

# 只設置 Python
.\scripts\common\setup\setup_environment.ps1 -SkipNode -SkipGo -SkipRust
```

**功能特性**:
- ✅ 自動檢測已安裝的語言
- ✅ 升級 pip、setuptools、wheel
- ✅ 安裝 AIVA 套件（可編輯模式）
- ✅ 安裝 Playwright 瀏覽器
- ✅ 處理 Go 模組依賴
- ✅ 驗證安裝結果

**執行時間**: 約 3-5 分鐘（取決於網速）

---

### 2️⃣ diagnose_system.ps1 - 系統診斷

**用途**: 全面診斷系統問題並提供修復建議

**基本用法**:
```powershell
# 運行診斷
.\scripts\common\validation\diagnose_system.ps1

# 匯出診斷報告
.\scripts\common\validation\diagnose_system.ps1 -ExportReport
```

**檢查項目**:
- 🔍 Python、Node.js、Go、Rust 安裝狀態
- 🐳 Docker 服務狀態
- 📦 Python 套件完整性
- 📁 專案結構完整性
- ⚙️ 配置文件存在性
- 🔌 端口占用情況
- 💻 系統資源（CPU、記憶體、磁碟）

**輸出示例**:
```
🔧 發現的問題:
   • Python 版本過舊 (3.8.0)
   • 缺少 Python 套件: playwright, redis
   • Docker 服務未運行

💡 修復建議:
   • 請升級到 Python 3.9 或更高版本
   • 執行: pip install playwright redis
   • 請啟動 Docker Desktop
```

---

### 3️⃣ start_system.ps1 - 系統啟動

**用途**: 一鍵啟動 AIVA 系統及所有相關服務

**啟動模式**:

```powershell
# 最小模式（開發用）- 僅 RabbitMQ + API
.\scripts\common\launcher\start_system.ps1 -Mode minimal

# 標準模式（推薦）- RabbitMQ + DB + Core Services
.\scripts\common\launcher\start_system.ps1 -Mode standard

# 完整模式 - 所有服務
.\scripts\common\launcher\start_system.ps1 -Mode full
```

**高級選項**:
```powershell
# 跳過基礎設施（如已在運行）
.\scripts\common\launcher\start_system.ps1 -SkipInfrastructure

# 詳細輸出
.\scripts\common\launcher\start_system.ps1 -Verbose
```

**啟動內容**:

| 模式 | 基礎設施 | Python 服務 | 適用場景 |
|------|---------|------------|---------|
| minimal | RabbitMQ | API 服務 | 快速開發、API 測試 |
| standard | RabbitMQ + PostgreSQL + Redis | Core + Integration | 日常開發（推薦）|
| full | 所有基礎設施 | 所有服務 | 完整功能測試 |

**啟動時間**:
- minimal: ~20 秒
- standard: ~40 秒
- full: ~60 秒

---

### 4️⃣ health_check.ps1 - 健康檢查

**用途**: 檢查所有服務的運行狀態和健康程度

**基本用法**:
```powershell
# 單次檢查
.\scripts\common\validation\health_check.ps1

# 持續監控（每30秒）
.\scripts\common\validation\health_check.ps1 -Continuous

# 自定義間隔（每60秒）
.\scripts\common\validation\health_check.ps1 -Continuous -Interval 60
```

**檢查項目**:
- 🐳 Docker 容器狀態
- 🐍 Python 服務健康端點
- 🔧 RabbitMQ、PostgreSQL、Redis、Neo4j
- 💻 CPU、記憶體、磁碟使用率
- 📊 Python 進程信息

**輸出示例**:
```
🐍 Python 服務
-----------------------------------
   ✅ AIVA Core: HTTP 200 - 45ms
   ✅ Integration Service: HTTP 200 - 32ms

🔧 基礎設施服務
-----------------------------------
   ✅ RabbitMQ Management: HTTP 200 - 12ms
   ✅ PostgreSQL: TCP localhost:5432 - 5ms
   ✅ Redis: TCP localhost:6379 - 3ms
```

---

### 5️⃣ stop_system.ps1 - 系統停止

**用途**: 安全停止所有 AIVA 服務

**基本用法**:
```powershell
# 正常停止
.\scripts\common\launcher\stop_system.ps1

# 強制停止（立即終止）
.\scripts\common\launcher\stop_system.ps1 -Force

# 保留基礎設施（僅停止服務）
.\scripts\common\launcher\stop_system.ps1 -KeepInfrastructure
```

**停止流程**:
1. 優雅關閉 Python 服務
2. 停止 Docker 容器
3. 清理臨時文件
4. 驗證停止狀態

---

## 🎯 常見使用場景

### 場景 A: 完整開發環境設置

```powershell
# 1. 初始化環境
.\scripts\common\setup\setup_environment.ps1

# 2. 驗證安裝
.\scripts\common\validation\diagnose_system.ps1

# 3. 啟動系統
.\scripts\common\launcher\start_system.ps1 -Mode standard

# 4. 檢查健康
.\scripts\common\validation\health_check.ps1
```

### 場景 B: 快速 API 開發

```powershell
# 啟動最小環境
.\scripts\common\launcher\start_system.ps1 -Mode minimal

# 持續監控服務
.\scripts\common\validation\health_check.ps1 -Continuous -Interval 10
```

### 場景 C: 問題排查

```powershell
# 1. 停止所有服務
.\scripts\common\launcher\stop_system.ps1 -Force

# 2. 運行診斷
.\scripts\common\validation\diagnose_system.ps1 -ExportReport

# 3. 根據建議修復問題

# 4. 重新啟動
.\scripts\common\launcher\start_system.ps1
```

### 場景 D: 每日工作流程

```powershell
# 早上
.\scripts\common\launcher\start_system.ps1 -Mode standard

# 開發中...

# 午休
.\scripts\common\launcher\stop_system.ps1 -KeepInfrastructure

# 下午
.\scripts\common\launcher\start_system.ps1 -SkipInfrastructure

# 下班
.\scripts\common\launcher\stop_system.ps1
```

---

## 🔧 進階技巧

### 創建快捷啟動別名

在 PowerShell Profile 中添加：

```powershell
# 編輯 Profile
notepad $PROFILE

# 添加別名
function Start-AIVA { cd C:\D\fold7\AIVA-git; .\scripts\common\launcher\start_system.ps1 -Mode standard }
function Stop-AIVA { cd C:\D\fold7\AIVA-git; .\scripts\common\launcher\stop_system.ps1 }
function Check-AIVA { cd C:\D\fold7\AIVA-git; .\scripts\common\validation\health_check.ps1 }

Set-Alias aiva-start Start-AIVA
Set-Alias aiva-stop Stop-AIVA
Set-Alias aiva-check Check-AIVA
```

使用別名：
```powershell
aiva-start   # 啟動
aiva-check   # 檢查
aiva-stop    # 停止
```

---

## ❓ 常見問題

**Q: 為什麼不用虛擬環境？**  
A: 為避免環境切換造成的套件不一致問題，統一使用全域環境。詳見 [遷移報告](../../guides/GLOBAL_ENVIRONMENT_MIGRATION_2025-11-25.md)。

**Q: 可以在其他目錄執行腳本嗎？**  
A: 建議在專案根目錄執行，腳本會自動處理路徑。

**Q: 如何查看詳細日誌？**  
A: 服務啟動在新窗口，可直接查看。Docker 使用 `docker-compose logs -f`。

**Q: 健康檢查失敗怎麼辦？**  
A: 運行 `diagnose_system.ps1` 獲取詳細診斷和修復建議。

**Q: 可以同時運行多個模式嗎？**  
A: 不建議。端口會衝突。請先停止再啟動新模式。

---

## 📊 腳本對比

### 新版 vs 舊版

| 項目 | 舊版（deprecated） | 新版（現代化） |
|------|-------------------|---------------|
| 環境要求 | 需要虛擬環境 | 全域環境 |
| 錯誤處理 | 基本 | 完善 |
| 用戶反饋 | 簡單 | 彩色、詳細 |
| 啟動模式 | 單一 | 三種模式 |
| 健康檢查 | 基礎 | 全面、實時 |
| 診斷功能 | 有限 | 詳盡、可匯出 |
| 參數選項 | 少 | 豐富 |

---

## 🎓 學習資源

- [AIVA 開發指南](../../guides/README.md)
- [系統架構文檔](../../docs/README.md)
- [全域環境遷移報告](../../guides/GLOBAL_ENVIRONMENT_MIGRATION_2025-11-25.md)
- [原 Common Scripts README](./README.md)

---

## 🔄 版本歷史

**v2.0 (2025-11-25)**
- ✅ 移除虛擬環境依賴
- ✅ 新增多模式啟動
- ✅ 增強健康檢查
- ✅ 改進診斷功能
- ✅ 優化用戶體驗

**v1.0 (2025-10-13)**
- 初始版本（已棄用）

---

## 📞 支持

如遇問題，請：
1. 運行診斷工具獲取報告
2. 查看日誌輸出
3. 提交 Issue 附上診斷報告

---

**祝開發愉快！** 🚀
