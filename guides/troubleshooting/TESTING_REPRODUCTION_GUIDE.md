---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Report
---

# AIVA 測試重現快速指南

## 📋 目錄

- [� 目錄](#目錄)
- [�🚀 一鍵重現所有成功測試](#一鍵重現所有成功測試)
  - [📋 前置檢查 (30秒)](#前置檢查-30秒)
  - [🔧 環境設置 (1分鐘)](#環境設置-1分鐘)
- [僅限特殊測試情境（Docker 環境）](#僅限特殊測試情境docker-環境)
  - [⚡ 基礎服務啟動 (2分鐘)](#基礎服務啟動-2分鐘)
  - [🎯 重現所有成功測試 (15分鐘)](#重現所有成功測試-15分鐘)
- [❌ 已知未修復問題](#已知未修復問題)
  - [動態組件管理失敗](#動態組件管理失敗)
  - [🛠️ 修復動態組件管理 (下次執行)](#修復動態組件管理-下次執行)
- [📊 當前測試狀態](#當前測試狀態)
- [🎉 重現成功標準](#重現成功標準)
- [🔗 相關資源](#相關資源)
  - [故障排除指南](#故障排除指南)
  - [修復指南](#修復指南)
  - [架構指南](#架構指南)
  - [開發指南](#開發指南)

**快速目錄**: `C:\D\fold7\AIVA-git`

## � 目錄

- [🚀 一鍵重現所有成功測試](#-一鍵重現所有成功測試)
- [🔧 環境設置](#-環境設置)
- [⚡ 基礎服務啟動](#-基礎服務啟動)
- [🧪 測試執行](#-測試執行)
- [📊 結果驗證](#-結果驗證)
- [🐛 故障排除](#-故障排除)
- [🔗 相關資源](#-相關資源)

## �🚀 一鍵重現所有成功測試

### 📋 前置檢查 (30秒)
```bash
cd C:\D\fold7\AIVA-git
ls docker-compose.yml, ai_autonomous_testing_loop.py, comprehensive_pentest_runner.py
docker run -d -p 3000:3000 bkimminch/juice-shop  # 啟動靶場
```

### 🔧 環境設置 (1分鐘)
```bash
# PowerShell 環境變數設置
# 測試環境配置

> ⚠️ **研發階段**: 以下環境變數設置**不需要**  
> 測試自動使用預設值: `postgresql://postgres:postgres@localhost:5432/aiva_db`

## 僅限特殊測試情境（Docker 環境）
```powershell
$env:POSTGRES_HOST = "postgres"
$env:POSTGRES_USER = "postgres"
$env:POSTGRES_PASSWORD = "aiva123"
$env:POSTGRES_DB = "aiva_db"
$env:AIVA_TARGET_URL = "http://localhost:3000"
```

### ⚡ 基礎服務啟動 (2分鐘)
```bash
# v2.0 架構: 已移除 RabbitMQ，使用直接數據合約通信
docker-compose up -d postgres redis neo4j aiva-core
Start-Sleep 60  # 等待服務健康
docker-compose ps  # 確認所有服務healthy
```

### 🎯 重現所有成功測試 (15分鐘)

#### 1. Scanner 組件測試 (3分鐘)
```bash
python sqli_test.py
# 預期: 發現Juice Shop SQL注入漏洞
```

#### 2. Testing 組件測試 (5分鐘)  
```bash
python autonomous_test.py
# 預期: 發現5個SQL注入點，功能測試通過
```

#### 3. Explorer 組件測試 (3分鐘)
```bash
python system_explorer.py  
# 預期: 發現17個目錄，22個端點
# ⚠️ 注意: system_explorer.py 是系統自我探索工具 (對內診斷)
# 不要與目標掃描工具 (對外偵測) 混淆
```

**📖 術語說明**: 
- `system_explorer.py` = **SystemSelfExplorer** = AIVA **自我診斷**工具 (對內)
- 用途: 掃描 AIVA 自身的模組、組件、端點
- 不是: 對外部目標的掃描工具

📚 **詳細說明**: 參見 [`../../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md`](../../AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md)

#### 4. 組件間通信測試 (1分鐘)
```bash
pip install pika  # 如果需要
python message_queue_test.py
# 預期: 6個隊列全部正常通信
```

#### 5. Validator 組件測試 (2分鐘)
```bash  
python ai_functionality_validator.py
# 預期: 5個腳本100%驗證通過
```

#### 6. 數據持久化測試 (1分鐘)
```bash
pip install psycopg2-binary redis  # 如果需要
python data_persistence_test.py
# 預期: PostgreSQL + Redis 全部測試通過
```

## ❌ 已知未修復問題

### 動態組件管理失敗
```bash
# 這個會失敗 - 需要修復Dockerfile.component
docker-compose --profile scanners up -d
# 錯誤: ModuleNotFoundError: No module named 'services.core'
```

### 🛠️ 修復動態組件管理 (下次執行)
```bash
# 1. 修復Dockerfile.component
notepad Dockerfile.component
# 替換 COPY 部分為: COPY services/ ./services/ 和 COPY *.py ./

# 2. 重新測試
docker-compose --profile testing build testing-autonomous
docker-compose --profile testing up -d testing-autonomous
docker logs aiva-testing-autonomous
```

## 📊 當前測試狀態
- ✅ **完成**: 9/13 個測試項目 (69.2%)
- ✅ **Layer 0 基礎服務**: 100% 成功
- ✅ **功能組件獨立測試**: 83.3% 成功  
- ❌ **動態組件管理**: 需要修復Docker配置

## 🎉 重現成功標準
執行上述6個測試都成功，即可確認AIVA系統核心功能完全正常！

---

## 🔗 相關資源

### 故障排除指南
- 📖 [導入問題解決](./IMPORT_ISSUES_RESOLUTION_GUIDE.md)
- 📖 [前向引用修復](./FORWARD_REFERENCE_REPAIR_GUIDE.md)
- 📖 [性能優化指南](./PERFORMANCE_OPTIMIZATION_GUIDE.md)
- 📖 [測試重現指南](./TESTING_REPRODUCTION_GUIDE.md)

### 修復指南
- �� [AI 修復指南](../repairs/AIVA_AI_REPAIR_GUIDE.md)

### 架構指南
- 📖 [兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)

### 開發指南
- 📖 [依賴管理指南](../development/DEPENDENCY_MANAGEMENT_GUIDE.md)

