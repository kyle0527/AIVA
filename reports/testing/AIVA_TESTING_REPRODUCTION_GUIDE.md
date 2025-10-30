---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Report
---

# AIVA 測試重現快速指南

**快速目錄**: `C:\D\fold7\AIVA-git`

## 🚀 一鍵重現所有成功測試

### 📋 前置檢查 (30秒)
```bash
cd C:\D\fold7\AIVA-git
ls docker-compose.yml, ai_autonomous_testing_loop.py, comprehensive_pentest_runner.py
docker run -d -p 3000:3000 bkimminch/juice-shop  # 啟動靶場
```

### 🔧 環境設置 (1分鐘)
```bash
# PowerShell 環境變數設置
$env:AIVA_POSTGRES_HOST = "postgres"
$env:AIVA_POSTGRES_USER = "postgres"
$env:AIVA_POSTGRES_PASSWORD = "aiva123"
$env:AIVA_POSTGRES_DB = "aiva_db"
$env:AIVA_TARGET_URL = "http://localhost:3000"
```

### ⚡ 基礎服務啟動 (2分鐘)
```bash
docker-compose up -d postgres redis rabbitmq neo4j aiva-core
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
```

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