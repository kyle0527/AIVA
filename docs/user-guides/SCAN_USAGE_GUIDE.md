# 🔍 AIVA 掃描功能使用指南

> **適用對象**: 需要執行漏洞掃描的使用者  
> **前置要求**: 已閱讀 [快速入門](GETTING_STARTED.md)  
> **最後驗證**: 2025-11-29

---

## 📑 目錄

1. [掃描功能簡介](#掃描功能簡介)
2. [快速開始](#快速開始)
3. [掃描方式](#掃描方式)
4. [掃描結果](#掃描結果)
5. [進階配置](#進階配置)
6. [故障排除](#故障排除)

---

## 掃描功能簡介

AIVA 掃描模組是系統中最大的功能模組，包含:
- **286 個掃描能力** (佔總能力 36.6%)
- **兩階段掃描**: Phase 0 (快速) + Phase 1 (深度)
- **智能調度**: AI 自動選擇合適的掃描工具
- **多語言支持**: Python, Rust, TypeScript, Go

**掃描架構** (v2.1):
```
AI 命令中心
    ↓
掃描模組 (scan/)
    ├── Phase 0: 快速偵察 (5-10 分鐘)
    │   ├── 端口掃描
    │   ├── 服務識別
    │   └── 基礎指紋
    │
    └── Phase 1: 深度掃描 (10-30 分鐘)
        ├── 漏洞檢測
        ├── 安全配置檢查
        └── 攻擊面評估
```

---

## 快速開始

### 方式 1: CLI 命令 (最簡單) ⭐

```bash
# 自動掃描（AI 決定掃描策略）
python aiva_cli.py --attack "掃描 http://localhost:3000"

# 完整滲透測試
python aiva_cli.py --attack "完整測試 http://localhost:3000"

# 特定漏洞掃描
python aiva_cli.py --attack "檢查 http://localhost:3000 是否有 SQL 注入"
```

**優點**:
- ✅ 一行命令完成
- ✅ AI 自動選擇工具
- ✅ 無需編寫代碼
- ✅ 適合新手

### 方式 2: API 調用

#### 啟動 API 服務
```bash
# Windows
啟動AI服務.bat

# 或命令行
python scripts/startup/start_ai_service.py --mode api
```

#### 調用掃描 API
```bash
# 快速掃描 (Phase 0)
curl -X POST http://localhost:8000/api/v1/scan \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "target": "http://localhost:3000",
    "scan_type": "quick"
  }'

# 深度掃描 (Phase 0 + Phase 1)
curl -X POST http://localhost:8000/api/v1/scan \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "target": "http://localhost:3000",
    "scan_type": "deep"
  }'
```

### 方式 3: 監控模式（自動化）

```bash
# 每小時自動掃描
python scripts/startup/start_ai_service.py \
  --mode monitor \
  --targets http://localhost:3000 \
  --interval 3600

# 每30分鐘掃描多個目標
python scripts/startup/start_ai_service.py \
  --mode monitor \
  --targets http://localhost:3000 http://example.com \
  --interval 1800
```

---

## 掃描方式對比

| 方式 | 適用場景 | 難度 | 自動化程度 |
|------|---------|------|-----------|
| **CLI 命令** | 臨時測試、學習 | ⭐ 簡單 | 中 |
| **API 調用** | 系統整合、批量掃描 | ⭐⭐ 中等 | 高 |
| **監控模式** | 持續監控、定期掃描 | ⭐⭐ 中等 | 高 |

---

## 掃描流程詳解

### Phase 0: 快速偵察 (5-10 分鐘)

**目的**: 快速了解目標基本情況

**掃描內容**:
```
1. 端口掃描
   - 常見端口 (Top 1000)
   - 服務版本識別
   
2. Web 服務檢測
   - HTTP/HTTPS 識別
   - Web 伺服器類型
   - 框架指紋
   
3. 基礎安全檢查
   - SSL/TLS 配置
   - 常見安全標頭
   - robots.txt 分析
```

**CLI 使用**:
```bash
# 快速掃描
python aiva_cli.py --attack "快速掃描 http://localhost:3000"
```

**預期輸出**:
```
🔍 開始 Phase 0 快速偵察...

[1/3] 端口掃描
✓ 發現開放端口: 80, 443, 3000
✓ 識別服務: HTTP (nginx 1.18)

[2/3] Web 服務檢測
✓ Web 框架: Express.js
✓ 技術棧: Node.js, JavaScript

[3/3] 基礎安全檢查
⚠ 缺少 X-Frame-Options 標頭
⚠ 缺少 Content-Security-Policy
✓ HTTPS 已啟用

Phase 0 完成 (用時: 8 分鐘)
發現 2 個潛在問題
```

### Phase 1: 深度掃描 (10-30 分鐘)

**目的**: 深入分析漏洞和攻擊面

**掃描內容**:
```
1. 漏洞檢測
   - SQL 注入
   - XSS (跨站腳本)
   - CSRF
   - 文件上傳漏洞
   
2. 配置檢查
   - 敏感檔案暴露
   - 目錄遍歷
   - 弱密碼
   
3. 攻擊面評估
   - 輸入點分析
   - API 端點掃描
   - 參數篡改測試
```

**CLI 使用**:
```bash
# 深度掃描
python aiva_cli.py --attack "深度掃描 http://localhost:3000"

# 完整測試（Phase 0 + Phase 1）
python aiva_cli.py --attack "完整測試 http://localhost:3000"
```

**預期輸出**:
```
🔎 開始 Phase 1 深度掃描...

[1/3] 漏洞檢測
✓ SQL 注入測試: 發現 3 個注入點
✓ XSS 測試: 發現 2 個反射型 XSS
✗ CSRF 測試: 未發現
✓ 文件上傳: 發現繞過漏洞

[2/3] 配置檢查
✓ 發現敏感檔案: /admin, /.git
⚠ 目錄遍歷可能存在
✗ 弱密碼測試: 未發現

[3/3] 攻擊面評估
✓ 識別 15 個 API 端點
✓ 發現 8 個輸入點
✓ 參數篡改: 發現 4 個可利用點

Phase 1 完成 (用時: 25 分鐘)
發現 11 個漏洞 (高危: 5, 中危: 4, 低危: 2)
```

---

## 掃描結果

### 結果位置

```
掃描結果保存位置:
├── logs/scan_*.log          # 掃描日誌
├── reports/scan_*.json      # JSON 格式報告
├── reports/scan_*.html      # HTML 格式報告（如果生成）
└── data/scan_results/       # 原始數據
```

### 查看結果

#### 方式 1: 命令行輸出
```bash
# 掃描時直接顯示
python aiva_cli.py --attack "掃描 http://localhost:3000"
# 結果會即時顯示在終端
```

#### 方式 2: 查看日誌檔案
```powershell
# Windows
Get-Content logs/scan_*.log -Tail 100

# Linux/macOS
tail -100 logs/scan_*.log
```

#### 方式 3: API 查詢結果
```bash
# 查詢掃描狀態
curl http://localhost:8000/api/v1/scan/{scan_id}/status

# 獲取掃描報告
curl http://localhost:8000/api/v1/scan/{scan_id}/report
```

### 結果格式

**JSON 範例**:
```json
{
  "scan_id": "scan_20251129_001",
  "target": "http://localhost:3000",
  "start_time": "2025-11-29T10:00:00Z",
  "end_time": "2025-11-29T10:15:00Z",
  "duration": 900,
  "phases": {
    "phase0": {
      "status": "completed",
      "findings": 5
    },
    "phase1": {
      "status": "completed",
      "findings": 11
    }
  },
  "vulnerabilities": [
    {
      "id": "vuln_001",
      "type": "SQL Injection",
      "severity": "high",
      "url": "http://localhost:3000/api/search",
      "parameter": "q",
      "description": "SQL 注入漏洞",
      "recommendation": "使用參數化查詢"
    }
  ],
  "summary": {
    "total": 16,
    "high": 5,
    "medium": 7,
    "low": 4
  }
}
```

---

## 常見掃描場景

### 場景 1: Web 應用測試

```bash
# 步驟 1: 啟動靶場
docker run -d -p 3000:3000 bkimminich/juice-shop

# 步驟 2: 快速掃描（Phase 0）
python aiva_cli.py --attack "快速掃描 http://localhost:3000"

# 步驟 3: 如果發現可疑，執行深度掃描
python aiva_cli.py --attack "深度掃描 http://localhost:3000"
```

### 場景 2: API 安全測試

```bash
# 掃描 REST API
python aiva_cli.py --attack "測試 API http://localhost:8000/api"

# 檢查特定漏洞
python aiva_cli.py --attack "檢查 http://localhost:8000/api 是否有注入漏洞"
```

### 場景 3: 批量掃描

```bash
# 創建目標列表
# targets.txt:
# http://localhost:3000
# http://localhost:8080
# http://testsite.local

# 批量掃描腳本
for target in $(cat targets.txt); do
    echo "Scanning $target..."
    python aiva_cli.py --attack "掃描 $target"
    sleep 60  # 避免過於頻繁
done
```

### 場景 4: 定期監控

```bash
# 啟動監控模式（每天掃描一次）
python scripts/startup/start_ai_service.py \
  --mode monitor \
  --targets http://production-site.com \
  --interval 86400
```

---

## 進階配置

### 自定義掃描參數

如果需要更細緻的控制，可以使用 API:

```bash
curl -X POST http://localhost:8000/api/v1/scan \
  -H "Content-Type: application/json" \
  -d '{
    "target": "http://localhost:3000",
    "scan_type": "custom",
    "options": {
      "phase0": {
        "enabled": true,
        "port_range": "1-10000",
        "timeout": 600
      },
      "phase1": {
        "enabled": true,
        "modules": ["sql_injection", "xss", "file_upload"],
        "depth": "deep"
      }
    }
  }'
```

### 掃描策略

**快速策略** (5-10 分鐘):
- ✅ Phase 0 only
- ✅ Top 1000 端口
- ✅ 基礎檢查

**標準策略** (15-20 分鐘):
- ✅ Phase 0 + Phase 1
- ✅ 常見漏洞掃描
- ✅ 推薦日常使用

**深度策略** (30-60 分鐘):
- ✅ Phase 0 + Phase 1
- ✅ 所有端口 (1-65535)
- ✅ 完整漏洞庫
- ✅ 推薦重要系統

---

## 故障排除

### 問題 1: 掃描無法啟動

**症狀**: 命令執行後無反應

**解決方法**:
```bash
# 檢查目標是否可達
curl http://target-url

# 檢查 AI 服務狀態
python aiva_cli.py --test

# 查看錯誤日誌
cat logs/aiva_error.log
```

### 問題 2: 掃描中斷

**症狀**: 掃描進行到一半停止

**解決方法**:
```bash
# 查看掃描日誌
cat logs/scan_*.log | tail -50

# 檢查是否被防火牆阻擋
# 降低掃描速度或使用 API 自定義參數
```

### 問題 3: 結果不準確

**症狀**: 漏報或誤報

**解決方法**:
```bash
# 使用深度掃描
python aiva_cli.py --attack "深度掃描 http://target"

# 同步最新能力
python aiva_cli.py --sync

# 手動驗證結果
curl -X POST http://target/vulnerable-endpoint -d "payload"
```

---

## 安全注意事項

⚠️ **重要提醒**:

1. **授權測試**
   - ❌ 禁止未經授權掃描他人系統
   - ✅ 僅掃描自己擁有或已獲得授權的系統

2. **生產環境**
   - ❌ 避免在生產環境執行深度掃描
   - ✅ 在測試環境或維護時段進行

3. **掃描頻率**
   - ❌ 避免過於頻繁掃描（可能被視為攻擊）
   - ✅ 建議間隔至少 1 小時

4. **目標負載**
   - ❌ 不要掃描負載過高的系統
   - ✅ 監控掃描對目標系統的影響

5. **資料保護**
   - ✅ 保護掃描結果（可能包含敏感資訊）
   - ✅ 適當設置檔案權限

---

## 效能優化

### 加快掃描速度

```bash
# 方法 1: 僅執行 Phase 0
python aiva_cli.py --attack "快速掃描 http://target"

# 方法 2: 限制掃描範圍
# 使用 API 自定義端口範圍

# 方法 3: 並行掃描多個目標
# 使用腳本啟動多個掃描進程
```

### 降低資源占用

```bash
# 使用監控模式（後台運行）
python scripts/startup/start_ai_service.py \
  --mode monitor \
  --targets http://target \
  --interval 3600

# 調整日誌級別
python scripts/startup/start_ai_service.py \
  --mode api \
  --log-level WARNING
```

---

## 下一步

✅ 已熟悉掃描功能？繼續學習:

1. **API 整合** → [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md)
2. **結果分析** → [ANALYSIS_GUIDE.md](ANALYSIS_GUIDE.md)
3. **常見問題** → [FAQ.md](FAQ.md)

---

**最後更新**: 2025-11-29  
**驗證狀態**: ✅ 基於實際系統架構  
**掃描能力**: 286 個 (已驗證)
