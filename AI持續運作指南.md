# AIVA AI 持續運作指南

## 🎯 快速開始

### 方式 1: 雙擊啟動 (最簡單)

直接雙擊 `啟動AI服務.bat` 即可啟動 API 服務模式。

### 方式 2: PowerShell 選單 (推薦)

```powershell
.\start_ai.ps1
```

會顯示互動式選單，支持 5 種運作模式。

### 方式 3: 命令行直接啟動

```powershell
# API 模式
python start_ai_service.py --mode api

# 監控模式
python start_ai_service.py --mode monitor --targets http://localhost:3000 --interval 1800

# 交互模式
python start_ai_service.py --mode interactive

# 守護進程模式
python start_ai_service.py --mode daemon
```

---

## 📋 運作模式詳解

### 1️⃣ API 服務模式 (推薦用於生產環境)

**功能**: REST API 持續監聽，提供完整的 Web API 介面

**啟動方式**:
```powershell
python start_ai_service.py --mode api --port 8000
```

**訪問地址**:
- 🌐 服務: `http://localhost:8000`
- 📚 API 文檔: `http://localhost:8000/docs`
- 🔐 健康檢查: `http://localhost:8000/health`

**預設帳號**:
- Admin: `admin` / `aiva-admin-2025`
- User: `user` / `aiva-user-2025`
- Viewer: `viewer` / `aiva-viewer-2025`

**適用場景**:
- ✅ 需要通過 HTTP API 調用 AIVA
- ✅ 與其他系統整合
- ✅ 團隊協作環境
- ✅ 持續運行的生產環境

**API 調用示例**:
```powershell
# 啟動掃描
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/scan/start" -Method POST -Headers @{"Authorization"="Bearer YOUR_TOKEN"} -Body (@{target="http://example.com"} | ConvertTo-Json) -ContentType "application/json"

# 查看掃描狀態
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/scan/status/scan_id" -Headers @{"Authorization"="Bearer YOUR_TOKEN"}
```

---

### 2️⃣ 後台監控模式 (推薦用於持續監控)

**功能**: 自動定期掃描指定目標，無需人工介入

**啟動方式**:
```powershell
python start_ai_service.py --mode monitor `
    --targets http://localhost:3000 http://example.com `
    --interval 1800
```

**參數說明**:
- `--targets`: 監控目標 URL 列表（可指定多個）
- `--interval`: 掃描間隔（秒），預設 3600 (1小時)

**適用場景**:
- ✅ 需要持續監控特定網站
- ✅ 定期安全掃描
- ✅ 自動化安全測試
- ✅ 夜間批量掃描

**日誌輸出**:
```
🔍 監控模式啟動
📋 監控目標: http://localhost:3000, http://example.com
⏱️  掃描間隔: 1800秒 (30.0分鐘)

🚀 開始第 1 次掃描 [15:30:00]
✅ 掃描完成: completed
📊 發現資產: 57
💤 等待 1800秒後進行下一次掃描...
```

---

### 3️⃣ 交互式模式 (推薦用於手動測試)

**功能**: 命令行交互界面，支持即時命令輸入

**啟動方式**:
```powershell
python start_ai_service.py --mode interactive
```

**可用命令**:
```
AIVA> scan http://example.com          # 標準掃描
AIVA> scan-fast http://localhost:3000  # 快速掃描
AIVA> status                            # 查看系統狀態
AIVA> engines                           # 查看引擎狀態
AIVA> help                              # 顯示幫助
AIVA> quit                              # 退出
```

**適用場景**:
- ✅ 開發測試
- ✅ 手動安全測試
- ✅ 學習 AIVA 功能
- ✅ 快速驗證目標

**使用示例**:
```
AIVA> scan http://localhost:3000
🚀 開始掃描: http://localhost:3000
✅ 掃描完成
📊 發現 57 個資產
前 5 個資產:
  [1] URL: http://localhost:3000/
  [2] URL: http://localhost:3000/api/users
  [3] Form: login_form
  [4] API: /rest/products/search
  [5] Header: X-Powered-By: Express
```

---

### 4️⃣ 守護進程模式 (推薦用於全功能運行)

**功能**: 同時運行 API 服務和後台監控

**啟動方式**:
```powershell
python start_ai_service.py --mode daemon `
    --port 8000 `
    --targets http://localhost:3000 `
    --interval 3600
```

**適用場景**:
- ✅ 需要同時提供 API 和自動監控
- ✅ 企業級持續運行
- ✅ 完整功能環境

---

### 5️⃣ Windows 服務模式 (推薦用於開機自啟)

**功能**: 註冊為 Windows 計劃任務，開機自動啟動

**啟動方式** (需要管理員權限):
```powershell
# 使用 PowerShell 選單
.\start_ai.ps1
# 選擇選項 5

# 或直接運行 (以管理員身份)
.\start_ai.ps1
```

**管理命令**:
```powershell
# 啟動服務
Start-ScheduledTask -TaskName "AIVA_AI_Service"

# 停止服務
Stop-ScheduledTask -TaskName "AIVA_AI_Service"

# 查看狀態
Get-ScheduledTask -TaskName "AIVA_AI_Service"

# 刪除服務
Unregister-ScheduledTask -TaskName "AIVA_AI_Service" -Confirm:$false
```

**適用場景**:
- ✅ 需要開機自動啟動
- ✅ 伺服器環境
- ✅ 完全無人值守運行

---

## 🔧 進階配置

### 修改監聽地址

```powershell
# 僅本機訪問
python start_ai_service.py --mode api --host 127.0.0.1 --port 8000

# 允許外部訪問
python start_ai_service.py --mode api --host 0.0.0.0 --port 8000
```

### 修改日誌級別

```powershell
python start_ai_service.py --mode api --log-level DEBUG
```

### 多目標監控

```powershell
python start_ai_service.py --mode monitor `
    --targets http://app1.com http://app2.com http://app3.com `
    --interval 1800
```

### 組合使用

```powershell
# API 在前台，監控在後台
Start-Process powershell -ArgumentList "-Command", "python start_ai_service.py --mode monitor --targets http://localhost:3000 --interval 1800"
python start_ai_service.py --mode api
```

---

## 📊 監控與日誌

### 日誌位置

- **主日誌**: `logs/aiva_service.log`
- **API 日誌**: `logs/api.log`
- **掃描日誌**: `logs/scan.log`

### 查看實時日誌

```powershell
# PowerShell
Get-Content logs\aiva_service.log -Wait -Tail 50

# CMD
tail -f logs\aiva_service.log
```

### 監控服務狀態

```powershell
# API 健康檢查
Invoke-RestMethod http://localhost:8000/health

# 查看系統統計
Invoke-RestMethod http://localhost:8000/api/v1/admin/stats
```

---

## 🛠️ 故障排除

### 問題 1: 端口已被占用

**錯誤**: `Address already in use`

**解決**:
```powershell
# 查找占用端口的進程
netstat -ano | findstr :8000

# 停止進程
taskkill /PID <PID> /F

# 或使用其他端口
python start_ai_service.py --mode api --port 8001
```

### 問題 2: Python 模組導入失敗

**錯誤**: `ModuleNotFoundError`

**解決**:
```powershell
# 確保環境變數正確
$env:PYTHONPATH = "C:\D\fold7\AIVA-git"

# 安裝依賴
pip install -r requirements.txt
```

### 問題 3: 服務無法啟動

**檢查步驟**:
```powershell
# 1. 檢查 Python 版本
python --version  # 需要 3.11+

# 2. 檢查依賴
pip list | Select-String "fastapi|uvicorn|httpx"

# 3. 測試基本功能
python -c "from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator; print('OK')"
```

### 問題 4: 引擎不可用

**檢查引擎狀態**:
```powershell
# 交互模式檢查
python start_ai_service.py --mode interactive
# 輸入: engines

# 或運行測試
python validate_scan_system.py
```

**修復方案**:
- **TypeScript 引擎**: 需要編譯
  ```powershell
  cd services\scan\engines\typescript_engine
  npm install
  npm run build
  ```

- **Rust 引擎**: 檢查二進制文件
  ```powershell
  dir services\scan\engines\rust_engine\target\release\*.exe
  ```

- **Go 引擎**: 檢查二進制文件
  ```powershell
  dir services\scan\engines\go_engine\*.exe
  ```

---

## 📈 性能優化

### 1. 增加並發數

修改 `config/settings.yaml`:
```yaml
scan:
  concurrency: 10  # 增加並發請求數
  timeout: 30      # 調整超時時間
```

### 2. 調整掃描策略

```python
# 在代碼中使用不同策略
result = await coordinator.execute_strategy_fast(...)      # 最快
result = await coordinator.execute_strategy_balanced(...)  # 平衡
result = await coordinator.execute_strategy_comprehensive(...) # 最全面
```

### 3. 資源限制

```powershell
# 限制 CPU 使用
Start-Process python -ArgumentList "start_ai_service.py --mode api" -Priority BelowNormal

# 限制記憶體（需要額外工具）
```

---

## 🔒 安全建議

1. **更改預設密碼**: 修改 `api/main.py` 中的預設帳號密碼
2. **啟用 HTTPS**: 配置反向代理（Nginx/Caddy）
3. **限制訪問**: 僅允許信任的 IP 訪問
4. **定期更新**: 保持 AIVA 和依賴更新

---

## 📚 相關文檔

- [API 文檔](http://localhost:8000/docs) - 啟動服務後訪問
- [架構設計](docs/ARCHITECTURE.md)
- [開發指南](docs/DEVELOPMENT.md)
- [測試指南](TESTING.md)

---

## 💡 最佳實踐

### 開發環境

```powershell
# 使用交互式模式或 API 模式 + 自動重載
python start_ai_service.py --mode interactive
```

### 測試環境

```powershell
# 使用監控模式定期測試
python start_ai_service.py --mode monitor --targets http://test.local --interval 600
```

### 生產環境

```powershell
# 使用守護進程模式 + Windows 服務
# 1. 註冊為計劃任務
.\start_ai.ps1  # 選項 5

# 2. 配置反向代理
# 3. 設置監控告警
```

---

## 🎯 快速參考

| 模式 | 命令 | 適用場景 |
|------|------|---------|
| API | `python start_ai_service.py --mode api` | 生產環境、團隊協作 |
| 監控 | `python start_ai_service.py --mode monitor` | 定期掃描、自動化 |
| 交互 | `python start_ai_service.py --mode interactive` | 開發測試、學習 |
| 守護 | `python start_ai_service.py --mode daemon` | 完整功能、持續運行 |
| 服務 | `.\start_ai.ps1` (選項5) | 開機自啟、無人值守 |

---

## 📞 需要幫助？

- 查看日誌: `logs/aiva_service.log`
- 運行診斷: `python diagnose.py`
- 檢查狀態: `python validate_scan_system.py`
