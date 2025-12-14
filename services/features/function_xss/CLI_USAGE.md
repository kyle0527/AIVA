# function_xss CLI 使用指南

## ✅ 修復完成

### 已修復的 Bug

1. **timeout 參數未定義錯誤** - 所有 `_execute_*_tool` 方法現在都正確接收 timeout 參數
2. **新增 CLI 入口** - 可直接測試而無需 MQ

---

## 🚀 快速開始

### 測試 Juice Shop (反射型 XSS)

```powershell
# 在專案根目錄執行
python -m services.features.function_xss `
    --url "http://localhost:3000/rest/products/search" `
    --param "q" `
    --type reflected `
    --timeout 30
```

### 測試 WebGoat (反射型 XSS)

```powershell
python -m services.features.function_xss `
    --url "http://localhost:8080/WebGoat/xss/simple" `
    --param "input" `
    --type reflected
```

### 測試 DOM XSS

```powershell
python -m services.features.function_xss `
    --url "http://localhost:3000/#/search" `
    --type dom
```

### 測試儲存型 XSS

```powershell
python -m services.features.function_xss `
    --url "http://localhost:3000/api/comments" `
    --param "comment" `
    --type stored `
    --method POST `
    --location body `
    --view-url "http://localhost:3000/comments"
```

---

## 📋 參數說明

| 參數 | 必填 | 預設值 | 說明 |
|------|------|--------|------|
| `--url` | ✅ | - | 目標 URL |
| `--type` | ❌ | reflected | 檢測類型 (reflected/dom/stored) |
| `--param` | ❌ | q | 測試參數名稱 |
| `--method` | ❌ | GET | HTTP 方法 (GET/POST) |
| `--location` | ❌ | query | 參數位置 (query/body/header) |
| `--timeout` | ❌ | 30 | 超時秒數 |
| `--view-url` | ❌ | - | 查看頁面 URL (僅 stored 類型) |

---

## 📊 輸出格式

```json
{
  "target": "http://localhost:3000/rest/products/search",
  "type": "reflected",
  "findings_count": 2,
  "vulnerable": true,
  "findings": [
    {
      "payload": "<script>alert(1)</script>",
      "status": 200,
      "vulnerable": true,
      "evidence": "...前 200 字的回應內容..."
    }
  ]
}
```

---

## 🔧 修復內容詳情

### 1. hackingtool_engine.py

**修復前**：
```python
async def _execute_go_tool(self, tool_config, target_url):
    # ...
    result = await self._run_command(command, timeout_seconds=timeout)
    # ❌ NameError: name 'timeout' is not defined
```

**修復後**：
```python
async def _execute_go_tool(self, tool_config, target_url, timeout: int = 300):
    # ...
    result = await self._run_command(command, timeout_seconds=timeout)
    # ✅ timeout 參數已正確定義
```

### 2. __main__.py (新增)

提供三種檢測模式的 CLI 接口：
- ✅ Reflected XSS
- ✅ DOM XSS
- ✅ Stored XSS

---

## 🎯 測試範例

### Juice Shop 完整測試

```powershell
# 1. 確保 Juice Shop 運行中
docker run -d -p 3000:3000 bkimminich/juice-shop

# 2. 測試搜尋框 (反射型)
python -m services.features.function_xss `
    --url "http://localhost:3000/rest/products/search" `
    --param "q" `
    --type reflected

# 預期結果: 找到多個有效的 XSS payloads

# 3. 測試留言板 (儲存型)
python -m services.features.function_xss `
    --url "http://localhost:3000/api/feedbacks" `
    --param "comment" `
    --type stored `
    --method POST `
    --location body `
    --view-url "http://localhost:3000/feedbacks"
```

---

## ⚠️ 注意事項

1. **只用於授權測試**：僅對自己擁有或獲得授權的系統進行測試
2. **避免過度請求**：timeout 設定合理，避免 DDoS
3. **Stored XSS 警告**：會實際寫入資料，測試前備份目標系統

---

## 🐛 已知限制

1. DOM XSS 檢測目前使用靜態分析，無 headless browser
2. 某些複雜的 payload 可能需要手動微調
3. WAF 可能導致部分檢測失敗

---

## 📝 與 function_bizlogic 的對比

| 特性 | function_bizlogic | function_xss |
|------|------------------|--------------|
| CLI 入口 | ✅ __main__.py | ✅ __main__.py |
| 無需 MQ | ✅ | ✅ |
| 直接測試 | ✅ | ✅ |
| JSON 輸出 | ✅ | ✅ |
| Bug 修復 | ✅ worker.py 移除 | ✅ timeout 參數修復 |
