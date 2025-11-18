# AIVA Scan Node - Playwright 動態掃描引擎

使用 Node.js + Playwright 實現的高性能動態網頁掃描引擎。

## 功能

- ✅ 使用 Playwright 進行真實瀏覽器掃描
- ✅ 支援深度爬取
- ✅ 自動提取表單、輸入框、API 端點
- ✅ 通過 RabbitMQ 接收任務
- ✅ TypeScript 型別安全

## 安裝

```powershell
# 安裝依賴
npm install

# 安裝 Playwright 瀏覽器
npm run install:browsers
```

## 開發

```powershell
# 開發模式 (自動重載)
npm run dev
```

## 生產

```powershell
# 編譯
npm run build

# 啟動
npm start
```

## 環境變數

```env
RABBITMQ_URL=amqp://aiva:dev_password@localhost:5672/
LOG_LEVEL=info
```

## 任務格式

```json
{
  "scan_id": "scan_xxx",
  "target_url": "https://example.com",
  "max_depth": 2,
  "max_pages": 10,
  "enable_javascript": true
}
```

## 結果格式

```json
{
  "scan_id": "scan_xxx",
  "assets": [
    {
      "type": "form",
      "value": "form_0",
      "metadata": {
        "url": "https://example.com/login",
        "action": "/api/login",
        "method": "POST"
      }
    }
  ],
  "vulnerabilities": [],
  "metadata": {
    "pages_scanned": 5,
    "duration_seconds": 12.5
  }
}
```
