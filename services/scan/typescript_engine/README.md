# 🟦 AIVA TypeScript Engine - 動態掃描引擎

> **版本**: v3.0 | **狀態**: ✅ Production Ready | **更新**: 2026-01-23

---

## 📋 概述

**TypeScript Engine** 是 AIVA 的動態掃描引擎，基於 **Playwright** 提供真實瀏覽器環境的安全檢測能力。

### 🎯 核心能力

- ✅ **DOM XSS 檢測** - Source-to-Sink 追蹤
- ✅ **SPA 路由分析** - 單頁應用安全檢測
- ✅ **WebSocket 安全** - 雙向通信漏洞檢測
- ✅ **PostMessage 檢測** - 跨窗口通信安全
- ✅ **DOM Clobbering** - DOM 污染攻擊檢測

---

## 🏗️ 架構設計

```
typescript_engine/
├── src/
│   ├── index.ts                              # CLI 任務處理器
│   ├── dom-security-analyzer.ts              # DOM 安全分析器
│   ├── spa-route-analyzer.ts                 # SPA 路由分析器
│   ├── websocket-security-analyzer.ts        # WebSocket 分析器
│   ├── services/
│   │   ├── scan-service.ts                   # 基礎掃描服務
│   │   ├── enhanced-dynamic-scan.service.ts  # 增強動態掃描
│   │   ├── enhanced-dom-xss-detector.service.ts  # DOM XSS 檢測
│   │   ├── network-interceptor.service.ts    # 網路攔截
│   │   ├── spa-state-crawler.service.ts      # SPA 狀態爬蟲
│   │   ├── interaction-simulator.service.ts  # 交互模擬器
│   │   └── enhanced-content-extractor.service.ts  # 內容提取
│   └── interfaces/                           # TypeScript 接口定義
├── types/
│   └── playwright.d.ts                       # Playwright 類型聲明
├── package.json                              # NPM 依賴
└── tsconfig.json                             # TypeScript 配置
```

---

## 🚀 快速開始

### 1️⃣ 安裝依賴

```bash
cd services/scan/typescript_engine
npm install
npm run install:browsers  # 安裝 Chromium
```

### 2️⃣ 編譯

```bash
npm run build
```

### 3️⃣ 運行

```bash
# 開發模式（熱重載）
npm run dev

# 生產模式
npm start
```

---

## 🔧 主要模組

### 1. DOM XSS 檢測器

**文件**: `src/services/enhanced-dom-xss-detector.service.ts`

**功能**:
- Source-to-Sink 數據流追蹤
- 動態注入點發現
- Payload 自動生成
- Sink 攔截與驗證

**檢測能力**:
- `innerHTML` / `outerHTML` / `document.write()`
- `eval()` / `Function()` / `setTimeout()`
- `location.href` / `location.replace()`

### 2. SPA 路由分析器

**文件**: `src/spa-route-analyzer.ts`

**功能**:
- 自動偵測 SPA 框架（React/Vue/Angular）
- 路由枚舉與爬取
- 客戶端權限繞過檢測
- 路由遍歷攻擊測試

### 3. WebSocket 安全分析器

**文件**: `src/websocket-security-analyzer.ts`

**功能**:
- WebSocket 連接監控
- 消息內容安全檢測
- 敏感數據洩露識別
- Origin 驗證檢查

### 4. 網路攔截器

**文件**: `src/services/network-interceptor.service.ts`

**功能**:
- HTTP/HTTPS 請求攔截
- AJAX 請求追蹤
- API 端點提取
- 請求模式分析

---

## 📊 集成方式

### CLI 輸入輸出

**標準輸入 (stdin)**: 接收 JSON 格式的任務
**標準輸出 (stdout)**: 輸出 JSON 格式的結果

**任務格式**:
```json
{
  "scan_id": "scan_001",
  "target_url": "https://example.com",
  "max_depth": 3,
  "max_pages": 100,
  "enable_javascript": true
}
```

**結果格式**:
```json
{
  "scan_id": "scan_001",
  "assets": [...],
  "dom_security_findings": [
    {
      "type": "DOM_XSS",
      "severity": "HIGH",
      "source": "location.hash",
      "sink": "innerHTML",
      "payload": "<img src=x onerror=alert(1)>"
    }
  ]
}
```

---

## 🔗 相關文檔

- [主掃描模組 README](../README.md)
- [Playwright 官方文檔](https://playwright.dev/)
- [TypeScript 官方文檔](https://www.typescriptlang.org/)

---

## 📝 許可證

MIT License - 詳見主專案 LICENSE 文件
