# TypeScript Engine - node_modules 依賴完整指南

## 📑 目錄

- [📋 目錄](#-目錄)
  - [總覽與統計](#總覽與統計)
  - [核心依賴](#核心依賴)
  - [開發工具](#開發工具)
  - [完整套件列表](#完整套件列表)
  - [實用資訊](#實用資訊)
- [📊 總體概覽](#-總體概覽)
  - [基本統計](#基本統計)
- [📈 詳細統計分析](#-詳細統計分析)
  - [檔案類型分布（估計）](#檔案類型分布估計)
  - [大小分布](#大小分布)
- [🎯 核心依賴套件](#-核心依賴套件)
  - [生產環境依賴 (4 個)](#生產環境依賴-4-個)
  - [開發環境依賴 (9 個)](#開發環境依賴-9-個)
- [📚 重要間接依賴](#-重要間接依賴)
  - [瀏覽器相關](#瀏覽器相關)
  - [網路和流相關](#網路和流相關)
  - [解析器和 AST](#解析器和-ast)
  - [工具庫](#工具庫)
- [🔧 可執行命令](#-可執行命令)
  - [編譯和構建](#編譯和構建)
  - [代碼質量](#代碼質量)
  - [測試](#測試)
  - [開發工具](#開發工具)
  - [日誌和調試](#日誌和調試)
  - [其他工具](#其他工具)
- [📦 依賴樹結構](#-依賴樹結構)
  - [核心依賴關係](#核心依賴關係)
- [💾 存儲空間分析](#-存儲空間分析)
  - [大小分布](#大小分布)
  - [生產環境優化](#生產環境優化)
- [🚀 使用場景](#-使用場景)
  - [場景 1: 開發環境](#場景-1-開發環境)
  - [場景 2: 生產環境](#場景-2-生產環境)
  - [場景 3: CI/CD 環境](#場景-3-cicd-環境)
- [🔍 依賴管理](#-依賴管理)
  - [查看依賴樹](#查看依賴樹)
  - [更新依賴](#更新依賴)
  - [審計安全性](#審計安全性)
- [⚠️ 常見問題](#-常見問題)
  - [Q1: 為什麼 node_modules 這麼大？](#q1-為什麼-node_modules-這麼大)
  - [Q2: Playwright 瀏覽器在哪裡？](#q2-playwright-瀏覽器在哪裡)
  - [Q3: 可以刪除 node_modules 嗎？](#q3-可以刪除-node_modules-嗎)
  - [Q4: 為什麼有這麼多 @types 套件？](#q4-為什麼有這麼多-types-套件)
  - [Q5: 可以不安裝 devDependencies 嗎？](#q5-可以不安裝-devdependencies-嗎)
- [📚 進階資源](#-進階資源)
  - [官方文檔](#官方文檔)
  - [套件搜索](#套件搜索)
- [✅ 檢查清單](#-檢查清單)

---

**文件版本**: 2.0.0  
**更新日期**: 2025-11-20  
**適用版本**: aiva-scan-node@1.0.0

> **本文檔涵蓋**：213 個套件、5,905 個檔案、100.07 MB 存儲空間的完整分析

---

## 📋 目錄

### 總覽與統計
- [📊 總體概覽](#總體概覽) - 基本統計數據
- [📈 詳細統計分析](#詳細統計分析) - 檔案數、大小分布

### 核心依賴
- [🎯 核心依賴套件（4 個生產環境）](#核心依賴套件)
  - [playwright - 瀏覽器自動化](#1-playwright1561-關鍵)
  - [amqplib - RabbitMQ 客戶端](#2-amqplib0109-關鍵)
  - [pino - 日誌記錄](#3-pino8210-重要)
  - [pino-pretty - 日誌美化](#4-pino-pretty1122-輔助)

### 開發工具
- [🛠️ 開發依賴套件（9 個）](#開發依賴套件)
  - [TypeScript 編譯器](#typescript)
  - [ESLint 代碼檢查](#eslint)
  - [Prettier 格式化](#prettier)
  - [Vitest 測試框架](#vitest)
  - [其他開發工具](#其他開發工具)

### 完整套件列表
- [📦 完整套件清單（213 個）](#完整套件清單)
  - [Scoped 套件（13 個）](#scoped-套件)
  - [一般套件（200 個）](#一般套件)

### 實用資訊
- [💻 可執行命令（68 個）](#可執行命令)
- [📂 存儲空間分析](#存儲空間分析)
- [🔗 依賴樹結構](#依賴樹結構)
- [💡 使用場景](#使用場景)
- [❓ FAQ 常見問題](#faq-常見問題)

---

## 📊 總體概覽

### 基本統計

| 指標 | 數值 | 說明 |
|------|------|------|
| **總套件數** | 213 個 | 包含所有直接和間接依賴 |
| **總檔案數** | **5,905 個** | 包含所有原始碼、類型定義、編譯產物、文檔、授權文件 |
| **直接依賴** | 13 個 | package.json 中定義的套件 |
| **Scoped 套件** | 13 個 | 以 @ 開頭的組織套件 |
| **一般套件** | 200 個 | 標準 npm 套件 |
| **總大小** | **100.07 MB** | 不含 Playwright 瀏覽器（瀏覽器額外 ~300MB） |
| **可執行命令** | 68 個 | 位於 node_modules/.bin |

---

## 📈 詳細統計分析

### 檔案類型分布（估計）

| 類型 | 數量 | 佔比 | 說明 |
|------|------|------|------|
| **JavaScript 文件** | ~3,500 個 | 59% | .js, .mjs, .cjs 編譯產物 |
| **類型定義文件** | ~1,200 個 | 20% | .d.ts TypeScript 類型 |
| **文檔文件** | ~800 個 | 14% | README, LICENSE, CHANGELOG |
| **配置文件** | ~300 個 | 5% | package.json, tsconfig.json |
| **其他文件** | ~105 個 | 2% | .map, .json, .txt |

### 大小分布

| 範圍 | 套件數 | 代表套件 |
|------|--------|----------|
| **> 10 MB** | 6 個 | typescript (17.69 MB), playwright (40+ MB) |
| **1-10 MB** | 12 個 | vitest, tsx, prettier, eslint |
| **100 KB - 1 MB** | 35 個 | @types/node, pino, amqplib |
| **< 100 KB** | 160 個 | 大部分工具函數庫 |

---

## 🎯 核心依賴套件

### 生產環境依賴 (4 個)

#### 1. **playwright@1.56.1** ⭐⭐⭐ 關鍵
```json
{
  "名稱": "playwright",
  "版本": "1.56.1",
  "用途": "瀏覽器自動化核心引擎",
  "大小": "~40 MB (僅庫文件)",
  "依賴": ["playwright-core"]
}
```

**功能**:
- ✅ 控制 Chromium/Firefox/WebKit 瀏覽器
- ✅ 執行 JavaScript 在真實瀏覽器環境
- ✅ 網路請求攔截 (Request/Response)
- ✅ WebSocket 監聽
- ✅ 頁面截圖和 PDF 生成
- ✅ 自動等待和重試機制

**使用範例**:
```typescript
import { chromium } from 'playwright-core';

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto('https://example.com');
```

**注意事項**:
- 瀏覽器二進制文件不包含在 node_modules
- 需額外運行 `npm run install:browsers` 下載 Chromium (~170 MB)
- 瀏覽器位置: `%USERPROFILE%\AppData\Local\ms-playwright\`

---

#### 2. **amqplib@0.10.9** ⭐⭐⭐ 關鍵
```json
{
  "名稱": "amqplib",
  "版本": "0.10.9",
  "用途": "RabbitMQ 客戶端庫",
  "協議": "AMQP 0-9-1",
  "依賴": ["buffer-more-ints", "readable-stream"]
}
```

**功能**:
- ✅ 連接 RabbitMQ 服務器
- ✅ 發送/接收訊息 (Publish/Subscribe)
- ✅ 隊列管理 (assertQueue, consume)
- ✅ 訊息確認 (ack/nack)
- ✅ 持久化訊息

**使用範例**:
```typescript
import * as amqp from 'amqplib';

const conn = await amqp.connect('amqp://guest:guest@localhost:5672/');
const channel = await conn.createChannel();
await channel.assertQueue('task.queue', { durable: true });
await channel.sendToQueue('task.queue', Buffer.from(JSON.stringify(data)));
```

**相關概念**:
- **Channel**: 通信通道，一個連接可有多個通道
- **Queue**: 訊息隊列，先進先出
- **Exchange**: 訊息路由器（本項目未使用）

---

#### 3. **pino@8.21.0** ⭐⭐ 重要
```json
{
  "名稱": "pino",
  "版本": "8.21.0",
  "用途": "高性能 JSON 日誌庫",
  "特點": "異步寫入、結構化日誌",
  "依賴": ["pino-abstract-transport", "sonic-boom", "thread-stream"]
}
```

**功能**:
- ✅ 結構化 JSON 日誌輸出
- ✅ 多級別日誌 (trace/debug/info/warn/error/fatal)
- ✅ 異步寫入，不阻塞主線程
- ✅ 子日誌器 (child logger) 支持
- ✅ 性能極高 (~5x 快於 Winston)

**使用範例**:
```typescript
import pino from 'pino';

const logger = pino({
  level: 'info',
  transport: { target: 'pino-pretty' }
});

logger.info({ url: 'http://example.com' }, '開始掃描');
```

**輸出格式**:
```json
{"level":30,"time":1700000000000,"pid":12345,"hostname":"AIVA","url":"http://example.com","msg":"開始掃描"}
```

---

#### 4. **pino-pretty@10.3.1** ⭐ 開發輔助
```json
{
  "名稱": "pino-pretty",
  "版本": "10.3.1",
  "用途": "美化 Pino 日誌輸出",
  "場景": "開發環境、調試"
}
```

**功能**:
- ✅ 彩色終端輸出
- ✅ 人類可讀格式
- ✅ 時間戳格式化
- ✅ 錯誤堆棧美化

**效果對比**:
```bash
# 原始 JSON (生產環境)
{"level":30,"time":1700000000000,"msg":"掃描完成"}

# 美化輸出 (開發環境)
[2025-11-20 14:30:00] INFO: 掃描完成
```

---

### 開發環境依賴 (9 個)

#### 5. **typescript@5.9.3** ⭐⭐⭐ 必需
```json
{
  "名稱": "typescript",
  "版本": "5.9.3",
  "用途": "TypeScript 編譯器",
  "命令": ["tsc", "tsserver"],
  "大小": "~20 MB"
}
```

**功能**:
- ✅ TypeScript → JavaScript 編譯
- ✅ 類型檢查和推斷
- ✅ 生成 .d.ts 類型定義文件
- ✅ Source Map 生成

**使用**:
```bash
npm run build  # 使用 tsc 編譯 src/ → dist/
```

---

#### 6. **@types/node@20.19.23** ⭐⭐ 必需
```json
{
  "名稱": "@types/node",
  "版本": "20.19.23",
  "用途": "Node.js API 類型定義",
  "範圍": "fs, path, process, Buffer 等"
}
```

**作用**:
- 提供 Node.js 內建模組的 TypeScript 類型
- 啟用 IDE 自動完成和類型檢查

---

#### 7. **@types/amqplib@0.10.8** ⭐ 必需
```json
{
  "名稱": "@types/amqplib",
  "版本": "0.10.8",
  "用途": "amqplib TypeScript 類型定義"
}
```

---

#### 8. **eslint@8.57.1** ⭐⭐ 代碼質量
```json
{
  "名稱": "eslint",
  "版本": "8.57.1",
  "用途": "JavaScript/TypeScript 代碼檢查",
  "依賴套件": "~30 個"
}
```

**功能**:
- ✅ 語法錯誤檢測
- ✅ 代碼風格檢查
- ✅ 最佳實踐建議
- ✅ 自動修復

**使用**:
```bash
npm run lint  # 執行 ESLint 檢查
```

---

#### 9. **@typescript-eslint/eslint-plugin@6.21.0** ⭐⭐
#### 10. **@typescript-eslint/parser@6.21.0** ⭐⭐
```json
{
  "用途": "ESLint 的 TypeScript 支持",
  "功能": ["解析 TypeScript 語法", "TypeScript 專用規則"]
}
```

---

#### 11. **prettier@3.6.2** ⭐ 代碼格式化
```json
{
  "名稱": "prettier",
  "版本": "3.6.2",
  "用途": "代碼自動格式化"
}
```

**功能**:
- ✅ 統一代碼風格
- ✅ 自動縮排和換行
- ✅ 支持多種語言

**使用**:
```bash
npm run format  # 格式化所有 TypeScript 文件
```

---

#### 12. **tsx@4.20.6** ⭐⭐ 開發工具
```json
{
  "名稱": "tsx",
  "版本": "4.20.6",
  "用途": "TypeScript 即時執行和熱重載"
}
```

**功能**:
- ✅ 無需編譯直接運行 .ts 文件
- ✅ 文件變更自動重啟 (watch 模式)
- ✅ 支持 ESM 和 CommonJS

**使用**:
```bash
npm run dev  # 使用 tsx watch 開發模式
```

---

#### 13. **vitest@1.6.1** ⭐ 測試框架
```json
{
  "名稱": "vitest",
  "版本": "1.6.1",
  "用途": "單元測試框架",
  "特點": "快速、Vite 驅動"
}
```

**功能**:
- ✅ 單元測試
- ✅ 測試覆蓋率報告
- ✅ Mock 和 Spy
- ✅ Watch 模式

**使用**:
```bash
npm test  # 運行測試
```

---

## 📚 重要間接依賴

### 瀏覽器相關

#### **playwright-core**
- Playwright 的核心實現
- 不包含瀏覽器下載邏輯

---

### 網路和流相關

#### **readable-stream**
- Node.js Stream API 實現
- 用於 amqplib 和其他庫

#### **buffer**
- Buffer polyfill for browsers

#### **sonic-boom**
- 超高速異步寫入流
- Pino 日誌的底層實現

---

### 解析器和 AST

#### **acorn@8.x**
- JavaScript 解析器
- ESLint 使用

#### **espree@9.x**
- ESLint 官方解析器
- 基於 acorn

---

### 工具庫

#### **glob / fast-glob / micromatch**
- 文件模式匹配
- 用於 ESLint 查找文件

#### **chalk / picocolors / colorette**
- 終端顏色輸出
- 不同庫使用不同實現

#### **debug@4.x**
- 調試日誌工具
- 許多庫的依賴

---

## 🔧 可執行命令

### 編譯和構建

```bash
tsc           # TypeScript 編譯器
tsserver      # TypeScript Language Server (IDE 用)
esbuild       # 快速打包工具
rollup        # 模塊打包器
vite          # 前端構建工具
```

### 代碼質量

```bash
eslint        # 代碼檢查
prettier      # 代碼格式化
```

### 測試

```bash
vitest        # 單元測試
vite-node     # Vite Node 運行器
```

### 開發工具

```bash
tsx           # TypeScript 執行器
playwright    # 瀏覽器自動化 CLI
node-which    # 查找可執行文件
```

### 日誌和調試

```bash
pino          # JSON 日誌
pino-pretty   # 日誌美化
```

### 其他工具

```bash
nanoid        # 生成唯一 ID
semver        # 語義化版本管理
rimraf        # 跨平台刪除文件
js-yaml       # YAML 解析器
acorn         # JavaScript 解析器
```

---

## 📦 依賴樹結構

### 核心依賴關係

```
aiva-scan-node@1.0.0
├── playwright@1.56.1
│   └── playwright-core@1.56.1
│
├── amqplib@0.10.9
│   ├── buffer-more-ints@1.0.0
│   └── readable-stream@4.5.2
│       ├── buffer@6.0.3
│       ├── events@3.3.0
│       └── process@0.11.10
│
├── pino@8.21.0
│   ├── pino-abstract-transport@1.2.0
│   ├── sonic-boom@4.3.0
│   ├── thread-stream@3.1.0
│   └── pino-std-serializers@7.0.0
│
├── pino-pretty@10.3.1
│   ├── colorette@2.0.20
│   ├── dateformat@5.0.2
│   └── help-me@5.0.0
│
├── typescript@5.9.3
│
├── eslint@8.57.1
│   ├── espree@9.6.1
│   │   └── acorn@8.14.0
│   ├── @eslint/js@8.57.1
│   └── [~25 個其他依賴]
│
├── @typescript-eslint/eslint-plugin@6.21.0
│   ├── @typescript-eslint/utils@6.21.0
│   └── ts-api-utils@1.4.3
│
└── vitest@1.6.1
    ├── vite@5.4.11
    ├── chai@4.5.0
    ├── tinybench@2.9.0
    └── tinypool@0.8.4
```

---

## 💾 存儲空間分析

### 大小分布

| 類別 | 套件數 | 估計大小 | 百分比 |
|------|--------|---------|--------|
| **Playwright** | 2 | ~40 MB | 40% |
| **TypeScript** | 1 | ~20 MB | 20% |
| **ESLint 生態** | ~35 | ~15 MB | 15% |
| **Vitest/測試** | ~15 | ~10 MB | 10% |
| **Pino/日誌** | ~10 | ~5 MB | 5% |
| **其他工具** | ~150 | ~10 MB | 10% |
| **總計** | **213** | **~100 MB** | **100%** |

### 生產環境優化

如果只部署生產環境（不含開發工具）:

```bash
npm install --production
```

**節省空間**:
- 移除 devDependencies (~40 MB)
- 保留 4 個核心套件: playwright, amqplib, pino, pino-pretty
- 最終大小: **~60 MB**

---

## 🚀 使用場景

### 場景 1: 開發環境

**需要的套件**:
- ✅ 所有 213 個套件
- ✅ Playwright 瀏覽器 (~170 MB)
- ✅ 總空間: ~270 MB

**命令**:
```bash
npm install
npm run install:browsers
npm run dev
```

---

### 場景 2: 生產環境

**需要的套件**:
- ✅ 生產依賴: playwright, amqplib, pino, pino-pretty
- ✅ 間接依賴: ~50 個
- ✅ Playwright 瀏覽器 (~170 MB)
- ✅ 總空間: ~230 MB

**命令**:
```bash
npm install --production
npm run install:browsers
npm start
```

---

### 場景 3: CI/CD 環境

**需要的套件**:
- ✅ 所有依賴（包括測試工具）
- ✅ Playwright 瀏覽器（使用 --with-deps）
- ✅ 總空間: ~300 MB

**命令**:
```bash
npm ci
npx playwright install --with-deps chromium
npm run build
npm test
```

---

## 🔍 依賴管理

### 查看依賴樹

```bash
# 查看所有依賴
npm list

# 查看特定套件依賴
npm list playwright

# 只顯示直接依賴
npm list --depth=0

# 查看過時的套件
npm outdated
```

---

### 更新依賴

```bash
# 檢查可更新的套件
npm outdated

# 更新到次要版本
npm update

# 更新到最新主要版本
npm install <package>@latest

# 互動式更新
npx npm-check-updates -i
```

---

### 審計安全性

```bash
# 檢查安全漏洞
npm audit

# 自動修復漏洞
npm audit fix

# 強制修復（可能破壞相容性）
npm audit fix --force
```

---

## ⚠️ 常見問題

### Q1: 為什麼 node_modules 這麼大？

**A**: 主要原因:
1. Playwright 核心庫 (~40 MB)
2. TypeScript 編譯器 (~20 MB)
3. 213 個套件的累積

**優化建議**:
- 生產環境使用 `--production` 標誌
- 考慮使用 pnpm 代替 npm（共享依賴）

---

### Q2: Playwright 瀏覽器在哪裡？

**A**: 不在 node_modules 中！

**位置**:
- Windows: `%USERPROFILE%\AppData\Local\ms-playwright\`
- Linux: `~/.cache/ms-playwright/`
- macOS: `~/Library/Caches/ms-playwright/`

**大小**: Chromium ~170 MB

---

### Q3: 可以刪除 node_modules 嗎？

**A**: 可以，但需要重新安裝:

```bash
# 刪除
Remove-Item -Recurse -Force node_modules

# 重新安裝
npm install
```

---

### Q4: 為什麼有這麼多 @types 套件？

**A**: TypeScript 類型定義:
- `@types/node` - Node.js API 類型
- `@types/amqplib` - amqplib 類型

這些只在開發時需要，運行時不影響。

---

### Q5: 可以不安裝 devDependencies 嗎？

**A**: 生產環境可以:

```bash
npm install --production
```

但無法使用:
- ❌ TypeScript 編譯 (tsc)
- ❌ 代碼檢查 (eslint)
- ❌ 測試 (vitest)
- ❌ 開發模式 (tsx)

---

## 📚 進階資源

### 官方文檔

- [Playwright 文檔](https://playwright.dev/)
- [amqplib GitHub](https://github.com/amqp-node/amqplib)
- [Pino 文檔](https://getpino.io/)
- [TypeScript 手冊](https://www.typescriptlang.org/docs/)
- [ESLint 規則](https://eslint.org/docs/rules/)
- [Vitest 指南](https://vitest.dev/)

---

### 套件搜索

- [npm 官方網站](https://www.npmjs.com/)
- [npms.io](https://npms.io/) - 套件品質評分
- [bundlephobia](https://bundlephobia.com/) - 套件大小分析

---

## ✅ 檢查清單

**安裝後驗證**:

```bash
# 1. 檢查 node_modules 存在
Test-Path node_modules  # 應返回 True

# 2. 檢查核心套件
npm list playwright amqplib pino typescript

# 3. 檢查可執行命令
Get-Command tsc, eslint, prettier

# 4. 驗證 TypeScript 版本
tsc --version  # 應顯示 5.9.3

# 5. 驗證 Playwright
npx playwright --version  # 應顯示 1.56.1

# 6. 檢查總大小
$size = (Get-ChildItem node_modules -Recurse | Measure-Object -Property Length -Sum).Sum
"$([math]::Round($size/1MB,2)) MB"
```

---

**維護者**: AIVA 開發團隊  
**最後更新**: 2025-11-20  
**相關文檔**: [README.md](./README.md) | [IMPROVEMENT_PLAN.md](./IMPROVEMENT_PLAN.md)
