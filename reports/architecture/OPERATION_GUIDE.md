# TypeScript Engine 操作指南

## 📑 目錄

- [📋 目錄](#-目錄)
- [🎯 環境準備](#-環境準備)
  - [系統需求](#系統需求)
  - [前置條件檢查](#前置條件檢查)
- [📦 安裝依賴](#-安裝依賴)
  - [Node.js 依賴](#nodejs-依賴)
  - [Playwright 瀏覽器](#playwright-瀏覽器)
- [🔨 編譯與啟動](#-編譯與啟動)
  - [TypeScript 編譯](#typescript-編譯)
  - [啟動引擎](#啟動引擎)
- [🧪 測試驗證](#-測試驗證)
  - [基本功能測試](#基本功能測試)
  - [靶場掃描測試](#靶場掃描測試)
- [🔧 故障排除](#-故障排除)
  - [常見問題 Q&A](#常見問題-qa)
  - [日誌分析](#日誌分析)
- [🚀 進階操作](#-進階操作)
  - [開發模式](#開發模式)
  - [性能調優](#性能調優)
- [✅ 操作檢查清單](#-操作檢查清單)
  - [安裝階段](#安裝階段)
  - [編譯階段](#編譯階段)
  - [運行階段](#運行階段)
  - [測試階段](#測試階段)
- [📞 需要幫助？](#-需要幫助)

---

> **文檔狀態**: ✅ 完整 | **最後更新**: 2025-11-22  
> **適用版本**: v2.0 | **預計時間**: 15-30 分鐘

**📚 返回**: [文檔中心](./INDEX.md) | [架構設計](./ARCHITECTURE.md) | [修復報告](./FIXES_SUMMARY.md)

---

## 📋 目錄

1. [環境準備](#環境準備)
   - [系統需求](#系統需求)
   - [前置條件檢查](#前置條件檢查)
2. [安裝依賴](#安裝依賴)
   - [Node.js 依賴](#nodejs-依賴)
   - [Playwright 瀏覽器](#playwright-瀏覽器)
3. [編譯與啟動](#編譯與啟動)
   - [TypeScript 編譯](#typescript-編譯)
   - [啟動引擎](#啟動引擎)
4. [測試驗證](#測試驗證)
   - [基本功能測試](#基本功能測試)
   - [靶場掃描測試](#靶場掃描測試)
5. [故障排除](#故障排除)
   - [常見問題 Q&A](#常見問題-qa)
   - [日誌分析](#日誌分析)
6. [進階操作](#進階操作)
   - [開發模式](#開發模式)
   - [性能調優](#性能調優)

---

## 🎯 環境準備

### 系統需求

| 項目 | 最低需求 | 推薦配置 |
|------|----------|----------|
| **作業系統** | Windows 10+ / Linux / macOS | Windows 11 |
| **CPU** | 雙核心 2.0GHz | 四核心 3.0GHz+ |
| **記憶體** | 4GB | 8GB+ |
| **硬碟空間** | 2GB 可用空間 | 5GB+ SSD |
| **Node.js** | v20.0.0+ | v20.11.0+ (LTS) |
| **npm** | v10.0.0+ | v10.2.0+ |

### 前置條件檢查

開始前請確認以下項目：

#### 1. 檢查 Node.js 版本

```powershell
node --version
# 預期輸出: v20.x.x 或更高
```

**如果版本過低**：
```powershell
# 下載最新 LTS 版本
# https://nodejs.org/en/download/
```

#### 2. 檢查 npm 版本

```powershell
npm --version
# 預期輸出: 10.x.x 或更高
```

#### 3. 檢查靶場運行狀態

```powershell
# 確認 Juice Shop 運行中
curl http://localhost:3000
# 預期: HTML 回應

# 確認 WebGoat 運行中
curl http://localhost:8080/WebGoat
# 預期: 30x 重定向或 HTML 回應
```

**如果靶場未運行**：
```powershell
# 返回專案根目錄
cd C:\D\fold7\AIVA-git

# 啟動靶場容器
docker-compose up -d juice-shop webgoat
```

#### 4. 確認目錄結構

```powershell
# 進入 TypeScript Engine 目錄
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine

# 確認關鍵檔案存在
Test-Path src/index.ts        # 應該返回 True
Test-Path package.json         # 應該返回 True
Test-Path tsconfig.json        # 應該返回 True
```

---

## 📦 安裝依賴

### Node.js 依賴

#### 步驟 1: 清理舊依賴 (可選)

如果之前已安裝過，建議清理：

```powershell
# 刪除舊的 node_modules
Remove-Item -Recurse -Force node_modules -ErrorAction SilentlyContinue

# 刪除 package-lock.json
Remove-Item package-lock.json -ErrorAction SilentlyContinue
```

#### 步驟 2: 安裝套件

```powershell
# 確保在 typescript_engine 目錄下
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine

# 安裝所有依賴 (約 1-2 分鐘)
npm install
```

**預期輸出**：
```
added 213 packages, and audited 214 packages in 45s

52 packages are looking for funding
  run `npm fund` for details

found 0 vulnerabilities
```

#### 步驟 3: 驗證安裝

```powershell
# 檢查 node_modules 是否存在
Test-Path node_modules
# 應該返回 True

# 確認關鍵套件
Test-Path node_modules/playwright
Test-Path node_modules/typescript
Test-Path node_modules/amqplib
# 全部應該返回 True
```

### Playwright 瀏覽器

Playwright 需要下載 Chromium 瀏覽器 (~300MB)：

```powershell
# 安裝 Chromium 瀏覽器
npm run install:browsers
```

**預期輸出**：
```
> aiva-scan-node@1.0.0 install:browsers
> playwright install --with-deps chromium

Downloading Chromium 123.0.6312.4 (playwright build v1097)
...
✔ Chromium 123.0.6312.4 downloaded
```

**驗證瀏覽器安裝**：
```powershell
# 檢查瀏覽器路徑
npx playwright --version
# 預期: Version 1.41.0 或類似
```

---

## 🔨 編譯與啟動

### TypeScript 編譯

#### 步驟 1: 執行編譯

```powershell
# 確保在 typescript_engine 目錄下
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine

# 編譯 TypeScript → JavaScript
npm run build
```

**預期輸出**：
```
> aiva-scan-node@1.0.0 build
> tsc

# 無錯誤訊息即為成功
```

#### 步驟 2: 驗證編譯產物

```powershell
# 確認 dist 目錄存在
Test-Path dist
# 應該返回 True

# 確認主程式存在
Test-Path dist/index.js
# 應該返回 True

# 查看檔案結構
ls dist
```

**預期結構**：
```
dist/
├── index.js
├── index.js.map
├── services/
│   ├── scan-service.js
│   ├── network-interceptor.service.js
│   ├── enhanced-content-extractor.service.js
│   └── interaction-simulator.service.js
├── interfaces/
│   └── dynamic-scan.interfaces.js
└── utils/
    └── logger.js
```

### 啟動引擎

#### 方式 1: 獨立模式 (推薦測試用)

```powershell
# 🚨 重要：必須在 typescript_engine 目錄下執行
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine

# 直接啟動 Node.js
node dist/index.js
```

**預期輸出**：
```json
{"level":30,"time":"2025-11-22T...","msg":"🚀 啟動 TypeScript 掃描引擎 v2.0"}
{"level":30,"time":"2025-11-22T...","msg":"🌐 啟動 Chromium 瀏覽器..."}
{"level":30,"time":"2025-11-22T...","msg":"✅ 瀏覽器已就緒"}
{"level":30,"time":"2025-11-22T...","msg":"⏳ 等待掃描任務..."}
```

**停止引擎**：
```powershell
# 按 Ctrl+C 停止
```

#### 方式 2: 通過 Python Worker (生產模式)

```powershell
# 返回專案根目錄
cd C:\D\fold7\AIVA-git

# 設置 PYTHONPATH
$env:PYTHONPATH="C:\D\fold7\AIVA-git"

# 啟動 Python Worker
python -m services.scan.engines.typescript_engine.worker
```

**預期輸出**：
```
[INFO] TypeScript Engine Worker 啟動
[INFO] 監聽任務隊列: task.scan.typescript
[INFO] Node.js 程序啟動成功 (PID: 12345)
```

---

## 🧪 測試驗證

### 基本功能測試

#### 測試 1: 驗證環境

```powershell
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine

# 執行環境檢查
node -e "console.log('Node.js:', process.version); console.log('Platform:', process.platform);"
```

**預期輸出**：
```
Node.js: v20.11.0
Platform: win32
```

#### 測試 2: 驗證編譯

```powershell
# 檢查編譯產物
if (Test-Path dist/index.js) { 
    Write-Host "✅ 編譯成功" -ForegroundColor Green 
} else { 
    Write-Host "❌ 編譯失敗" -ForegroundColor Red 
}
```

#### 測試 3: 驗證依賴

```powershell
# 檢查關鍵依賴
$dependencies = @('playwright', 'typescript', 'amqplib', 'pino')
foreach ($dep in $dependencies) {
    if (Test-Path "node_modules/$dep") {
        Write-Host "✅ $dep 已安裝" -ForegroundColor Green
    } else {
        Write-Host "❌ $dep 缺失" -ForegroundColor Red
    }
}
```

### 靶場掃描測試

#### 測試 4: Docker 容器狀態檢查

```powershell
# 檢查靶場容器運行狀態
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | Select-String -Pattern "juice-shop|vigilant_shockle|laughing_jang|ecstatic_ritchie"
```

**如果發現 laughing_jang 顯示 (unhealthy)**：
```powershell
# 檢查容器健康狀態詳情
docker inspect laughing_jang --format '{{json .State.Health}}' | ConvertFrom-Json | ConvertTo-Json -Depth 5
```

#### 測試 5: 完整靶場連接測試 (依照實際驗證流程)

**步驟 1: 測試 Juice Shop 主要實例**
```powershell
curl -s http://localhost:3000 | Select-String -Pattern "title|Juice" | Select-Object -First 3
```

**步驟 2: 測試 Juice Shop 備用實例**
```powershell
curl -s http://localhost:3003 | Select-String -Pattern "title|Juice" | Select-Object -First 3
```

**步驟 3: 測試 WebGoat 根路徑 (預期 404)**
```powershell
curl -s http://localhost:8080 | Select-String -Pattern "title|Goat" | Select-Object -First 3
```
> 💡 **注意**: WebGoat 根路徑返回 404 是正常行為

**步驟 4: 測試 WebGoat 正確端點**
```powershell
curl -s http://localhost:8080/WebGoat/login | Select-String -Pattern "WebGoat|title" | Select-Object -First 5
```

**步驟 5: 測試 Juice Shop 第三個實例**
```powershell
curl -s http://localhost:3001 | Select-String -Pattern "title|Juice" | Select-Object -First 3
```

**步驟 6: 使用 UseBasicParsing 進行最終驗證**
```powershell
curl -s http://localhost:3001 -UseBasicParsing | Select-String -Pattern "title|Juice" | Select-Object -First 3
```

#### 預期測試結果

| 端點 | 預期狀態 | 成功標誌 |
|------|---------|---------|
| `localhost:3000` | ✅ 正常 | 返回 `<title>OWASP Juice Shop</title>` |
| `localhost:3003` | ✅ 正常 | 返回包含 "Juice" 的標題 |
| `localhost:8080` | ⚠️ 404 | 返回 404 錯誤頁面 (正常) |
| `localhost:8080/WebGoat/login` | ✅ 正常 | 返回 `<title>Login Page</title>` |
| `localhost:3001` | ✅ 正常 | 返回包含 "Juice" 的標題 |

**重要提醒**：
- 🚫 **不要使用 `test_typescript_engine.py`** - 該腳本依賴完整系統架構，獨立測試會失敗
- ✅ **使用上述 curl 命令組合進行完整驗證**
- ✅ **所有 Juice Shop 實例 (3000, 3001, 3003) 都應該響應**

---

## 🔧 故障排除

### 常見問題 Q&A

#### Q1: `Error: Cannot find module 'C:\D\fold7\AIVA-git\dist\index.js'`

**原因**: 當前工作目錄錯誤

**解決**：
```powershell
# 必須在 typescript_engine 目錄下執行
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine
node dist/index.js
```

#### Q2: `npm install` 失敗

**可能原因**：
1. 網路問題
2. npm 快取損壞
3. 權限不足

**解決**：
```powershell
# 方案 1: 清理 npm 快取
npm cache clean --force
npm install

# 方案 2: 使用國內鏡像
npm config set registry https://registry.npmmirror.com
npm install

# 方案 3: 以管理員身份執行
Start-Process powershell -Verb RunAs
cd C:\D\fold7\AIVA-git\services\scan\engines\typescript_engine
npm install
```

#### Q3: Playwright 瀏覽器下載失敗

**解決**：
```powershell
# 手動下載瀏覽器
npx playwright install --force chromium

# 檢查瀏覽器路徑
$env:PLAYWRIGHT_BROWSERS_PATH = "$env:USERPROFILE\.cache\ms-playwright"
npx playwright install chromium
```

#### Q4: TypeScript 編譯錯誤

**常見錯誤**：
```
error TS2307: Cannot find module 'playwright' or its corresponding type declarations.
```

**解決**：
```powershell
# 重新安裝依賴
npm install

# 確認 tsconfig.json 正確
cat tsconfig.json | Select-String "moduleResolution"
# 應該包含: "moduleResolution": "node"
```

#### Q5: 掃描卡住或超時

**原因**：
- 靶場未響應
- 網路問題
- 記憶體不足

**解決**：
```powershell
# 檢查靶場狀態
curl http://localhost:3000

# 檢查記憶體
Get-Process node | Select-Object WS

# 重啟引擎
# Ctrl+C 停止後重新啟動
```

#### Q6: 日誌輸出亂碼

**解決**：
```powershell
# 設置編碼
$OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 或使用 pino-pretty 美化
node dist/index.js | npx pino-pretty
```

### 日誌分析

#### 正常啟動日誌

```json
{"level":30,"time":"...","msg":"🚀 啟動 TypeScript 掃描引擎 v2.0"}
{"level":30,"time":"...","msg":"🌐 啟動 Chromium 瀏覽器..."}
{"level":30,"time":"...","msg":"✅ 瀏覽器已就緒"}
```

#### 錯誤日誌範例

**錯誤 1: 瀏覽器啟動失敗**
```json
{"level":50,"time":"...","msg":"❌ 瀏覽器啟動失敗","error":"Executable doesn't exist"}
```
→ 執行 `npm run install:browsers`

**錯誤 2: 靶場連接失敗**
```json
{"level":40,"time":"...","msg":"❌ 掃描頁面失敗","url":"http://localhost:3000","error":"net::ERR_CONNECTION_REFUSED"}
```
→ 檢查靶場是否運行

**錯誤 3: 記憶體不足**
```json
{"level":50,"time":"...","msg":"❌ 頁面崩潰","error":"Target closed"}
```
→ 增加系統記憶體或減少 max_pages

---

## 🚀 進階操作

### 開發模式

#### 熱重載開發

```powershell
# 使用 tsx watch 模式（修改代碼自動重啟）
npm run dev
```

**適用場景**：
- 修改 src/ 下的 TypeScript 代碼
- 實時查看變更效果
- 快速迭代開發

#### 代碼品質檢查

```powershell
# ESLint 檢查
npm run lint

# Prettier 格式化
npm run format

# TypeScript 類型檢查
npx tsc --noEmit
```

### 性能調優

#### 調整掃描參數

編輯任務配置 (如通過 API 或 RabbitMQ)：

```json
{
  "scan_id": "perf-test-001",
  "targets": ["http://localhost:3000"],
  "max_depth": 2,        // 降低深度減少時間
  "max_pages": 20,       // 限制頁面數
  "timeout_ms": 60000    // 60 秒超時
}
```

#### 監控記憶體使用

```powershell
# 持續監控 Node.js 進程
while ($true) {
    Get-Process node | Select-Object ProcessName, WS, PM, CPU | Format-Table
    Start-Sleep -Seconds 5
}
```

#### 優化 Chromium 設置

編輯 `src/services/scan-service.ts`：

```typescript
const context = await this.browser.newContext({
  viewport: { width: 1280, height: 720 },  // 降低解析度
  ignoreHTTPSErrors: true,                  // 忽略 SSL 錯誤
  javaScriptEnabled: true,
  userAgent: 'AIVA-Scanner/2.0'
});
```

---

## ✅ 操作檢查清單

完成以下清單確保引擎正常運作：

### 安裝階段
- [ ] Node.js >= 20.0.0
- [ ] npm >= 10.0.0
- [ ] `npm install` 完成
- [ ] Playwright 瀏覽器已下載
- [ ] node_modules 目錄存在

### 編譯階段
- [ ] TypeScript 編譯無錯誤
- [ ] dist/index.js 存在
- [ ] dist/services/ 目錄完整

### 運行階段
- [ ] 靶場容器運行中
- [ ] 引擎啟動無錯誤
- [ ] 瀏覽器正常打開
- [ ] 可成功訪問靶場

### 測試階段
- [ ] test_typescript_engine.py 通過
- [ ] 掃描結果包含資產
- [ ] 無記憶體洩漏
- [ ] 性能符合預期

---

## 📞 需要幫助？

- **查看日誌**: 所有錯誤都會記錄在控制台
- **參考架構文檔**: [ARCHITECTURE.md](./ARCHITECTURE.md)
- **查看修復記錄**: [FIXES_SUMMARY.md](./FIXES_SUMMARY.md)
- **返回文檔中心**: [INDEX.md](./INDEX.md)

---

**文檔維護**: AIVA 開發團隊  
**最後更新**: 2025-11-22  
**下次審查**: 2026-02-22
