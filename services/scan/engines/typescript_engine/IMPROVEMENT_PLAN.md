# TypeScript Engine 改善計劃

**日期**: 2025-11-20  
**當前狀態**: 🟡 70% 完成 - 需要修復核心問題才能使用  
**預計工作量**: 4-6 小時

---

## 📋 目錄

### 現狀分析
- [📊 功能完整度分析](#功能完整度分析)
  - [✅ 已完成的核心功能 (70%)](#已完成的核心功能-70)
  - [❌ 未完成/有問題的功能 (30%)](#未完成有問題的功能-30)
- [🔴 Critical 問題詳解](#critical-問題詳解)
  - [問題 1: Worker 與 Node.js 通信機制錯誤](#問題-1-worker-與-nodejs-通信機制錯誤)
  - [問題 2: 隊列名稱不一致](#問題-2-隊列名稱不一致)
  - [問題 3: 資產格式不匹配](#問題-3-資產格式不匹配)

### 解決方案
- [🎯 改善計劃](#改善計劃)
  - [Phase A: 修復核心通信機制 (2-3 小時)](#phase-a-修復核心通信機制-2-3-小時)
  - [Phase B: 優化與增強 (2-3 小時)](#phase-b-優化與增強-2-3-小時)
- [📋 實施優先級](#實施優先級)

### 完成後狀態
- [🎯 完成後的狀態](#完成後的狀態)
- [🚀 使用方式（完成後）](#使用方式完成後)
- [📊 投資回報分析](#投資回報分析)
- [✅ 檢查清單](#檢查清單)

### 相關文件
- [📦 依賴說明](./NODE_MODULES_GUIDE.md) - 213 個套件 / 5,905 檔案完整分析
- [📖 使用手冊](./README.md) - 完整使用說明和架構設計

---

## 📊 功能完整度分析

### ✅ 已完成的核心功能 (70%)

| 功能模塊 | 完成度 | 說明 |
|---------|--------|------|
| **Playwright 整合** | ✅ 100% | 瀏覽器啟動、頁面導航、等待策略 |
| **SPA 框架檢測** | ✅ 100% | React/Vue/Angular/Svelte 檢測邏輯 |
| **SPA 路由提取** | ✅ 100% | History API 監聽、Hash 路由提取 |
| **網路請求攔截** | ✅ 100% | Request/Response 監聽、分類過濾 |
| **WebSocket 檢測** | ✅ 100% | WebSocket 連接監聽 |
| **表單提取** | ✅ 100% | 表單、輸入框、動作屬性提取 |
| **連結爬取** | ✅ 100% | 同域連結提取、深度控制 |
| **TypeScript 編譯** | ✅ 100% | 編譯配置正確，無錯誤 |
| **依賴安裝** | ✅ 100% | Playwright 1.56.1、amqplib、pino |

### ❌ 未完成/有問題的功能 (30%)

| 問題 | 嚴重性 | 影響 |
|------|--------|------|
| **RabbitMQ 整合不完整** | 🔴 Critical | Worker 無法與 index.ts 通信 |
| **任務傳遞方式錯誤** | 🔴 Critical | 使用臨時文件而非 RabbitMQ |
| **隊列名稱不統一** | 🟡 Medium | `task.scan.dynamic` vs `TASK_SCAN_PHASE1` |
| **缺少 AIVA Common 整合** | 🟡 Medium | Asset 格式轉換不完整 |
| **缺少去重邏輯** | 🟡 Medium | 可能產生重複資產 |
| **缺少錯誤恢復** | 🟠 Low | 瀏覽器崩潰後無法恢復 |

---

## 🔴 Critical 問題詳解

### 問題 1: Worker 與 Node.js 通信機制錯誤

**當前實現**:
```python
# worker.py line 260+
# 使用臨時文件傳遞任務 ❌
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(task, f)
    task_file = f.name

env = {
    **os.environ,
    "AIVA_SCAN_TASK_FILE": task_file,  # 環境變數傳遞文件路徑
}

proc = await asyncio.create_subprocess_exec(
    NODE_EXECUTABLE,
    str(dist_dir / "index.js"),
    env=env,
)
```

**問題**:
1. `index.ts` 不讀取 `AIVA_SCAN_TASK_FILE` 環境變數
2. `index.ts` 設計為監聽 RabbitMQ 隊列 `task.scan.dynamic`
3. Worker 啟動 Node.js 子進程後立即退出，沒有等待結果
4. Node.js 進程會持續監聽隊列，而不是執行一次掃描就退出

**影響**: 🔴 **完全無法工作** - Worker 無法獲取掃描結果

---

### 問題 2: 隊列名稱不一致

**index.ts** (Node.js):
```typescript
const TASK_QUEUE = 'task.scan.dynamic';  // 舊的隊列名稱
const RESULT_QUEUE = 'findings.new';     // 舊的隊列名稱
```

**worker.py** (Python):
```python
await broker.subscribe(Topic.TASK_SCAN_PHASE1)  # 新的標準：task.scan.phase1
await broker.publish(Topic.RESULTS_SCAN_COMPLETED)  # 新的標準：results.scan.completed
```

**影響**: 🔴 **無法接收任務** - 監聽的隊列不同，Worker 發送到 Phase1 隊列，Node.js 監聽 dynamic 隊列

---

### 問題 3: 資產格式不匹配

**index.ts 輸出**:
```typescript
interface Asset {
  type: string;
  value: string;
  metadata: Record<string, any>;  // 簡單格式
}
```

**worker.py 期望**:
```python
Asset(
    asset_id=new_id("asset"),      # ❌ index.ts 沒有生成
    type=raw_asset.get("type"),
    value=raw_asset.get("value"),
    confidence=1.0,                 # ❌ index.ts 沒有提供
    **raw_asset.get("metadata", {}),
)
```

**影響**: 🟡 **資產轉換可能失敗** - 缺少必需字段

---

## 🎯 改善計劃

### Phase A: 修復核心通信機制 (2-3 小時)

#### A1: 統一架構設計 - 選擇其中一種方案

**方案 1: 獨立 Node.js 服務 (推薦)** ⭐
```
┌─────────────────┐
│ RabbitMQ Server │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐  ┌────────────────┐
│ Worker │  │ Node.js Service│
│ (啟動) │  │  (長期運行)    │
└────────┘  └────────────────┘
                    │
                    ▼
          監聽 task.scan.phase1
          執行掃描
          發送結果到 results.scan.completed
```

**優點**:
- ✅ 符合微服務架構
- ✅ 一個 Node.js 進程處理多個掃描任務
- ✅ 資源利用率高（瀏覽器可復用）
- ✅ 與 Rust Engine 架構一致

**實施步驟**:
1. 修改 `index.ts` 隊列名稱為 `task.scan.phase1`
2. 修改結果隊列為 `results.scan.completed`
3. 調整資產格式，添加 `asset_id` 和 `confidence`
4. Worker 只負責啟動 Node.js 服務（不需要每次掃描都啟動）

---

**方案 2: Python 調用 Node.js 腳本 (簡單但低效)**
```
Worker 收到任務
    ↓
啟動 Node.js 子進程
    ↓
執行一次掃描
    ↓
輸出 JSON 到 stdout
    ↓
Worker 解析結果
    ↓
發送到 RabbitMQ
```

**優點**:
- ✅ 實施簡單
- ✅ Worker 完全控制流程

**缺點**:
- ❌ 每次掃描都要啟動瀏覽器（~3秒開銷）
- ❌ 資源浪費
- ❌ 不符合微服務架構

**實施步驟**:
1. 創建新的 `scanner.ts` 腳本（不監聽 RabbitMQ）
2. 從 `process.argv` 讀取任務參數或 stdin
3. 執行掃描後輸出 JSON 到 stdout
4. Worker 讀取 stdout 並解析

---

#### A2: 實施方案 1（推薦）

**需要修改的文件**:

1. **src/index.ts** (3 處修改):
```typescript
// 修改 1: 隊列名稱
const TASK_QUEUE = 'task.scan.phase1';      // 改
const RESULT_QUEUE = 'results.scan.completed'; // 改

// 修改 2: 任務接口
interface ScanTask {
  scan_id: string;
  targets: string[];        // 改：支持多目標
  max_depth: number;
  max_pages: number;
  enable_javascript: boolean;
}

// 修改 3: 資產格式
interface Asset {
  asset_id: string;         // 新增
  type: string;
  value: string;
  confidence: number;       // 新增
  metadata: Record<string, any>;
}
```

2. **worker.py** (簡化):
```python
async def run() -> None:
    """
    TypeScript Worker 主函數
    只負責啟動 Node.js 服務（如果尚未運行）
    """
    broker = await get_broker()
    
    # 檢查 Node.js 服務是否運行
    if not await _is_node_service_running():
        logger.info("[TypeScript] Starting Node.js service...")
        await _start_node_service()
    
    logger.info("[TypeScript] Node.js service is ready")
    # Worker 只需確保服務運行，不需要處理任務
```

**預計時間**: 2 小時

---

### Phase B: 優化和完善 (2 小時)

#### B1: 添加資產去重邏輯

**問題**: 
- 同一個 API 端點可能被多次請求
- 同一個表單可能在多個頁面出現
- SPA 路由可能重複

**解決方案**:
```typescript
// scan-service.ts
private deduplicateAssets(assets: Asset[]): Asset[] {
  const seen = new Map<string, Asset>();
  
  for (const asset of assets) {
    const key = `${asset.type}:${asset.value}`;
    
    // 保留最詳細的資產（metadata 最多的）
    if (!seen.has(key) || 
        Object.keys(asset.metadata).length > Object.keys(seen.get(key)!.metadata).length) {
      seen.set(key, asset);
    }
  }
  
  return Array.from(seen.values());
}
```

**預計時間**: 30 分鐘

---

#### B2: 改善錯誤處理

**當前問題**:
- 瀏覽器崩潰後無法恢復
- 單個頁面失敗不應影響整個掃描
- 超時處理不完善

**解決方案**:
```typescript
// 添加重試邏輯
private async scanPageWithRetry(
  page: Page, 
  url: string, 
  maxRetries: number = 3
): Promise<Asset[]> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await this.scanSinglePage(page, url);
    } catch (error) {
      logger.warn({ url, attempt: i + 1, error }, '⚠️ 掃描失敗，重試中...');
      
      if (i === maxRetries - 1) {
        logger.error({ url }, '❌ 掃描失敗，放棄');
        return [];
      }
      
      await page.waitForTimeout(1000 * (i + 1)); // 指數退避
    }
  }
  return [];
}
```

**預計時間**: 1 小時

---

#### B3: 優化性能

**問題**:
- `waitUntil: 'networkidle'` 太嚴格（等待所有網路請求完成）
- 每個頁面固定等待 1000ms

**解決方案**:
```typescript
// 使用自適應等待策略
const response = await page.goto(url, {
  waitUntil: 'domcontentloaded',  // 只等待 DOM 載入
  timeout: 30000,
});

// 自適應等待（檢測動態內容）
await this.waitForDynamicContent(page);

private async waitForDynamicContent(page: Page): Promise<void> {
  let previousHeight = 0;
  let stableCount = 0;
  
  for (let i = 0; i < 5; i++) {
    const currentHeight = await page.evaluate(() => document.body.scrollHeight);
    
    if (currentHeight === previousHeight) {
      stableCount++;
      if (stableCount >= 2) break;  // 連續 2 次不變，認為穩定
    } else {
      stableCount = 0;
    }
    
    previousHeight = currentHeight;
    await page.waitForTimeout(500);
  }
}
```

**預計時間**: 30 分鐘

---

### Phase C: 測試與驗證 (1-2 小時)

#### C1: 單元測試

**測試目標**:
- ✅ RabbitMQ 連接正常
- ✅ 接收 Phase1 任務
- ✅ 掃描 Juice Shop 成功
- ✅ 資產格式正確
- ✅ 結果發送到正確隊列

**測試腳本** (已存在 `test_typescript_engine.py`):
```bash
# 運行測試
python services/scan/engines/typescript_engine/test_typescript_engine.py
```

**預期結果**:
```
✅ Node.js 服務啟動
✅ 連接到 RabbitMQ
✅ 監聽 task.scan.phase1
✅ 掃描 http://localhost:3000
✅ 發現 30-50 個資產
   - SPA 路由: 10-15
   - API 端點: 15-20
   - 表單: 3-5
   - WebSocket: 0-1
✅ 發送結果到 results.scan.completed
```

**預計時間**: 1 小時

---

#### C2: 整合測試

**測試場景**:
1. 啟動完整 AIVA 系統
2. 通過 Web API 提交掃描請求
3. 選擇 TypeScript 引擎
4. 驗證端到端流程

**預計時間**: 1 小時

---

## 📋 實施優先級

| 優先級 | 任務 | 預計時間 | 影響 |
|-------|------|---------|------|
| **P0** | 修復 RabbitMQ 整合（方案 1） | 2 小時 | 🔴 無此無法工作 |
| **P0** | 統一隊列名稱 | 30 分鐘 | 🔴 無此無法接收任務 |
| **P0** | 調整資產格式 | 30 分鐘 | 🟡 資產可能無法存儲 |
| **P1** | 添加去重邏輯 | 30 分鐘 | 🟡 提高資產質量 |
| **P1** | 改善錯誤處理 | 1 小時 | 🟠 提高穩定性 |
| **P2** | 優化性能 | 30 分鐘 | 🟢 提高速度 |
| **P2** | 測試驗證 | 2 小時 | 🟢 確保質量 |

**總計**: 4-6 小時

---

## 🎯 完成後的狀態

### 架構圖

```
                    ┌─────────────────┐
                    │  RabbitMQ Server │
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
                ▼            ▼            ▼
         ┌──────────┐  ┌──────────┐  ┌──────────────┐
         │  Python  │  │   Rust   │  │  TypeScript  │
         │  Worker  │  │  Worker  │  │  Node.js     │
         └──────────┘  └──────────┘  └──────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Playwright      │
                                    │ Chromium        │
                                    │ - SPA 路由      │
                                    │ - AJAX 攔截     │
                                    │ - WebSocket     │
                                    └─────────────────┘
```

### 功能對比

| 功能 | Python Engine | Rust Engine | TypeScript Engine |
|------|--------------|-------------|-------------------|
| **靜態爬取** | ✅ | ✅ | ✅ |
| **SPA 路由** | ❌ | ❌ | ✅ **獨有** |
| **動態 AJAX** | ⚠️ 有限 | ❌ | ✅ **最優** |
| **JavaScript 執行** | ✅ Playwright | ❌ | ✅ **更快** |
| **WebSocket** | ❌ | ❌ | ✅ **獨有** |
| **性能** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **內存** | ~100MB | ~5MB | ~300MB |

---

## 🚀 使用方式（完成後）

### 啟動服務

```bash
# 終端 1: 啟動 RabbitMQ
docker-compose up -d rabbitmq

# 終端 2: 啟動 TypeScript Worker（自動啟動 Node.js 服務）
cd C:\D\fold7\AIVA-git
python -m services.scan.engines.typescript_engine.worker
```

### 提交掃描任務

```python
# 通過 RabbitMQ 發送任務
from services.aiva_common.schemas import Phase1StartPayload

payload = Phase1StartPayload(
    scan_id="scan-001",
    targets=["http://localhost:3000"],
    selected_engines=["typescript"],
    max_depth=3,
    timeout=300,
)

await broker.publish(
    Topic.TASK_SCAN_PHASE1,
    payload.model_dump_json().encode(),
)
```

### 預期輸出

```
[TypeScript] Node.js service started
[TypeScript] Connected to RabbitMQ
[TypeScript] Listening on queue: task.scan.phase1
[TypeScript] Received scan task: scan-001
[TypeScript] Target: http://localhost:3000
[TypeScript] Detected SPA framework: Angular
[TypeScript] Found 15 SPA routes
[TypeScript] Intercepted 23 AJAX requests
[TypeScript] Found 4 forms
[TypeScript] Total assets: 42
[TypeScript] Scan completed in 18.5s
[TypeScript] Published results to: results.scan.completed
```

---

## 📊 投資回報分析

| 指標 | 當前 | 完成後 | 改善 |
|------|------|--------|------|
| **可用性** | 0% | 100% | ∞ |
| **資產發現** | 0 | 30-50/靶場 | +50 |
| **SPA 路由** | 0 | 10-15/靶場 | **獨有** |
| **AJAX 端點** | 0 | 15-20/靶場 | **最優** |
| **掃描速度** | N/A | ~20秒/靶場 | 中等 |
| **開發時間** | N/A | 4-6 小時 | 可接受 |

---

## ✅ 檢查清單

**在開始修復前確認**:
- [ ] Node.js >= 20.0.0 已安裝
- [ ] `npm install` 完成
- [ ] `npm run build` 編譯成功
- [ ] Playwright 瀏覽器已安裝（`npm run install:browsers`）
- [ ] RabbitMQ 運行中
- [ ] Python 虛擬環境已激活
- [ ] AIVA Common 可導入

**修復完成後驗證**:
- [ ] Node.js 服務可以啟動
- [ ] 連接到 RabbitMQ 成功
- [ ] 監聽正確的隊列（`task.scan.phase1`）
- [ ] 掃描 Juice Shop 成功
- [ ] 發現 30+ 資產
- [ ] 結果發送到正確隊列（`results.scan.completed`）
- [ ] 資產格式符合 AIVA Common 規範
- [ ] Worker 可以解析結果
- [ ] 端到端測試通過

---

**維護者**: AIVA 開發團隊  
**最後更新**: 2025-11-20  
**狀態**: 待實施
