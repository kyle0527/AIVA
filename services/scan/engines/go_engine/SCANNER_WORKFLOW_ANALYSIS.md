# SSRF 掃描器工作流程分析報告

## 📊 完整工作流程圖

```
┌─────────────────────────────────────────────────────────────────────┐
│                         1. 程式啟動階段                               │
│                      (cmd/ssrf-scanner/main.go)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  讀取 STDIN JSON 輸入          │
                    │  json.NewDecoder(os.Stdin)    │
                    │  --------------------------------│
                    │  輸入格式：                     │
                    │  {                             │
                    │    "scan_id": "...",           │
                    │    "targets": ["http://..."],  │
                    │    "concurrency": 1            │
                    │  }                             │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      2. Worker Pool 創建階段                          │
│                    (internal/common/worker_pool.go)                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  NewWorkerPool(concurrency)   │
                    │  --------------------------------│
                    │  • 創建 taskQueue channel      │
                    │  • 創建 resultChan channel     │
                    │  • 啟動 N 個 worker goroutines │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            ┌───────────────┐              ┌───────────────┐
            │   Worker 0    │              │   Worker N    │
            │   (goroutine) │     ...      │   (goroutine) │
            └───────────────┘              └───────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                        等待 taskQueue 中的任務
                                    
┌─────────────────────────────────────────────────────────────────────┐
│                       3. 任務提交階段                                 │
│                      (cmd/ssrf-scanner/main.go)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  for target in targets:       │
                    │    pool.Submit(task)          │
                    │  --------------------------------│
                    │  Task 包含：                   │
                    │  • Target URL                  │
                    │  • Scanner (SSRFDetector)     │
                    │  • Config                      │
                    └───────────────────────────────┘
                                    │
                                    ▼
                            放入 taskQueue
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                       4. Worker 執行階段                              │
│                   (internal/common/worker_pool.go)                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  task := <-taskQueue          │
                    │  --------------------------------│
                    │  執行：                         │
                    │  assets, err = task.Scanner.  │
                    │    Scan(ctx, [target], config)│
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     5. SSRF 掃描執行階段                              │
│                (internal/ssrf/detector/ssrf.go)                     │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  SSRFDetector.Scan()          │
                    │  --------------------------------│
                    │  for target in targets:       │
                    │    scanSingleTarget(target)   │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  scanSingleTarget()           │
                    │  --------------------------------│
                    │  1. 解析目標 URL               │
                    │  2. 準備 16 個參數名稱         │
                    │  3. 準備 8 個 Payloads        │
                    │  4. 雙層循環測試               │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │   雙層循環：16 × 8 = 128 次   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │  for param in [url, uri, path, redirect, ...]:    │
        │    for payload in [AWS IMDS, GCP, localhost, ...]:│
        │      testSSRF(testURL, param, payload)            │
        └───────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              ⚠️ 6. HTTP 請求發送階段 [關鍵點]                         │
│                (internal/ssrf/detector/ssrf.go)                     │
│                      Line 202-280                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────────────────────────┐
        │           testSSRF() 函數流程                  │
        │  =============================================  │
        │                                                │
        │  1️⃣ 構造測試 URL                               │
        │     testURL = "http://localhost:8080/         │
        │               WebGoat?url=<payload>"          │
        │                                                │
        │  2️⃣ 創建 HTTP 請求對象                         │
        │     req = http.NewRequestWithContext(         │
        │             ctx, "GET", testURL, nil)         │
        │                                                │
        │  3️⃣ 設置 Headers                               │
        │     req.Header.Set("User-Agent",              │
        │       "AIVA-SSRF-Scanner/1.0")                │
        │                                                │
        │  4️⃣ 執行 HTTP 請求 ⬇️ [對外發送訊息的關鍵點]    │
        │     resp, err = d.client.Do(req)              │
        │     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^          │
        │     這裡負責發送 HTTP 請求到靶場                │
        │                                                │
        │     HTTP Client 配置：                         │
        │     • Timeout: 10 秒                           │
        │     • Transport: 使用 Go 默認 Transport       │
        │     • 無 Proxy                                 │
        │     • CheckRedirect: 最多 3 次重定向           │
        │                                                │
        └───────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                ▼ err == nil                   ▼ err != nil
        ┌─────────────────┐          ┌──────────────────────┐
        │  請求成功        │          │  請求失敗            │
        │  --------------  │          │  ------------------  │
        │  • 讀取響應體    │          │  • 檢查錯誤類型      │
        │  • 判斷是否漏洞  │          │  • 返回 nil          │
        │  • 創建 Asset   │          │  （沒有發現漏洞）    │
        └─────────────────┘          └──────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                            返回 Asset 或 nil
                                    │
┌─────────────────────────────────────────────────────────────────────┐
│                    7. 結果收集階段                                    │
│                  (cmd/ssrf-scanner/main.go)                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  taskResult := <-pool.Results()│
                    │  --------------------------------│
                    │  • 收集 Assets                 │
                    │  • 統計成功/失敗數             │
                    │  • 記錄錯誤                    │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     8. 結果輸出階段                                   │
│                   (cmd/ssrf-scanner/main.go)                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  json.NewEncoder(os.Stdout)   │
                    │  --------------------------------│
                    │  輸出到 STDOUT：                │
                    │  {                             │
                    │    "scan_id": "...",           │
                    │    "status": "success",        │
                    │    "requests_made": 0,  ❌     │
                    │    "assets": []                │
                    │  }                             │
                    └───────────────────────────────┘
```

---

## 🔍 關鍵組件職責分析

### 📤 負責「對外發送訊息」的組件

#### 1. **HTTP Client** (`d.client` in `SSRFDetector`)
**位置**: `internal/ssrf/detector/ssrf.go`, Line 46-55

```go
client := &http.Client{
    Timeout: 10 * time.Second,
    CheckRedirect: func(req *http.Request, via []*http.Request) error {
        if len(via) >= 3 {
            return fmt.Errorf("too many redirects")
        }
        return nil
    },
}
```

**職責**:
- ✅ 發送 HTTP GET 請求到靶場
- ✅ 管理超時（10 秒）
- ✅ 處理重定向（最多 3 次）
- ✅ 使用 Go 默認 `http.DefaultTransport`

**關鍵執行點**: Line 225
```go
resp, err := d.client.Do(req)
```

**這是唯一真正對外發送 HTTP 請求的代碼**

---

#### 2. **HTTP Request 對象** (`*http.Request`)
**創建位置**: `internal/ssrf/detector/ssrf.go`, Line 214

```go
req, err := http.NewRequestWithContext(ctx, "GET", testURL, nil)
```

**包含的信息**:
- Method: `GET`
- URL: 構造的測試 URL（如 `http://localhost:8080/WebGoat?url=http://169.254.169.254/...`）
- Headers: `User-Agent: AIVA-SSRF-Scanner/1.0`
- Context: 用於超時和取消控制

---

### 📥 負責「接收外界訊息」的組件

#### 1. **HTTP Response** (`*http.Response`)
**接收位置**: `internal/ssrf/detector/ssrf.go`, Line 225

```go
resp, err := d.client.Do(req)
if err == nil {
    defer resp.Body.Close()
    body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024*1024))
    bodyStr := string(body)
    // 分析響應...
}
```

**職責**:
- ✅ 接收 HTTP 響應
- ✅ 讀取響應體（最多 1MB）
- ✅ 提供響應狀態碼 (`resp.StatusCode`)
- ✅ 提供響應 Headers (`resp.Header`)

**接收的數據**:
- 靶場返回的 HTML/JSON
- HTTP 狀態碼（200, 404, 500 等）
- 響應頭（Content-Type, Set-Cookie 等）

---

#### 2. **漏洞判斷邏輯** (`isSSRFVulnerable()`)
**位置**: `internal/ssrf/detector/ssrf.go`, Line 285+

```go
func (d *SSRFDetector) isSSRFVulnerable(statusCode int, body, payload string) bool {
    // 檢查響應內容是否包含 SSRF 指標
    if strings.Contains(payload, "169.254.169.254") {
        return d.isAWSMetadataResponse(body, statusCode)
    }
    // ... 其他判斷邏輯
}
```

**職責**:
- ✅ 分析響應內容
- ✅ 檢測 SSRF 漏洞指標
- ✅ 判斷是否為誤報（登錄頁、錯誤頁等）

---

## ⚠️ 當前問題診斷

### 問題現象
```json
{
  "requests_made": 0,     // ❌ 應該是 128
  "assets": [],           // ❌ 沒有發現任何漏洞
  "execution_time": 0.86  // ⚠️ 太快了（應該 >5 秒）
}
```

### 可能原因分析

#### 原因 A: HTTP 請求被靜默忽略
**位置**: `internal/ssrf/detector/ssrf.go`, Line 225-280

```go
resp, err := d.client.Do(req)
if err == nil {
    // 處理成功響應...
} else {
    // ❌ 錯誤處理邏輯：
    if d.isSSRFIndicatorError(err) {
        // 只記錄 Debug 日誌，沒有任何實際作用
    }
    return nil  // ← 直接返回，請求失敗被吞掉
}
```

**問題**: 如果 `client.Do()` 返回錯誤，函數直接返回 `nil`，沒有任何統計或反饋。

---

#### 原因 B: requests_made 計數器從未實現
**位置**: `internal/common/types.go`, Line 46

```go
type ScanResult struct {
    // ...
    RequestsMade   int     `json:"requests_made"`  // ← 定義了但從未賦值
    // ...
}
```

**問題**: 
- 字段只是定義，沒有任何代碼增加這個計數器
- 無論發送多少請求，都永遠是 0

---

#### 原因 C: Context 可能被提前取消
**位置**: `internal/ssrf/detector/ssrf.go`, Line 162-166

```go
for _, param := range ssrfParams {
    for _, payload := range testPayloads {
        select {
        case <-ctx.Done():        // ← 檢查 Context
            return assets, ctx.Err()
        default:
        }
        // ...
    }
}
```

**疑點**: 
- 如果 `ctx` 在循環開始前就被取消，所有測試會被跳過
- 但日誌顯示有 160 條「測試 SSRF」，說明循環有執行

---

#### 原因 D: HTTP Client 配置問題
**位置**: `internal/ssrf/detector/ssrf.go`, Line 46-55

```go
client := &http.Client{
    Timeout: 10 * time.Second,
    // ❌ 沒有設置 Transport
    // ❌ 使用默認 http.DefaultTransport
}
```

**可能問題**:
- 默認 Transport 可能有 DNS 解析問題
- 默認 Transport 可能有連接池限制
- 沒有自定義 DialContext，可能被系統防火牆阻擋

---

## 🎯 數據流向圖

```
外部輸入 (STDIN JSON)
    │
    ▼
┌──────────────────────┐
│  main.go             │
│  • 解析 JSON         │
│  • 創建 Worker Pool  │
└──────────────────────┘
    │
    ▼ (通過 taskQueue channel)
┌──────────────────────┐
│  worker_pool.go      │
│  • 接收任務          │
│  • 調用 Scanner.Scan │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  ssrf.go (Scan)      │
│  • 循環處理 targets  │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  ssrf.go             │
│  (scanSingleTarget)  │
│  • 雙層循環測試      │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│  ssrf.go (testSSRF)  │
│  • 構造請求          │
│  • client.Do(req) ⬅️ │ 📤 發送 HTTP 到靶場
│  • 接收響應     ➡️   │ 📥 接收靶場響應
│  • 分析漏洞          │
└──────────────────────┘
    │
    ▼
返回 Asset 或 nil
    │
    ▼ (通過 resultChan channel)
┌──────────────────────┐
│  main.go             │
│  (executeScan)       │
│  • 收集結果          │
│  • 統計計數          │
└──────────────────────┘
    │
    ▼
外部輸出 (STDOUT JSON)
```

---

## 📋 關鍵變量追蹤

### 1. 請求計數器 `requests_made`
**定義位置**: `internal/common/types.go:46`
```go
RequestsMade   int     `json:"requests_made"`
```

**❌ 問題**: 從未被賦值或遞增

**應該在哪裡更新**:
- Option A: 在 `testSSRF()` 中，每次調用 `client.Do()` 後遞增
- Option B: 在 `scanSingleTarget()` 中統計
- Option C: 在 `Scan()` 方法中維護

---

### 2. HTTP Client `d.client`
**創建位置**: `internal/ssrf/detector/ssrf.go:46`
```go
client := &http.Client{
    Timeout: 10 * time.Second,
    CheckRedirect: func(req *http.Request, via []*http.Request) error {
        if len(via) >= 3 {
            return fmt.Errorf("too many redirects")
        }
        return nil
    },
}
```

**使用位置**: Line 225
```go
resp, err := d.client.Do(req)
```

---

### 3. Context `ctx`
**來源**: Worker Pool 傳入
```go
assets, err := task.Scanner.Scan(p.ctx, []string{task.Target}, task.Config)
```

**生命週期**: 
- 創建於 `WorkerPool.New()`: `context.WithCancel(context.Background())`
- 傳遞到所有 worker goroutines
- 在 `Shutdown()` 時取消

---

## 🔧 建議修復方向

### Priority 1: 驗證 HTTP 請求是否真的發送

創建一個最小測試來驗證：
```go
package main

import (
    "fmt"
    "net/http"
    "time"
)

func main() {
    client := &http.Client{Timeout: 10 * time.Second}
    
    req, _ := http.NewRequest("GET", "http://localhost:8080/WebGoat", nil)
    req.Header.Set("User-Agent", "Test")
    
    fmt.Println("發送請求...")
    resp, err := client.Do(req)
    
    if err != nil {
        fmt.Printf("❌ 錯誤: %v\n", err)
    } else {
        fmt.Printf("✅ 成功: HTTP %d\n", resp.StatusCode)
        resp.Body.Close()
    }
}
```

---

### Priority 2: 實現請求計數器

在 `SSRFDetector` 中添加：
```go
type SSRFDetector struct {
    logger        *zap.Logger
    client        *http.Client
    blockedRanges []*net.IPNet
    requestCount  *int64  // ← 新增
}

func (d *SSRFDetector) testSSRF(...) *common.Asset {
    // ...
    atomic.AddInt64(d.requestCount, 1)  // ← 遞增
    resp, err := d.client.Do(req)
    // ...
}
```

---

### Priority 3: 檢查錯誤處理邏輯

確保所有錯誤都被適當記錄和處理，而不是靜默返回 `nil`。

---

## 📊 時間軸分析

根據日誌時間戳：
```
16:16:13.320  Scanner started
16:16:13.355  開始 SSRF 掃描
16:16:13.355  測試 SSRF (第 1 次)
16:16:13.442  測試 SSRF (第 2 次)  ← 間隔 87ms
16:16:13.456  測試 SSRF (第 3 次)  ← 間隔 14ms
...
16:16:14.249  測試 SSRF (最後)
16:16:14.249  SSRF 掃描完成        ← 總耗時 0.9 秒
```

**異常點**:
- 160 次測試只用了 0.9 秒
- 平均每次測試 5.6ms
- **正常的 HTTP 請求至少需要 20-50ms**

**結論**: HTTP 請求很可能沒有真正發送，或者立即失敗返回。
