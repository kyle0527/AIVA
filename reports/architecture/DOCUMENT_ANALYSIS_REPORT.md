# Word 文檔與 Go Engine 實際狀態比對分析報告

## 📑 目錄

- [📋 執行摘要](#執行摘要)
  - [✅ 文檔準確描述的部分](#文檔準確描述的部分)
  - [❌ 文檔與實際代碼的重大差異](#文檔與實際代碼的重大差異)
  - [⚠️ 當前系統的真實問題](#當前系統的真實問題)
- [🔍 詳細比對分析](#詳細比對分析)
  - [1. API 設計與函數簽名](#1-api-設計與函數簽名)
    - [文檔聲稱的設計](#文檔聲稱的設計)
    - [實際代碼現狀](#實際代碼現狀)
  - [2. HTTP 請求構造邏輯](#2-http-請求構造邏輯)
    - [文檔聲稱的實現](#文檔聲稱的實現)
    - [實際代碼現狀](#實際代碼現狀-1)
  - [3. 數據流與參數傳遞鏈](#3-數據流與參數傳遞鏈)
    - [文檔描述的數據流](#文檔描述的數據流)
    - [實際代碼的數據流](#實際代碼的數據流)
  - [4. 測試工具與腳本](#4-測試工具與腳本)
    - [文檔提到的測試腳本](#文檔提到的測試腳本)
      - [`test_webgoat_auth.ps1`](#testwebgoatauthps1)
      - [`test_multi_targets.ps1`](#testmultitargetsps1)
  - [5. HTTP Client 配置](#5-http-client-配置)
    - [文檔提到的問題與修復](#文檔提到的問題與修復)
- [🐛 當前系統的真實問題](#當前系統的真實問題-1)
  - [問題 1: 請求計數器缺失](#問題-1-請求計數器缺失)
  - [問題 2: HTTP 請求可能未真正執行](#問題-2-http-請求可能未真正執行)
  - [問題 3: 靶場連接性未驗證](#問題-3-靶場連接性未驗證)
- [📊 建議的合適度評分](#建議的合適度評分)
- [🎯 實施建議](#實施建議)
  - [短期行動（修復當前問題）](#短期行動修復當前問題)
    - [1. 添加請求計數器](#1-添加請求計數器)
    - [2. 添加請求執行日誌](#2-添加請求執行日誌)
    - [3. 驗證靶場連接性](#3-驗證靶場連接性)
  - [中期行動（實現文檔描述的功能）](#中期行動實現文檔描述的功能)
    - [階段 1: 修改數據結構](#階段-1-修改數據結構)
    - [階段 2: 修改函數簽名](#階段-2-修改函數簽名)
    - [階段 3: 注入 Headers](#階段-3-注入-headers)
    - [階段 4: 更新主程序](#階段-4-更新主程序)
    - [階段 5: 創建測試腳本](#階段-5-創建測試腳本)
  - [長期行動（架構優化）](#長期行動架構優化)
- [🔚 結論](#結論)

---


**生成時間**: 2025年11月21日  
**分析對象**: `Untitled.docx` 轉換後的修復指南  
**目標系統**: `C:\D\fold7\AIVA-git\services\scan\engines\go_engine`

---

## 📋 執行摘要

Word 文檔描述了一個**理想化的修復方案**，主要聚焦於「為 SSRF 掃描器添加 HTTP Headers（特別是 Session Cookie）支持」的功能增強。然而，與當前 Go Engine 的實際代碼狀態進行比對後發現：

### ✅ 文檔準確描述的部分
1. **問題診斷正確**: 描述的「302 重定向到登錄頁導致漏報」問題確實存在
2. **修復理念合理**: 通過注入 Cookie/Headers 穿透身份驗證的思路正確
3. **測試流程完整**: 提供的驗證步驟（獲取 JSESSIONID → 執行掃描 → 分析結果）符合實務

### ❌ 文檔與實際代碼的重大差異
1. **核心修復尚未實現**: 當前代碼**不支持** Headers 參數傳遞
2. **API 簽名不匹配**: `Scan()` 和 `scanSingleTarget()` 函數簽名中沒有 `headers` 參數
3. **HTTP 請求未注入**: `testSSRF()` 函數構造請求時未調用 `req.Header.Set()`
4. **測試腳本不存在**: 文檔提到的 `test_webgoat_auth.ps1` 和 `test_multi_targets.ps1` 在目錄中未找到

### ⚠️ 當前系統的真實問題
根據最近的測試日誌（`fix_test.log`），系統存在更底層的問題：
- **`requests_made: 0`**: 掃描器構造了 160 個測試 URL，但實際發送的 HTTP 請求數為 0
- **HTTP Client 配置問題**: 雖然已移除 DialContext 阻擋，但請求仍未執行
- **缺少請求計數器**: 代碼中沒有增加 `RequestsMade` 統計的邏輯

---

## 🔍 詳細比對分析

### 1. API 設計與函數簽名

#### 文檔聲稱的設計
```go
// 文檔描述：Scan 方法接收 headers 參數
func (d *SSRFDetector) Scan(
    ctx context.Context, 
    targets []string, 
    headers map[string]string,  // ← 文檔說這個參數存在
    config *common.ScannerConfig
) ([]common.Asset, error)
```

#### 實際代碼現狀
```go
// 文件: internal/ssrf/detector/ssrf.go, Line 66
func (d *SSRFDetector) Scan(
    ctx context.Context, 
    targets []string, 
    config *common.ScannerConfig  // ← 實際沒有 headers 參數
) ([]common.Asset, error)
```

**結論**: ❌ **不匹配** - 文檔描述的 API 升級尚未實現

---

### 2. HTTP 請求構造邏輯

#### 文檔聲稱的實現
```go
// 文檔描述：在 testSSRF 函數中注入自定義 Headers
req, err := http.NewRequest("GET", testURL, nil)
if err != nil {
    return nil
}

// 注入自定義 Headers（來自上層傳遞的 headers map）
for key, value := range headers {
    req.Header.Set(key, value)  // ← 文檔說有這段邏輯
}
```

#### 實際代碼現狀
```go
// 文件: internal/ssrf/detector/ssrf.go, Line ~200-230
func (d *SSRFDetector) testSSRF(...) *common.Asset {
    // 構造測試 URL
    testURL := fmt.Sprintf("%s?%s=%s", originalTarget, param, url.QueryEscape(payload.url))
    
    // 創建 HTTP 請求
    req, err := http.NewRequest("GET", testURL, nil)
    if err != nil {
        d.logger.Error("無法創建請求", zap.Error(err))
        return nil
    }
    
    // ❌ 沒有任何 Headers 注入邏輯
    
    // 執行請求
    resp, err := d.client.Do(req)
    // ...
}
```

**結論**: ❌ **未實現** - 代碼中沒有任何 `req.Header.Set()` 調用

---

### 3. 數據流與參數傳遞鏈

#### 文檔描述的數據流
```
JSON 輸入 (含 headers)
    ↓
main.go 解析
    ↓
Scan(targets, headers, config)  ← headers 被傳遞
    ↓
scanSingleTarget(target, headers)
    ↓
testSSRF(target, param, payload, headers)  ← headers 注入到 HTTP 請求
    ↓
http.Client.Do(req)
```

#### 實際代碼的數據流
```
JSON 輸入 (可能含 headers)
    ↓
main.go 解析 (ScanRequest 結構體可能有 headers 字段，但未使用)
    ↓
Scan(targets, config)  ← ❌ headers 被丟棄
    ↓
scanSingleTarget(target)  ← ❌ 沒有 headers 參數
    ↓
testSSRF(target, param, payload)  ← ❌ 沒有 headers 參數
    ↓
http.Client.Do(req)  ← 請求不包含自定義 Headers
```

**結論**: ❌ **架構斷層** - 整個參數傳遞鏈未建立

---

### 4. 測試工具與腳本

#### 文檔提到的測試腳本

##### `test_webgoat_auth.ps1`
```powershell
# 文檔描述的功能：
# 1. 提示用戶輸入 JSESSIONID
# 2. 構造包含 Cookie 的 JSON
# 3. 注入到掃描器 stdin
# 4. 捕獲並分析結果
```

**實際狀況**: ❌ **文件不存在**  
目錄中只有這些測試腳本：
- `test_ssrf_real.ps1`
- `test_webgoat_ssrf.ps1`
- `test-docker-targets.ps1`

##### `test_multi_targets.ps1`
```powershell
# 文檔描述的功能：
# 並發測試 httpbin.org、example.com、localhost:8080
```

**實際狀況**: ❌ **文件不存在**

---

### 5. HTTP Client 配置

#### 文檔提到的問題與修復
> "原先 DialContext 阻擋了 localhost 訪問，已移除"

**實際驗證**: ✅ **部分正確**

```go
// 文件: internal/ssrf/detector/ssrf.go, Line 45-55
// 當前代碼確實移除了 DialContext 阻擋邏輯
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

但根據最近測試結果（`fix_test.log`），雖然移除了阻擋，**HTTP 請求仍未真正發送**（`requests_made: 0`）。

---

## 🐛 當前系統的真實問題

### 問題 1: 請求計數器缺失

**證據**: 測試輸出
```json
{
  "requests_made": 0,  // ← 應該是 160 (20 params × 8 payloads)
  "targets_scanned": 1,
  "success_count": 1
}
```

**日誌證據**: 160 條 "測試 SSRF" Debug 日誌存在，證明循環執行了，但 `requests_made` 仍為 0

**根本原因**: 代碼中沒有遞增 `requests_made` 計數器的邏輯

**查找結果**:
```bash
grep -r "requests_made" internal/ssrf/
# 結果: 沒有找到任何匹配
```

---

### 問題 2: HTTP 請求可能未真正執行

**疑點分析**:

1. **`testSSRF()` 函數返回值處理**
   ```go
   // Line ~170-180: scanSingleTarget 中調用 testSSRF
   if asset := d.testSSRF(ctx, target, param, payload); asset != nil {
       assets = append(assets, *asset)
   }
   ```
   
   如果 `testSSRF()` 提前返回 `nil`（例如在 `http.NewRequest` 失敗後），請求就不會執行。

2. **錯誤處理邏輯**
   ```go
   // Line ~215: testSSRF 函數
   req, err := http.NewRequest("GET", testURL, nil)
   if err != nil {
       d.logger.Error("無法創建請求", zap.Error(err))
       return nil  // ← 這裡直接返回，不會執行 client.Do()
   }
   ```

3. **可能的 Context 取消**
   ```go
   req = req.WithContext(ctx)
   if ctx.Err() != nil {
       return nil  // ← Context 被取消也會跳過請求
   }
   ```

---

### 問題 3: 靶場連接性未驗證

**文檔假設**: WebGoat 運行在 `localhost:8080`  
**實際情況**: 未知

**建議驗證命令**:
```powershell
# 測試 WebGoat 是否可訪問
Invoke-WebRequest -Uri "http://localhost:8080/WebGoat" -Method GET
```

---

## 📊 建議的合適度評分

| 評估維度 | 評分 | 說明 |
|---------|------|------|
| **問題診斷準確性** | ⭐⭐⭐⭐☆ (4/5) | 正確識別了「302 重定向」和「Session 缺失」問題 |
| **修復方案可行性** | ⭐⭐⭐⭐⭐ (5/5) | Headers 注入方案在技術上完全可行 |
| **實際實現程度** | ⭐☆☆☆☆ (1/5) | **幾乎完全未實現**，只移除了部分阻擋邏輯 |
| **文檔與代碼一致性** | ⭐☆☆☆☆ (1/5) | API 簽名、函數邏輯完全不匹配 |
| **測試工具完整性** | ⭐☆☆☆☆ (1/5) | 提到的測試腳本不存在 |
| **故障排除實用性** | ⭐⭐⭐⭐☆ (4/5) | FAQ 部分描述的問題和解決方案有參考價值 |

**總體評分**: ⭐⭐⭐☆☆ (3/5)  
**評語**: **文檔是優秀的設計規劃書，但與當前實現嚴重脫節**

---

## 🎯 實施建議

### 短期行動（修復當前問題）

#### 1. 添加請求計數器
```go
// 在 scanSingleTarget 或 testSSRF 中添加
atomic.AddInt64(&requestsCount, 1)  // 使用 atomic 保證並發安全
```

#### 2. 添加請求執行日誌
```go
// 在 client.Do(req) 前後添加
d.logger.Debug("準備發送 HTTP 請求", zap.String("url", testURL))
resp, err := d.client.Do(req)
d.logger.Debug("請求完成", 
    zap.Error(err),
    zap.Int("status", resp.StatusCode),
)
```

#### 3. 驗證靶場連接性
```powershell
# 在執行掃描前先確認 WebGoat 可訪問
Test-NetConnection -ComputerName localhost -Port 8080
```

---

### 中期行動（實現文檔描述的功能）

#### 階段 1: 修改數據結構
```go
// internal/common/types.go
type ScanRequest struct {
    ScanID      string            `json:"scan_id"`
    Targets     []string          `json:"targets"`
    Headers     map[string]string `json:"headers"`  // ← 新增
    Concurrency int               `json:"concurrency"`
    Timeout     int               `json:"timeout"`
}
```

#### 階段 2: 修改函數簽名
```go
// internal/ssrf/detector/ssrf.go
func (d *SSRFDetector) Scan(
    ctx context.Context, 
    targets []string,
    headers map[string]string,  // ← 新增參數
    config *common.ScannerConfig,
) ([]common.Asset, error)

func (d *SSRFDetector) scanSingleTarget(
    ctx context.Context, 
    target string,
    headers map[string]string,  // ← 新增參數
) ([]common.Asset, error)
```

#### 階段 3: 注入 Headers
```go
// 在 testSSRF 函數中
req, err := http.NewRequest("GET", testURL, nil)
if err != nil {
    return nil
}

// 注入自定義 Headers
for key, value := range headers {
    req.Header.Set(key, value)
    d.logger.Debug("注入 Header", 
        zap.String("key", key),
        zap.String("value", value[:min(len(value), 20)] + "..."),
    )
}
```

#### 階段 4: 更新主程序
```go
// cmd/ssrf-scanner/main.go
assets, err := scanner.Scan(
    ctx, 
    request.Targets,
    request.Headers,  // ← 傳遞 Headers
    config,
)
```

#### 階段 5: 創建測試腳本
```powershell
# test_webgoat_auth.ps1
$sessionId = Read-Host "請輸入 JSESSIONID"
$json = @{
    scan_id = "auth_test"
    targets = @("http://localhost:8080/WebGoat/SSRF/task1")
    headers = @{
        Cookie = "JSESSIONID=$sessionId"
    }
    concurrency = 1
    timeout = 10
} | ConvertTo-Json

$json | .\bin\ssrf-scanner.exe 2>&1 | Tee-Object test_auth.log
```

---

### 長期行動（架構優化）

1. **引入配置文件支持**: 將 Headers、代理設置等移到 YAML/JSON 配置文件
2. **會話管理器**: 自動處理 Cookie 過期和刷新
3. **認證插件系統**: 支持 OAuth2、JWT、API Key 等多種認證方式
4. **請求記錄器**: 保存所有 HTTP 請求和響應到文件，便於事後分析

---

## 🔚 結論

**Word 文檔的價值**:
- ✅ 提供了清晰的產品願景
- ✅ 詳細描述了預期的用戶體驗
- ✅ 包含完整的測試流程和故障排除指南

**實施現狀**:
- ❌ 核心功能（Headers 支持）完全未實現
- ⚠️ 存在更底層的問題（請求未發送，計數器為 0）
- ❌ 測試工具鏈不完整

**下一步行動**:
1. **優先級 P0**: 修復 `requests_made: 0` 問題，確保 HTTP 請求真正發送
2. **優先級 P1**: 實現 Headers 參數傳遞鏈（API 簽名 → 函數調用 → HTTP 請求）
3. **優先級 P2**: 創建文檔中提到的測試腳本
4. **優先級 P3**: 完善文檔，使其與實際代碼保持同步

**建議**:  
文檔可作為開發路線圖使用，但**不應作為當前系統功能的說明文檔**，需明確標註為「設計規劃」或「開發中功能」。
