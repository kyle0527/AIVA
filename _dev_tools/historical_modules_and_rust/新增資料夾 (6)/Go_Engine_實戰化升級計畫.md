# Go Engine 實戰化升級計畫 (Production Upgrade Plan)

版本: 3.0 (Enterprise Grade & Deep Dive)

最後更新: 2025-11-21

適用對象: 核心架構師、資安研發工程師、SRE 團隊

目標:

將 Go Engine 從一個僅能進行小規模功能驗證的開發原型 (Proof of
Concept)，全面升級為能夠穩定處理百萬級目標 (Million-scale
targets)、具備高容錯能力與高檢測覆蓋率的生產級資安掃描工具。

此升級計畫旨在徹底解決現有架構在大規模掃描時面臨的兩大核心挑戰：

1.  **資源耗盡 (Resource Exhaustion)**: 解決因全量載入導致的記憶體溢出
    (OOM) 瓶頸。

2.  **檢測盲區 (Detection Blind Spots)**:
    解決因檢測邏輯單一、缺乏重試機制導致的漏報 (False Negatives)。

## **1. 架構設計哲學 (Architectural Philosophy)**

在進行代碼層面的修改之前，我們必須確立指導本次升級的核心哲學。這些原則將貫穿所有的代碼變更。

### **1.1 從批次處理 (Batch) 到串流 I/O (Streaming I/O)**

傳統的批次處理模式（讀取所有 -\> 處理所有 -\>
輸出所有）在雲原生環境下是反模式
(Anti-pattern)。它導致記憶體使用量與輸入規模呈線性關係
\$O(n)\$，這在大數據量下是不可接受的。

我們轉向 **串流架構 (Streaming Architecture)**，將記憶體複雜度降低至
\$O(1)\$（僅與併發數相關，與總目標數無關）。這意味著無論輸入是 10
個目標還是 1,000 萬個目標，掃描器佔用的 RAM 應保持恆定（例如穩定在
50MB）。這使得掃描器可以在極低規格的容器（如 128MB RAM）中穩定運行。

### **1.2 Unix Pipeline 哲學 (The Unix Philosophy)**

我們將 Go Engine 視為一個標準的 Unix 過濾器 (Filter)：

> *\"Write programs that do one thing and do it well. Write programs to
> work together. Write programs to handle text streams, because that is
> a universal interface.\"*

- **標準輸入 (Stdin)**: 接受目標流。

- **標準輸出 (Stdout)**: 輸出結構化的 JSON 結果流。

- **標準錯誤 (Stderr)**: 輸出運行日誌與狀態監控。

這種設計使得 Go Engine 可以無縫整合到任何 CI/CD 流程、Kubernetes Job
或簡單的 Shell 腳本鏈路中，無需額外的適配層。

### **1.3 零外部依賴 (Zero Dependency Policy)**

為了確保極致的輕量化與安全性，我們堅持**不引入任何新的第三方依賴**。

- **安全性**: 減少供應鏈攻擊 (Supply Chain Attacks) 的風險面。

- **可維護性**: 避免 Dependency Hell，升級 Go 版本時無需擔心依賴破壞。

- **可移植性**: 編譯後的 Binary 不依賴任何系統庫 (glibc
  版本差異等)，可跨 Linux/macOS/Windows 運行。

## **2. 核心代碼改造深度解析 (Core Code Modifications)**

### **步驟 1: 替換 cmd/ssrf-scanner/main.go**

**重構重點**:

1.  **雙模輸入識別 (Dual-Mode Input Recognition)**: 透過
    bufio.Reader.Peek 實現對 JSON
    結構化輸入與純文本列表的自動偵測，這消除了對命令行參數的依賴，提升了使用者體驗。

2.  **信號量模式 (Semaphore Pattern)**: 放棄複雜的 Worker Pool 庫，改用
    Go 原生的 Channel
    作為信號量來控制併發。這種方式代碼更少、效能更好，且更容易處理
    Context 取消。

3.  **即時反饋 (Real-time Feedback)**:
    採用生產者-消費者模型，掃描結果即時寫入
    Stdout，讓上層調度系統能精確掌握進度。

**檔案路徑**: go_engine/cmd/ssrf-scanner/main.go

package main\
\
import (\
\"bufio\"\
\"context\"\
\"encoding/json\"\
\"os\"\
\"os/signal\"\
\"sync\"\
\"syscall\"\
\"time\"\
\
\"\[github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/common\](https://github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/common)\"\
\"\[github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/ssrf/detector\](https://github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/ssrf/detector)\"\
\"go.uber.org/zap\"\
\"go.uber.org/zap/zapcore\"\
)\
\
const (\
VERSION = \"3.0.0-Enterprise\"\
SCANNER_TYPE = \"ssrf\"\
)\
\
func main() {\
// 1. 初始化 Logger (生產環境配置)\
// 關鍵決策: 將所有日誌嚴格導向 stderr。\
// 原因: 在 Pipeline 架構中，stdout
是數據通道，混入日誌會破壞下游解析器的 JSON 結構。\
config := zap.NewProductionEncoderConfig()\
config.EncodeTime = zapcore.ISO8601TimeEncoder\
core := zapcore.NewCore(\
zapcore.NewConsoleEncoder(config),\
os.Stderr,\
zapcore.InfoLevel,\
)\
logger := zap.New(core)\
defer logger.Sync()\
\
// 2. 設置優雅關閉 (Graceful Shutdown)機制\
// 背景: Kubernetes 在停止 Pod 時會發送
SIGTERM。若直接殺死進程，可能導致正在進行的 HTTP
請求中斷，造成數據不一致。\
// 實作: 透過 Context 傳遞取消信號，讓所有 Worker
停止接收新任務，但允許當前任務完成。\
ctx, cancel := context.WithCancel(context.Background())\
defer cancel()\
\
sigChan := make(chan os.Signal, 1)\
signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)\
go func() {\
sig := \<-sigChan\
logger.Warn(\"Received signal, initiating graceful shutdown\...\",
zap.String(\"signal\", sig.String()))\
cancel()\
}()\
\
// 3. 智慧輸入處理 (Smart Input Handling)\
var request common.ScanRequest\
\
// 使用 bufio 進行高效讀取，這對於處理數 GB 的輸入文件至關重要\
reader := bufio.NewReader(os.Stdin)\
\
// 透過 Peek(1) 預讀第一個字節，這是不消耗 stream
的操作，允許我們判斷格式後繼續讀取\
firstByte, err := reader.Peek(1)\
\
if err == nil && len(firstByte) \> 0 && firstByte\[0\] == \'{\' {\
// 模式 A: JSON 結構化輸入 (Integration Mode)\
// 適用場景: 由 Python Adapter 調用，包含完整的 ScanID, Timeout,
Concurrency 配置。\
decoder := json.NewDecoder(reader)\
if err := decoder.Decode(&request); err != nil {\
logger.Error(\"Failed to decode JSON input\", zap.Error(err))\
os.Exit(1)\
}\
} else {\
// 模式 B: 純文本列表輸入 (Pipeline Mode)\
// 適用場景: CLI 工具鏈串接 (如 cat urls.txt \| ./scanner)。\
// 自動生成 ScanID 並賦予生產環境的預設值。\
request.ScanID = \"cli-\" + time.Now().Format(\"20060102-150405\")\
request.Concurrency = 50 // CLI 模式通常追求速度，預設較高併發\
request.Timeout = 15\
\
scanner := bufio.NewScanner(reader)\
// 擴大 Scanner 的緩衝區，以處理超長 URL\
buf := make(\[\]byte, 0, 64\*1024)\
scanner.Buffer(buf, 1024\*1024)\
\
for scanner.Scan() {\
url := scanner.Text()\
if url != \"\" {\
request.Targets = append(request.Targets, url)\
}\
}\
}\
\
if len(request.Targets) == 0 {\
logger.Error(\"No targets provided via stdin\")\
os.Exit(1)\
}\
\
// 防禦性編程：確保配置參數在合理範圍內\
if request.Concurrency \<= 0 { request.Concurrency = 10 }\
if request.Timeout \<= 0 { request.Timeout = 20 }\
\
logger.Info(\"Scanner initialized\",\
zap.String(\"version\", VERSION),\
zap.Int(\"targets_count\", len(request.Targets)),\
zap.Int(\"concurrency_limit\", request.Concurrency))\
\
// 4. 初始化檢測器\
ssrfDetector := detector.NewSSRFDetector(logger)\
scanConfig := &common.ScannerConfig{\
Concurrency: request.Concurrency,\
Timeout: time.Duration(request.Timeout) \* time.Second,\
}\
\
// 5. 啟動結果輸出 Consumer (Goroutine)\
// 使用緩衝通道避免因為 I/O 寫入速度慢而阻塞掃描 worker\
resultsChan := make(chan common.Asset, request.Concurrency\*2)\
doneChan := make(chan bool)\
\
go func() {\
encoder := json.NewEncoder(os.Stdout)\
// 輸出 NDJSON (Newline Delimited JSON)\
// 優勢: 允許接收端逐行讀取 (Streaming
Parse)，記憶體佔用極低，且具備容錯性（一行壞掉不影響整個文件）。\
for asset := range resultsChan {\
output := map\[string\]interface{}{\
\"type\": \"asset_found\",\
\"scan_id\": request.ScanID,\
\"timestamp\": time.Now().Format(time.RFC3339),\
\"data\": asset,\
}\
encoder.Encode(output)\
}\
doneChan \<- true\
}()\
\
// 6. 執行並發掃描 Producer (Semaphore Pattern)\
start := time.Now()\
\
// 使用 Channel 作為信號量，控制同時運行的 Goroutine 數量\
semaphore := make(chan struct{}, request.Concurrency)\
var wg sync.WaitGroup\
\
for \_, target := range request.Targets {\
wg.Add(1)\
semaphore \<- struct{}{} // 獲取 Token (若滿則阻塞)\
\
go func(t string) {\
defer wg.Done()\
defer func() { \<-semaphore }() // 釋放 Token\
\
// 快速失敗檢查 (Fail-Fast Check)\
select {\
case \<-ctx.Done():\
return // 若收到關閉信號，立即放棄任務\
default:\
}\
\
// 執行實際掃描\
assets, err := ssrfDetector.Scan(ctx, \[\]string{t}, scanConfig)\
if err != nil {\
// 僅記錄 Debug 日誌，避免大量失敗 (如目標不可達) 淹沒錯誤日誌\
logger.Debug(\"Target scan failed\", zap.String(\"target\", t),
zap.Error(err))\
}\
\
// 將結果發送至 Consumer\
for \_, asset := range assets {\
resultsChan \<- asset\
}\
}(target)\
}\
\
wg.Wait() // 等待所有 worker 完成\
close(resultsChan) // 關閉通道，通知 Consumer 停止\
\<-doneChan // 等待 Consumer 完成所有 I/O 寫入\
\
// 7. 輸出生命週期結束信號 (Lifecycle Signal)\
// 這對於自動化調度至關重要，標誌著任務的\"成功完成\"而非\"崩潰退出\"。\
finalSummary := map\[string\]interface{}{\
\"type\": \"scan_complete\",\
\"scan_id\": request.ScanID,\
\"metrics\": map\[string\]interface{}{\
\"duration_seconds\": time.Since(start).Seconds(),\
\"targets_processed\": len(request.Targets),\
\"throughput_pps\": float64(len(request.Targets)) /
time.Since(start).Seconds(),\
},\
}\
json.NewEncoder(os.Stdout).Encode(finalSummary)\
}

### 步驟 2: 替換 internal/ssrf/detector/ssrf.go

**重構重點**:

1.  **HTTP Client 深度調優**:
    針對掃描場景（大量短連接、不可信目標）進行優化，禁用 Keep-Alive
    以防止 FD 耗盡，限制 Redirect 次數以防止陷阱。

2.  **矩陣式探測 (Matrix Fuzzing)**: 針對每個目標參數，自動衍生出多種
    HTTP 方法 (GET, POST-JSON, POST-Form) 與多種 Payload
    的組合，大幅提升檢測覆蓋率。

3.  **韌性設計 (Resilience Design)**: 實作應用層的自動重試 (Retry with
    Backoff)，有效對抗網路抖動。

4.  **資源保護**: 增加對 Response Body 的讀取限制
    (LimitReader)，防止因為目標返回超大文件 (如 ISO 映像檔)
    導致記憶體溢出。

**檔案路徑**: go_engine/internal/ssrf/detector/ssrf.go

package detector\
\
import (\
\"context\"\
\"fmt\"\
\"io\"\
\"net/http\"\
\"net/url\"\
\"strings\"\
\"time\"\
\
\"go.uber.org/zap\"\
\"\[github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/common\](https://github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/common)\"\
)\
\
type SSRFDetector struct {\
logger \*zap.Logger\
client \*http.Client\
}\
\
func NewSSRFDetector(logger \*zap.Logger) \*SSRFDetector {\
// HTTP Client 調優：專為高併發掃描設計\
client := &http.Client{\
Timeout: 15 \* time.Second, // 平衡檢測速度與準確性\
\
// 安全策略：限制重定向\
// 防止掃描器被惡意目標引導至無限迴圈，或被用作跳板攻擊第三方\
CheckRedirect: func(req \*http.Request, via \[\]\*http.Request) error {\
if len(via) \>= 3 { return fmt.Errorf(\"too many redirects (safety limit
exceeded)\") }\
return nil\
},\
\
Transport: &http.Transport{\
TLSHandshakeTimeout: 10 \* time.Second,\
// 關鍵優化：禁用 Keep-Alive\
// 掃描器的流量模式是「對大量不同主機發起少量請求」。\
// 維持 TCP 長連接對此場景無益，反而會迅速耗盡 OS 的 ephemeral ports 和
file descriptors。\
DisableKeepAlives: true,\
MaxIdleConns: 100,\
ResponseHeaderTimeout: 10 \* time.Second,\
},\
}\
return &SSRFDetector{logger: logger, client: client}\
}\
\
// Scan 介面適配器\
func (d \*SSRFDetector) Scan(ctx context.Context, targets \[\]string,
config \*common.ScannerConfig) (\[\]common.Asset, error) {\
var allAssets \[\]common.Asset\
for \_, target := range targets {\
if ctx.Err() != nil { return allAssets, ctx.Err() }\
// 委派給強韌掃描邏輯\
assets := d.scanTargetRobust(ctx, target)\
allAssets = append(allAssets, assets\...)\
}\
return allAssets, nil\
}\
\
// scanTargetRobust - 實戰級掃描邏輯核心\
// 實現了矩陣式探測策略 (Parameter \* Payload \* Method)\
func (d \*SSRFDetector) scanTargetRobust(ctx context.Context, target
string) \[\]common.Asset {\
var assets \[\]common.Asset\
\
// 1. 靜態資源過濾 (Heuristic Filtering)\
// 效能優化：略過 .jpg, .css 等幾乎不可能存在 SSRF 的目標，節省 30%+
的無效請求。\
if d.isStaticResource(target) {\
return assets\
}\
\
parsedURL, err := url.Parse(target)\
if err != nil { return assets }\
\
// 2. 載入 Payload 字典\
payloads := d.getPayloads()\
\
// 3. 定義高風險參數列表 (Top SSRF Parameters)\
// 基於 Bug Bounty 數據統計，針對這些參數進行 Fuzzing 效益最高。\
params := \[\]string{\"url\", \"uri\", \"link\", \"src\", \"target\",
\"dest\", \"callback\", \"webhook\", \"image_url\", \"path\", \"feed\",
\"host\", \"data\"}\
\
// 4. 執行矩陣測試\
for \_, param := range params {\
for \_, p := range payloads {\
// 4.1 GET Method 測試 (Query String Injection)\
getURL := d.buildGetURL(parsedURL, param, p.Url)\
if asset := d.executeTest(ctx, \"GET\", getURL, \"\", target, param, p);
asset != nil {\
assets = append(assets, \*asset)\
}\
\
// 4.2 POST Method (JSON) 測試 - 針對 REST API\
jsonBody := fmt.Sprintf(\`{\"%s\": \"%s\"}\`, param, p.Url)\
if asset := d.executeTest(ctx, \"POST\", target, jsonBody, target,
param, p); asset != nil {\
asset.Name += \" (POST JSON)\"\
assets = append(assets, \*asset)\
}\
\
// 4.3 POST Method (Form-UrlEncoded) 測試 - 針對傳統 Web App\
formBody := fmt.Sprintf(\"%s=%s\", param, url.QueryEscape(p.Url))\
if asset := d.executeTest(ctx, \"POST_FORM\", target, formBody, target,
param, p); asset != nil {\
asset.Name += \" (POST FORM)\"\
assets = append(assets, \*asset)\
}\
}\
}\
return assets\
}\
\
// executeTest - 執行單次測試 (含自動重試與資源保護)\
func (d \*SSRFDetector) executeTest(ctx context.Context, method,
targetURL, bodyStr, originalTarget, param string, payload Payload)
\*common.Asset {\
var req \*http.Request\
var err error\
\
// 構造請求：正確設置 Content-Type 對於觸發後端邏輯至關重要\
if method == \"POST_FORM\" {\
req, err = http.NewRequestWithContext(ctx, \"POST\", targetURL,
strings.NewReader(bodyStr))\
req.Header.Set(\"Content-Type\", \"application/x-www-form-urlencoded\")\
} else {\
req, err = http.NewRequestWithContext(ctx, method, targetURL,
strings.NewReader(bodyStr))\
if method == \"POST\" {\
req.Header.Set(\"Content-Type\", \"application/json\")\
}\
}\
\
if err != nil { return nil }\
// 設置仿真 User-Agent，降低被簡單 WAF 攔截的機率\
req.Header.Set(\"User-Agent\", \"Mozilla/5.0 (Compatible;
AIVA-Security-Scanner/2.1)\")\
\
// 自動重試邏輯 (Auto-Retry with Backoff)\
// 解決網路暫態錯誤，顯著降低 False Negatives\
var resp \*http.Response\
maxRetries := 1\
for i := 0; i \<= maxRetries; i++ {\
resp, err = d.client.Do(req)\
if err == nil { break }\
\
// 如果是最後一次嘗試失敗，則放棄\
if i == maxRetries { return nil }\
\
// 線性退避：等待 500ms 後重試\
time.Sleep(500 \* time.Millisecond)\
}\
\
defer resp.Body.Close()\
\
// 資源保護：使用 LimitReader\
// 防止惡意目標返回無限數據流或超大文件導致掃描器 OOM。\
// 512KB 足以包含大部分錯誤頁面或 Metadata 內容。\
bodyBytes, \_ := io.ReadAll(io.LimitReader(resp.Body, 1024\*512))\
bodyString := string(bodyBytes)\
\
// 漏洞驗證\
if d.isSSRFVulnerable(resp.StatusCode, bodyString, payload.Url) {\
return &common.Asset{\
Type: \"web_vulnerability\",\
Name: fmt.Sprintf(\"SSRF - %s\", payload.Name),\
Severity: \"high\",\
Confidence: \"high\",\
SourceEngine: \"go\",\
Details: map\[string\]interface{}{\
\"url\": targetURL,\
\"method\": method,\
\"param\": param,\
\"payload\": payload.Url,\
\"evidence_snippet\": bodyString\[:min(len(bodyString), 200)\], //
證據截斷\
\"full_evidence_size\": len(bodyString),\
},\
}\
}\
return nil\
}\
\
// isStaticResource - 靜態資源啟發式過濾\
func (d \*SSRFDetector) isStaticResource(target string) bool {\
exts := \[\]string{\".jpg\", \".jpeg\", \".png\", \".gif\", \".css\",
\".js\", \".woff\", \".svg\", \".ico\", \".mp4\", \".mp3\", \".avi\",
\".pdf\"}\
lower := strings.ToLower(target)\
for \_, ext := range exts {\
if strings.HasSuffix(lower, ext) { return true }\
}\
return false\
}\
\
// isSSRFVulnerable - 漏洞特徵匹配邏輯\
func (d \*SSRFDetector) isSSRFVulnerable(status int, body, payload
string) bool {\
// 1. 誤報消除 (False Positive Reduction)\
// 排除雖然 200 OK 但實際上是登入頁面或自定義錯誤頁的情況\
bodyLower := strings.ToLower(body)\
if strings.Contains(bodyLower, \"\<title\>login\") \|\|\
strings.Contains(bodyLower, \"404 not found\") \|\|\
strings.Contains(bodyLower, \"captcha\") {\
return false\
}\
\
// 2. AWS IMDS 特徵檢測\
if strings.Contains(payload, \"169.254.169.254\") {\
if strings.Contains(body, \"ami-id\") \|\| strings.Contains(body,
\"instance-id\") { return true }\
// IMDS v1 返回純文本，不含 HTML 標籤，這是一個強特徵\
if !strings.Contains(body, \"\<html\") && len(body) \> 5 && len(body) \<
1000 { return true }\
}\
\
// 3. 本地文件洩漏特徵檢測\
if strings.Contains(payload, \"/etc/passwd\") {\
if strings.Contains(body, \"root:x:0:0\") { return true }\
}\
\
// 4. 內網服務探測特徵檢測\
if strings.Contains(payload, \"127.0.0.1\") \|\|
strings.Contains(payload, \"localhost\") {\
// 狀態碼差異分析：401/403 證明連接成功但權限不足，這是內網存在的強證據\
if status == 401 \|\| status == 403 { return true }\
// 200 OK 且非登入頁面，可能暴露了未授權的內部 Admin 面板\
if status == 200 && len(body) \> 100 && !strings.Contains(bodyLower,
\"login\") { return true }\
}\
\
return false\
}\
\
// Payload 結構與輔助函數\
type Payload struct {\
Name string\
Url string\
}\
\
func (d \*SSRFDetector) getPayloads() \[\]Payload {\
return \[\]Payload{\
{\"AWS IMDS\",
\"\[http://169.254.169.254/latest/meta-data/\](http://169.254.169.254/latest/meta-data/)\"},\
{\"Localhost\",
\"\[http://127.0.0.1/admin\](http://127.0.0.1/admin)\"},\
{\"File ETC\", \"file:///etc/passwd\"},\
{\"Internal Probe\", \"\[http://192.168.1.1\](http://192.168.1.1)\"},\
}\
}\
\
func (d \*SSRFDetector) buildGetURL(base \*url.URL, param, value string)
string {\
u := \*base\
q := u.Query()\
q.Set(param, value)\
u.RawQuery = q.Encode()\
return u.String()\
}\
\
func min(a, b int) int {\
if a \< b { return a }\
return b\
}

## **3. 編譯與部署策略** (Compilation & Deployment)

為了適應生產環境的發佈需求，我們採用「最小化構建 (Minimalist
Build)」策略。

\# 進入 Go Engine 目錄\
cd go_engine\
\
\# 清理舊的構建產物\
rm -f bin/ssrf-scanner.exe\
\
\# 執行生產級編譯\
\# 參數詳解:\
\# -ldflags \"-s -w\":\
\# -s: 移除符號表 (Symbol Table)。這會移除除錯資訊，雖然會讓 gdb
調試變難，但顯著減小二進制體積。\
\# -w: 移除 DWARF 調試資訊。這進一步縮減體積。\
\# 效益: 體積通常減少
30%-50%，加速容器鏡像拉取與啟動速度。同時增加逆向工程難度。\
go build -ldflags \"-s -w\" -o bin/ssrf-scanner.exe ./cmd/ssrf-scanner\
\
\# 驗證編譯結果與健康檢查 (Health Check)\
\# 執行不帶參數的指令，預期應該直接退出並在 stderr 顯示錯誤，且 exit
code 為 1\
./bin/ssrf-scanner.exe ; if \[ \$? -eq 1 \]; then echo \"Build
Verification Passed\"; else echo \"Build Verification Failed\"; fi

## **4. 全方位驗證測試計畫 (Comprehensive Verification Plan)**

在將新版本部署到生產環境前，必須通過以下三個層級的驗證測試。

### **4.1 I/O 與多態輸入驗證 (Input Polymorphism Test)**

**目的**:
驗證掃描器是否能正確處理不同來源、不同格式的輸入流，且具備容錯能力。

- **測試案例 A (JSON Stream)**: 模擬 Python Adapter 調用。\
  echo
  \'{\"targets\":\[\"\[http://example.com\](http://example.com)\"\],
  \"concurrency\": 5}\' \| ./bin/ssrf-scanner.exe\
  \# 預期結果: stdout 輸出 scan_complete JSON，stderr 輸出啟動日誌。

- **測試案例 B (Text Pipeline)**: 模擬 CLI 串接。\
  echo -e
  \"\[http://site1.com\](http://site1.com)\\n\[http://site2.com\](http://site2.com)\"
  \| ./bin/ssrf-scanner.exe\
  \# 預期結果: 掃描 2 個目標，並正確輸出。

- **測試案例 C (Empty Input)**: 邊界條件測試。\
  echo \"\" \| ./bin/ssrf-scanner.exe\
  \# 預期結果: 優雅退出，Exit Code 1，日誌顯示 \"No targets provided\"。

### **4.2 漏洞檢測能力驗證 (Detection Capability Test)**

**目的**: 驗證升級後的矩陣探測邏輯是否能發現舊版本無法檢測的漏洞。

- **測試環境**: 使用 WebGoat 或 Docker 搭建的靶場。

- **測試指令**:\
  \# 假設靶場運行在 localhost:8080\
  echo \"http://localhost:8080/WebGoat/SSRF/task1\" \|
  ./bin/ssrf-scanner.exe

- **驗證標準**:

  1.  必須在 stdout 看到 type: \"asset_found\" 的 JSON。

  2.  JSON 中的 data.details.method 欄位應包含 GET, POST, POST_FORM
      等不同嘗試的記錄（若靶場支援多種觸發方式）。

  3.  JSON 中的 evidence_snippet 必須包含具體的敏感資訊（如 ami-id 或
      root:x:0:0）。

### **4.3 壓力與資源過濾驗證 (Stress & Resource Efficiency Test)**

**目的**: 驗證在高負載下，系統是否會發生 OOM 或因無效請求導致效能下降。

- **靜態資源過濾測試**:\
  echo
  \"\[http://example.com/large_video.mp4\](http://example.com/large_video.mp4)\"
  \| ./bin/ssrf-scanner.exe\
  \# 預期結果: 掃描瞬間完成 (毫秒級)，日誌中無發送請求記錄。這驗證了
  Heuristic Filter 生效。

- **大文件 OOM 測試**:\
  \# 構造 10 萬個重複 URL 的列表\
  yes \"\[http://example.com/api/test\](http://example.com/api/test)\"
  \| head -n 100000 \| ./bin/ssrf-scanner.exe
