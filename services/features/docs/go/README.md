# 🐹 Go開發模組指南

**導航**: [← 返回文檔中心](../README.md) | [← 返回主模組](../../README.md)

---

## 📑 目錄

- [Go模組架構](#go模組架構)
- [開發環境配置](#開發環境配置)
- [編碼規範與最佳實踐](#編碼規範與最佳實踐)
- [併發程式設計](#併發程式設計)
- [測試與基準測試](#測試與基準測試)
- [效能優化](#效能優化)
- [部署與建置](#部署與建置)

---

## 🏗️ Go模組架構

AIVA Features的Go模組專注於高併發、高效能的檢測任務，特別適合需要大量並行處理的場景。

### 📊 **Go代碼統計**
- **總檔案數**: 11個Go檔案
- **總代碼行數**: 1,796行 (占13%)
- **平均檔案大小**: 163行/檔案
- **主要模組**: 1個認證檢測模組 (function_authn_go)

### **標準目錄結構**
```
function_authn_go/           # Go認證檢測模組
├── go.mod                  # Go模組定義
├── go.sum                  # 依賴鎖定檔案
├── Dockerfile             # Docker建置檔案
├── README.md              # 模組文檔
├── cmd/                   # 命令行程式
│   └── worker/
│       └── main.go       # Worker主程式
├── internal/              # 內部實現(私有)
│   ├── brute_force/      # 暴力破解檢測
│   │   ├── detector.go
│   │   └── config.go
│   ├── token_test/       # 令牌測試
│   │   ├── jwt.go
│   │   └── oauth.go
│   ├── weak_config/      # 弱配置檢測
│   │   ├── analyzer.go
│   │   └── rules.go
│   └── common/           # 共用組件
│       ├── types.go
│       ├── errors.go
│       └── utils.go
├── pkg/                   # 公開API(可導出)
│   ├── client/           # 客戶端接口
│   ├── models/           # 數據模型
│   └── config/           # 配置管理
├── tests/                 # 測試檔案
│   ├── integration/      # 整合測試
│   ├── unit/            # 單元測試
│   └── benchmarks/      # 基準測試
└── scripts/              # 建置腳本
    ├── build.sh
    └── test.sh
```

---

## ⚙️ 開發環境配置

### **Go版本要求**
- **最低版本**: Go 1.19+
- **推薦版本**: Go 1.21+
- **支援平台**: Windows, Linux, macOS

### **環境設置**
```bash
# 安裝Go (Windows)
# 下載並安裝 https://golang.org/dl/

# 驗證安裝
go version

# 設置環境變數
export GO111MODULE=on
export GOPROXY=https://goproxy.cn,direct  # 中國用戶
export GOPRIVATE=github.com/yourcompany/*  # 私有模組

# 工作目錄設置
mkdir -p $GOPATH/src/github.com/aiva/features
cd $GOPATH/src/github.com/aiva/features/function_authn_go
```

### **模組初始化**
```bash
# 初始化Go模組
go mod init github.com/aiva/features/function_authn_go

# 添加依賴
go get github.com/golang-jwt/jwt/v5
go get github.com/gin-gonic/gin
go get github.com/stretchr/testify
go get go.uber.org/zap          # 日誌庫
go get github.com/spf13/viper   # 配置管理
go get golang.org/x/crypto      # 加密庫

# 整理依賴
go mod tidy
```

### **開發工具**
```bash
# 安裝開發工具
go install golang.org/x/tools/cmd/goimports@latest
go install golang.org/x/lint/golint@latest
go install honnef.co/go/tools/cmd/staticcheck@latest
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# VS Code Go擴展
code --install-extension golang.go
```

### **VS Code配置**
```json
{
    "go.useLanguageServer": true,
    "go.formatTool": "goimports",
    "go.lintTool": "golangci-lint",
    "go.lintFlags": [
        "--fast"
    ],
    "go.testFlags": ["-v"],
    "go.testTimeout": "10s",
    "go.coverOnSave": true,
    "go.coverageDecorator": {
        "type": "gutter",
        "coveredHighlightColor": "rgba(64,128,128,0.5)",
        "uncoveredHighlightColor": "rgba(128,64,64,0.25)"
    },
    "[go]": {
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

---

## 📝 編碼規範與最佳實踐

### **命名規範**

#### **包命名**
```go
// 好的包名 - 簡潔、描述性
package detector
package bruteforce
package tokentest

// 避免的包名
package detectorutils  // 太長
package util          // 太通用
package pkg           // 無意義
```

#### **變數和函數命名**
```go
// 變數命名 - camelCase
var userName string
var maxRetryCount int
var isAuthEnabled bool

// 常數命名 - PascalCase或ALL_CAPS
const (
    DefaultTimeout = 30 * time.Second
    MaxWorkers     = 100
)

// 函數命名 - PascalCase(公開) 或 camelCase(私有)
func DetectVulnerability(target string) error {  // 公開
    return detectInternal(target)
}

func detectInternal(target string) error {  // 私有
    // 實現
}

// 結構體命名 - PascalCase
type VulnerabilityResult struct {
    Type        string    `json:"type"`
    Severity    string    `json:"severity"`
    Confidence  float64   `json:"confidence"`
    Location    string    `json:"location"`
    Timestamp   time.Time `json:"timestamp"`
}
```

### **錯誤處理模式**
```go
package errors

import (
    "errors"
    "fmt"
)

// 自定義錯誤類型
var (
    ErrInvalidTarget    = errors.New("invalid target URL")
    ErrConnectionFailed = errors.New("connection failed")
    ErrTimeout         = errors.New("operation timeout")
)

// 錯誤包裝
func DetectVulnerability(target string) (*Result, error) {
    if target == "" {
        return nil, fmt.Errorf("target cannot be empty: %w", ErrInvalidTarget)
    }
    
    result, err := performDetection(target)
    if err != nil {
        return nil, fmt.Errorf("detection failed for %s: %w", target, err)
    }
    
    return result, nil
}

// 錯誤檢查模式
func processTargets(targets []string) error {
    for _, target := range targets {
        if err := validateTarget(target); err != nil {
            return fmt.Errorf("invalid target %s: %w", target, err)
        }
        
        result, err := DetectVulnerability(target)
        if err != nil {
            // 決定是否繼續處理其他目標
            if errors.Is(err, ErrTimeout) {
                continue // 超時錯誤可以跳過
            }
            return err // 其他錯誤停止處理
        }
        
        handleResult(result)
    }
    return nil
}
```

### **並發安全模式**
```go
package detector

import (
    "context"
    "sync"
    "time"
)

// 線程安全的檢測器
type SafeDetector struct {
    mu       sync.RWMutex
    cache    map[string]*Result
    config   *Config
    workers  int
}

func NewSafeDetector(config *Config) *SafeDetector {
    return &SafeDetector{
        cache:   make(map[string]*Result),
        config:  config,
        workers: config.Workers,
    }
}

// 線程安全的快取操作
func (d *SafeDetector) GetFromCache(key string) (*Result, bool) {
    d.mu.RLock()
    defer d.mu.RUnlock()
    
    result, exists := d.cache[key]
    return result, exists
}

func (d *SafeDetector) SetCache(key string, result *Result) {
    d.mu.Lock()
    defer d.mu.Unlock()
    
    d.cache[key] = result
}

// Worker Pool 模式
func (d *SafeDetector) ProcessTargets(ctx context.Context, targets []string) <-chan *Result {
    resultChan := make(chan *Result, len(targets))
    jobs := make(chan string, len(targets))
    
    // 啟動workers
    var wg sync.WaitGroup
    for i := 0; i < d.workers; i++ {
        wg.Add(1)
        go d.worker(ctx, &wg, jobs, resultChan)
    }
    
    // 發送任務
    go func() {
        defer close(jobs)
        for _, target := range targets {
            select {
            case jobs <- target:
            case <-ctx.Done():
                return
            }
        }
    }()
    
    // 關閉結果通道
    go func() {
        wg.Wait()
        close(resultChan)
    }()
    
    return resultChan
}

func (d *SafeDetector) worker(ctx context.Context, wg *sync.WaitGroup, jobs <-chan string, results chan<- *Result) {
    defer wg.Done()
    
    for {
        select {
        case target, ok := <-jobs:
            if !ok {
                return
            }
            
            // 檢查快取
            if cached, exists := d.GetFromCache(target); exists {
                results <- cached
                continue
            }
            
            // 執行檢測
            result, err := d.detect(ctx, target)
            if err != nil {
                // 處理錯誤
                continue
            }
            
            // 更新快取
            d.SetCache(target, result)
            
            // 發送結果
            select {
            case results <- result:
            case <-ctx.Done():
                return
            }
            
        case <-ctx.Done():
            return
        }
    }
}
```

---

## 🚀 併發程式設計

### **Goroutine管理**
```go
package concurrent

import (
    "context"
    "runtime"
    "sync"
    "time"
)

// 資源池管理
type WorkerPool struct {
    workers    int
    jobQueue   chan Job
    resultPool sync.Pool
    wg         sync.WaitGroup
    ctx        context.Context
    cancel     context.CancelFunc
}

type Job struct {
    ID     string
    Target string
    Config *Config
}

type Result struct {
    JobID       string
    Target      string
    Found       bool
    Severity    string
    Confidence  float64
    ProcessTime time.Duration
}

func NewWorkerPool(workers int) *WorkerPool {
    if workers <= 0 {
        workers = runtime.NumCPU()
    }
    
    ctx, cancel := context.WithCancel(context.Background())
    
    return &WorkerPool{
        workers:  workers,
        jobQueue: make(chan Job, workers*2), // 緩衝佇列
        resultPool: sync.Pool{
            New: func() interface{} {
                return &Result{}
            },
        },
        ctx:    ctx,
        cancel: cancel,
    }
}

func (wp *WorkerPool) Start() <-chan *Result {
    resultChan := make(chan *Result, wp.workers)
    
    // 啟動worker goroutines
    for i := 0; i < wp.workers; i++ {
        wp.wg.Add(1)
        go wp.worker(resultChan)
    }
    
    // 監控goroutine，負責關閉resultChan
    go func() {
        wp.wg.Wait()
        close(resultChan)
    }()
    
    return resultChan
}

func (wp *WorkerPool) worker(resultChan chan<- *Result) {
    defer wp.wg.Done()
    
    for {
        select {
        case job, ok := <-wp.jobQueue:
            if !ok {
                return // jobQueue已關閉
            }
            
            // 從對象池獲取結果對象
            result := wp.resultPool.Get().(*Result)
            defer wp.resultPool.Put(result) // 使用完畢後歸還
            
            // 重置結果對象
            *result = Result{
                JobID:  job.ID,
                Target: job.Target,
            }
            
            start := time.Now()
            
            // 執行檢測任務
            if wp.detectVulnerability(job) {
                result.Found = true
                result.Severity = "High"
                result.Confidence = 0.95
            }
            
            result.ProcessTime = time.Since(start)
            
            // 發送結果
            select {
            case resultChan <- result:
            case <-wp.ctx.Done():
                return
            }
            
        case <-wp.ctx.Done():
            return
        }
    }
}

func (wp *WorkerPool) Submit(job Job) error {
    select {
    case wp.jobQueue <- job:
        return nil
    case <-wp.ctx.Done():
        return context.Canceled
    default:
        return errors.New("job queue is full")
    }
}

func (wp *WorkerPool) Shutdown() {
    close(wp.jobQueue) // 關閉任務佇列
    wp.cancel()        // 取消context
}
```

### **限流和速率控制**
```go
package ratelimit

import (
    "context"
    "golang.org/x/time/rate"
    "time"
)

// 令牌桶限流器
type RateLimiter struct {
    limiter *rate.Limiter
    burst   int
}

func NewRateLimiter(requestsPerSecond int, burst int) *RateLimiter {
    return &RateLimiter{
        limiter: rate.NewLimiter(rate.Limit(requestsPerSecond), burst),
        burst:   burst,
    }
}

func (rl *RateLimiter) Allow() bool {
    return rl.limiter.Allow()
}

func (rl *RateLimiter) Wait(ctx context.Context) error {
    return rl.limiter.Wait(ctx)
}

// 滑動窗口限流器
type SlidingWindowLimiter struct {
    windowSize time.Duration
    maxCount   int
    requests   []time.Time
    mu         sync.Mutex
}

func NewSlidingWindowLimiter(windowSize time.Duration, maxCount int) *SlidingWindowLimiter {
    return &SlidingWindowLimiter{
        windowSize: windowSize,
        maxCount:   maxCount,
        requests:   make([]time.Time, 0, maxCount),
    }
}

func (swl *SlidingWindowLimiter) Allow() bool {
    swl.mu.Lock()
    defer swl.mu.Unlock()
    
    now := time.Now()
    cutoff := now.Add(-swl.windowSize)
    
    // 清理過期請求
    var validRequests []time.Time
    for _, req := range swl.requests {
        if req.After(cutoff) {
            validRequests = append(validRequests, req)
        }
    }
    swl.requests = validRequests
    
    // 檢查是否超過限制
    if len(swl.requests) >= swl.maxCount {
        return false
    }
    
    // 記錄新請求
    swl.requests = append(swl.requests, now)
    return true
}
```

---

## 🧪 測試與基準測試

### **單元測試**
```go
package detector

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/mock"
)

// 測試結構體
func TestVulnerabilityDetector(t *testing.T) {
    detector := NewDetector(&Config{
        Timeout:    10 * time.Second,
        MaxWorkers: 5,
    })
    
    t.Run("有效目標檢測", func(t *testing.T) {
        target := "http://example.com/vulnerable"
        result, err := detector.Detect(context.Background(), target)
        
        require.NoError(t, err)
        assert.NotNil(t, result)
        assert.Equal(t, target, result.Target)
    })
    
    t.Run("無效目標處理", func(t *testing.T) {
        invalidTarget := "not-a-url"
        result, err := detector.Detect(context.Background(), invalidTarget)
        
        assert.Error(t, err)
        assert.Nil(t, result)
        assert.Contains(t, err.Error(), "invalid target")
    })
    
    t.Run("超時處理", func(t *testing.T) {
        ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
        defer cancel()
        
        target := "http://slow-server.com"
        result, err := detector.Detect(ctx, target)
        
        assert.Error(t, err)
        assert.Nil(t, result)
        assert.Contains(t, err.Error(), "timeout")
    })
}

// 表格驅動測試
func TestValidateURL(t *testing.T) {
    tests := []struct {
        name     string
        url      string
        expected bool
    }{
        {"有效HTTP URL", "http://example.com", true},
        {"有效HTTPS URL", "https://example.com", true},
        {"有效帶路徑URL", "https://example.com/path", true},
        {"無效協議", "ftp://example.com", false},
        {"無效格式", "not-a-url", false},
        {"空字符串", "", false},
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            result := ValidateURL(tt.url)
            assert.Equal(t, tt.expected, result)
        })
    }
}

// Mock測試
type MockHTTPClient struct {
    mock.Mock
}

func (m *MockHTTPClient) Get(url string) (*http.Response, error) {
    args := m.Called(url)
    return args.Get(0).(*http.Response), args.Error(1)
}

func TestDetectorWithMock(t *testing.T) {
    mockClient := new(MockHTTPClient)
    detector := &Detector{
        client: mockClient,
    }
    
    // 設置mock期望
    mockResponse := &http.Response{
        StatusCode: 200,
        Body:       ioutil.NopCloser(strings.NewReader("vulnerable response")),
    }
    mockClient.On("Get", "http://example.com").Return(mockResponse, nil)
    
    result, err := detector.Detect(context.Background(), "http://example.com")
    
    require.NoError(t, err)
    assert.True(t, result.Found)
    mockClient.AssertExpectations(t)
}
```

### **基準測試**
```go
package detector

import (
    "context"
    "testing"
)

func BenchmarkDetectSingle(b *testing.B) {
    detector := NewDetector(&Config{MaxWorkers: 1})
    target := "http://example.com"
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := detector.Detect(context.Background(), target)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkDetectConcurrent(b *testing.B) {
    detector := NewDetector(&Config{MaxWorkers: 10})
    targets := generateTargets(100)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        resultChan := detector.ProcessTargets(context.Background(), targets)
        for range resultChan {
            // 消費結果
        }
    }
}

// 記憶體使用基準測試
func BenchmarkMemoryUsage(b *testing.B) {
    b.ReportAllocs()
    
    detector := NewDetector(&Config{MaxWorkers: 5})
    
    for i := 0; i < b.N; i++ {
        result := &Result{
            Target:     "http://example.com",
            Found:      true,
            Confidence: 0.95,
        }
        
        // 模擬處理
        _ = result
    }
}

// 不同併發級別的比較
func BenchmarkConcurrencyLevels(b *testing.B) {
    concurrencyLevels := []int{1, 5, 10, 20, 50}
    targets := generateTargets(100)
    
    for _, workers := range concurrencyLevels {
        b.Run(fmt.Sprintf("Workers_%d", workers), func(b *testing.B) {
            detector := NewDetector(&Config{MaxWorkers: workers})
            
            b.ResetTimer()
            for i := 0; i < b.N; i++ {
                resultChan := detector.ProcessTargets(context.Background(), targets)
                for range resultChan {
                    // 消費結果
                }
            }
        })
    }
}
```

---

## 🎯 效能優化

### **記憶體優化**
```go
package optimization

import (
    "sync"
)

// 對象池優化
var resultPool = sync.Pool{
    New: func() interface{} {
        return &Result{
            Details: make(map[string]interface{}, 8), // 預分配容量
        }
    },
}

func GetResult() *Result {
    return resultPool.Get().(*Result)
}

func PutResult(r *Result) {
    // 重置對象狀態
    r.Reset()
    resultPool.Put(r)
}

func (r *Result) Reset() {
    r.Target = ""
    r.Found = false
    r.Confidence = 0
    
    // 清空map但保留容量
    for k := range r.Details {
        delete(r.Details, k)
    }
}

// 字串建構器優化
func BuildReport(results []*Result) string {
    var builder strings.Builder
    
    // 預估容量
    estimatedSize := len(results) * 100
    builder.Grow(estimatedSize)
    
    builder.WriteString("Detection Report\n")
    builder.WriteString("================\n")
    
    for _, result := range results {
        builder.WriteString(fmt.Sprintf("Target: %s\n", result.Target))
        builder.WriteString(fmt.Sprintf("Found: %v\n", result.Found))
        builder.WriteString("---\n")
    }
    
    return builder.String()
}

// slice預分配優化
func ProcessLargeDataset(data []string) []*Result {
    // 預分配容量，避免多次擴容
    results := make([]*Result, 0, len(data))
    
    for _, item := range data {
        if result := processItem(item); result != nil {
            results = append(results, result)
        }
    }
    
    return results
}
```

### **網路優化**
```go
package network

import (
    "crypto/tls"
    "net"
    "net/http"
    "time"
)

// 高效能HTTP客戶端
func NewOptimizedHTTPClient() *http.Client {
    transport := &http.Transport{
        // 連接池設置
        MaxIdleConns:        100,
        MaxIdleConnsPerHost: 20,
        MaxConnsPerHost:     30,
        
        // 超時設置
        IdleConnTimeout:     90 * time.Second,
        TLSHandshakeTimeout: 10 * time.Second,
        
        // 保持連接活躍
        DisableKeepAlives: false,
        
        // 自定義撥號器
        DialContext: (&net.Dialer{
            Timeout:   5 * time.Second,
            KeepAlive: 30 * time.Second,
        }).DialContext,
        
        // TLS配置優化
        TLSClientConfig: &tls.Config{
            InsecureSkipVerify: true, // 僅測試環境
            MinVersion:         tls.VersionTLS12,
        },
        
        // 響應標頭超時
        ResponseHeaderTimeout: 10 * time.Second,
        
        // 期望100Continue超時
        ExpectContinueTimeout: 1 * time.Second,
    }
    
    return &http.Client{
        Transport: transport,
        Timeout:   30 * time.Second,
        CheckRedirect: func(req *http.Request, via []*http.Request) error {
            // 限制重定向次數
            if len(via) >= 3 {
                return http.ErrUseLastResponse
            }
            return nil
        },
    }
}

// 連接池管理
type ConnectionPool struct {
    pool    chan net.Conn
    factory func() (net.Conn, error)
    close   func(net.Conn) error
}

func NewConnectionPool(size int, factory func() (net.Conn, error)) *ConnectionPool {
    return &ConnectionPool{
        pool:    make(chan net.Conn, size),
        factory: factory,
        close: func(conn net.Conn) error {
            return conn.Close()
        },
    }
}

func (cp *ConnectionPool) Get() (net.Conn, error) {
    select {
    case conn := <-cp.pool:
        return conn, nil
    default:
        return cp.factory()
    }
}

func (cp *ConnectionPool) Put(conn net.Conn) {
    select {
    case cp.pool <- conn:
    default:
        cp.close(conn) // 池已滿，關閉連接
    }
}
```

---

## 📦 部署與建置

### **建置腳本**
```bash
#!/bin/bash
# build.sh

set -e

# 環境設置
export CGO_ENABLED=0
export GOOS=linux
export GOARCH=amd64

# 版本信息
VERSION=$(git describe --tags --always --dirty)
BUILD_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse HEAD)

# 建置標誌
LDFLAGS="-X main.Version=${VERSION} -X main.BuildTime=${BUILD_TIME} -X main.GitCommit=${GIT_COMMIT} -s -w"

echo "Building AIVA Authentication Detector..."
echo "Version: ${VERSION}"
echo "Build Time: ${BUILD_TIME}"
echo "Git Commit: ${GIT_COMMIT}"

# 清理舊的建置
rm -rf dist/
mkdir -p dist/

# 建置不同平台的二進制檔案
PLATFORMS=("linux/amd64" "windows/amd64" "darwin/amd64" "darwin/arm64")

for PLATFORM in "${PLATFORMS[@]}"; do
    GOOS=${PLATFORM%/*}
    GOARCH=${PLATFORM#*/}
    OUTPUT_NAME="aiva-authn-detector-${GOOS}-${GOARCH}"
    
    if [ $GOOS = "windows" ]; then
        OUTPUT_NAME+='.exe'
    fi
    
    echo "Building for ${GOOS}/${GOARCH}..."
    env GOOS=$GOOS GOARCH=$GOARCH go build \
        -ldflags="$LDFLAGS" \
        -o dist/$OUTPUT_NAME \
        cmd/worker/main.go
done

echo "Build completed successfully!"
ls -la dist/
```

### **Docker建置**
```dockerfile
# Multi-stage build
FROM golang:1.21-alpine AS builder

# 安裝build依賴
RUN apk add --no-cache git ca-certificates tzdata

# 設置工作目錄
WORKDIR /app

# 複製go mod檔案
COPY go.mod go.sum ./

# 下載依賴
RUN go mod download

# 複製源代碼
COPY . .

# 建置應用程式
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-w -s" \
    -o main \
    cmd/worker/main.go

# 運行時映像
FROM scratch

# 從builder複製ca證書
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# 從builder複製時區資料
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo

# 從builder複製應用程式
COPY --from=builder /app/main /

# 暴露端口
EXPOSE 8080

# 健康檢查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["/main", "--health-check"]

# 設置用戶
USER 65534:65534

# 啟動應用程式
ENTRYPOINT ["/main"]
```

### **Makefile**
```makefile
.PHONY: build test clean docker help

# 變數定義
BINARY_NAME=aiva-authn-detector
VERSION?=$(shell git describe --tags --always --dirty)
BUILD_DIR=dist
DOCKER_TAG=aiva/authn-detector:$(VERSION)

# 預設目標
.DEFAULT_GOAL := help

## build: 建置應用程式
build:
	@echo "Building $(BINARY_NAME) version $(VERSION)..."
	@mkdir -p $(BUILD_DIR)
	@go build -ldflags="-X main.Version=$(VERSION)" \
		-o $(BUILD_DIR)/$(BINARY_NAME) \
		cmd/worker/main.go

## test: 運行測試
test:
	@echo "Running tests..."
	@go test -v -race -coverprofile=coverage.out ./...
	@go tool cover -html=coverage.out -o coverage.html

## benchmark: 運行基準測試
benchmark:
	@echo "Running benchmarks..."
	@go test -bench=. -benchmem ./...

## clean: 清理建置檔案
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR)
	@rm -f coverage.out coverage.html

## docker: 建置Docker映像
docker:
	@echo "Building Docker image $(DOCKER_TAG)..."
	@docker build -t $(DOCKER_TAG) .

## lint: 運行程式碼檢查
lint:
	@echo "Running linters..."
	@golangci-lint run

## fmt: 格式化程式碼
fmt:
	@echo "Formatting code..."
	@go fmt ./...
	@goimports -w .

## mod: 整理模組依賴
mod:
	@echo "Tidying module dependencies..."
	@go mod tidy

## help: 顯示幫助信息
help:
	@echo "Available targets:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | sort
```

---

## 🔗 相關連結

### **開發指南**
- [🐍 Python開發指南](../python/README.md) - Python模組開發
- [🛡️ 安全模組](../security/README.md) - 完整實現參考
- [🔧 開發中模組](../development/README.md) - 進行中的開發

### **Go語言資源**
- [Go官方文檔](https://golang.org/doc/) - Go語言官方文檔
- [Effective Go](https://golang.org/doc/effective_go.html) - Go編程指南
- [Go by Example](https://gobyexample.com/) - Go範例學習
- [The Go Blog](https://blog.golang.org/) - Go官方部落格

### **工具與庫**
- [Gin Web Framework](https://gin-gonic.com/) - HTTP Web框架
- [Cobra CLI](https://cobra.dev/) - 命令行應用程式庫
- [Viper](https://github.com/spf13/viper) - 配置管理
- [Zap](https://github.com/uber-go/zap) - 結構化日誌庫

### **測試工具**
- [Testify](https://github.com/stretchr/testify) - 測試斷言庫
- [GoMock](https://github.com/golang/mock) - Mock框架
- [GoConvey](http://goconvey.co/) - 測試Web UI

---

*最後更新: 2025年11月7日*  
*維護團隊: AIVA Go Development Team*