# AIVA Features - Go 開發指南 🐹

## 📑 目錄

- [🎯 **Go 在 AIVA 中的角色**](#go-在-aiva-中的角色)
  - [**🚀 高效能定位**](#高效能定位)
  - [**⚡ Go 組件統計**](#go-組件統計)
- [🏗️ **Go 架構模式**](#go-架構模式)
  - [**🔐 認證服務架構**](#認證服務架構)
  - [**☁️ CSPM 雲端安全管理**](#cspm-雲端安全管理)
  - [**📦 SCA 軟體組件分析**](#sca-軟體組件分析)
- [🛠️ **Go 開發環境設定**](#go-開發環境設定)
  - [**📦 Go Modules 配置**](#go-modules-配置)
  - [**🚀 快速開始**](#快速開始)
- [🧪 **測試策略**](#測試策略)
  - [**🔍 單元測試範例**](#單元測試範例)
  - [**🧪 整合測試**](#整合測試)
- [📈 **效能優化指南**](#效能優化指南)
  - [**⚡ 並發最佳實踐**](#並發最佳實踐)
  - [**📊 監控與指標**](#監控與指標)
- [🔧 **部署與維運**](#部署與維運)
  - [**🐳 Docker 多階段建置**](#docker-多階段建置)
  - [**📊 Kubernetes 部署**](#kubernetes-部署)
- [🔗 相關資源](#相關資源)
  - [模組開發指南](#模組開發指南)
  - [架構指南](#架構指南)
  - [開發指南](#開發指南)
  - [服務文檔](#服務文檔)

---
---
---
---

## 🎯 **Go 在 AIVA 中的角色**

### **🚀 高效能定位**
Go 在 AIVA Features 模組中扮演「**高效能服務提供者**」的角色：

```
🐹 Go 高效能服務架構
├── 🔐 認證安全服務 (AUTHN)
│   ├── JWT 令牌驗證 (15組件)
│   ├── OAuth 2.0 流程處理
│   └── 多因素認證 (MFA)
├── ☁️ 雲端安全管理 (CSPM)  
│   ├── AWS 配置檢查 (15組件)
│   ├── Azure 安全評估
│   └── GCP 合規性掃描
├── 📦 軟體組件分析 (SCA)
│   ├── 依賴項漏洞掃描 (20組件)
│   ├── License 合規檢查
│   └── 供應鏈安全分析
└── 🌐 網路安全防護 (SSRF Guard)
    ├── 內網請求防護 (19組件)
    ├── DNS 重綁定防護
    └── 協議濫用檢測
```

### **⚡ Go 組件統計**
- **認證服務**: 44 個組件 (高效能身份驗證)
- **雲端安全**: 15 個組件 (CSPM 合規檢查)  
- **組件分析**: 40 個組件 (SCA 依賴掃描)
- **網路防護**: 66 個組件 (SSRF 和網路安全)

---

## 🏗️ **Go 架構模式**

### **🔐 認證服務架構**

```go
package authn

import (
    "context"
    "time"
    "sync"
    "github.com/golang-jwt/jwt/v5"
    "golang.org/x/time/rate"
)

// AuthenticationService 高效能認證服務
type AuthenticationService struct {
    jwtSecretKey    []byte
    rateLimiter     *rate.Limiter
    tokenCache      sync.Map
    metrics         *AuthMetrics
}

// AuthRequest 認證請求結構
type AuthRequest struct {
    Username    string            `json:"username"`
    Password    string            `json:"password"`
    MFAToken    string            `json:"mfa_token,omitempty"`
    ClientInfo  map[string]string `json:"client_info"`
}

// AuthResponse 認證回應結構  
type AuthResponse struct {
    Success     bool      `json:"success"`
    AccessToken string    `json:"access_token,omitempty"`
    ExpiresAt   time.Time `json:"expires_at,omitempty"`
    UserID      string    `json:"user_id,omitempty"`
    Permissions []string  `json:"permissions,omitempty"`
}

// NewAuthenticationService 創建認證服務
func NewAuthenticationService(secretKey []byte) *AuthenticationService {
    return &AuthenticationService{
        jwtSecretKey: secretKey,
        rateLimiter:  rate.NewLimiter(rate.Limit(100), 200), // 100 req/sec, burst 200
        metrics:      NewAuthMetrics(),
    }
}

// Authenticate 執行高效能身份認證
func (as *AuthenticationService) Authenticate(ctx context.Context, req *AuthRequest) (*AuthResponse, error) {
    startTime := time.Now()
    defer func() {
        as.metrics.RecordAuthDuration(time.Since(startTime))
    }()
    
    // 1. 速率限制檢查
    if !as.rateLimiter.Allow() {
        as.metrics.IncrementRateLimited()
        return &AuthResponse{Success: false}, ErrRateLimitExceeded
    }
    
    // 2. 快取檢查 (避免重複計算)
    cacheKey := as.generateCacheKey(req)
    if cached, ok := as.tokenCache.Load(cacheKey); ok {
        as.metrics.IncrementCacheHit()
        return cached.(*AuthResponse), nil
    }
    
    // 3. 並發認證流程
    userChan := make(chan *User, 1)
    mfaChan := make(chan bool, 1)
    
    // 並行驗證用戶憑證和 MFA
    go as.validateUserCredentials(ctx, req.Username, req.Password, userChan)
    go as.validateMFAToken(ctx, req.Username, req.MFAToken, mfaChan)
    
    // 4. 等待驗證結果
    var user *User
    var mfaValid bool
    
    for i := 0; i < 2; i++ {
        select {
        case user = <-userChan:
            if user == nil {
                as.metrics.IncrementAuthFailure("invalid_credentials")
                return &AuthResponse{Success: false}, ErrInvalidCredentials
            }
        case mfaValid = <-mfaChan:
            if !mfaValid && req.MFAToken != "" {
                as.metrics.IncrementAuthFailure("invalid_mfa")
                return &AuthResponse{Success: false}, ErrInvalidMFA
            }
        case <-ctx.Done():
            as.metrics.IncrementAuthFailure("timeout")
            return &AuthResponse{Success: false}, ctx.Err()
        }
    }
    
    // 5. 生成 JWT 令牌
    token, expiresAt, err := as.generateJWTToken(user)
    if err != nil {
        as.metrics.IncrementAuthFailure("token_generation")
        return &AuthResponse{Success: false}, err
    }
    
    // 6. 快取結果
    response := &AuthResponse{
        Success:     true,
        AccessToken: token,
        ExpiresAt:   expiresAt,
        UserID:      user.ID,
        Permissions: user.Permissions,
    }
    
    as.tokenCache.Store(cacheKey, response)
    as.metrics.IncrementAuthSuccess()
    
    return response, nil
}

// validateUserCredentials 並發驗證用戶憑證
func (as *AuthenticationService) validateUserCredentials(ctx context.Context, username, password string, result chan<- *User) {
    defer close(result)
    
    // 使用 bcrypt 或 Argon2 進行安全密碼驗證
    user, err := as.userRepository.GetByUsername(ctx, username)
    if err != nil {
        result <- nil
        return
    }
    
    if !as.verifyPassword(password, user.PasswordHash) {
        result <- nil
        return
    }
    
    result <- user
}
```

### **☁️ CSPM 雲端安全管理**

```go
package cspm

import (
    "context"
    "sync"
    "github.com/aws/aws-sdk-go-v2/aws"
    "github.com/aws/aws-sdk-go-v2/service/ec2"
    "github.com/aws/aws-sdk-go-v2/service/s3"
)

// CSPMScanner 雲端安全態勢管理掃描器
type CSPMScanner struct {
    awsConfig   aws.Config
    rules       []SecurityRule
    workers     int
    results     chan ScanResult
    metrics     *CSPMMetrics
}

// SecurityRule 安全規則介面
type SecurityRule interface {
    ID() string
    Description() string
    Severity() string
    Check(ctx context.Context, resource CloudResource) (*RuleResult, error)
}

// CloudResource 雲端資源介面
type CloudResource interface {
    Type() string
    ID() string
    Region() string
    Tags() map[string]string
}

// ScanConfig CSPM 掃描配置
type ScanConfig struct {
    Regions         []string          `json:"regions"`
    ResourceTypes   []string          `json:"resource_types"`
    Rules           []string          `json:"rules"`
    Concurrency     int              `json:"concurrency"`
    Timeout         time.Duration    `json:"timeout"`
}

// ScanResult 掃描結果
type ScanResult struct {
    ResourceID    string                `json:"resource_id"`
    ResourceType  string                `json:"resource_type"`
    Region        string                `json:"region"`
    RuleResults   []RuleResult         `json:"rule_results"`
    Timestamp     time.Time            `json:"timestamp"`
}

// PerformCSPMScan 執行 CSPM 掃描
func (cs *CSPMScanner) PerformCSPMScan(ctx context.Context, config *ScanConfig) (<-chan ScanResult, error) {
    // 1. 初始化掃描通道
    resourceChan := make(chan CloudResource, 100)
    
    // 2. 啟動資源發現 goroutines
    var discoveryWG sync.WaitGroup
    for _, region := range config.Regions {
        for _, resourceType := range config.ResourceTypes {
            discoveryWG.Add(1)
            go func(region, resourceType string) {
                defer discoveryWG.Done()
                cs.discoverResources(ctx, region, resourceType, resourceChan)
            }(region, resourceType)
        }
    }
    
    // 3. 關閉資源通道當所有發現完成
    go func() {
        discoveryWG.Wait()
        close(resourceChan)
    }()
    
    // 4. 啟動規則檢查 workers
    var scanWG sync.WaitGroup
    for i := 0; i < config.Concurrency; i++ {
        scanWG.Add(1)
        go func() {
            defer scanWG.Done()
            cs.scanWorker(ctx, resourceChan, config.Rules)
        }()
    }
    
    // 5. 關閉結果通道當所有掃描完成
    go func() {
        scanWG.Wait()
        close(cs.results)
    }()
    
    return cs.results, nil
}

// scanWorker 掃描 worker
func (cs *CSPMScanner) scanWorker(ctx context.Context, resources <-chan CloudResource, ruleIDs []string) {
    for resource := range resources {
        select {
        case <-ctx.Done():
            return
        default:
            result := cs.scanResource(ctx, resource, ruleIDs)
            if result != nil {
                cs.results <- *result
            }
        }
    }
}

// scanResource 掃描單一資源
func (cs *CSPMScanner) scanResource(ctx context.Context, resource CloudResource, ruleIDs []string) *ScanResult {
    result := &ScanResult{
        ResourceID:   resource.ID(),
        ResourceType: resource.Type(),
        Region:       resource.Region(),
        Timestamp:    time.Now(),
        RuleResults:  make([]RuleResult, 0),
    }
    
    // 並行執行多個規則檢查
    var mu sync.Mutex
    var wg sync.WaitGroup
    
    for _, ruleID := range ruleIDs {
        rule := cs.getRuleByID(ruleID)
        if rule == nil {
            continue
        }
        
        wg.Add(1)
        go func(r SecurityRule) {
            defer wg.Done()
            
            ruleResult, err := r.Check(ctx, resource)
            if err != nil {
                cs.metrics.IncrementRuleError(r.ID())
                return
            }
            
            mu.Lock()
            result.RuleResults = append(result.RuleResults, *ruleResult)
            mu.Unlock()
            
            cs.metrics.IncrementRuleCheck(r.ID())
        }(rule)
    }
    
    wg.Wait()
    return result
}
```

### **📦 SCA 軟體組件分析**

```go
package sca

import (
    "bufio"
    "context"
    "encoding/json"
    "os"
    "path/filepath"
    "regexp"
)

// SCAScanner 軟體組件分析掃描器
type SCAScanner struct {
    vulnDB          VulnerabilityDatabase
    licenseDB       LicenseDatabase
    packageParsers  map[string]PackageParser
    workers         int
    metrics         *SCAMetrics
}

// Dependency 依賴項結構
type Dependency struct {
    Name            string   `json:"name"`
    Version         string   `json:"version"`
    PackageManager  string   `json:"package_manager"`
    FilePath        string   `json:"file_path"`
    Licenses        []string `json:"licenses"`
    Direct          bool     `json:"direct"`
}

// Vulnerability 漏洞結構
type Vulnerability struct {
    CVE         string  `json:"cve"`
    CVSS        float64 `json:"cvss"`
    Severity    string  `json:"severity"`
    Description string  `json:"description"`
    References  []string `json:"references"`
}

// SCAResult SCA 掃描結果
type SCAResult struct {
    ProjectPath      string                    `json:"project_path"`
    Dependencies     []Dependency             `json:"dependencies"`
    Vulnerabilities  map[string][]Vulnerability `json:"vulnerabilities"`
    LicenseIssues    []LicenseIssue           `json:"license_issues"`
    Statistics       SCAStatistics           `json:"statistics"`
}

// ScanProject 掃描專案依賴項
func (sca *SCAScanner) ScanProject(ctx context.Context, projectPath string) (*SCAResult, error) {
    result := &SCAResult{
        ProjectPath:     projectPath,
        Dependencies:    make([]Dependency, 0),
        Vulnerabilities: make(map[string][]Vulnerability),
        LicenseIssues:   make([]LicenseIssue, 0),
    }
    
    // 1. 發現所有套件管理檔案
    manifestFiles, err := sca.discoverManifestFiles(projectPath)
    if err != nil {
        return nil, err
    }
    
    // 2. 並行解析依賴項
    depChan := make(chan []Dependency, len(manifestFiles))
    var wg sync.WaitGroup
    
    for _, manifestFile := range manifestFiles {
        wg.Add(1)
        go func(file string) {
            defer wg.Done()
            deps := sca.parseDependencies(ctx, file)
            depChan <- deps
        }(manifestFile)
    }
    
    // 3. 收集所有依賴項
    go func() {
        wg.Wait()
        close(depChan)
    }()
    
    for deps := range depChan {
        result.Dependencies = append(result.Dependencies, deps...)
    }
    
    // 4. 並行漏洞掃描和 License 檢查
    vulnChan := make(chan map[string][]Vulnerability, 1)
    licenseChan := make(chan []LicenseIssue, 1)
    
    go sca.scanVulnerabilities(ctx, result.Dependencies, vulnChan)
    go sca.checkLicenses(ctx, result.Dependencies, licenseChan)
    
    // 5. 收集結果
    result.Vulnerabilities = <-vulnChan
    result.LicenseIssues = <-licenseChan
    
    // 6. 計算統計資訊
    result.Statistics = sca.calculateStatistics(result)
    
    return result, nil
}

// parseDependencies 解析依賴項 (支援多種套件管理器)
func (sca *SCAScanner) parseDependencies(ctx context.Context, manifestFile string) []Dependency {
    ext := filepath.Ext(manifestFile)
    
    switch {
    case strings.Contains(manifestFile, "package.json"):
        return sca.parseNodeJSDependencies(manifestFile)
    case strings.Contains(manifestFile, "requirements.txt"), strings.Contains(manifestFile, "Pipfile"):
        return sca.parsePythonDependencies(manifestFile)
    case strings.Contains(manifestFile, "go.mod"):
        return sca.parseGoDependencies(manifestFile)
    case strings.Contains(manifestFile, "Cargo.toml"):
        return sca.parseRustDependencies(manifestFile)
    default:
        return []Dependency{}
    }
}

// parseGoDependencies 解析 Go 依賴項
func (sca *SCAScanner) parseGoDependencies(goModFile string) []Dependency {
    file, err := os.Open(goModFile)
    if err != nil {
        return []Dependency{}
    }
    defer file.Close()
    
    dependencies := make([]Dependency, 0)
    scanner := bufio.NewScanner(file)
    
    // 匹配 Go 模組依賴項格式
    depRegex := regexp.MustCompile(`^\s*([^/\s]+(?:/[^/\s]+)*)\s+v([0-9]+\.[0-9]+\.[0-9]+(?:-[^/\s]+)?)`)
    
    for scanner.Scan() {
        line := scanner.Text()
        matches := depRegex.FindStringSubmatch(line)
        
        if len(matches) >= 3 {
            dep := Dependency{
                Name:           matches[1],
                Version:        matches[2],
                PackageManager: "go",
                FilePath:       goModFile,
                Direct:         true, // Go mod 中的都是直接依賴
            }
            dependencies = append(dependencies, dep)
        }
    }
    
    return dependencies
}
```

---

## 🛠️ **Go 開發環境設定**

### **📦 Go Modules 配置**
```go
// go.mod
module github.com/aiva/features

go 1.21

require (
    github.com/golang-jwt/jwt/v5 v5.0.0
    github.com/aws/aws-sdk-go-v2 v1.21.0
    github.com/aws/aws-sdk-go-v2/service/ec2 v1.118.0
    github.com/aws/aws-sdk-go-v2/service/s3 v1.40.0
    github.com/redis/go-redis/v9 v9.2.1
    github.com/prometheus/client_golang v1.17.0
    golang.org/x/time v0.3.0
    golang.org/x/crypto v0.14.0
)

require (
    // 測試依賴
    github.com/stretchr/testify v1.8.4
    github.com/testcontainers/testcontainers-go v0.25.0
    go.uber.org/goleak v1.2.1
)
```

### **🚀 快速開始**
```bash
# 1. Go 環境設定
cd services/features/
export GO111MODULE=on
export GOPROXY=https://proxy.golang.org,direct

# 2. 依賴安裝
go mod tidy
go mod download

# 3. 建置所有 Go 服務
make build-go-all

# 4. 執行測試
go test ./... -v -race -cover

# 5. 效能基準測試
go test ./... -bench=. -benchmem

# 6. 靜態分析
go vet ./...
golangci-lint run
```

---

## 🧪 **測試策略**

### **🔍 單元測試範例**
```go
package authn

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestAuthenticationService_Authenticate(t *testing.T) {
    tests := []struct {
        name     string
        request  *AuthRequest
        want     *AuthResponse
        wantErr  bool
    }{
        {
            name: "successful_authentication",
            request: &AuthRequest{
                Username: "testuser",
                Password: "correctpassword",
                MFAToken: "123456",
            },
            want: &AuthResponse{
                Success: true,
                UserID:  "user123",
            },
            wantErr: false,
        },
        {
            name: "invalid_credentials",
            request: &AuthRequest{
                Username: "testuser",
                Password: "wrongpassword",
            },
            want: &AuthResponse{
                Success: false,
            },
            wantErr: true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            // 設定測試環境
            service := NewAuthenticationService([]byte("test-secret"))
            ctx := context.Background()
            
            // 執行測試
            got, err := service.Authenticate(ctx, tt.request)
            
            // 驗證結果
            if tt.wantErr {
                assert.Error(t, err)
            } else {
                assert.NoError(t, err)
                assert.Equal(t, tt.want.Success, got.Success)
                if tt.want.UserID != "" {
                    assert.Equal(t, tt.want.UserID, got.UserID)
                }
            }
        })
    }
}

// 基準測試
func BenchmarkAuthenticationService_Authenticate(b *testing.B) {
    service := NewAuthenticationService([]byte("test-secret"))
    ctx := context.Background()
    request := &AuthRequest{
        Username: "benchuser",
        Password: "benchpassword",
    }
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _, err := service.Authenticate(ctx, request)
            if err != nil {
                b.Fatal(err)
            }
        }
    })
}
```

### **🧪 整合測試**
```go
package integration

import (
    "context"
    "testing"
    "github.com/testcontainers/testcontainers-go"
    "github.com/testcontainers/testcontainers-go/wait"
)

func TestCSPMIntegration(t *testing.T) {
    if testing.Short() {
        t.Skip("skipping integration test")
    }
    
    // 使用 Testcontainers 啟動 LocalStack (AWS 模擬器)
    ctx := context.Background()
    
    req := testcontainers.ContainerRequest{
        Image:        "localstack/localstack:latest",
        ExposedPorts: []string{"4566/tcp"},
        Env: map[string]string{
            "SERVICES": "ec2,s3,iam",
        },
        WaitingFor: wait.ForLog("Ready."),
    }
    
    container, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
        ContainerRequest: req,
        Started:          true,
    })
    require.NoError(t, err)
    defer container.Terminate(ctx)
    
    // 獲取容器端點
    endpoint, err := container.Endpoint(ctx, "")
    require.NoError(t, err)
    
    // 建立 CSMP 掃描器連接到 LocalStack
    scanner := NewCSPMScanner(CSPMConfig{
        Endpoint: endpoint,
        Region:   "us-east-1",
    })
    
    // 執行整合測試
    config := &ScanConfig{
        Regions:       []string{"us-east-1"},
        ResourceTypes: []string{"ec2"},
        Concurrency:   4,
        Timeout:       time.Minute * 5,
    }
    
    results, err := scanner.PerformCSPMScan(ctx, config)
    require.NoError(t, err)
    
    // 驗證結果
    resultCount := 0
    for result := range results {
        assert.NotEmpty(t, result.ResourceID)
        assert.Equal(t, "ec2", result.ResourceType)
        resultCount++
    }
    
    assert.Greater(t, resultCount, 0)
}
```

---

## 📈 **效能優化指南**

### **⚡ 並發最佳實踐**
```go
// ✅ 良好實踐: 使用 worker pool 模式
func ProcessConcurrently[T any, R any](
    ctx context.Context,
    items []T,
    processor func(context.Context, T) (R, error),
    workerCount int,
) ([]R, error) {
    
    jobs := make(chan T, len(items))
    results := make(chan R, len(items))
    errors := make(chan error, len(items))
    
    // 啟動 workers
    var wg sync.WaitGroup
    for i := 0; i < workerCount; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for job := range jobs {
                select {
                case <-ctx.Done():
                    return
                default:
                    result, err := processor(ctx, job)
                    if err != nil {
                        errors <- err
                    } else {
                        results <- result
                    }
                }
            }
        }()
    }
    
    // 發送工作
    go func() {
        defer close(jobs)
        for _, item := range items {
            jobs <- item
        }
    }()
    
    // 等待完成
    go func() {
        wg.Wait()
        close(results)
        close(errors)
    }()
    
    // 收集結果
    var finalResults []R
    var finalErrors []error
    
    for {
        select {
        case result, ok := <-results:
            if !ok {
                results = nil
            } else {
                finalResults = append(finalResults, result)
            }
        case err, ok := <-errors:
            if !ok {
                errors = nil
            } else {
                finalErrors = append(finalErrors, err)
            }
        }
        
        if results == nil && errors == nil {
            break
        }
    }
    
    if len(finalErrors) > 0 {
        return finalResults, finalErrors[0] // 返回第一個錯誤
    }
    
    return finalResults, nil
}

// ✅ 記憶體池最佳化
var (
    bufferPool = sync.Pool{
        New: func() interface{} {
            return make([]byte, 4096)
        },
    }
)

func ProcessWithBufferPool(data []byte) error {
    // 從池中獲取緩衝區
    buffer := bufferPool.Get().([]byte)
    defer bufferPool.Put(buffer)
    
    // 使用緩衝區處理資料
    // ... 處理邏輯
    
    return nil
}
```

### **📊 監控與指標**
```go
package monitoring

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    // HTTP 請求指標
    httpRequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "aiva_go_http_requests_total",
            Help: "Total HTTP requests processed",
        },
        []string{"method", "endpoint", "status"},
    )
    
    // 請求持續時間
    httpRequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "aiva_go_http_request_duration_seconds",
            Help:    "HTTP request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )
    
    // 活躍 Goroutines
    activeGoroutines = promauto.NewGauge(
        prometheus.GaugeOpts{
            Name: "aiva_go_active_goroutines",
            Help: "Number of active goroutines",
        },
    )
    
    // 記憶體使用
    memoryUsage = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "aiva_go_memory_usage_bytes",
            Help: "Memory usage in bytes",
        },
        []string{"type"}, // heap, stack, gc
    )
)

// MonitoringMiddleware HTTP 監控中間件
func MonitoringMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        
        // 包裝 ResponseWriter 以捕捉狀態碼
        wrapped := &responseWriter{ResponseWriter: w, statusCode: 200}
        
        // 處理請求
        next.ServeHTTP(wrapped, r)
        
        // 記錄指標
        duration := time.Since(start)
        httpRequestDuration.WithLabelValues(r.Method, r.URL.Path).Observe(duration.Seconds())
        httpRequestsTotal.WithLabelValues(r.Method, r.URL.Path, fmt.Sprintf("%d", wrapped.statusCode)).Inc()
    })
}
```

---

## 🔧 **部署與維運**

### **🐳 Docker 多階段建置**
```dockerfile
# Dockerfile.go
# 階段 1: 建置
FROM golang:1.21-alpine AS builder

WORKDIR /app

# 安裝依賴
RUN apk add --no-cache git ca-certificates tzdata

# 複製 go mod 檔案
COPY go.mod go.sum ./
RUN go mod download

# 複製原始碼
COPY . .

# 建置二進位檔案
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/server

# 階段 2: 執行
FROM alpine:latest

RUN apk --no-cache add ca-certificates tzdata
WORKDIR /root/

# 複製二進位檔案
COPY --from=builder /app/main .

# 複製配置檔案
COPY --from=builder /app/configs ./configs

# 設定時區
ENV TZ=UTC

# 健康檢查
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# 執行
EXPOSE 8080
CMD ["./main"]
```

### **📊 Kubernetes 部署**
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aiva-go-services
  labels:
    app: aiva-go-services
spec:
  replicas: 3
  selector:
    matchLabels:
      app: aiva-go-services
  template:
    metadata:
      labels:
        app: aiva-go-services
    spec:
      containers:
      - name: aiva-go
        image: aiva/go-services:latest
        ports:
        - containerPort: 8080
        env:
        - name: GO_ENV
          value: "production"
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: aiva-secrets
              key: db-host
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: aiva-go-service
spec:
  selector:
    app: aiva-go-services
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

---

**📝 版本**: v2.0 - Go Development Guide  
**🔄 最後更新**: 2024-10-24  
**🐹 Go 版本**: 1.21+  
**👥 維護團隊**: AIVA Go Development Team

*這是 AIVA Features 模組 Go 組件的完整開發指南，專注於高效能服務、並發處理和雲端安全功能的實現。*

---

## 🔗 相關資源

### 模組開發指南
- 📖 [Python 開發指南](./PYTHON_DEVELOPMENT_GUIDE.md)
- 📖 [Go 開發指南](./GO_DEVELOPMENT_GUIDE.md)
- 📖 [Rust 開發指南](./RUST_DEVELOPMENT_GUIDE.md)
- 📖 [AI 引擎指南](./AI_ENGINE_GUIDE.md)
- 📖 [功能模組開發指南](./FEATURE_MODULES_DEVELOPMENT_GUIDE.md)
- 📖 [模組遷移指南](./MODULE_MIGRATION_GUIDE.md)

### 架構指南
- 📖 [跨語言 Schema 指南](../architecture/CROSS_LANGUAGE_SCHEMA_GUIDE.md)
- 📖 [兼容性指南](../architecture/CROSS_LANGUAGE_COMPATIBILITY_GUIDE.md)

### 開發指南
- 📖 [開發快速指南](../development/DEVELOPMENT_QUICK_START_GUIDE.md)
- 📖 [依賴管理指南](../development/DEPENDENCY_MANAGEMENT_GUIDE.md)

### 服務文檔
- 🔧 [Features 模組](../../services/features/README.md)
- 🔧 [Scan 引擎文檔](../../services/scan/README.md)

