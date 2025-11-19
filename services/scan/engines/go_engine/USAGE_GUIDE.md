# AIVA Go Engine 使用指南

## 📋 目錄

1. [快速開始](#快速開始)
2. [SSRF 掃描器使用](#ssrf-掃描器使用)
3. [CSPM 掃描器使用](#cspm-掃描器使用)
4. [SCA 掃描器使用](#sca-掃描器使用)
5. [命令行參數](#命令行參數)
6. [配置文件](#配置文件)
7. [Python 集成](#python-集成)
8. [實戰範例](#實戰範例)
9. [故障排除](#故障排除)
10. [性能調優](#性能調優)

---

## 快速開始

### 1. 編譯所有掃描器

```bash
# 使用 Makefile
make build

# 或手動編譯
go build -o bin/ssrf-scanner.exe ./cmd/ssrf-scanner
go build -o bin/cspm-scanner.exe ./cmd/cspm-scanner
go build -o bin/sca-scanner.exe ./cmd/sca-scanner
```

### 2. 驗證編譯結果

```powershell
# 檢查編譯產物
Get-ChildItem bin/*.exe | Select-Object Name, Length

# 預期輸出:
# Name              Length
# ----              ------
# ssrf-scanner.exe  7864320
# cspm-scanner.exe  5816320
# sca-scanner.exe   5816320
```

### 3. 基礎運行測試

```bash
# SSRF 掃描器
./bin/ssrf-scanner.exe

# CSPM 掃描器
./bin/cspm-scanner.exe

# SCA 掃描器
./bin/sca-scanner.exe
```

---

## SSRF 掃描器使用

### 架構概覽

SSRF 掃描器由三個核心模塊組成：

```
internal/ssrf/
├── detector/              # 核心檢測引擎
│   ├── ssrf.go           # 主要 SSRF 檢測邏輯
│   ├── cloud_metadata_scanner.go  # 雲端元數據掃描
│   └── internal_microservice_probe.go  # 內部微服務探測
├── oob/                  # Out-of-Band 驗證
│   └── monitor.go        # OOB 監控器
└── verifier/             # 驗證器
    └── verifier.go       # 結果驗證邏輯
```

### 使用方式

#### 方式 1: 獨立運行（需要手動實現任務輸入）

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/ssrf/detector"
    "go.uber.org/zap"
)

func main() {
    // 初始化 Logger
    logger, _ := zap.NewProduction()
    defer logger.Sync()
    
    // 創建 SSRF 檢測器
    ssrfDetector := detector.NewSSRFDetector(logger)
    
    // 構造掃描任務
    task := &detector.ScanTask{
        TaskID: "scan_001",
        Module: "ssrf",
        Target: "http://vulnerable-app.com/api",
        Metadata: map[string]string{
            "priority": "high",
        },
    }
    
    // 執行掃描
    ctx := context.Background()
    findings, err := ssrfDetector.Scan(ctx, task)
    if err != nil {
        log.Fatal(err)
    }
    
    // 輸出結果
    fmt.Printf("發現 %d 個漏洞\n", len(findings))
    for _, finding := range findings {
        fmt.Printf("漏洞: %s (嚴重性: %s)\n", 
            finding.Vulnerability.Name, 
            finding.Vulnerability.Severity)
    }
}
```

#### 方式 2: 通過 Python 調度器

```python
from dispatcher.worker import GoEngineWorker

# 初始化 Worker
worker = GoEngineWorker(scanner_type="ssrf")

# 提交任務
task = {
    "task_id": "scan_001",
    "module": "ssrf",
    "target": "http://vulnerable-app.com/api",
    "metadata": {
        "priority": "high"
    }
}

# 執行掃描
results = worker.execute_scan(task)

# 處理結果
for finding in results["findings"]:
    print(f"發現漏洞: {finding['vulnerability']['name']}")
```

### 檢測邏輯說明

#### 1. 內網 IP 阻擋

掃描器自動阻擋以下 IP 範圍：

```go
blockedCIDRs := []string{
    "10.0.0.0/8",           // 私有網絡 A
    "172.16.0.0/12",        // 私有網絡 B
    "192.168.0.0/16",       // 私有網絡 C
    "127.0.0.0/8",          // Localhost
    "169.254.169.254/32",   // AWS IMDS
    "fd00::/8",             // IPv6 ULA
}
```

#### 2. 雲端元數據服務檢測

支持檢測以下雲端提供商的元數據服務：

| 提供商 | 元數據端點 | 風險等級 |
|--------|-----------|---------|
| AWS | `http://169.254.169.254/latest/meta-data/` | HIGH |
| GCP | `http://metadata.google.internal/computeMetadata/v1/` | HIGH |
| Azure | `http://169.254.169.254/metadata/instance?api-version=2021-02-01` | HIGH |
| Alibaba Cloud | `http://100.100.100.200/latest/meta-data/` | HIGH |

#### 3. 敏感資訊關鍵字

響應內容包含以下關鍵字時會標記為漏洞：

- `ami-id`
- `instance-id`
- `iam/security-credentials`
- `computeMetadata`
- `config`
- `password`
- `secret`
- `token`
- `api_key`

### Payload 範例

```go
// 測試 AWS IMDS
target := "http://target.com/api?url=http://169.254.169.254/latest/meta-data/"

// 測試 localhost 繞過
target := "http://target.com/api?url=http://127.0.0.1:8080/admin"

// 測試 IPv6 localhost
target := "http://target.com/api?url=http://[::1]/"

// 測試內網掃描
target := "http://target.com/api?url=http://192.168.1.1/"
```

---

## CSPM 掃描器使用

### 架構概覽

CSPM（Cloud Security Posture Management）掃描器用於雲端配置審計：

```
internal/cspm/
├── audit/
│   └── aws.go            # AWS 審計邏輯
└── scanner/
    └── scanner.go        # 掃描器入口
```

### AWS S3 Bucket 審計

#### 完整使用範例

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"
    
    "github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/cspm/audit"
)

func main() {
    // 從環境變數讀取 AWS 憑證
    // export AWS_ACCESS_KEY_ID="your_key"
    // export AWS_SECRET_ACCESS_KEY="your_secret"
    // export AWS_DEFAULT_REGION="us-east-1"
    
    ctx := context.Background()
    region := os.Getenv("AWS_DEFAULT_REGION")
    if region == "" {
        region = "us-east-1"
    }
    
    // 創建審計器
    auditor, err := audit.NewAWSAuditor(ctx, region)
    if err != nil {
        log.Fatalf("無法創建審計器: %v", err)
    }
    
    // 執行 S3 Bucket 審計
    fmt.Println("開始 S3 Bucket 審計...")
    riskBuckets, err := auditor.AuditS3Buckets()
    if err != nil {
        log.Fatalf("審計失敗: %v", err)
    }
    
    // 輸出結果
    if len(riskBuckets) == 0 {
        fmt.Println("✓ 未發現風險 Bucket")
    } else {
        fmt.Printf("⚠️  發現 %d 個風險 Bucket:\n", len(riskBuckets))
        for i, bucket := range riskBuckets {
            fmt.Printf("  %d. %s\n", i+1, bucket)
        }
    }
}
```

#### 執行完整 CIS Benchmark 審計

```go
// 執行所有 AWS 服務審計
results, err := auditor.RunFullAudit()
if err != nil {
    log.Fatal(err)
}

// 輸出各服務的審計結果
fmt.Println("\n=== AWS CIS Benchmark 審計報告 ===\n")
for service, risks := range results {
    fmt.Printf("服務: %s\n", service)
    if len(risks) == 0 {
        fmt.Println("  ✓ 未發現風險配置")
    } else {
        fmt.Printf("  ⚠️  發現 %d 個風險項:\n", len(risks))
        for _, risk := range risks {
            fmt.Printf("    - %s\n", risk)
        }
    }
    fmt.Println()
}
```

### AWS 認證配置

#### 方式 1: 環境變數

```bash
# Linux/macOS
export AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
export AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
export AWS_DEFAULT_REGION="us-east-1"

# Windows PowerShell
$env:AWS_ACCESS_KEY_ID="AKIAIOSFODNN7EXAMPLE"
$env:AWS_SECRET_ACCESS_KEY="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
$env:AWS_DEFAULT_REGION="us-east-1"
```

#### 方式 2: AWS CLI 配置

```bash
# 使用 AWS CLI 配置
aws configure

# 驗證配置
aws sts get-caller-identity
```

#### 方式 3: IAM Role（推薦用於 EC2/ECS）

```go
// SDK 自動從 Instance Metadata 獲取憑證
cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion("us-east-1"))
```

### 檢查項目列表

#### ✅ 已實現

| 檢查項 | 描述 | 嚴重性 |
|--------|------|--------|
| S3 Bucket ACL | 檢查是否存在公開訪問權限 | HIGH |
| Public Access Block | 驗證 PAB 配置是否啟用 | HIGH |

#### 🚧 待實現

| 檢查項 | 描述 | 嚴重性 |
|--------|------|--------|
| IAM 用戶審計 | 檢查權限過大、未使用的訪問密鑰 | HIGH |
| Security Group | 檢查 0.0.0.0/0 開放的高風險端口 | HIGH |
| CloudTrail | 驗證日誌審計配置 | MEDIUM |
| KMS 密鑰 | 檢查密鑰輪換策略 | MEDIUM |

---

## SCA 掃描器使用

### 架構概覽

SCA（Software Composition Analysis）掃描器用於依賴分析：

```
internal/sca/
├── scanner/
│   └── scanner.go        # 掃描器主邏輯
├── analyzer/
│   └── analyzer.go       # 依賴分析器
└── fs/
    └── walker.go         # 文件系統遍歷
```

### 使用方式

```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/sca/scanner"
    "go.uber.org/zap"
)

func main() {
    logger, _ := zap.NewProduction()
    defer logger.Sync()
    
    // 創建 SCA 掃描器
    scaScanner := scanner.NewSCAScanner(logger)
    
    // 掃描目標目錄
    ctx := context.Background()
    targetPath := "/path/to/project"
    
    results, err := scaScanner.ScanDirectory(ctx, targetPath)
    if err != nil {
        log.Fatal(err)
    }
    
    // 輸出結果
    fmt.Printf("發現 %d 個依賴項\n", len(results.Dependencies))
    fmt.Printf("發現 %d 個已知漏洞\n", len(results.Vulnerabilities))
}
```

---

## 命令行參數

### 通用參數

所有掃描器支持以下參數：

```bash
# 顯示版本
./bin/ssrf-scanner.exe --version

# 顯示幫助
./bin/ssrf-scanner.exe --help

# 啟用詳細日誌
./bin/ssrf-scanner.exe --verbose

# 指定配置文件
./bin/ssrf-scanner.exe --config /path/to/config.yaml
```

### SSRF 掃描器專用參數

```bash
# 指定目標 URL
./bin/ssrf-scanner.exe --target "http://example.com/api"

# 指定 Payload 文件
./bin/ssrf-scanner.exe --payloads /path/to/payloads.txt

# 設置超時時間
./bin/ssrf-scanner.exe --timeout 30s

# 設置並發數
./bin/ssrf-scanner.exe --concurrency 10
```

### CSPM 掃描器專用參數

```bash
# 指定 AWS Region
./bin/cspm-scanner.exe --region us-east-1

# 指定服務類型
./bin/cspm-scanner.exe --service s3,iam,ec2

# 僅掃描特定 Bucket
./bin/cspm-scanner.exe --bucket my-bucket-name

# 輸出格式
./bin/cspm-scanner.exe --format json
```

---

## 配置文件

### SSRF 掃描器配置

創建 `config/ssrf.yaml`:

```yaml
# SSRF 掃描器配置
scanner:
  timeout: 30s
  max_redirects: 3
  concurrency: 5
  
# 阻擋的 IP 範圍
blocked_cidrs:
  - "10.0.0.0/8"
  - "172.16.0.0/12"
  - "192.168.0.0/16"
  - "127.0.0.0/8"
  - "169.254.169.254/32"
  - "fd00::/8"

# Payload 列表
payloads:
  - name: "AWS IMDS"
    url: "http://169.254.169.254/latest/meta-data/"
    risk: "HIGH"
  - name: "GCP Metadata"
    url: "http://metadata.google.internal/computeMetadata/v1/"
    risk: "HIGH"
  - name: "Localhost Admin"
    url: "http://127.0.0.1:80/admin"
    risk: "MEDIUM"

# 敏感關鍵字
sensitive_keywords:
  - "ami-id"
  - "instance-id"
  - "iam/security-credentials"
  - "password"
  - "secret"
  - "token"
  - "api_key"

# 日誌配置
logging:
  level: "info"
  format: "json"
  output: "stdout"
```

### CSPM 掃描器配置

創建 `config/cspm.yaml`:

```yaml
# CSPM 掃描器配置
aws:
  region: "us-east-1"
  profile: "default"
  
  # 要掃描的服務
  services:
    - s3
    - iam
    - ec2
    - cloudtrail
    - kms
  
  # S3 配置
  s3:
    check_acl: true
    check_public_access_block: true
    check_encryption: true
    check_versioning: true
  
  # IAM 配置
  iam:
    check_unused_credentials: true
    check_password_policy: true
    check_mfa: true

# 合規框架
compliance:
  frameworks:
    - "CIS AWS Foundations Benchmark v1.4.0"
    - "AWS Well-Architected Framework"
  
# 報告配置
reporting:
  format: "json"
  output_dir: "./reports"
  include_recommendations: true
```

---

## Python 集成

### 使用 dispatcher/worker.py

```python
import json
from pathlib import Path
from dispatcher.worker import GoEngineWorker

class ScanOrchestrator:
    def __init__(self):
        self.ssrf_worker = GoEngineWorker(scanner_type="ssrf")
        self.cspm_worker = GoEngineWorker(scanner_type="cspm")
        self.sca_worker = GoEngineWorker(scanner_type="sca")
    
    def scan_target(self, target_url, scan_types=["ssrf"]):
        """
        對目標執行多種類型的掃描
        
        Args:
            target_url: 目標 URL
            scan_types: 掃描類型列表 ['ssrf', 'cspm', 'sca']
        
        Returns:
            dict: 掃描結果
        """
        results = {}
        
        if "ssrf" in scan_types:
            task = {
                "task_id": f"ssrf_{target_url}",
                "target": target_url,
                "module": "ssrf"
            }
            results["ssrf"] = self.ssrf_worker.execute_scan(task)
        
        if "cspm" in scan_types:
            task = {
                "task_id": f"cspm_{target_url}",
                "target": target_url,
                "module": "cspm"
            }
            results["cspm"] = self.cspm_worker.execute_scan(task)
        
        if "sca" in scan_types:
            task = {
                "task_id": f"sca_{target_url}",
                "target": target_url,
                "module": "sca"
            }
            results["sca"] = self.sca_worker.execute_scan(task)
        
        return results
    
    def save_results(self, results, output_file):
        """保存掃描結果到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

# 使用範例
if __name__ == "__main__":
    orchestrator = ScanOrchestrator()
    
    # 掃描目標
    target = "http://vulnerable-app.com"
    results = orchestrator.scan_target(target, scan_types=["ssrf", "cspm"])
    
    # 保存結果
    orchestrator.save_results(results, "scan_results.json")
    
    # 輸出摘要
    print(f"掃描完成: {target}")
    print(f"SSRF 漏洞: {len(results['ssrf']['findings'])}")
    print(f"CSPM 風險: {len(results['cspm']['risks'])}")
```

---

## 實戰範例

### 範例 1: 批量掃描多個目標

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dispatcher.worker import GoEngineWorker

async def scan_multiple_targets(targets):
    """並行掃描多個目標"""
    worker = GoEngineWorker(scanner_type="ssrf")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for target in targets:
            task = {
                "task_id": f"scan_{target}",
                "target": target,
                "module": "ssrf"
            }
            future = executor.submit(worker.execute_scan, task)
            futures.append((target, future))
        
        results = {}
        for target, future in futures:
            try:
                result = future.result(timeout=60)
                results[target] = result
            except Exception as e:
                print(f"掃描失敗 {target}: {e}")
                results[target] = {"error": str(e)}
        
        return results

# 使用
targets = [
    "http://app1.example.com",
    "http://app2.example.com",
    "http://app3.example.com",
]

results = asyncio.run(scan_multiple_targets(targets))
```

### 範例 2: CI/CD 集成

```yaml
# .github/workflows/security-scan.yml
name: Security Scan

on:
  pull_request:
    branches: [main]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Go
        uses: actions/setup-go@v4
        with:
          go-version: '1.23'
      
      - name: Build Scanners
        run: |
          cd services/scan/engines/go_engine
          make build
      
      - name: Run SSRF Scan
        run: |
          ./bin/ssrf-scanner.exe --target ${{ secrets.TEST_TARGET }}
      
      - name: Run CSPM Audit
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          ./bin/cspm-scanner.exe --region us-east-1
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: scan-results
          path: ./reports/
```

### 範例 3: Docker 容器掃描

```bash
# 掃描運行中的 Docker 容器
#!/bin/bash

echo "掃描 Docker 容器..."

# 列出所有運行中的容器
containers=$(docker ps --format "{{.Names}}")

for container in $containers; do
    echo "掃描容器: $container"
    
    # 獲取容器 IP
    ip=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $container)
    
    # 執行 SSRF 掃描
    ./bin/ssrf-scanner.exe --target "http://$ip" --output "scan_$container.json"
done

echo "掃描完成"
```

---

## 故障排除

### 問題 1: 編譯失敗

```bash
# 錯誤: cannot find package
go: downloading github.com/...

# 解決方案
cd services/scan/engines/go_engine
go work sync
go mod download
```

### 問題 2: AWS 認證失敗

```bash
# 錯誤: UnauthorizedOperation

# 解決方案 1: 檢查環境變數
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# 解決方案 2: 驗證 IAM 權限
aws sts get-caller-identity

# 解決方案 3: 使用正確的 Profile
export AWS_PROFILE=your-profile
```

### 問題 3: SSRF 掃描無結果

```bash
# 原因 1: 目標無 SSRF 漏洞（正常）
# 原因 2: 目標參數名不匹配

# 解決方案: 自定義參數名
./bin/ssrf-scanner.exe --params "url,target,redirect,link"
```

### 問題 4: 記憶體佔用過高

```bash
# 解決方案: 降低並發數
./bin/ssrf-scanner.exe --concurrency 2

# 或限制 Go 運行時記憶體
export GOGC=50
```

---

## 性能調優

### 1. 並發控制

```yaml
# config/ssrf.yaml
scanner:
  concurrency: 10  # 根據系統資源調整
  timeout: 30s
  retry: 3
```

### 2. 記憶體優化

```bash
# 設置 Go GC 策略
export GOGC=100  # 預設值，降低會更頻繁 GC

# 限制最大記憶體
export GOMEMLIMIT=2GiB
```

### 3. 網絡優化

```yaml
# config/ssrf.yaml
scanner:
  keep_alive: true
  idle_conn_timeout: 90s
  max_idle_conns: 100
```

### 4. 日誌優化

```yaml
# config/logging.yaml
logging:
  level: "warn"  # 生產環境使用 warn 或 error
  output: "file"
  file_path: "/var/log/aiva/scan.log"
  max_size: 100  # MB
  max_backups: 5
```

---

## 附錄

### A. 完整 API 參考

參考 Go 源碼註釋和 GoDoc。

### B. 漏洞數據庫

- CVE: https://cve.mitre.org/
- NVD: https://nvd.nist.gov/
- GitHub Advisory: https://github.com/advisories

### C. 合規框架

- CIS AWS Foundations Benchmark
- NIST Cybersecurity Framework
- OWASP Top 10

### D. 相關資源

- [OWASP SSRF](https://owasp.org/www-community/attacks/Server_Side_Request_Forgery)
- [AWS Security Best Practices](https://aws.amazon.com/security/best-practices/)
- [Go Security Checklist](https://github.com/guardrailsio/awesome-golang-security)

---

**文檔版本**: 1.0.0  
**最後更新**: 2025-11-20  
**維護者**: AIVA Team
