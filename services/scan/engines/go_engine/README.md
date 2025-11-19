# Go Engine - AIVA 掃描器引擎

重構後的 Go 掃描器引擎，採用 Go Workspace 多模塊架構。

## 📁 目錄結構

```
go_engine/
├── bin/                    # 編譯產物
│   ├── ssrf-scanner.exe
│   ├── cspm-scanner.exe
│   └── sca-scanner.exe
├── cmd/                    # 命令入口點
│   ├── ssrf-scanner/       # SSRF 掃描器主程序
│   ├── cspm-scanner/       # CSPM 掃描器主程序
│   └── sca-scanner/        # SCA 掃描器主程序
├── internal/               # 內部實現邏輯
│   ├── ssrf/              # SSRF 檢測邏輯
│   │   ├── detector/      # 核心檢測器
│   │   ├── oob/           # Out-of-Band 驗證
│   │   └── verifier/      # 驗證器
│   ├── cspm/              # 雲端安全態勢管理
│   │   ├── audit/         # 審計邏輯
│   │   └── scanner/       # 掃描器
│   ├── sca/               # 軟體組成分析
│   │   ├── scanner/       # 掃描器
│   │   ├── analyzer/      # 分析器
│   │   └── fs/            # 文件系統工具
│   └── common/            # 共用組件（已棄用）
├── pkg/                    # 共享模型
│   └── models/            # 數據結構定義
├── dispatcher/             # Python 協調器
│   ├── worker.py          # 主協調器（609行）
│   ├── build.py           # 構建腳本（392行）
│   └── dispatcher_legacy.py  # 舊版調度器（57行）
├── go.work                 # Go Workspace 配置
├── go.work.sum            # 依賴校驗和
├── Makefile               # 構建腳本
└── __init__.py            # Python 包標識
```

## 🏗️ 架構設計

### Go Workspace 管理
使用 `go.work` 管理 8 個獨立 Go 模塊：
- 3 個命令模塊（cmd/）
- 4 個內部邏輯模塊（internal/）
- 1 個共享模型模塊（pkg/models）

### 模塊路徑
所有模塊使用統一路徑前綴：
```
github.com/kyle0527/aiva/services/scan/engines/go_engine/
```

## 🛠️ 構建方式

### 使用 Makefile（推薦）
```bash
# 構建所有掃描器
make build

# 構建單一掃描器
make ssrf    # SSRF 掃描器
make cspm    # CSPM 掃描器
make sca     # SCA 掃描器

# 清理編譯產物
make clean

# 清理並重建
make rebuild

# 檢查環境
make check

# 初始化開發環境
make setup

# 查看構建狀態
make status

# 查看幫助
make help
```

### 手動構建
```bash
# 進入掃描器目錄
cd cmd/ssrf-scanner

# 編譯
go build -ldflags="-s -w" -trimpath -o ../../bin/ssrf-scanner.exe .
```

## 📊 編譯產物

| 掃描器 | 文件名 | 大小 |
|--------|--------|------|
| SSRF | ssrf-scanner.exe | ~5.3 MB |
| CSPM | cspm-scanner.exe | ~3.9 MB |
| SCA | sca-scanner.exe | ~3.9 MB |

## 🔧 開發環境需求

- Go 1.23.1+
- Make（可選，用於使用 Makefile）

## 📝 模塊依賴

### 共同依賴
- `go.uber.org/zap` - 日誌庫
- `github.com/rabbitmq/amqp091-go` - RabbitMQ 客戶端

### aiva_common_go
所有掃描器依賴共享的 `aiva_common_go` 模塊：
```
github.com/kyle0527/aiva/services/function/common/go/aiva_common_go
```

## 🚀 使用方式

### 1. SSRF 掃描器

#### 基礎使用
```bash
# 直接運行（進入等待模式）
./bin/ssrf-scanner.exe

# 通過 Go 運行並傳入參數
go run ./cmd/ssrf-scanner/main.go
```

#### 測試場景
```bash
# 測試 AWS IMDS 漏洞
curl "http://target.com/api?url=http://169.254.169.254/latest/meta-data/"

# 測試內網訪問
curl "http://target.com/api?url=http://192.168.1.1/"

# 測試 localhost 繞過
curl "http://target.com/api?url=http://127.0.0.1:8080/"
```

#### 檢測邏輯
- 自動阻擋內網 IP 範圍 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- 檢測雲端元數據服務 (AWS/GCP)
- 驗證響應內容是否包含敏感資訊 (ami-id, instance-id, credentials)

---

### 2. CSPM 掃描器

#### 基礎使用
```bash
# 運行掃描器
./bin/cspm-scanner.exe
```

#### AWS S3 審計範例
```go
// 創建審計器
auditor, err := audit.NewAWSAuditor(ctx, "us-east-1")
if err != nil {
    log.Fatal(err)
}

// 執行 S3 Bucket 審計
riskBuckets, err := auditor.AuditS3Buckets()
if err != nil {
    log.Fatal(err)
}

// 輸出風險 Bucket
for _, bucket := range riskBuckets {
    fmt.Printf("⚠️ Risk Bucket: %s\n", bucket)
}
```

#### 完整 AWS 審計
```go
// 執行所有 CIS Benchmark 檢查
results, err := auditor.RunFullAudit()
if err != nil {
    log.Fatal(err)
}

// 輸出審計結果
for service, risks := range results {
    fmt.Printf("Service: %s, Risks: %d\n", service, len(risks))
}
```

#### 檢測內容
- ✅ **S3 Bucket ACL 檢查** - 檢測公開訪問權限
- ✅ **Public Access Block** - 驗證 PAB 配置
- 🚧 **IAM 用戶審計** - 檢查權限過大問題
- 🚧 **Security Group** - 檢查 0.0.0.0/0 開放
- 🚧 **CloudTrail** - 驗證日誌審計配置
- 🚧 **KMS** - 檢查密鑰管理策略

#### AWS 認證配置
```bash
# 環境變數方式
export AWS_ACCESS_KEY_ID="your_access_key"
export AWS_SECRET_ACCESS_KEY="your_secret_key"
export AWS_DEFAULT_REGION="us-east-1"

# 或使用 AWS CLI 配置
aws configure
```

---

### 3. SCA 掃描器

#### 基礎使用
```bash
# 運行掃描器
./bin/sca-scanner.exe
```

#### 檢測邏輯
- 文件系統遍歷
- 依賴分析
- 漏洞數據庫匹配

---

### 4. 多目標批量掃描

#### 創建掃描腳本
```powershell
# scan-targets.ps1
$targets = @(
    "http://localhost:3000",  # juice-shop-live
    "http://localhost:3001",  # ecstatic_ritchie
    "http://localhost:3003",  # vigilant_shockle
    "http://webgoat:8080"     # laughing_jang
)

foreach ($target in $targets) {
    Write-Host "Scanning: $target" -ForegroundColor Cyan
    
    # SSRF 測試 Payloads
    $payloads = @(
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://127.0.0.1:80/admin",
        "http://192.168.1.1/"
    )
    
    foreach ($payload in $payloads) {
        $testUrl = "$target/api?url=$payload"
        Write-Host "  Testing: $payload"
        
        try {
            $response = Invoke-WebRequest -Uri $testUrl -TimeoutSec 5
            if ($response.StatusCode -eq 200) {
                Write-Host "    ⚠️ Potential SSRF: $payload" -ForegroundColor Yellow
            }
        } catch {
            Write-Host "    ✓ Blocked or No Response" -ForegroundColor Green
        }
    }
}
```

#### 執行批量掃描
```powershell
# 運行腳本
.\scan-targets.ps1
```

---

### 5. Docker 容器掃描

#### 掃描運行中的容器
```bash
# 列出所有容器
docker ps

# 針對特定容器掃描
docker exec juice-shop-live curl http://169.254.169.254/latest/meta-data/

# 掃描容器內部端口
docker exec -it ecstatic_ritchie netstat -tuln
```

#### 容器網絡分析
```bash
# 檢查容器網絡
docker network inspect bridge

# 測試容器間通信
docker exec juice-shop-live ping -c 3 vigilant_shockle
```

---

### 6. Python 協調器集成

#### 使用 dispatcher/worker.py
```python
from dispatcher.worker import GoEngineWorker

# 初始化 Worker
worker = GoEngineWorker(scanner_type="ssrf")

# 提交掃描任務
task = {
    "task_id": "scan_001",
    "target": "http://localhost:3000",
    "payloads": [
        "http://169.254.169.254/latest/meta-data/",
        "http://127.0.0.1:8080/"
    ]
}

# 執行掃描
results = worker.scan(task)
print(f"Found {len(results)} vulnerabilities")
```

---

### 7. 結果輸出格式

#### SSRF Finding 範例
```json
{
    "finding_id": "finding_scan_001_1763584750",
    "task_id": "scan_001",
    "scan_id": "ssrf_scan",
    "status": "confirmed",
    "vulnerability": {
        "name": "SSRF",
        "cwe": "CWE-918",
        "severity": "HIGH",
        "confidence": "FIRM",
        "description": "Server-Side Request Forgery vulnerability detected"
    },
    "target": {
        "url": "http://localhost:3000"
    },
    "evidence": {
        "request": "http://localhost:3000/api?url=http://169.254.169.254/latest/meta-data/",
        "response": "ami-id\ninstance-id\n...",
        "proof": "Status: 200, Body (前100字): ami-id\ninstance-id\n..."
    },
    "created_at": "2025-11-20T10:32:30Z",
    "updated_at": "2025-11-20T10:32:30Z"
}
```

#### CSPM Audit 結果範例
```json
{
    "service": "s3",
    "risk_buckets": [
        "my-public-bucket",
        "test-open-bucket"
    ],
    "checks_performed": [
        "ACL Configuration",
        "Public Access Block"
    ],
    "timestamp": "2025-11-20T10:35:00Z"
}
```

## ⚠️ 當前狀態

### ✅ 已完成
- [x] 架構重構（cmd/ + internal/ + pkg/）
- [x] Go Workspace 配置
- [x] 所有掃描器編譯成功
- [x] Makefile 構建腳本
- [x] 模塊路徑統一

### 🚧 待實現
- [ ] RabbitMQ Worker 實現
- [ ] 命令行參數處理（-payload flag）
- [ ] Candidate 驗證者模式
- [ ] Docker 支持（需創建新 Dockerfile）
- [ ] 完整的 Python 調度器集成

## 📚 相關文檔

- [Go Workspace 文檔](https://go.dev/doc/tutorial/workspaces)
- [AIVA 架構設計](../../../README.md)
- [重構計畫](../../../../Downloads/新增資料夾%20(6)/重構規劃.md)

## 🔄 遷移說明

從舊架構遷移：
- ~~`ssrf_scanner/`~~ → `cmd/ssrf-scanner/` + `internal/ssrf/`
- ~~`cspm_scanner/`~~ → `cmd/cspm-scanner/` + `internal/cspm/`
- ~~`sca_scanner/`~~ → `cmd/sca-scanner/` + `internal/sca/`
- ~~`common/`~~ → `internal/common/` + `pkg/models/`
- ~~`shared/`~~ → 已刪除
- ~~`build_scanners.ps1`~~ → Makefile
- ~~`docker-compose.yml`~~ → 待重新設計
