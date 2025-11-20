# Go Engine - AIVA 掃描器引擎

重構後的 Go 掃描器引擎，採用 Go Workspace 多模塊架構。

## 📋 目錄

1. [架構概覽](#架構概覽)
2. [目錄結構](#目錄結構)
3. [模塊說明](#模塊說明)
4. [技術架構](#技術架構)
5. [運作流程](#運作流程)
6. [開發狀態](#開發狀態)
7. [相關文檔](#相關文檔)

**💡 實際操作請參考**: **[使用指南 (USAGE_GUIDE.md)](./USAGE_GUIDE.md)**

---

## 架構概覽

Go Engine 是 AIVA 安全掃描平台的高性能掃描引擎，負責執行以下三類安全掃描：

- **SSRF Scanner** - 服務端請求偽造檢測
- **CSPM Scanner** - 雲端安全態勢管理
- **SCA Scanner** - 軟體組成分析

### 核心特性

✅ **Go Workspace 多模塊架構** - 8 個獨立模塊統一管理  
✅ **高性能並發掃描** - 支持大規模目標批量掃描  
✅ **標準化輸出** - 使用 `aiva_common_go` 統一數據格式  
✅ **雲端原生支持** - 深度集成 AWS/GCP/Azure API  
✅ **Python 協調器集成** - 無縫對接現有調度系統

---

## 目錄結構

```
go_engine/
├── bin/                    # 編譯產物目錄
│   ├── ssrf-scanner.exe   # SSRF 掃描器 (7.5 MB)
│   ├── cspm-scanner.exe   # CSPM 掃描器 (5.6 MB)
│   └── sca-scanner.exe    # SCA 掃描器 (5.6 MB)
│
├── cmd/                    # 命令入口點 (3 個模塊)
│   ├── ssrf-scanner/      # SSRF 掃描器主程序
│   │   ├── main.go        # 程序入口
│   │   └── go.mod         # 模塊依賴
│   ├── cspm-scanner/      # CSPM 掃描器主程序
│   └── sca-scanner/       # SCA 掃描器主程序
│
├── internal/               # 內部實現邏輯 (4 個模塊)
│   ├── ssrf/              # SSRF 檢測邏輯
│   │   ├── detector/      # 核心檢測引擎
│   │   │   ├── ssrf.go                    # 主檢測邏輯
│   │   │   ├── cloud_metadata_scanner.go  # 雲端元數據
│   │   │   └── internal_microservice_probe.go  # 微服務探測
│   │   ├── oob/           # Out-of-Band 驗證
│   │   │   └── monitor.go
│   │   ├── verifier/      # 結果驗證器
│   │   │   └── verifier.go
│   │   └── go.mod
│   │
│   ├── cspm/              # 雲端安全態勢管理
│   │   ├── audit/         # 審計邏輯
│   │   │   └── aws.go     # AWS 服務審計
│   │   ├── scanner/       # 掃描器實現
│   │   │   └── scanner.go
│   │   └── go.mod
│   │
│   ├── sca/               # 軟體組成分析
│   │   ├── scanner/       # 掃描器核心
│   │   │   └── scanner.go
│   │   ├── analyzer/      # 依賴分析器
│   │   │   └── analyzer.go
│   │   ├── fs/            # 文件系統工具
│   │   │   └── walker.go
│   │   └── go.mod
│   │
│   └── common/            # 共用組件 (已棄用)
│       └── go.mod
│
├── pkg/                    # 共享模型
│   └── models/            # 數據結構定義
│       └── go.mod
│
├── dispatcher/             # Python 協調器
│   ├── worker.py          # 主協調器 (609 行)
│   ├── build.py           # 構建腳本 (392 行)
│   └── dispatcher_legacy.py  # 舊版調度器 (57 行)
│
├── go.work                 # Go Workspace 配置
├── go.work.sum            # 依賴校驗和
├── Makefile               # 構建腳本
├── README.md              # 本文件
├── USAGE_GUIDE.md         # 詳細使用指南
└── __init__.py            # Python 包標識
```

---

## 模塊說明

### 1. SSRF Scanner (Server-Side Request Forgery)

**路徑**: `cmd/ssrf-scanner/` + `internal/ssrf/`

**功能**:
- 檢測服務端請求偽造漏洞
- 雲端元數據服務探測 (AWS IMDS, GCP Metadata, Azure IMDS)
- 內網 IP 阻擋與繞過檢測
- Out-of-Band (OOB) 驗證

**檢測覆蓋**:
```
✅ AWS IMDS (169.254.169.254)
✅ GCP Metadata (metadata.google.internal)
✅ Azure IMDS (169.254.169.254 + 特定 Header)
✅ Localhost 繞過 (127.0.0.1, [::1], 0.0.0.0)
✅ 內網掃描 (10.x.x.x, 172.16-31.x.x, 192.168.x.x)
✅ 敏感資訊檢測 (credentials, tokens, secrets)
```

**編譯產物**: `bin/ssrf-scanner.exe` (7.5 MB)

---

### 2. CSPM Scanner (Cloud Security Posture Management)

**路徑**: `cmd/cspm-scanner/` + `internal/cspm/`

**功能**:
- AWS 雲端配置審計
- CIS Benchmark 合規檢查
- 資源安全配置驗證
- 多雲環境支持 (規劃中)

**已實現檢查項**:
```
✅ S3 Bucket ACL 檢查
✅ S3 Public Access Block 驗證
🚧 IAM 用戶權限審計
🚧 Security Group 規則檢查
🚧 CloudTrail 日誌配置
🚧 KMS 密鑰管理審計
```

**支持雲端提供商**:
- ✅ AWS (S3 完整支持)
- 🚧 GCP (規劃中)
- 🚧 Azure (規劃中)

**編譯產物**: `bin/cspm-scanner.exe` (5.6 MB)

---

### 3. SCA Scanner (Software Composition Analysis)

**路徑**: `cmd/sca-scanner/` + `internal/sca/`

**功能**:
- 依賴庫掃描與分析
- 已知漏洞匹配 (CVE/NVD)
- License 合規檢查
- 過時依賴檢測

**支持語言**:
```
🚧 JavaScript/Node.js (package.json)
🚧 Python (requirements.txt, Pipfile)
🚧 Go (go.mod)
🚧 Java (pom.xml, build.gradle)
```

**編譯產物**: `bin/sca-scanner.exe` (5.6 MB)

---

## 技術架構

### Go Workspace 管理

使用 `go.work` 統一管理 8 個獨立 Go 模塊：

```
go.work
├── ./cmd/ssrf-scanner       # 命令模塊 1
├── ./cmd/cspm-scanner       # 命令模塊 2
├── ./cmd/sca-scanner        # 命令模塊 3
├── ./internal/ssrf          # 邏輯模塊 1
├── ./internal/cspm          # 邏輯模塊 2
├── ./internal/sca           # 邏輯模塊 3
├── ./internal/common        # 共用模塊 (已棄用)
└── ./pkg/models             # 共享模型
```

### 模塊路徑規範

所有模塊使用統一路徑前綴：
```
github.com/kyle0527/aiva/services/scan/engines/go_engine/
```

範例:
```go
import (
    "github.com/kyle0527/aiva/services/scan/engines/go_engine/internal/ssrf/detector"
    "github.com/kyle0527/aiva/services/scan/engines/go_engine/pkg/models"
)
```

### 共享依賴

所有掃描器依賴 `aiva_common_go` 統一數據格式：

```go
import (
    schemas "github.com/kyle0527/aiva/services/function/common/go/aiva_common_go/schemas/generated"
)

// 使用標準 FindingPayload
type Finding = schemas.FindingPayload
```

### 核心依賴庫

| 依賴 | 用途 | 版本 |
|------|------|------|
| `go.uber.org/zap` | 結構化日誌 | 1.26.0 |
| `github.com/aws/aws-sdk-go-v2/*` | AWS 服務集成 | 最新 |
| `github.com/google/uuid` | UUID 生成 | 1.6.0 |
| `github.com/sirupsen/logrus` | 日誌庫 | 1.9.3 |
| `aiva_common_go` | 標準數據格式 | 內部 |

---

## 運作流程

```
┌─────────────────┐
│  Python 調度器   │
│ (dispatcher)    │
└────────┬────────┘
         │ 1. 提交任務 (ScanTask)
         ↓
┌─────────────────┐
│  Go 掃描器引擎   │
│  (go_engine)    │
├─────────────────┤
│ • SSRF Scanner  │ ← 2. 執行掃描
│ • CSPM Scanner  │
│ • SCA Scanner   │
└────────┬────────┘
         │ 3. 返回 Finding (JSON)
         ↓
┌─────────────────┐
│  結果處理        │
│ (aiva_common)   │
└─────────────────┘
         │ 4. 存入數據庫
         ↓
┌─────────────────┐
│  MongoDB        │
└─────────────────┘
```

### 詳細流程說明

#### 1. 任務提交階段
- Python 調度器接收外部掃描請求
- 構造標準化任務結構 (`ScanTask`)
- 通過 subprocess 或 RabbitMQ 調用 Go 掃描器

```python
task = {
    "task_id": "scan_001",
    "module": "ssrf",
    "target": "http://target.com/api",
    "metadata": {"priority": "high"}
}
```

#### 2. 掃描執行階段
- Go 掃描器初始化檢測引擎
- 執行具體掃描邏輯 (SSRF/CSPM/SCA)
- 生成標準化 Finding 結構

```go
findings, err := detector.Scan(ctx, task)
```

#### 3. 結果返回階段
- Finding 序列化為 JSON
- 返回給 Python 調度器
- 驗證數據格式完整性

```json
{
    "finding_id": "finding_scan_001_...",
    "vulnerability": {"name": "SSRF", "severity": "HIGH"},
    "target": {"url": "http://target.com/api"},
    "evidence": {...}
}
```

#### 4. 數據持久化階段
- Python 調度器處理結果
- 存入 MongoDB 數據庫
- 觸發後續工作流 (通知、報告生成等)

---

## 開發狀態

### ✅ 已完成

- [x] 架構重構 (cmd/ + internal/ + pkg/)
- [x] Go Workspace 配置
- [x] 所有掃描器編譯成功
- [x] SSRF 檢測邏輯完整實現
- [x] CSPM S3 審計功能
- [x] 標準化 Finding 輸出
- [x] Makefile 構建腳本
- [x] 完整使用指南文檔

### 🚧 進行中

- [ ] RabbitMQ Worker 實現
- [ ] 命令行參數處理 (--target, --payload 等)
- [ ] CSPM 其他 AWS 服務審計 (IAM, EC2, CloudTrail, KMS)
- [ ] SCA 依賴掃描實現

### 📋 待規劃

- [ ] Docker 支持 (需創建新 Dockerfile)
- [ ] 完整的 Python 調度器集成
- [ ] GCP/Azure 雲端支持
- [ ] 性能基準測試
- [ ] 單元測試覆蓋率提升

---

## 相關文檔

### 📖 內部文檔

- **[使用指南 (USAGE_GUIDE.md)](./USAGE_GUIDE.md)** - 完整操作手冊
  - 快速開始
  - 各掃描器詳細使用
  - 命令行參數
  - 配置文件
  - Python 集成
  - 實戰範例
  - 故障排除
  - 性能調優

### 🔗 外部資源

- [Go Workspace 文檔](https://go.dev/doc/tutorial/workspaces)
- [AIVA 主項目 README](../../../../README.md)
- [OWASP SSRF](https://owasp.org/www-community/attacks/Server_Side_Request_Forgery)
- [AWS Security Best Practices](https://aws.amazon.com/security/best-practices/)
- [CIS AWS Foundations Benchmark](https://www.cisecurity.org/benchmark/amazon_web_services)

### ⚙️ 技術規範

- **Go 版本**: 1.23.1+
- **編譯目標**: Windows/Linux/macOS
- **Python 兼容**: 3.8+
- **依賴管理**: Go Modules + Go Workspace

---

## 快速構建

詳細操作請參考 [使用指南](./USAGE_GUIDE.md)。

```bash
# 構建所有掃描器
make build

# 構建單一掃描器
make ssrf
make cspm
make sca

# 清理並重建
make clean && make build

# 查看構建狀態
make status
```

---

## 貢獻指南

### 開發環境設置

```bash
# 1. 克隆倉庫
git clone https://github.com/kyle0527/aiva.git
cd aiva/services/scan/engines/go_engine

# 2. 初始化依賴
go work sync
go mod download

# 3. 編譯驗證
make build

# 4. 運行測試
go test ./...
```

### 代碼規範

- 遵循 Go 官方代碼風格
- 使用 `gofmt` 格式化代碼
- 所有公開函數需要註釋
- 錯誤處理使用 `fmt.Errorf` 包裝

### 提交流程

1. 創建功能分支
2. 編寫單元測試
3. 確保編譯通過
4. 提交 Pull Request
5. 等待 Code Review

---

**版本**: 1.0.0  
**最後更新**: 2025-11-20  
**維護者**: AIVA Team
