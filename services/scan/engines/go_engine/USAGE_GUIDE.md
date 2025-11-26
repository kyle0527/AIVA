# AIVA Go Engine 使用指南

> **架構版本**: v2.1 適配器模式  
> **通信方式**: JSON stdin/stdout (無 RabbitMQ 依賴)  
> **最後更新**: 2025-11-21

## 📋 目錄

1. [快速開始](#快速開始) ✅ 已驗證 2025-11-23
2. [架構概覽](#架構概覽)
3. [SSRF 掃描器使用](#ssrf-掃描器使用) ✅ 已驗證 2025-11-23
4. [JSON 通信協議](#json-通信協議) ✅ 已驗證 2025-11-23
5. [Python 集成](#python-集成) ✅ 已修復 2025-11-23 (原有同步代碼錯誤)
6. [實戰測試](#實戰測試) ✅ 已驗證 2025-11-23
7. [檢測邏輯詳解](#檢測邏輯詳解)
8. [故障排除](#故障排除)
9. [性能調優](#性能調優)

---

## 快速開始

### 1. 編譯掃描器

```powershell
# Windows PowerShell
cd C:\D\fold7\AIVA-git\services\scan\engines\go_engine
go build -o bin/ssrf-scanner.exe ./cmd/ssrf-scanner
```

### 2. 驗證編譯結果

```powershell
# 檢查編譯產物
Get-ChildItem bin/*.exe | Select-Object Name, Length, LastWriteTime

# 預期輸出:
# Name              Length    LastWriteTime
# ----              ------    -------------
# ssrf-scanner.exe  7864320   2025-11-21 ...
```

### 3. 基礎測試（JSON stdin/stdout）

```powershell
# 創建測試輸入
$input = '{"scan_id":"test001","targets":["http://example.com"],"concurrency":5,"timeout":10}'

# 執行掃描
echo $input | .\bin\ssrf-scanner.exe

# 預期輸出: JSON 格式的掃描結果
# {
#   "scan_id": "test001",
#   "scanner_type": "ssrf",
#   "status": "success",
#   "assets": [...]
# }
```

---

## 架構概覽

### 設計原則

Go Engine 採用 **v2.1 適配器模式**，特點：

- ✅ **無外部依賴**: 不依賴 RabbitMQ、Redis 等
- ✅ **簡單通信**: JSON stdin/stdout，易於調試
- ✅ **高性能**: 並發執行，Worker Pool 管理
- ✅ **統一接口**: 所有掃描器使用相同的 JSON 協議

---

### 目錄結構

```
services/scan/engines/go_engine/
├── cmd/
│   └── ssrf-scanner/
│       └── main.go              # 掃描器入口
├── internal/
│   ├── common/
│   │   ├── types.go            # 共享類型定義
│   │   └── worker_pool.go      # 並發 Worker Pool
│   └── ssrf/
│       └── detector/
│           └── ssrf.go         # SSRF 檢測邏輯
├── bin/
│   └── ssrf-scanner.exe        # 編譯產物
└── USAGE_GUIDE.md              # 本文檔
```

---

## SSRF 掃描器使用

### 核心特性

✅ **智能檢測**: 基於 OWASP/PortSwigger 最佳實踐  
✅ **避免誤判**: 精確的響應格式驗證，排除登錄頁面  
✅ **多 Payload**: 支持 AWS/GCP/Azure 元數據、File Protocol、內網探測  
✅ **高性能**: 並發執行，支持自定義並發數

### 基本使用

#### 方式 1: 直接調用（PowerShell）

```powershell
# 基礎測試
$input = @'
{
  "scan_id": "test001",
  "targets": ["http://localhost:8080/WebGoat/SSRF/task1"],
  "concurrency": 5,
  "timeout": 10
}
'@

$input | .\bin\ssrf-scanner.exe 2>$null | ConvertFrom-Json

# 輸出結果
# {
#   "scan_id": "test001",
#   "scanner_type": "ssrf",
#   "status": "success",
#   "execution_time": 1.23,
#   "targets_scanned": 1,
#   "assets": []
# }
```

#### 方式 2: 通過 Python 適配器（推薦）

```python
# 使用 services/scan/coordinators/engines/go_adapter.py
import subprocess
import json

def scan_with_go_engine(targets, concurrency=5, timeout=10):
    """使用 Go 引擎掃描"""
    
    # 構造輸入
    scan_input = {
        "scan_id": "python_scan",
        "targets": targets,
        "concurrency": concurrency,
        "timeout": timeout
    }
    
    # 調用 Go 掃描器
    process = subprocess.Popen(
        ["./bin/ssrf-scanner.exe"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    stdout, stderr = process.communicate(
        input=json.dumps(scan_input).encode()
    )
    
    # 解析結果
    if process.returncode == 0:
        return json.loads(stdout)
    else:
        raise Exception(f"Scan failed: {stderr.decode()}")

# 使用
results = scan_with_go_engine(["http://example.com/api"])
print(f"Found {len(results['assets'])} vulnerabilities")
```

---

## JSON 通信協議

### 輸入格式 (stdin)

```json
{
  "scan_id": "unique_scan_id",
  "targets": [
    "http://target1.com/api",
    "http://target2.com/page"
  ],
  "concurrency": 5,
  "timeout": 10
}
```

**字段說明**:
- `scan_id` (string): 唯一掃描 ID
- `targets` (array): 目標 URL 列表
- `concurrency` (int, 可選): 並發數，默認 5
- `timeout` (int, 可選): 超時秒數，默認 10

### 輸出格式 (stdout)

```json
{
  "scan_id": "unique_scan_id",
  "scanner_type": "ssrf",
  "status": "success",
  "execution_time": 2.543,
  "targets_scanned": 2,
  "requests_made": 144,
  "success_count": 2,
  "failure_count": 0,
  "assets": [
    {
      "type": "web_vulnerability",
      "name": "SSRF - File Protocol",
      "severity": "high",
      "confidence": "high",
      "source_engine": "go",
      "details": {
        "finding_id": "ssrf_abc123",
        "vulnerability_type": "SSRF",
        "cwe": "CWE-918",
        "description": "Attempt to read local files via file protocol",
        "affected_url": "http://target.com/api?url=file:///etc/passwd",
        "vulnerable_param": "url",
        "payload_used": "file:///etc/passwd",
        "payload_type": "File Protocol",
        "http_method": "GET",
        "response_status": 200,
        "response_time_ms": 45,
        "response_length": 2048,
        "response_preview": "root:x:0:0:root:/root:/bin/bash...",
        "evidence": {
          "request_url": "http://target.com/api?url=file:///etc/passwd",
          "response_status": 200,
          "response_headers": {...},
          "indicators_found": [
            "Found /etc/passwd root entry",
            "Unix password file format detected"
          ]
        }
      }
    }
  ]
}
```

**狀態碼**:
- `success`: 掃描成功完成
- `error`: 掃描過程發生錯誤
- `partial`: 部分目標掃描失敗

---

## Python 集成

### 使用 go_adapter.py（推薦）

```python
# services/scan/coordinators/engines/go_adapter.py
# ⚠️ 注意: 實際的 go_adapter.py 使用異步(async/await)
# 以下為簡化的同步示例,僅供理解 JSON 通信協議

import asyncio
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

class GoAdapter:
    """Go 引擎適配器 - 異步 JSON stdin/stdout 通信
    
    注意: AIVA 使用異步架構,實際代碼必須使用 async/await
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.go_scanner_path = None
    
    async def is_available(self) -> bool:
        """檢查 Go 引擎是否可用"""
        try:
            base_path = Path(__file__).parent.parent.parent / "engines" / "go_engine"
            possible_paths = [
                base_path / "bin" / "ssrf-scanner.exe",
                base_path / "bin" / "cspm-scanner.exe",
                base_path / "bin" / "sca-scanner.exe"
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.go_scanner_path = path
                    self.logger.info(f"找到 Go 掃描器: {path}")
                    return True
            
            self.logger.warning("Go 掃描器二進制文件不存在")
            return False
        except Exception as e:
            self.logger.warning(f"Go 引擎檢查失敗: {e}")
            return False
    
    async def scan(self, targets: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        """執行異步掃描"""
        
        if not await self.is_available():
            return {
                "assets": [],
                "metadata": {"engine": "go"},
                "error": "Go 引擎不可用"
            }
        
        # 構造輸入
        scan_input = {
            "scan_id": options.get("scan_id", "default_scan"),
            "targets": targets,
            "concurrency": options.get("concurrency", 10),
            "timeout": options.get("timeout", 30)
        }
        
        self.logger.info(f"🚀 Go 引擎開始掃描: {len(targets)} 個目標")
        
        # 異步調用 Go 掃描器
        try:
            proc = await asyncio.create_subprocess_exec(
                str(self.go_scanner_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=json.dumps(scan_input).encode('utf-8')),
                timeout=options.get("timeout", 30) + 10
            )
            
            if proc.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                self.logger.error(f"Go 掃描器執行失敗: {error_msg}")
                return {
                    "assets": [],
                    "metadata": {"engine": "go"},
                    "error": error_msg
                }
            
            # 解析結果
            result = json.loads(stdout.decode('utf-8', errors='ignore'))
            assets = result.get("assets", [])
            
            self.logger.info(f"✅ Go 引擎完成: {len(assets)} 個資產")
            
            return {
                "assets": assets,
                "metadata": {
                    "engine": "go",
                    "scanner_type": result.get("scanner_type", "ssrf"),
                    "execution_time": result.get("execution_time", 0)
                },
                "error": None
            }
            
        except asyncio.TimeoutError:
            self.logger.error("Go 掃描器執行超時")
            return {
                "assets": [],
                "metadata": {"engine": "go"},
                "error": "超時"
            }
        except Exception as e:
            self.logger.error(f"Go 引擎錯誤: {str(e)}")
            return {
                "assets": [],
                "metadata": {"engine": "go"},
                "error": str(e)
            }

# 使用範例 (異步)
async def main():
    adapter = GoAdapter()
    results = await adapter.scan(
        targets=["http://localhost:3000"],
        options={"scan_id": "test_001", "concurrency": 10}
    )
    print(f"Found {len(results['assets'])} vulnerabilities")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 實戰測試

### 測試環境準備

```powershell
# 1. 啟動 WebGoat 靶場（Docker）
docker run -d -p 8080:8080 webgoat/webgoat

# 2. 啟動 OWASP Juice Shop
docker run -d -p 3000:3000 bkimminich/juice-shop

# 3. 驗證靶場運行
curl http://localhost:8080/WebGoat
curl http://localhost:3000
```

### 測試 1: WebGoat SSRF 檢測

```powershell
# 創建測試腳本
$testScript = @'
$input = @"
{
  "scan_id": "webgoat_ssrf_test",
  "targets": ["http://localhost:8080/WebGoat/SSRF/task1"],
  "concurrency": 5,
  "timeout": 15
}
"@

Write-Host "=== WebGoat SSRF 掃描測試 ===" -ForegroundColor Cyan
$input | .\bin\ssrf-scanner.exe 2>$null | ConvertFrom-Json | ForEach-Object {
    Write-Host "`n狀態: $($_.status)" -ForegroundColor Green
    Write-Host "掃描耗時: $($_.execution_time)s"
    Write-Host "目標數: $($_.targets_scanned)"
    Write-Host "請求數: $($_.requests_made)"
    Write-Host "發現資產: $($_.assets.Count)" -ForegroundColor Yellow
    
    if ($_.assets.Count -gt 0) {
        Write-Host "`n🚨 檢測到漏洞:" -ForegroundColor Red
        $_.assets | Select-Object -First 3 | ForEach-Object {
            Write-Host "  [$($_.severity)] $($_.name)"
            Write-Host "  置信度: $($_.confidence)"
            Write-Host "  參數: $($_.details.vulnerable_param)"
            Write-Host "  證據: $($_.details.evidence.indicators_found -join ', ')" -ForegroundColor Gray
            Write-Host ""
        }
    } else {
        Write-Host "`n✅ 未發現 SSRF 漏洞（或目標未登錄）" -ForegroundColor Green
    }
}
'@

$testScript | Out-File -Encoding UTF8 test_webgoat_ssrf.ps1

# 執行測試
.\test_webgoat_ssrf.ps1
```

### 測試 2: Juice Shop 誤判驗證

```powershell
# 測試修復後的檢測器是否還會誤判登錄頁面
$input = @'
{
  "scan_id": "juice_shop_test",
  "targets": ["http://localhost:3000"],
  "concurrency": 3,
  "timeout": 10
}
'@

Write-Host "=== Juice Shop 誤判測試 ===" -ForegroundColor Cyan
$result = $input | .\bin\ssrf-scanner.exe 2>$null | ConvertFrom-Json

if ($result.assets.Count -eq 0) {
    Write-Host "✅ 測試通過: 正確識別為正常應用，無誤報" -ForegroundColor Green
} else {
    Write-Host "❌ 測試失敗: 仍有誤報" -ForegroundColor Red
    $result.assets | ForEach-Object {
        Write-Host "  誤報: $($_.name)"
    }
}
```

### 測試 3: 並發性能測試

```powershell
# 測試不同並發數的性能
$targets = @(
    "http://localhost:8080/WebGoat/SSRF/task1",
    "http://localhost:8080/WebGoat/SSRF/task2",
    "http://localhost:3000",
    "http://localhost:3001"
)

foreach ($concurrency in @(1, 5, 10)) {
    $input = @{
        scan_id = "perf_test_$concurrency"
        targets = $targets
        concurrency = $concurrency
        timeout = 15
    } | ConvertTo-Json
    
    Write-Host "`n測試並發數: $concurrency" -ForegroundColor Yellow
    $result = $input | .\bin\ssrf-scanner.exe 2>$null | ConvertFrom-Json
    Write-Host "  耗時: $($result.execution_time)s"
    Write-Host "  請求數: $($result.requests_made)"
    Write-Host "  成功率: $([math]::Round($result.success_count / $result.targets_scanned * 100, 2))%"
}
```

---

## 檢測邏輯詳解

### 1. 避免誤判機制（核心修復）

基於 **PortSwigger** 和 **OWASP** 最佳實踐，實現了精確的響應驗證：

```go
// ❌ 舊邏輯（會誤判）
if strings.Contains(body, "password") {
    return true  // 登錄頁面也有 password 字段！
}

// ✅ 新邏輯（精確驗證）
func isSSRFVulnerable(statusCode int, body, payload string) bool {
    // 1. 排除常見錯誤頁面
    if isCommonErrorPage(body) {  // 檢查 <title>login</title>
        return false
    }
    
    // 2. 針對不同 payload 精確驗證
    if strings.Contains(payload, "169.254.169.254") {
        return isAWSMetadataResponse(body, statusCode)  // 驗證 IMDS 格式
    }
    
    if strings.Contains(payload, "file://") {
        return isFileProtocolResponse(body, statusCode)  // 驗證 /etc/passwd 格式
    }
    
    // 3. 內網地址檢查響應差異
    if isInternalIP(payload) {
        if statusCode == 401 || statusCode == 403 {
            return true  // 成功到達內部服務
        }
        if statusCode == 200 && !isLoginPage(body) {
            return containsInternalServiceIndicators(body)
        }
    }
    
    return false
}
```

### 2. AWS IMDS 精確檢測

```go
func isAWSMetadataResponse(body string, statusCode int) bool {
    if statusCode != 200 {
        return false
    }
    
    // AWS IMDS 響應特徵：純文本，無 HTML
    if strings.Contains(body, "<html") || strings.Contains(body, "<body") {
        return false
    }
    
    // 檢查 IMDS 特定內容
    if strings.Contains(body, "ami-id") || 
       strings.Contains(body, "instance-id") ||
       strings.Contains(body, "security-credentials") {
        return true
    }
    
    // 簡短純文本（如 ami-id 值）
    if len(body) < 500 && !strings.Contains(body, "<") {
        return true
    }
    
    return false
}
```

### 3. File Protocol 檢測

```go
func isFileProtocolResponse(body string, statusCode int) bool {
    if statusCode != 200 {
        return false
    }
    
    // 檢查 /etc/passwd 格式：root:x:0:0:...
    if strings.Contains(body, "root:x:0:0") {
        return true
    }
    
    // 驗證 Unix 用戶列表格式
    lines := strings.Split(body, "\n")
    for _, line := range lines {
        // username:x:uid:gid:info:home:shell
        if strings.Count(line, ":") >= 6 && !strings.Contains(line, "<") {
            return true
        }
    }
    
    return false
}
```

### 4. 支持的 Payload 類型

| Payload 類型 | 目標 | 檢測方法 | 嚴重性 |
|-------------|------|---------|--------|
| AWS IMDS v1 | `http://169.254.169.254/latest/meta-data/` | 驗證元數據格式 | HIGH |
| AWS IMDS v2 | `http://169.254.169.254/latest/api/token` | 檢查 token 響應 | HIGH |
| GCP Metadata | `http://metadata.google.internal/...` | 驗證 JSON 格式 | HIGH |
| File Protocol | `file:///etc/passwd` | 驗證文件格式 | HIGH |
| Localhost Admin | `http://127.0.0.1/admin` | 檢查內部服務 | MEDIUM |
| Private Network | `http://192.168.1.1/` | 檢查響應差異 | MEDIUM |

### 5. 證據收集

每個檢測到的漏洞都包含完整證據：

```json
{
  "evidence": {
    "request_url": "http://target.com/api?url=...",
    "response_status": 200,
    "response_headers": {...},
    "indicators_found": [
      "Found /etc/passwd root entry",
      "Unix password file format detected"
    ]
  }
}
```

---

## 故障排除

### 問題 1: 編譯失敗

```powershell
# 錯誤: cannot find module
go: module github.com/kyle0527/aiva@latest found

# 解決方案
cd C:\D\fold7\AIVA-git\services\scan\engines\go_engine
go mod tidy
go build -o bin/ssrf-scanner.exe ./cmd/ssrf-scanner
```

### 問題 2: JSON 解析失敗

```powershell
# 錯誤: invalid character 'e' after top-level value

# 原因: stderr 混入 stdout
# 解決方案: 重定向 stderr
$input | .\bin\ssrf-scanner.exe 2>$null | ConvertFrom-Json
```

### 問題 3: 掃描無結果

```powershell
# 原因 1: 目標確實無漏洞（正常）
# 原因 2: 靶場需要登錄
# 原因 3: 參數名不匹配

# 解決方案: 檢查目標是否可訪問
curl http://localhost:8080/WebGoat/SSRF/task1

# 查看完整日誌
$input | .\bin\ssrf-scanner.exe 2>&1 | Out-File -Encoding UTF8 debug.log
cat debug.log
```

### 問題 4: 編碼錯誤

```powershell
# 錯誤: invalid UTF-8 sequence

# 原因: 中文字符編碼問題（已修復）
# 現版本使用英文描述，不應出現此問題

# 如仍出現，檢查 PowerShell 編碼
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

### 問題 5: 靶場連接失敗

```powershell
# 檢查 Docker 容器
docker ps | Select-String "webgoat|juice-shop"

# 重啟容器
docker restart <container_id>

# 查看容器日誌
docker logs <container_id>
```

---

## 性能調優

### 1. 並發控制

```json
{
  "concurrency": 10  // 根據系統資源調整
}
```

**建議值**:
- 本地測試: 1-5
- 生產環境: 10-20
- 高性能服務器: 50+

### 2. 超時設置

```json
{
  "timeout": 15  // 秒
}
```

**建議值**:
- 內網掃描: 5-10秒
- 外網掃描: 15-30秒
- 慢速目標: 60秒+

### 3. Worker Pool 配置

Worker Pool 自動管理並發執行：

```go
// internal/common/worker_pool.go
pool := NewWorkerPool(concurrency)
pool.Start()
defer pool.Stop()

// 提交任務
pool.Submit(func() {
    // 執行掃描任務
})
```

### 4. 記憶體優化

```powershell
# 設置 Go GC 策略
$env:GOGC = "100"  # 預設值，降低會更頻繁 GC

# 限制最大記憶體
$env:GOMEMLIMIT = "2GiB"

# 執行掃描
$input | .\bin\ssrf-scanner.exe
```

### 5. 日誌級別

```go
// 生產環境: 僅輸出 Error 和 Warn
logger, _ := zap.NewProduction()

// 開發環境: 輸出所有級別
logger, _ := zap.NewDevelopment()
```

---

## 常見問題 (FAQ)

### Q1: 為什麼 WebGoat 掃描結果為 0？

**A**: 這是正常的！修復後的檢測器會正確識別登錄頁面，不會誤報。如果 WebGoat 返回登錄頁面（未登錄狀態），檢測器會自動排除，避免誤判。

要測試真實檢測，需要：
1. 登錄 WebGoat
2. 或使用已登錄的 session cookie
3. 或測試其他已認證的 SSRF 端點

### Q2: 如何驗證檢測器是否正常工作？

**A**: 使用多個測試場景：

```powershell
# 1. 測試登錄頁面（應該 0 個結果）
echo '{"scan_id":"test1","targets":["http://localhost:3000"]}' | .\bin\ssrf-scanner.exe

# 2. 測試明顯的 SSRF endpoint（如果存在）
echo '{"scan_id":"test2","targets":["http://vulnerable-app.com/api?url=..."]}' | .\bin\ssrf-scanner.exe

# 3. 檢查日誌輸出
.\bin\ssrf-scanner.exe 2>&1 | Out-File debug.log
```

### Q3: 如何添加自定義 Payload？

**A**: 修改 `internal/ssrf/detector/ssrf.go`:

```go
testPayloads := []struct {
    name        string
    url         string
    description string
}{
    // ... 現有 payload
    {
        name:        "Custom Internal API",
        url:         "http://internal-api.local/v1/",
        description: "Attempt to access custom internal API",
    },
}
```

### Q4: 為什麼不支持 POST 請求？

**A**: 當前版本聚焦於 GET 請求的 SSRF 檢測。POST 支持將在後續版本添加。

### Q5: 如何與 CI/CD 集成？

**A**: 使用 PowerShell 腳本或 Python 適配器：

```yaml
# .github/workflows/security-scan.yml
- name: Run SSRF Scan
  run: |
    $input = '{"scan_id":"ci_scan","targets":["${{ secrets.TEST_URL }}"]}}'
    $result = $input | .\bin\ssrf-scanner.exe 2>$null | ConvertFrom-Json
    if ($result.assets.Count -gt 0) {
      Write-Error "Found $($result.assets.Count) vulnerabilities"
      exit 1
    }
```

---

## 參考資源

### 安全最佳實踐
- [OWASP SSRF](https://owasp.org/www-community/attacks/Server_Side_Request_Forgery)
- [PortSwigger SSRF](https://portswigger.net/web-security/ssrf)
- [CWE-918](https://cwe.mitre.org/data/definitions/918.html)

### Go 開發資源
- [Go Concurrency Patterns](https://go.dev/blog/pipelines)
- [Effective Go](https://go.dev/doc/effective_go)
- [Go Security Checklist](https://github.com/guardrailsio/awesome-golang-security)

### AIVA 相關文檔
- [架構設計](../../README.md)
- [Python 適配器](../coordinators/engines/go_adapter.py)
- [多引擎協調](../coordinators/multi_engine_coordinator.py)

---

**文檔版本**: 2.1.0  
**最後更新**: 2025-11-21  
**維護者**: AIVA Security Team  
**支持**: 如有問題請提交 Issue
