# 🌐 SSRF 檢測模組

**版本**: v2.0 | **狀態**: ✅ 生產就緒 | **更新**: 2025-12-12

**什麼是 SSRF？**  
Server-Side Request Forgery（伺服器端請求偽造）是一種攻擊技術，政擊者誘使伺服器對內部或第三方系統發起非預期的HTTP請求。本模組整合了內網探測、雲端元數據訪問、協議測試、DNS rebinding 和帶外（OAST）驗證等多重檢測技術。

## 📚 快速導航

- [🚀 CLI 使用方式](#-cli-使用方式) - **推薦：無需 MQ，直接測試**
- [⚡ 新增強化功能](#-新增強化功能) - v2.0 特性
- [⚙️ 運作流程](#️-運作流程)
- [🔧 核心能力](#-核心能力)

## 🏗️ 架構圖
```
┌─────────────────────────────────────────────────────────────┐
│                   智能 SSRF 檢測架構                          │
├─────────────────────────────────────────────────────────────┤
│ AI Command      │command_handler │ SmartSSRFDetector│ OAST    │
│ Interface       │               │                  │ Service │
│       ↓         │       ↓       │        ↓         │    ↓    │
│ FEATURE_SSRF_   │ FunctionTask  │ ParamSemantics   │ oast_   │
│ TEST            │ Payload       │ Analyzer         │ dispatcher│
│       │         │               │internal_address_ │         │
│       └─────────┼───────────────┼─ detector        │    ↓    │
│                 │               │        ↓         │ Callback│
│                 ↓               │ 多協議測試 +      │ 驗證    │
│         SsrfTestVector          │ 雲端探測         │         │
│         (param_semantics)       │                  │         │
└─────────────────────────────────────────────────────────────┘
```

## ⚡ 新增強化功能

### v2.0 (2025-12-12) - 重大更新

- ✅ **移除降級功能**: 所有錯誤現在都會拋出異常，不再默默失敗
- ✅ **IP 編碼繞過**: 支援 Decimal、Hex、Octal、Mixed encoding
- ✅ **DNS Rebinding**: 支援 rebind.it、rbndr.us 等服務
- ✅ **雲端元數據強化**: 新增 AWS ECS、GCP、Azure、阿里雲、騰訊雲
- ✅ **協議繞過**: CRLF 注入、URL encoding、@ symbol bypass
- ✅ **更多協議**: 支援 gopher://、dict://、tftp:// 等 5+ 協議

## ⚡️ 運作流程
1. **參數語義分析** - 識別可能的 SSRF 注入點（URL、檔案路徑參數）
2. **測試向量生成** - 根據參數特性生成對應的測試 payload
3. **多層級檢測** - 執行分層檢測策略：
   - **內網探測**: 掃描 127.0.0.1、169.254.169.254（AWS 元數據）等
   - **協議測試**: file://, gopher://, dict://, ldap:// 等協議
   - **IP 繞過**: Decimal/Hex/Octal 編碼、DNS rebinding
   - **帶外驗證**: 使用 OAST 服務確認盲 SSRF
4. **結果整合** - 綜合直接響應和帶外回調確認漏洞

## 🚀 CLI 使用方式

### ⭐ 推薦：CLI 直接測試（無需 MQ）

**版本**: v2.0 新增 | **狀態**: ✅ 完整支援

```powershell
# 在專案根目錄執行

# 1. 基本 SSRF 檢測
python -m services.features.function_ssrf `
    --url "http://localhost:3000/api/fetch" `
    --param "url" `
    --timeout 30

# 2. 進階檢測（含 IP 繞過）
python -m services.features.function_ssrf `
    --url "http://localhost:3000/api/fetch" `
    --param "url" `
    --advanced `
    --timeout 30

# 3. DNS Rebinding 檢測
python -m services.features.function_ssrf `
    --url "http://localhost:3000/api/fetch" `
    --param "url" `
    --dns-rebinding `
    --timeout 60

# 4. 雲端 metadata 檢測
python -m services.features.function_ssrf `
    --url "http://vulnerable-app.com/import" `
    --param "source_url" `
    --method POST `
    --location body `
    --cloud aws,gcp,azure
```

**輸出格式** (JSON):
```json
{
  "target": "http://localhost:3000/api/fetch",
  "findings_count": 2,
  "vulnerable": true,
  "findings": [
    {
      "type": "SSRF_INTERNAL_ACCESS",
      "severity": "HIGH",
      "payload": "http://127.0.0.1/admin",
      "evidence": "Successfully accessed internal service"
    },
    {
      "type": "SSRF_CLOUD_METADATA",
      "severity": "CRITICAL",
      "payload": "http://169.254.169.254/latest/meta-data/",
      "evidence": "AWS metadata endpoint accessible"
    }
  ]
}
```

---

### 方式二：程式化調用（開發用）

**適用場景**: 整合測試、自動化腳本

```python
# 方式 A: 使用 worker.process_task（推薦）
import asyncio
from services.aiva_common.schemas import FunctionTaskPayload, FunctionTaskTarget
from services.features.function_ssrf.worker import process_task
from services.features.function_ssrf.param_semantics_analyzer import ParamSemanticsAnalyzer
from services.features.function_ssrf.internal_address_detector import InternalAddressDetector
from services.features.function_ssrf.oast_dispatcher import OastDispatcher
import httpx

async def test_ssrf():
    # 建立任務結構
    task = FunctionTaskPayload(
        task_id="task_ssrf_001",
        scan_id="scan_001",
        target=FunctionTaskTarget(
            url="http://localhost:3000/api/fetch",
            parameter="url",
            method="GET",
            parameter_location="query"
        ),
        strategy="normal"
    )
    
    async with httpx.AsyncClient(timeout=30) as client:
        result = await process_task(
            task,
            client=client,
            analyzer=ParamSemanticsAnalyzer(),
            detector=InternalAddressDetector(),
            dispatcher=OastDispatcher()
        )
    
    # 輸出結果
    print(f\"發現 {len(result.findings)} 個漏洞\")
    print(f\"嘗試次數: {result.telemetry.attempts}\")
    print(f\"OAST 回調: {result.telemetry.oast_callbacks}\")
    
    for finding in result.findings:
        print(f\"\\n漏洞類型: {finding.vulnerability.type}\")
        print(f\"嚴重程度: {finding.vulnerability.severity}\")
        print(f\"Payload: {finding.payload.request}\")

asyncio.run(test_ssrf())
```

**任務參數說明**:
- `task_id`: 任務唯一識別碼，必須以 `task_` 開頭
- `scan_id`: 掃描會話 ID，用於關聯多個任務
- `target.url`: 目標 URL，完整的 HTTP/HTTPS 位址
- `target.parameter`: 要測試的參數名稱（如 `url`, `file`, `path`）
- `target.method`: HTTP 方法 (`GET`, `POST`, `PUT`, `DELETE`)
- `target.parameter_location`: 參數位置 (`query`, `body`, `header`, `cookie`)
- `strategy`: 檢測策略 (`normal`, `aggressive`, `stealth`)

**process_task 函數參數**:
- `task`: FunctionTaskPayload 對象，包含檢測目標信息
- `client`: httpx.AsyncClient 實例，用於發送 HTTP 請求
- `analyzer`: ParamSemanticsAnalyzer 實例，用於分析參數語義
- `detector`: InternalAddressDetector 實例，用於檢測內網地址
- `dispatcher`: OastDispatcher 實例，用於處理帶外回調

**自動檢測功能**:
- ✅ **內網掃描**: 自動測試 127.0.0.1, localhost, 192.168.x.x
- ✅ **雲端 metadata**: 自動探測 AWS/GCP/Azure/阿里雲/騰訊雲
- ✅ **協議繞過**: 自動測試 file://, gopher://, dict:// 等
- ✅ **IP 編碼**: 自動生成 Decimal/Hex/Octal 編碼 payload
- ✅ **DNS Rebinding**: 自動使用 rebind.it, rbndr.us 服務
- ✅ **OAST 驗證**: 自動註冊並檢查帶外回調

**參數變化範例**:
```python
# GET 參數測試
target=FunctionTaskTarget(
    url="http://example.com/api/fetch",
    parameter="url",
    method="GET",
    parameter_location="query"
)

# POST Body 測試
target=FunctionTaskTarget(
    url="http://example.com/api/import",
    parameter="file_path",
    method="POST",
    parameter_location="body",
    json_data={"format": "json"}
)

# 圖片上傳 URL 測試
target=FunctionTaskTarget(
    url="http://example.com/api/upload_from_url",
    parameter="image_url",
    method="POST",
    parameter_location="body"
)
```

```python
import asyncio
import httpx
from services.aiva_common.schemas import FunctionTaskPayload, FunctionTaskTarget
from services.features.function_ssrf.smart_ssrf_detector import SmartSSRFDetector
from services.features.function_ssrf.param_semantics_analyzer import ParamSemanticsAnalyzer
from services.features.function_ssrf.internal_address_detector import InternalAddressDetector
from services.features.function_ssrf.oast_dispatcher import OastDispatcher

async def test_ssrf():
    task = FunctionTaskPayload(
        task_id="task_ssrf_001",
        scan_id="scan_001",
        target=FunctionTaskTarget(
            url="http://localhost:3000/api/fetch",
            parameter="url",
            method="GET",
            parameter_location="query"
        )
    )
    
    detector = SmartSSRFDetector()
    async with httpx.AsyncClient(timeout=30) as client:
        findings, metrics = await detector.detect_vulnerabilities(
            task,
            client=client,
            analyzer=ParamSemanticsAnalyzer(),
            detector=InternalAddressDetector(),
            dispatcher=OastDispatcher()
        )
    
    print(f"Found {len(findings)} vulnerabilities")
    print(f"Metrics: {metrics}")

asyncio.run(test_ssrf())
```

---

### 方式三：透過 Message Queue（已棄用）

**狀態**: ⚠️ 已棄用，請使用 CLI 方式

<details>
<summary>點擊查看舊版 MQ 方式（不推薦）</summary>

```python
# 🎯 正確的 SSRF 命令處理器使用方式 (2025-12-03 修正)
import asyncio
from services.features.function_ssrf.command_handler import SSRFCommandHandler
from services.aiva_common.schemas.commands import AICommand, CommandType

async def test_ssrf_command_handler():
    \"\"\"標準 SSRF 命令處理器執行示例\"\"\"
    
    # 1. 創建命令處理器
    handler = SSRFCommandHandler()
    
    # 2. 構建 AI 命令 (必須包含所有必填參數)
    command = AICommand(
        command_id=\"ssrf_test_001\",
        command_type=CommandType.FEATURE_SSRF_TEST,
        target_module=\"features.ssrf\",
        payload={
            \"target_url\": \"http://localhost:3000/api/fetch\",
            \"parameters\": {\"url\": \"http://example.com\"},
            \"detection_methods\": [\"callback\"],  # callback, blind, internal_scan
            \"callback_server\": None  # 可選回調服務器
        },
        timeout=60,  # 1分鐘超時
        
        # 必填追蹤參數 (符合 aiva_common 規範)
        trace_id=\"trace_ssrf_001\",
        session_id=\"session_001\", 
        parent_command_id=None,
        callback_url=None
    )
    
    # 3. 執行命令
    result = await handler.handle_command(command)
    
    # 4. 解析結果
    print(f\"✅ 執行狀態: {result.status}\")
    print(f\"✅ 執行成功: {result.success}\")
    print(f\"⏱️ 執行時間: {result.execution_time:.2f}秒\")
    
    if result.success:
        vulnerability_found = result.result.get(\"vulnerability_found\", False)
        detection_results = result.result.get(\"detection_results\", [])
        print(f\"🎯 發現漏洞: {vulnerability_found}\")
        print(f\"🔢 檢測結果: {len(detection_results)} 個\")
        
        for result_item in detection_results:
            print(f\"  - 方法: {result_item.get('method')}\")
            print(f\"  - 載荷: {result_item.get('payload')}\")
            print(f\"  - 證據: {result_item.get('evidence')}\")
    else:
        print(f\"❌ 執行失敗: {result.error}\")
        print(f\"🔍 錯誤代碼: {result.error_code}\")
        
    return result

# 執行測試
asyncio.run(test_ssrf_command_handler())
```

**修正重點 (2025-12-03)**:
- ✅ AICommand 必須包含 `trace_id`, `session_id` 等追蹤參數
- ✅ AICommandResult 使用 `started_at`, `completed_at` 而非舊的 `timestamp`
- ✅ 錯誤處理包含 `error_code` 和 `error_details`
- ✅ 檢測器方法調用: `detect_vulnerabilities(task, client=client)`
- ✅ 參數分析器調用: `analyzer.analyze(task)` 而非 `analyze_parameters()`

**實際測試結果範例：**
```
=== SSRF 引擎實際執行測試 ===

【測試 1】內網掃描...
  ✅ 內網掃描完成，發現 1 個問題
    - SSRF_INTERNAL_ACCESS: Internal host 127.0.0.1 is reachable via SSRF

【測試 2】檔案協議檢測...
  ✅ 檔案協議檢測完成，發現 1 個問題
    - SSRF_FILE_PROTOCOL: file:// access via SSRF may expose local files

=== 測試完成 ===
```

</details>

---

## 🔧 核心能力

- ✅ **多雲端支援**: AWS、GCP、Azure、阿里雲、騰訊雲 metadata 檢測
- ✅ **IP 編碼繞過**: Decimal、Hex、Octal、Mixed encoding
- ✅ **DNS Rebinding**: 支援 rebind.it、rbndr.us 等服務
- ✅ **協議繞過**: CRLF 注入、URL encoding、@ symbol bypass
- ✅ **多協議測試**: gopher://、dict://、tftp://、ldap://、smb://
- ✅ **OAST 驗證**: 帶外回調確認盲 SSRF
- ✅ **虛假回應過濾**: 內部服務驗證、WAF 干擾檢測
- ✅ **無降級模式**: 所有錯誤拋出異常，不會默默失敗

## 📝 更新日誌

### v2.0 (2025-12-12)
- ✅ 移除所有降級功能（continue、return []）
- ✅ 新增 DNS rebinding 檢測器
- ✅ 新增 IP 編碼繞過技術（17+ payloads）
- ✅ 擴展雲端 metadata 端點（AWS ECS、GCP、Azure、阿里雲、騰訊雲）
- ✅ 強化協議繞過（CRLF、URL encoding、@ bypass）
- ✅ 新增 CLI 入口
- ✅ 移除 MQ 依賴

### v1.0
- 初始版本，支援基本 SSRF 檢測

## 🎯 使用場景

### ✅ 適用場景

- **URL 參數功能**: 圖片獲取、網頁預覽、文件下載
- **Webhook 回調**: API 通知、第三方整合
- **代理服務**: 內容代理、API 轉發
- **雲端服務**: 檔案處理、數據同步

### ⚠️ 使用注意

- 避免掃描生產環境的關鍵內部服務
- 注意可能觸發的安全警報
- 謹慎測試雲端元數據端點
- 確保 OAST 回調域名的安全性
- 僅在授權的目標上進行測試

## 📚 相關文檔

- [dns_rebinding_detector.py](./dns_rebinding_detector.py) - DNS Rebinding 檢測器實現
- [param_semantics_analyzer.py](./param_semantics_analyzer.py) - 參數語義分析與 payload 生成
- [smart_ssrf_detector.py](./smart_ssrf_detector.py) - 智能 SSRF 檢測器
- [FALSE_POSITIVE_ANALYSIS.md](../FALSE_POSITIVE_ANALYSIS.md) - 虛假回應分析報告

### 如何使用？
```python
# 1. 基本內網探測
internal_scan = {
    "target_url": "https://webapp.com/proxy",
    "test_parameters": {"target_url": "http://example.com"},
    "internal_scan": True,
    "targets": [
        "127.0.0.1:22",      # SSH
        "127.0.0.1:3306",    # MySQL
        "127.0.0.1:6379",    # Redis
        "192.168.1.1"        # 內網閘道
    ]
}

# 2. 雲端元數據檢測
cloud_metadata = {
    "target_url": "https://cloud-app.com/fetch-image",
    "test_parameters": {"image_url": "https://example.com/pic.jpg"},
    "cloud_metadata": ["aws", "gcp", "azure"],
    "metadata_endpoints": [
        "http://169.254.169.254/latest/meta-data/",  # AWS
        "http://metadata.google.internal/",          # GCP
        "http://169.254.169.254/metadata/instance"   # Azure
    ]
}

# 3. 協議測試
protocol_test = {
    "target_url": "https://app.com/load-config",
    "test_parameters": {"config_path": "/app/config.json"},
    "protocols": ["file", "gopher", "dict", "ldap"],
    "payloads": [
        "file:///etc/passwd",
        "file:///c:/windows/win.ini",
        "gopher://127.0.0.1:25/",
        "dict://127.0.0.1:11211/stat"
    ]
}

# 4. 帶外（Blind SSRF）檢測
blind_ssrf = {
    "target_url": "https://app.com/webhook",
    "test_parameters": {"callback_url": "https://legitimate.com"},
    "oast_callback": True,
    "oast_service_url": "http://localhost:8083",
    "blind_payloads": [
        "http://{oast_token}.your-domain.com",
        "https://{oast_token}.your-domain.com/ssrf-test"
    ]
}

# 5. 綜合檢測（推薦）
comprehensive_ssrf = {
    "target_url": "https://complex-app.com/api/process",
    "test_parameters": {
        "data_source": "https://external-api.com/data",
        "backup_url": "https://backup.com/save",
        "template_path": "/templates/default.html"
    },
    "internal_scan": True,
    "cloud_metadata": ["aws", "gcp", "azure"],
    "protocols": ["http", "https", "file", "gopher", "dict"],
    "oast_callback": True,
    "follow_redirects": False,  # 避免重定向干擾
    "custom_headers": {
        "X-Forwarded-For": "127.0.0.1",
        "X-Real-IP": "169.254.169.254"
    }
}
```

## 🔧 核心能力
- **多協議支援**: HTTP/HTTPS/File/Gopher/Dict/LDAP 全覆蓋
- **雲端元數據檢測**: AWS/GCP/Azure 元數據服務探測
- **帶外驗證**: OAST 服務實現 blind SSRF 確認
- **智能參數分析**: 根據參數名稱和語義選擇測試策略
- **內網掃描**: 自動化內部服務發現和端口探測
- **繞過技術**: IP 編碼、重定向利用、DNS 重綁定

## 🎯 後續發展方向
- [ ] **容器環境檢測** - Docker socket、Kubernetes API 探測
- [ ] **IPv6 支援** - 擴展到 IPv6 網路環境檢測
- [ ] **時間盲注檢測** - 基於延遲的 SSRF 檢測技術
- [ ] **機器學習參數識別** - AI 自動識別潛在 SSRF 參數
- [ ] **響應差異分析** - 更智能的虛假陽性過濾
- [ ] **自動化 payload 優化** - 根據目標特徵動態調整測試向量