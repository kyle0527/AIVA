# AIVA Integration Capability Tools (數位鑑識與輔助工具)

> **⚠️ 架構更新注意 (AIVA v2.0):**
> AIVA 系統已經棄用了原有的 `CapabilityRegistry` 與 `BaseCapability` 註冊系統。所有的服務整合現在皆遵循 `CommandHandler` 架構（參見 [aiva_common/core/command_center.py](../../aiva_common/core/command_center.py)）或轉型為獨立 CLI 工具。
>
> 詳情請參閱 [上層整合架構文件](../README.md)。

本目錄包含了 AIVA 系統中用於特定領域的獨立輔助工具集，包含數位鑑識 (Forensics)、隱寫術 (Steganography)、逆向工程 (Reverse Engineering) 以及 Bug Bounty 報告生成等。這些工具目前主要以 CLI 或腳本形式提供給安全研究員或 AI 進行調用。

## 📑 目錄

- [📋 概述](#概述)
- [✨ 包含的工具集](#包含的工具集)
- [🏗️ 系統架構](#系統架構)
- [🛠️ 安裝和配置](#安裝和配置)
- [🧩 整合 aiva_common](#整合-aivacommon)
- [🐛 故障排除](#故障排除)
- [🤝 貢獻指南](#貢獻指南)
  - [使用的工具和插件](#使用的工具和插件)
  - [遵循的標準](#遵循的標準)
- [📊 監控和診斷](#監控和診斷)
  - [健康檢查](#健康檢查)
  - [記分卡系統](#記分卡系統)
  - [日誌和追蹤](#日誌和追蹤)
- [🔧 開發和擴展](#開發和擴展)
  - [添加新的能力類型](#添加新的能力類型)
  - [自定義發現邏輯](#自定義發現邏輯)
  - [添加新的測試類型](#添加新的測試類型)
- [🐛 故障排除](#故障排除)
---

## 📋 概述

本模組集成了從開源安全工具（如 HackingTool）移植而來的多種進階安全分析工具。它們已被重構以符合 AIVA 的資料模型，並可作為獨立管理器運行。

## ✨ 包含的工具集

### 🔍 數位鑑識工具 (`forensic_tools.py`)
- **Autopsy**: 開源數位鑑識平台。
- **Wireshark**: 網路封包分析工具。
- **BulkExtractor**: 快速資料提取工具。
- **Guymager**: 磁碟映像建立工具。

### 🎭 隱寫術工具 (`steganography_tools.py`)
- 提供對多媒體檔案進行資料隱藏與檢測的工具。
- 支援 Steghide、Stegcracker 等後端。

### 📱 逆向工程工具 (`reverse_engineering_tools.py`)
- 行動應用程式（主要是 Android）的逆向與分析工具。

### 💰 Bug Bounty 報告系統 (`bug_bounty_reporting.py`)
- 協助安全研究員生成標準化漏洞報告。
- 支援各層級漏洞的 PoC 自動生成。

## 🏗️ 系統架構

```
AIVA Capability Tools
├── 📄 models.py                       # 統一資料模型 (基於 aiva_common)
├── 🔧 forensic_tools.py               # 數位鑑識管理器
├── 🔧 steganography_tools.py          # 隱寫術管理器
├── 🔧 reverse_engineering_tools.py    # 逆向工程管理器
└── 🔧 bug_bounty_reporting.py         # Bug Bounty 報告管理器
```

## 🛠️ 安裝和配置

### 前置需求

請確保系統已經安裝對應的後端工具，如 nmap, wireshark, git 等。

### 安裝步驟

1. **確保 `aiva_common` 可用**
   ```bash
   # 檢查 aiva_common 路徑
   ls services/aiva_common/
   ```

2. **配置系統**
   ```bash
   # 複製預設配置檔案
   cp capability_registry.yaml config/
   
   # 根據需要編輯配置
   nano config/capability_registry.yaml
   ```

3. **初始化資料庫**
   ```bash
   # 自動建立資料庫表格（首次啟動時）
   python -m services.integration.capability.start_registry --info
   ```

## 🚀 快速開始

### 1. 快速啟動模式

```bash
# 快速啟動並自動發現能力
python -m services.integration.capability.start_registry --quick-start
```

### 2. 正常啟動服務

```bash
# 使用預設配置啟動
python -m services.integration.capability.start_registry

# 使用自訂配置啟動
python -m services.integration.capability.start_registry --config my_config.yaml

# 開發模式啟動
python -m services.integration.capability.start_registry --dev
```

### 3. 僅執行能力發現

```bash
# 掃描並顯示發現的能力
python -m services.integration.capability.start_registry --discover-only
```

## 🖥️ 命令行工具

能力註冊中心提供了豐富的 CLI 工具：

### 發現和註冊

```bash
# 發現系統中的能力
python -m services.integration.capability.cli discover

# 發現並自動註冊
python -m services.integration.capability.cli discover --auto-register
```

### 查看和管理

```bash
# 列出所有能力
python -m services.integration.capability.cli list

# 按語言篩選
python -m services.integration.capability.cli list --language python

# 按類型篩選
python -m services.integration.capability.cli list --type scanner

# 詳細檢查特定能力
python -m services.integration.capability.cli inspect security.sqli.boolean_detection
```

### 測試和驗證

```bash
# 測試能力連接性
python -m services.integration.capability.cli test security.sqli.boolean_detection

# 驗證能力定義檔案
python -m services.integration.capability.cli validate capability.yaml
```

### 文件和綁定產生

```bash
# 產生特定能力的文件
python -m services.integration.capability.cli docs security.sqli.boolean_detection

# 產生系統摘要報告
python -m services.integration.capability.cli docs --all --output report.md

# 產生跨語言綁定
python -m services.integration.capability.cli bindings security.sqli.boolean_detection --languages python go rust
```

### 統計資訊

```bash
# 顯示系統統計
python -m services.integration.capability.cli stats
```

## 🔌 API 使用

啟動服務後，API 文件可在以下位置存取：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### 主要 API 端點

#### 能力管理
```http
GET    /capabilities              # 列出所有能力
POST   /capabilities              # 註冊新能力
GET    /capabilities/{id}         # 獲取特定能力
```

#### 發現和統計
```http
POST   /discover                  # 手動觸發能力發現
GET    /stats                     # 獲取系統統計資訊
```

### API 使用示例

```python
import httpx
import asyncio

async def example_api_usage():
    async with httpx.AsyncClient() as client:
        # 獲取所有 Python 能力
        response = await client.get(
            "http://localhost:8000/capabilities",
            params={"language": "python"}
        )
        capabilities = response.json()
        
        # 觸發能力發現
        response = await client.post("http://localhost:8000/discover")
        discovery_stats = response.json()
        
        print(f"發現了 {discovery_stats['discovered_count']} 個能力")

# 執行示例
asyncio.run(example_api_usage())
```

## 📝 能力定義格式

### YAML 格式示例

```yaml
id: "security.sqli.boolean_detection"
name: "SQL 注入布爾盲注檢測"
description: "檢測 Web 應用中的 SQL 注入布爾盲注漏洞"
version: "1.0.0"
module: "function_sqli"
language: "python"
entrypoint: "services.features.function_sqli.worker:run_boolean_sqli"
capability_type: "scanner"

inputs:
  - name: "url"
    type: "str"
    required: true
    description: "目標 URL"
    validation_rules:
      format: "url"
  
  - name: "timeout"
    type: "int"
    required: false
    description: "超時時間(秒)"
    default: 30
    validation_rules:
      min: 1
      max: 300

outputs:
  - name: "vulnerabilities"
    type: "List[Dict]"
    description: "發現的漏洞列表"
    sample_value:
      - type: "sqli_boolean"
        severity: "high"
        parameter: "id"

tags: ["security", "sqli", "web", "injection"]
category: "vulnerability_scanner"
prerequisites: ["network.connectivity"]
dependencies: ["security.http.client"]
timeout_seconds: 300
priority: 80
```

### JSON 格式示例

```json
{
  "id": "network.scanner.port_scan",
  "name": "高性能端口掃描器",
  "description": "使用 Go 實現的高性能 TCP 端口掃描器",
  "version": "1.0.0",
  "module": "port_scanner_go",
  "language": "go",
  "entrypoint": "http://localhost:8081/scan",
  "capability_type": "scanner",
  "inputs": [
    {
      "name": "target",
      "type": "str",
      "required": true,
      "description": "目標主機或 IP 地址"
    },
    {
      "name": "ports",
      "type": "List[int]",
      "required": true,
      "description": "要掃描的端口列表"
    }
  ],
  "outputs": [
    {
      "name": "open_ports", 
      "type": "List[int]",
      "description": "開放的端口列表"
    }
  ],
  "tags": ["network", "port", "scan", "tcp"],
  "timeout_seconds": 120
}
```

## 🔧 配置選項

### 資料庫配置

```yaml
database:
  path: "capability_registry.db"          # SQLite 資料庫路徑
  backup_enabled: true                    # 是否啟用備份
  backup_interval_hours: 24               # 備份間隔
  max_backups: 7                          # 最大備份數量
```

### 發現配置

```yaml
discovery:
  auto_discovery_enabled: true            # 自動發現開關
  discovery_interval_minutes: 60          # 發現間隔
  scan_directories:                       # 掃描目錄
    - "services/features"
    - "services/scan"
  exclude_patterns:                       # 排除模式
    - "__pycache__"
    - "*.pyc"
```

### 監控配置

```yaml
monitoring:
  health_check_enabled: true              # 健康檢查開關
  health_check_interval_minutes: 15       # 檢查間隔
  alert_thresholds:                       # 告警閾值
    max_latency_ms: 5000
    min_success_rate: 95.0
```

### API 配置

```yaml
api:
  host: "0.0.0.0"                        # 綁定主機
  port: 8000                             # 綁定端口
  debug: false                           # 調試模式
  docs_enabled: true                     # API 文件
  cors_enabled: true                     # CORS 支援
```

## 🧩 整合 aiva_common

本系統深度整合了 `aiva_common` 的功能：

### 使用的標準化列舉

- `ProgrammingLanguage`: 程式語言定義
- `Severity`: 嚴重級別定義  
- `Confidence`: 信心等級定義
- `TaskStatus`: 任務狀態定義

### 使用的工具和插件

- `schema_validator.py`: 模式驗證工具
- `schema_codegen_tool.py`: 程式碼產生工具
- `module_connectivity_tester.py`: 連接性測試工具

### 遵循的標準

- **結構化日誌**: 使用 `aiva_common.utils.logging`
- **追蹤ID**: 使用 `aiva_common.utils.ids`
- **統一配置**: 遵循 12-Factor App 原則
- **錯誤處理**: 標準化的異常處理模式

## 📊 監控和診斷

### 健康檢查

系統提供多層次的健康檢查：

1. **基本連接性測試**: 檢查能力的入口點是否可達
2. **功能性測試**: 使用示例輸入測試能力的基本功能
3. **性能測試**: 監控響應時間和資源使用情況
4. **依賴檢查**: 驗證所有依賴項是否可用

### 記分卡系統

每個能力都有詳細的記分卡，包括：

- **可用性**: 過去一段時間的可用性百分比
- **成功率**: 成功執行的百分比
- **性能指標**: 平均延遲、P95/P99 延遲
- **錯誤統計**: 錯誤計數和分類
- **趨勢分析**: 性能趨勢評估

### 日誌和追蹤

- **結構化日誌**: JSON Lines 格式，便於解析和分析
- **分散式追蹤**: 每個操作都有唯一的追蹤 ID
- **操作審計**: 記錄所有重要的系統操作

## 🔧 開發和擴展

### 添加新的能力類型

```python
from services.integration.capability.models import CapabilityType

# 擴展能力類型列舉
class ExtendedCapabilityType(CapabilityType):
    AI_MODEL = "ai_model"
    DATA_PIPELINE = "data_pipeline"
    WORKFLOW = "workflow"
```

### 自定義發現邏輯

```python
from services.integration.capability.registry import CapabilityRegistry

class CustomRegistry(CapabilityRegistry):
    async def discover_custom_capabilities(self):
        # 實現自定義發現邏輯
        pass
```

### 添加新的測試類型

```python
from services.integration.capability.toolkit import CapabilityToolkit

class ExtendedToolkit(CapabilityToolkit):
    async def custom_health_check(self, capability):
        # 實現自定義健康檢查
        pass
```

## 🐛 故障排除

### 常見問題

1. **資料庫鎖定錯誤**
   ```bash
   # 檢查並終止占用資料庫的進程
   lsof capability_registry.db
   ```

2. **端口被占用**
   ```bash
   # 更改配置檔案中的端口
   api:
     port: 8001
   ```

3. **能力發現失敗**
   ```bash
   # 檢查掃描目錄是否存在
   python -m services.integration.capability.cli discover --verbose
   ```

4. **依賴項缺失**
   ```bash
   # 檢查 aiva_common 路徑
   ls -la services/aiva_common/
   ```

### 調試模式

```bash
# 啟用詳細日誌
python -m services.integration.capability.start_registry --dev

# 檢查配置
python -m services.integration.capability.start_registry --info
```

## 📚 API 參考

完整的 API 參考文件請參閱：
- Swagger UI: http://localhost:8000/docs
- 或查看自動產生的 OpenAPI 規範檔案

## 🤝 貢獻指南

1. 遵循 `aiva_common` 的程式碼風格和標準
2. 確保所有新功能都有相應的測試
3. 更新文件和範例
4. 使用結構化日誌記錄重要操作

## 📄 許可證

本專案遵循 AIVA 系統的整體許可證條款。

## 📞 支援

如需技術支援，請：
1. 查看本文件的故障排除部分
2. 檢查系統日誌檔案
3. 使用 CLI 工具進行診斷
4. 聯繫 AIVA 開發團隊

---

**AIVA 能力註冊中心** - 讓多語言服務管理變得簡單而強大 🚀