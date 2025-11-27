# AIVA 能力註冊中心

## 📑 目錄

- [📋 概述](#概述)
- [✨ 主要功能](#主要功能)
  - [🔍 智能能力發現](#智能能力發現)
  - [📊 統一能力管理](#統一能力管理)
  - [💚 健康監控](#健康監控)
  - [🔧 開發者工具](#開發者工具)
  - [🚀 高性能架構](#高性能架構)
- [🏗️ 系統架構](#系統架構)
- [🛠️ 安裝和配置](#安裝和配置)
  - [前置需求](#前置需求)
  - [安裝步驟](#安裝步驟)
- [🚀 快速開始](#快速開始)
  - [1. 快速啟動模式](#1-快速啟動模式)
  - [2. 正常啟動服務](#2-正常啟動服務)
  - [3. 僅執行能力發現](#3-僅執行能力發現)
- [🖥️ 命令行工具](#命令行工具)
  - [發現和註冊](#發現和註冊)
  - [查看和管理](#查看和管理)
  - [測試和驗證](#測試和驗證)
  - [文件和綁定產生](#文件和綁定產生)
  - [統計資訊](#統計資訊)
- [🔌 API 使用](#api-使用)
  - [主要 API 端點](#主要-api-端點)
    - [能力管理](#能力管理)
    - [發現和統計](#發現和統計)
  - [API 使用示例](#api-使用示例)
- [📝 能力定義格式](#能力定義格式)
  - [YAML 格式示例](#yaml-格式示例)
  - [JSON 格式示例](#json-格式示例)
- [🔧 配置選項](#配置選項)
  - [資料庫配置](#資料庫配置)
  - [發現配置](#發現配置)
  - [監控配置](#監控配置)
  - [API 配置](#api-配置)
- [🧩 整合 aiva_common](#整合-aivacommon)
  - [使用的標準化列舉](#使用的標準化列舉)
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
  - [常見問題](#常見問題)
  - [調試模式](#調試模式)
- [📚 API 參考](#api-參考)
- [🤝 貢獻指南](#貢獻指南)
- [📄 許可證](#許可證)
- [📞 支援](#支援)

---


## 📋 概述

AIVA 能力註冊中心是一個統一的服務發現和管理平台，專為 AIVA 系統的多語言、多模組架構設計。它提供了自動化的能力發現、註冊、監控和管理功能，完全遵循 `aiva_common` 的標準和最佳實踐。

## ✨ 主要功能

### 🔍 智能能力發現
- **自動掃描**: 自動發現 Python、Go、Rust、TypeScript 模組中的能力
- **動態註冊**: 實時註冊新發現的能力，無需手動配置
- **依賴分析**: 自動分析能力間的依賴關係

### 📊 統一能力管理
- **標準化介面**: 基於 `aiva_common` 規範的統一資料模型
- **類型安全**: 完整的 Pydantic v2 類型驗證和序列化
- **豐富的元數據**: 支援詳細的能力描述、參數定義和配置選項

### 💚 健康監控
- **即時監控**: 持續監控所有已註冊能力的健康狀態
- **性能分析**: 收集延遲、成功率、資源使用等關鍵指標
- **智能告警**: 基於可配置閾值的自動告警機制

### 🔧 開發者工具
- **跨語言綁定**: 自動產生多種語言的客戶端程式碼
- **API 文件**: 自動產生 OpenAPI/Swagger 文件
- **CLI 工具**: 豐富的命令行管理介面

### 🚀 高性能架構
- **異步處理**: 基於 FastAPI 和 asyncio 的高性能異步架構
- **SQLite 儲存**: 輕量級但功能完整的持久化存儲
- **RESTful API**: 標準 REST API，支援 JSON 和 YAML 格式

## 🏗️ 系統架構

```
AIVA 能力註冊中心
├── 📦 registry.py          # 核心註冊中心服務
├── 📄 models.py            # 統一資料模型 (基於 aiva_common)
├── 🔧 toolkit.py           # 能力管理工具集
├── ⚙️ config.py            # 配置管理
├── 🖥️ cli.py               # 命令行介面
├── 🚀 start_registry.py    # 服務啟動腳本
└── 📚 examples.py          # 使用示例
```

## 🛠️ 安裝和配置

### 前置需求

```bash
# Python 3.9+
python --version

# 必要的依賴套件
pip install fastapi uvicorn pydantic sqlite3 aiofiles aiohttp
```

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