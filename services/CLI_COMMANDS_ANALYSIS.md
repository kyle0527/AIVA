# AIVA CLI 指令分析報告
## 人類下令 vs AI 下令的命令體系

分析時間: 2025-12-14  
分析角度: 人類操作 CLI vs AI 調用 CommandHandler

---

## 📊 執行摘要

AIVA 系統存在**兩套獨立的命令體系**:

| 命令體系 | 使用者 | 界面類型 | 目的 |
|---------|-------|---------|------|
| **人類 CLI** | 👤 開發者/維運人員 | 終端機命令行 | 開發、測試、維護 |
| **AI Command** | 🤖 AI Commander | Python API | 生產環境自動化 |

**關鍵區別**:
- 人類 CLI: 面向**開發者**的工具集合
- AI Command: 面向**AI**的統一接口

---

## 👤 人類 CLI 命令 (開發者工具)

### 1. 系統管理類

#### 1.1 Lifecycle CLI (工具生命週期管理)
**文件**: `services/integration/capability/lifecycle_cli.py`

```bash
# 互動式界面管理工具
python -m services.integration.capability.lifecycle_cli

# 功能選單
1. 📦 安裝工具 (install)
2. 🔄 更新工具 (update)
3. 🗑️ 卸載工具 (uninstall)
4. 📊 查看工具狀態 (status)
5. 📝 列出所有工具 (list)
6. 🔍 搜尋工具 (search)
7. 💊 修復工具 (repair)
8. 🧹 清理無用工具 (cleanup)
```

**特性**:
- ✅ Rich UI 美化界面
- ✅ 互動式選單系統
- ✅ 進度條顯示
- ✅ 彩色輸出
- 🎯 **用途**: 開發者手動管理工具安裝

**使用場景**:
```bash
# 開發者安裝新工具
$ python lifecycle_cli.py install sqlmap

# 開發者查看工具狀態
$ python lifecycle_cli.py status

# 開發者更新工具
$ python lifecycle_cli.py update xsstrike
```

---

#### 1.2 AIVA Rich CLI (主系統界面)
**文件**: `services/core/ui/rich_cli.py`

```bash
# 啟動主系統 CLI
python -m services.core.ui.rich_cli

# 功能
1. 🎯 能力管理 (Capability Management)
   - 列出所有能力
   - 查看能力詳情
   - 啟用/停用能力
   - 測試能力

2. 🔧 系統管理 (System Management)
   - 系統狀態
   - 配置查看
   - 日誌查看

3. 🚀 快速操作 (Quick Actions)
   - 健康檢查
   - 工具測試
   - 系統診斷
```

**特性**:
- ✅ 基於 HackingTool 的 Rich UI 設計
- ✅ 樹狀目錄顯示
- ✅ 彩色表格
- ✅ 互動式提示
- 🎯 **用途**: 開發者系統管理和監控

---

#### 1.3 備份與清理腳本

**Backup CLI**:
```bash
# services/integration/scripts/backup.py
python backup.py --config path/to/config.json
python backup.py --dry-run  # 測試模式
python backup.py --verify   # 驗證備份
```

**Cleanup CLI**:
```bash
# services/integration/scripts/cleanup.py
python cleanup.py
```

🎯 **用途**: 維運人員進行系統維護

---

### 2. 開發測試類

#### 2.1 Payload Generator CLI
**文件**: `services/integration/capability/payload_generator.py`

```bash
# 生成測試 Payload
python -m services.integration.capability.payload_generator

# 互動式選單
1. 生成 XSS Payload
2. 生成 SQLi Payload
3. 生成 SSRF Payload
4. 生成 IDOR Payload
5. 生成混合 Payload
```

🎯 **用途**: 測試人員生成測試數據

---

#### 2.2 Recon CLI
**文件**: `services/integration/capability/function_recon.py`

```bash
# 偵察工具命令行
python -m services.integration.capability.function_recon --target example.com
```

🎯 **用途**: 安全測試人員手動偵察

---

#### 2.3 Live Target Scanner CLI
**文件**: `services/scan/coordinators/target_generators/live_target_scanner.py`

```bash
# 即時目標掃描
python live_target_scanner.py --url https://example.com
python live_target_scanner.py --urls "url1,url2,url3"
python live_target_scanner.py --strategy fast
python live_target_scanner.py --max-depth 5
python live_target_scanner.py --exclude "/admin,/private"
```

🎯 **用途**: 測試人員手動掃描目標

---

#### 2.4 Go Engine Builder CLI
**文件**: `services/scan/engines/go_engine/dispatcher/build.py`

```bash
# 構建 Go 掃描引擎
python build.py --platform linux
python build.py --platform windows --arch amd64
python build.py --optimize size
python build.py --verbose
python build.py --output ./bin/scanner
```

🎯 **用途**: 開發者構建多平台二進制文件

---

#### 2.5 SQL 工具 CLI 集合

**HackingTool SQL CLI**:
```bash
# services/features/function_sqli/hackingtool_sql_cli.py
python hackingtool_sql_cli.py
```

**SQL Injection CLI**:
```bash
# services/features/function_sqli/integration_tools/sql_tools.py
python sql_tools.py
```

**Bounty Hunter CLI**:
```bash
# services/features/function_sqli/integration_tools/bounty_hunter.py
python bounty_hunter.py
```

🎯 **用途**: 測試人員手動進行 SQL 注入測試

---

#### 2.6 Web Attack CLI
**文件**: `services/features/function_web_scanner/integration_tools/web_tools.py`

```bash
# Web 攻擊工具
python web_tools.py
```

🎯 **用途**: 測試人員手動進行 Web 攻擊測試

---

### 3. 探索與分析類

#### 3.1 AIVA Exploration Pipeline
**文件**: 
- `services/core/aiva_core/internal_exploration/aiva_exploration_pipeline.py`
- `services/core/aiva_core/internal_exploration/python_tools/aiva_exploration_pipeline.py`

```bash
# 執行系統內部探索
python aiva_exploration_pipeline.py --target-dir /path/to/analyze
python aiva_exploration_pipeline.py --verbose
```

🎯 **用途**: 開發者分析代碼庫

---

#### 3.2 Self-Healing Analyzer
**文件**: `services/core/aiva_core/internal_exploration/self_healing/analyze_results.py`

```bash
# 分析代碼健康度
python analyze_results.py
```

🎯 **用途**: 開發者進行代碼質量分析

---

### 4. 外部工具整合類

#### 4.1 Forensic Tools CLI
**文件**: `services/features/function_forensic/external_tools/volatility3/volatility3/cli/__init__.py`

```bash
# Volatility3 記憶體分析
vol.py -f memory.dump windows.pslist
vol.py -f memory.dump windows.cmdline
```

🎯 **用途**: 數位鑑識人員手動分析

---

#### 4.2 其他工具 CLI
- **Steganography Tools**: 隱寫術工具
- **Reverse Engineering Tools**: 逆向工程工具
- **Registry Tools**: 註冊表工具

🎯 **用途**: 各類專業測試人員

---

## 🤖 AI Command (AI 調用接口)

### 核心架構

```python
AI Commander (任務指揮官)
    ↓
AICommandCenter (命令中心)
    ↓
CommandHandler (命令處理器)
    ↓
具體功能實現
```

### AI 可用的命令類型

**文件**: `services/aiva_common/command_center.py`

#### 1. Scan 掃描命令

```python
# Phase 0: 快速掃描
command = AICommand(
    command_type=CommandType.SCAN_PHASE0,
    payload={
        "target_urls": ["https://example.com"],
        "strategy": "fast"
    }
)

# Phase 1: 深度掃描
command = AICommand(
    command_type=CommandType.SCAN_PHASE1,
    payload={
        "target_urls": ["https://example.com"],
        "scan_depth": "deep"
    }
)

# Phase 2: 綜合掃描
command = AICommand(
    command_type=CommandType.SCAN_COMPREHENSIVE,
    payload={
        "target_urls": ["https://example.com"],
        "full_analysis": True
    }
)
```

**Handler**: `services/scan/command_handler.py` - `ScanCommandHandler`

---

#### 2. XSS 檢測命令

```python
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={
        "target_url": "https://example.com/search?q=",
        "test_payloads": ["<script>alert(1)</script>", ...],
        "scan_mode": "aggressive"
    }
)
```

**Handler**: `services/features/function_xss/command_handler.py` - `XSSCommandHandler`

---

#### 3. SQL 注入檢測命令

```python
command = AICommand(
    command_type=CommandType.FEATURE_SQLI_TEST,
    payload={
        "target_url": "https://example.com/login",
        "parameters": ["username", "password"],
        "dbms": "MySQL"
    }
)
```

**Handler**: `services/features/function_sqli/command_handler.py` - `SQLiCommandHandler`

---

#### 4. SSRF 檢測命令

```python
command = AICommand(
    command_type=CommandType.FEATURE_SSRF_TEST,
    payload={
        "target_url": "https://example.com/api/fetch",
        "test_internal_ips": True
    }
)
```

**Handler**: `services/features/function_ssrf/command_handler.py` - `SSRFCommandHandler`

---

#### 5. IDOR 檢測命令

```python
command = AICommand(
    command_type=CommandType.FEATURE_IDOR_TEST,
    payload={
        "target_url": "https://example.com/api/user/123",
        "id_parameter": "user_id",
        "test_range": [1, 1000]
    }
)
```

**Handler**: `services/features/function_idor/command_handler.py` - `IDORCommandHandler`

---

#### 6. Search 搜尋命令

```python
# Google 搜尋
command = AICommand(
    command_type=CommandType.SEARCH_GOOGLE,
    payload={"query": "site:example.com vulnerabilities"}
)

# DuckDuckGo 搜尋
command = AICommand(
    command_type=CommandType.SEARCH_DUCKDUCKGO,
    payload={"query": "security bug"}
)

# GitHub 搜尋
command = AICommand(
    command_type=CommandType.SEARCH_GITHUB,
    payload={"query": "XSS vulnerability", "language": "python"}
)
```

**Handler**: `services/integration/search_command_handler.py` - `SearchCommandHandler`

---

#### 7. Health Check 健康檢查

```python
command = AICommand(
    command_type=CommandType.HEALTH_CHECK,
    payload={}
)
```

**Handler**: 各個 CommandHandler 都支持

---

### AI 命令執行流程

```python
# 1. AI Commander 分析任務
task = "掃描 example.com 尋找 XSS 漏洞"

# 2. 生成 AICommand
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={
        "target_url": "https://example.com",
        "scan_mode": "deep"
    }
)

# 3. 通過 CommandCenter 執行
result = await command_center.execute(command)

# 4. AI 獲得結果並繼續任務
if result.status == CommandStatus.SUCCESS:
    vulnerabilities = result.data['vulnerabilities']
    # AI 分析結果並規劃下一步
```

---

## 🔄 兩套體系對比

### 架構對比

| 特性 | 人類 CLI | AI Command |
|------|---------|------------|
| **界面** | 終端機命令行 | Python API |
| **輸入** | 文字參數 | 結構化 Payload |
| **輸出** | 美化的文字/表格 | JSON 結構 |
| **錯誤處理** | 友好的錯誤消息 | 異常/狀態碼 |
| **交互性** | 互動式提示 | 完全自動化 |
| **適用場景** | 開發/測試/維運 | 生產環境 |

---

### 功能對比

#### XSS 檢測範例

**人類 CLI** (不存在，需手動調用 Python):
```bash
# 假設有這樣的 CLI (實際不存在)
$ aiva xss --url https://example.com --payload "<script>alert(1)</script>"
```

**AI Command**:
```python
command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={"target_url": "https://example.com"}
)
result = await command_center.execute(command)
```

**關鍵差異**:
- 人類: 需要知道具體語法和參數
- AI: 只需要知道 CommandType，CommandHandler 內部處理細節

---

### 使用場景對比

#### 場景 1: 工具安裝

**人類**:
```bash
$ python lifecycle_cli.py
選擇: 1 (安裝工具)
輸入工具 ID: sqlmap
✅ 安裝成功
```

**AI**:
```python
# AI 不需要安裝工具
# CommandHandler 內部已經整合好所有工具
command = AICommand(
    command_type=CommandType.FEATURE_SQLI_TEST,
    payload={...}
)
# 直接執行，無需安裝
```

---

#### 場景 2: 掃描目標

**人類**:
```bash
$ python live_target_scanner.py \
  --url https://example.com \
  --strategy comprehensive \
  --max-depth 5

[進度條顯示]
✅ 掃描完成
[彩色表格顯示結果]
```

**AI**:
```python
command = AICommand(
    command_type=CommandType.SCAN_COMPREHENSIVE,
    payload={
        "target_urls": ["https://example.com"],
        "max_depth": 5
    }
)
result = await command_center.execute(command)
# AI 自動分析 result.data
```

---

#### 場景 3: 系統監控

**人類**:
```bash
$ python rich_cli.py
選擇: 系統狀態
[顯示彩色儀表板]
- CPU: 45%
- 記憶體: 2.1GB
- 活躍工具: 15
```

**AI**:
```python
# AI 通過 Observability 系統監控
# 不需要 CLI
from services.core.aiva_core.observability import metrics_collector
metrics = await metrics_collector.get_system_metrics()
```

---

## 📋 完整命令清單

### 人類 CLI 命令 (15+)

| 分類 | 命令 | 文件 | 用途 |
|------|------|------|------|
| **系統管理** | lifecycle_cli | lifecycle_cli.py | 工具生命週期管理 |
| | rich_cli | rich_cli.py | 主系統界面 |
| | backup | backup.py | 系統備份 |
| | cleanup | cleanup.py | 系統清理 |
| **開發測試** | payload_generator | payload_generator.py | 生成測試數據 |
| | recon_cli | function_recon.py | 偵察工具 |
| | live_scanner | live_target_scanner.py | 即時掃描 |
| | go_builder | build.py | 構建 Go 引擎 |
| **SQL 測試** | hackingtool_sql_cli | hackingtool_sql_cli.py | SQL 工具集 |
| | sql_injection_cli | sql_tools.py | SQL 注入測試 |
| | bounty_hunter_cli | bounty_hunter.py | 漏洞獎勵工具 |
| **其他** | web_attack_cli | web_tools.py | Web 攻擊工具 |
| | exploration_pipeline | aiva_exploration_pipeline.py | 代碼探索 |
| | self_healing_analyzer | analyze_results.py | 代碼分析 |
| | volatility_cli | volatility3/cli | 記憶體分析 |

---

### AI Command 類型 (7+)

| CommandType | Handler | 功能 |
|------------|---------|------|
| `SCAN_PHASE0` | ScanCommandHandler | 快速掃描 |
| `SCAN_PHASE1` | ScanCommandHandler | 深度掃描 |
| `SCAN_COMPREHENSIVE` | ScanCommandHandler | 綜合掃描 |
| `FEATURE_XSS_TEST` | XSSCommandHandler | XSS 檢測 |
| `FEATURE_SQLI_TEST` | SQLiCommandHandler | SQL 注入檢測 |
| `FEATURE_SSRF_TEST` | SSRFCommandHandler | SSRF 檢測 |
| `FEATURE_IDOR_TEST` | IDORCommandHandler | IDOR 檢測 |
| `SEARCH_GOOGLE` | SearchCommandHandler | Google 搜尋 |
| `SEARCH_DUCKDUCKGO` | SearchCommandHandler | DuckDuckGo 搜尋 |
| `SEARCH_GITHUB` | SearchCommandHandler | GitHub 搜尋 |
| `HEALTH_CHECK` | All Handlers | 健康檢查 |

---

## 🎯 設計理念

### 1. 清晰的職責劃分

**人類 CLI**:
- 🎯 目的: 開發、測試、維護
- 👥 用戶: 開發者、測試人員、維運人員
- 🖥️ 界面: 友好的命令行界面
- 🎨 特色: Rich UI、互動式、彩色輸出

**AI Command**:
- 🎯 目的: 生產環境自動化
- 🤖 用戶: AI Commander
- 🔌 界面: Python API
- ⚡ 特色: 高效、結構化、可編程

---

### 2. 無重疊設計

**關鍵原則**: 
> 人類 CLI 和 AI Command **不互相調用**

```
人類 CLI ❌ 不調用 ❌ AI Command
AI Command ❌ 不調用 ❌ 人類 CLI

兩者平行共存，各司其職
```

**例外情況**:
- 開發者可以使用 CLI 測試 CommandHandler
- 但這是開發測試行為，不是生產行為

---

### 3. 抽象層次清晰

```
┌─────────────────────┬─────────────────────┐
│   人類 CLI 層級      │   AI Command 層級    │
├─────────────────────┼─────────────────────┤
│ 終端機命令           │ AI Commander        │
│   ↓                 │   ↓                 │
│ Python CLI 腳本      │ AICommand           │
│   ↓                 │   ↓                 │
│ 直接調用工具         │ CommandCenter       │
│                     │   ↓                 │
│                     │ CommandHandler      │
│   ↓                 │   ↓                 │
└─────────────────────┴─────────────────────┘
         底層工具實現 (共享)
```

---

## 📚 使用指南

### 給開發者: 使用人類 CLI

#### 安裝新工具
```bash
cd services/integration/capability
python lifecycle_cli.py

# 選擇: 1 - 安裝工具
# 輸入工具 ID: xsstrike
```

#### 查看系統狀態
```bash
cd services/core/ui
python rich_cli.py

# 選擇: 系統管理 → 系統狀態
```

#### 測試掃描功能
```bash
cd services/scan/coordinators/target_generators
python live_target_scanner.py --url https://example.com --strategy fast
```

#### 分析代碼健康度
```bash
cd services/core/aiva_core/internal_exploration/self_healing
python analyze_results.py
```

---

### 給 AI: 使用 AI Command

#### 執行 XSS 掃描
```python
from services.aiva_common.command_center import AICommandCenter
from services.aiva_common.ai_command import AICommand, CommandType

command_center = AICommandCenter()

command = AICommand(
    command_type=CommandType.FEATURE_XSS_TEST,
    payload={
        "target_url": "https://example.com",
        "scan_mode": "comprehensive"
    }
)

result = await command_center.execute(command)

if result.status == CommandStatus.SUCCESS:
    vulnerabilities = result.data.get('vulnerabilities', [])
    # 處理結果
```

#### 執行綜合掃描
```python
command = AICommand(
    command_type=CommandType.SCAN_COMPREHENSIVE,
    payload={
        "target_urls": ["https://example.com"],
        "scan_options": {
            "check_xss": True,
            "check_sqli": True,
            "check_ssrf": True
        }
    }
)

result = await command_center.execute(command)
```

---

## ⚠️ 常見誤區

### 誤區 1: 混淆兩套體系

❌ **錯誤做法**:
```python
# AI 試圖調用人類 CLI
import subprocess
subprocess.run(["python", "lifecycle_cli.py", "install", "sqlmap"])
```

✅ **正確做法**:
```python
# AI 使用 CommandHandler
command = AICommand(
    command_type=CommandType.FEATURE_SQLI_TEST,
    payload={...}
)
```

---

### 誤區 2: 為 AI 創建 CLI

❌ **錯誤設計**:
```bash
# 為 AI 創建命令行工具
$ aiva-ai-command execute --type xss --payload '{...}'
```

✅ **正確設計**:
```python
# AI 直接調用 Python API
command = AICommand(command_type=CommandType.FEATURE_XSS_TEST, ...)
result = await command_center.execute(command)
```

**理由**: 
- AI 不需要命令行界面
- Python API 更高效、類型安全
- 避免序列化/反序列化開銷

---

### 誤區 3: 重複功能

❌ **不好的做法**:
```python
# 為同一個功能創建兩套接口
class XSSScanner:
    def cli_scan(self, args):  # 給 CLI 用
        ...
    
    def api_scan(self, payload):  # 給 AI 用
        ...
```

✅ **好的做法**:
```python
# CommandHandler 是統一的內部實現
class XSSCommandHandler:
    async def handle_command(self, command: AICommand):
        # 內部實現，被 AI 調用
        ...

# CLI 單獨實現（如果需要）
def cli_main():
    # 直接調用工具，不通過 CommandHandler
    scanner = XSSManager()
    scanner.scan(...)
```

---

## 🔮 未來擴展

### 可能添加的人類 CLI

1. **部署 CLI**: 自動化部署腳本
2. **監控 CLI**: 實時監控儀表板
3. **配置 CLI**: 系統配置管理
4. **報告 CLI**: 生成測試報告

### 可能添加的 AI Command

1. **配置管理命令**: AI 動態調整配置
2. **學習命令**: AI 學習新的攻擊模式
3. **協調命令**: 多工具協同執行

---

## 📊 統計數據

### 人類 CLI 統計

| 類別 | 數量 | 百分比 |
|------|------|--------|
| 系統管理類 | 4 | 27% |
| 開發測試類 | 4 | 27% |
| 安全測試類 | 4 | 27% |
| 探索分析類 | 2 | 13% |
| 外部工具類 | 1 | 6% |
| **總計** | **15** | **100%** |

### AI Command 統計

| 類別 | 數量 | 百分比 |
|------|------|--------|
| 掃描命令 | 3 | 27% |
| 漏洞檢測命令 | 4 | 36% |
| 搜尋命令 | 3 | 27% |
| 系統命令 | 1 | 10% |
| **總計** | **11** | **100%** |

---

## ✨ 總結

### 核心要點

1. ✅ **兩套獨立體系**: 人類 CLI 和 AI Command 各司其職
2. ✅ **清晰的職責**: 開發維護 vs 生產自動化
3. ✅ **無重疊設計**: 不互相調用，平行共存
4. ✅ **正確的抽象**: 人類友好 vs API 高效

### 設計原則

| 原則 | 說明 |
|------|------|
| **分離關注點** | CLI 關注開發者體驗，API 關注自動化 |
| **單一職責** | 每個工具只做一件事 |
| **抽象層次** | 人類和 AI 看到不同的抽象層次 |
| **可擴展性** | 兩套體系可以獨立擴展 |

### 關鍵教訓

> "CLI Registry 試圖讓 AI 使用 CLI 工具，這違反了設計原則。AI 應該使用 CommandHandler API，而不是命令行工具。"

**正確的設計**:
- 人類: 使用 CLI 工具進行開發和測試
- AI: 使用 CommandHandler API 進行自動化任務
- 兩者平行共存，各自優化

---

**報告版本**: v1.0  
**分析完成時間**: 2025-12-14  
**核心結論**: AIVA 擁有清晰分離的雙命令體系，人類 CLI 用於開發維護，AI Command 用於生產自動化
