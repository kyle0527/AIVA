# AIVA Scan 模組使用者手冊

> **版本**: v2.1 (適配器模式)  
> **最後更新**: 2025年11月21日  
> **適用對象**: AIVA 系統管理員、安全測試人員  
> **架構**: AI 命令中心 (取代 RabbitMQ)

---

## 📋 目錄

1. [快速開始](#快速開始) ✅ 已驗證 2025-11-23
2. [架構概覽](#架構概覽)
3. [兩階段掃描流程](#兩階段掃描流程)
4. [使用 AI 命令接口](#使用-ai-命令接口) ✅ 已驗證 2025-11-23
5. ~~監控掃描進度~~ ⚠️ 內容缺失,待補充
6. ~~查看掃描結果~~ ⚠️ 內容缺失,待補充
7. ~~故障排除~~ ⚠️ 內容缺失,待補充
8. ~~進階配置~~ ⚠️ 內容缺失,待補充

> **⚠️ 重要提示**: 本文檔目前不完整,第5-8章節內容尚未編寫。  
> 相關內容可參考 `SCAN_MODULE_RESTORATION_PLAN.md`。

---

## 🚀 快速開始

### 前置要求

```bash
# 1. 確認環境
✅ Python 3.11+
✅ 所有依賴已安裝 (pip install -r requirements.txt)

# 2. 檢查測試目標（可選）
docker ps | grep -E "juice-shop|webgoat"
```

### 30 秒快速測試

> ✅ **已驗證**: 2025年11月23日 - 命令流程正確運作

```python
# test_quick_scan.py
import asyncio
from services.aiva_common.command_center import get_command_center
from services.aiva_common.schemas import AICommand, CommandType
from services.scan.command_handler import ScanCommandHandler

async def quick_test():
    # 1. 取得命令中心
    command_center = get_command_center()
    
    # 2. 註冊 Scan 模組處理器
    scan_handler = ScanCommandHandler()
    command_center.register_module("scan", scan_handler)
    
    # 3. 建立 Phase 0 命令
    command = AICommand(
        command_id="scan_test_001_phase0",      # 必填: 唯一命令 ID
        command_type=CommandType.SCAN_PHASE0,
        target_module="scan",
        payload={
            "scan_id": "scan_test_001",         # 必須以 'scan_' 開頭
            "targets": ["http://localhost:3000"]  # Juice Shop 靶場
        }
    )
    
    # 4. 執行掃描
    result = await command_center.execute(command)
    
    # 5. 檢查結果
    print(f"狀態: {result.status}")
    print(f"執行時間: {result.execution_time:.2f}秒")
    if result.result:
        print(f"掃描結果: {result.result}")

# 執行測試
asyncio.run(quick_test())
```

**重要注意事項**:
- ✅ `command_id` 是必填欄位
- ✅ `scan_id` 必須以 `scan_` 開頭
- ✅ 必須先註冊 Scan 處理器
- ✅ targets 使用實際可訪問的 URL

---

## 🏗️ 架構概覽

### 核心組件 (v2.1 - 適配器模式)

```
┌─────────────────────────────────────────────────────────────┐
│                    AI 命令中心 (Core 模組)                   │
│                                                              │
│  Phase 0 決策 → Rust 快速偵察 → 分析結果 → Phase 1 決策     │
└─────────────────────────────────────────────────────────────┘
                            ↓ ↑
                    數據合約 (Pydantic)
                            ↓ ↑
┌─────────────────────────────────────────────────────────────┐
│              Scan 模組 - MultiEngineCoordinator              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        適配器層 (coordinators/engines/)              │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  Python Adapter  │  TypeScript Adapter               │  │
│  │  Rust Adapter    │  Go Adapter                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           四引擎並行執行 (asyncio.gather)            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 數據流向

```
用戶 → AI 命令中心 → Scan 模組 → AI 命令中心 → 後續處理
           ↓                          ↑
    SCAN_PHASE0 命令          Phase 0 結果
           ↓                          ↑
    SCAN_PHASE1 命令          Phase 1 結果
```

---

## 🎯 兩階段掃描流程

### Phase 0: 快速偵察 (5-10 分鐘)

**目標**: 快速獲取目標的基本資訊

**執行內容**:
- ✅ 目標可達性檢測
- ✅ 技術棧指紋識別 (Web Server, Framework, CMS)
- ✅ 敏感資訊掃描 (API Keys, Passwords, Tokens)
- ✅ 基礎端點發現 (深度 1，最多 50 個 URL)
- ✅ 初步攻擊面評估

**使用引擎**: Rust (高性能)

**輸出數據**:
```json
{
  "scan_id": "scan_abc123",
  "status": "success",
  "execution_time": 450.5,
  "assets": [
    {
      "asset_id": "asset_001",
      "type": "url",
      "value": "https://example.com/api/users",
      "has_form": false
    }
  ],
  "fingerprints": {
    "web_server": {"nginx": "1.21.0"},
    "frameworks": {"react": "18.2.0"},
    "technologies": ["JavaScript", "REST API"]
  },
  "summary": {
    "urls_found": 45,
    "forms_found": 3,
    "apis_found": 8
  }
}
```

### Phase 1: 深度掃描 (10-30 分鐘，按需)

**觸發條件** (Core 模組 AI 決策):
- 發現大量 JavaScript (使用 TypeScript 引擎)
- 發現 HTML 表單 (使用 Python 引擎)
- 發現 REST API (使用 Python 引擎)
- 需要高並發掃描 (使用 Go 引擎)

**執行內容**:
- ✅ 深度爬取 (深度 3-5)
- ✅ 動態內容渲染 (SPA, React, Vue)
- ✅ 表單參數提取
- ✅ API 端點深度分析
- ✅ 入口點完整發現

**使用引擎**: Python, TypeScript, Go, Rust (組合使用)

**輸出數據**:
```json
{
  "scan_id": "scan_abc123",
  "status": "success",
  "execution_time": 1250.8,
  "assets": [
    {
      "asset_id": "asset_100",
      "type": "form",
      "value": "https://example.com/login",
      "parameters": ["username", "password", "csrf_token"],
      "has_form": true
    }
  ],
  "engine_results": {
    "python": {"status": "completed", "findings": 120},
    "typescript": {"status": "completed", "findings": 85}
  },
  "phase0_summary": {
    "urls": 45,
    "execution_time": 450.5
  }
}
```

---

## 🎯 使用 AI 命令接口

> ✅ **已驗證**: 2025年11月23日 - 命令接口使用方式正確

### 基本掃描流程

#### 1. 取得命令中心並註冊處理器

```python
from services.aiva_common.command_center import get_command_center
from services.scan.command_handler import ScanCommandHandler

# 取得命令中心
command_center = get_command_center()

# 註冊 Scan 模組處理器
scan_handler = ScanCommandHandler()
command_center.register_module("scan", scan_handler)
```

#### 2. 建立掃描命令

**Phase 0 掃描 (快速偵察)**:
```python
from services.aiva_common.schemas import AICommand, CommandType

phase0_command = AICommand(
    command_id="scan_001_phase0",           # 必填: 唯一命令 ID
    command_type=CommandType.SCAN_PHASE0,
    target_module="scan",
    payload={
        "scan_id": "scan_001",              # 必須以 'scan_' 開頭
        "targets": ["http://localhost:3000"],
        "timeout": 600,                      # 10 分鐘
        "max_depth": 1                       # 基礎端點發現
    }
)
```

**Phase 1 掃描 (深度掃描)**:
```python
phase1_command = AICommand(
    command_id="scan_001_phase1",           # 必填: 唯一命令 ID
    command_type=CommandType.SCAN_PHASE1,
    target_module="scan",
    payload={
        "scan_id": "scan_001",              # 必須以 'scan_' 開頭
        "targets": ["http://localhost:3000"],
        "selected_engines": ["python", "typescript"],  # AI 決定的引擎組合
        "timeout": 1800,                     # 30 分鐘
        "max_depth": 3                       # 深度爬取
    }
)
```

#### 3. 執行掃描

```python
import asyncio

async def run_scan():
    # 執行 Phase 0
    phase0_result = await command_center.execute(phase0_command)
    print(f"Phase 0 狀態: {phase0_result.status}")
    print(f"Phase 0 耗時: {phase0_result.execution_time:.2f}秒")
    
    # 根據 Phase 0 結果決定是否執行 Phase 1
    if phase0_result.status in ["success", "completed"]:
        result_data = phase0_result.result
        if result_data and result_data.get("summary", {}).get("urls_found", 0) > 0:
            phase1_result = await command_center.execute(phase1_command)
            print(f"Phase 1 狀態: {phase1_result.status}")
            print(f"Phase 1 耗時: {phase1_result.execution_time:.2f}秒")

# 執行
asyncio.run(run_scan())
```

### 完整範例腳本

> ✅ **已驗證**: 2025年11月23日 - 代碼邏輯正確

**example_scan.py**:
```python
"""
AIVA Scan 完整掃描範例
"""
import asyncio
import time
from services.aiva_common.command_center import get_command_center
from services.aiva_common.schemas import AICommand, CommandType
from services.scan.command_handler import ScanCommandHandler

async def complete_scan(target_url: str):
    """執行完整的兩階段掃描"""
    # 初始化命令中心
    command_center = get_command_center()
    scan_handler = ScanCommandHandler()
    command_center.register_module("scan", scan_handler)
    
    scan_id = f"scan_{int(time.time())}"
    
    # Phase 0: 快速偵察
    print(f"[Phase 0] 開始快速偵察: {target_url}")
    phase0_cmd = AICommand(
        command_id=f"{scan_id}_phase0",         # 必填: 唯一命令 ID
        command_type=CommandType.SCAN_PHASE0,
        target_module="scan",
        payload={
            "scan_id": scan_id,                  # 必須 scan_ 前綴
            "targets": [target_url]
        }
    )
    
    phase0_result = await command_center.execute(phase0_cmd)
    if phase0_result.status not in ["success", "completed"]:
        print(f"[Phase 0] 失敗: {phase0_result.error}")
        return
    
    result_data = phase0_result.result
    assets_count = len(result_data.get("assets", [])) if result_data else 0
    print(f"[Phase 0] 完成，發現 {assets_count} 個資產")
    
    # Phase 1: 深度掃描（根據 Phase 0 結果決定）
    if assets_count > 0 and result_data.get("summary", {}).get("urls_found", 0) > 5:
        print(f"[Phase 1] 開始深度掃描")
        
        phase1_cmd = AICommand(
            command_id=f"{scan_id}_phase1",      # 必填: 唯一命令 ID
            command_type=CommandType.SCAN_PHASE1,
            target_module="scan",
            payload={
                "scan_id": scan_id,              # 必須 scan_ 前綴
                "targets": [target_url],
                "selected_engines": ["python", "typescript"]  # 正確欄位名
            }
        )
        
        phase1_result = await command_center.execute(phase1_cmd)
        if phase1_result.status in ["success", "completed"]:
            phase1_data = phase1_result.result
            phase1_assets = len(phase1_data.get("assets", [])) if phase1_data else 0
            print(f"[Phase 1] 完成，總計 {phase1_assets} 個資產")
        else:
            print(f"[Phase 1] 失敗: {phase1_result.error}")
    else:
        print("[Phase 1] Phase 0 結果已足夠，跳過深度掃描")

if __name__ == "__main__":
    asyncio.run(complete_scan("http://localhost:3000"))
```

### 驗證掃描執行

```bash
# 執行範例腳本
python example_scan.py

# 預期輸出:
# [Phase 0] 開始快速偵察: http://localhost:3000
# [Phase 0] 完成，發現 15 個資產
# [Phase 1] 開始深度掃描
# [Phase 1] 完成，總計 47 個資產
```

---
