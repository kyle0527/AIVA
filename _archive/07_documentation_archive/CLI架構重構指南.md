# AIVA CLI 架構重構計劃

> **⚠️ 文檔狀態**: ⏸️ **已完成部分目標 - 後續以 AI 排序器方案為準**  
> **完成項目**: ✅ 移除 Coordinator/Dispatcher 冗餘層  
> **新方向**: 🆕 採用 AI 排序器方案（詳見 [SIMPLIFIED_DUAL_CLI_DESIGN.md](docs/05_implementation_guides/SIMPLIFIED_DUAL_CLI_DESIGN.md)）  
> **建議**: 本文件的重構目標已部分達成，後續實施以最新的 AI 排序器方案為主

**制定日期**: 2026年1月11日  
**目標**: 統一 CLI 架構，簡化模組間通訊，去除冗餘組件

---

## 📋 現狀分析

### 當前架構問題

```
❌ 複雜的三層架構（未被使用）
┌─────────────────────────────────────┐
│ Layer 3: Orchestrator (AI決策層)    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Layer 2: Coordinator (協調層)       │  ← 冗餘！
│  - CoreServiceCoordinator           │
│  - IntegrationCoordinator           │
│  - AttackCoordinator                │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ Layer 1: Dispatcher (消息路由層)    │  ← 冗餘！
│  - BaseDispatcher                   │
│  - CognitiveDispatcher              │
│  - ExplorationDispatcher            │
└─────────────────────────────────────┘
```

**實際情況**：
- ✅ AI 模組已使用 `subprocess + CLI + JSON` 直接通訊
- ❌ Dispatcher/Coordinator 存在但未被調用
- ❌ CLI 系統（276 Flows）完全孤立
- ❌ Features/Scan 模組 CLI 接口不完整

### 正確的架構（已被 AI 模組實現）

```
✅ 精簡的兩層架構
┌─────────────────────────────────────────────┐
│  AI (CapabilityOrchestrator)                │
│  - 發出需求到任務編排                       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  任務編排 (ExecutionPlanner)                │
│  - create_execution_plan() 生成計劃         │
│  - 返回計劃給 AI（不直接執行）              │
└─────────────────────────────────────────────┘
                    ↓ (返回計劃)
┌─────────────────────────────────────────────┐
│  AI (CapabilityOrchestrator)                │
│  - 審查/修改計劃                            │
│  - execute(plan) 直接用 subprocess 執行     │
└─────────────────────────────────────────────┘
                    ↓ (subprocess CLI)
┌──────────┬──────────┬──────────┬────────────┐
│ Features │ Scan     │ Core     │ Integration│
│ 模組     │ 模組     │ 模組     │ 模組       │
│          │          │          │            │
│ CLI入口  │ CLI入口  │ CLI入口  │ CLI入口    │
│ 返回JSON │ 返回JSON │ 返回JSON │ 返回JSON   │
└──────────┴──────────┴──────────┴────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  AI (CapabilityOrchestrator)                │
│  - json.loads(stdout) 接收結果              │
│  - 整合、學習、優化                         │
└─────────────────────────────────────────────┘
```

---

## 🎯 重構目標（基於實際驗證）

### 1. 統一 CLI 接口標準（✅ 正確方向）

**原則**：所有模組遵循相同的 CLI 模式，輸出簡單 JSON 給 AI 直接處理

**實際架構**（已在運行）：
```
AI (CapabilityOrchestrator)
  ↓ subprocess
CLI 模組
  ↓ stdout (JSON)
AI 直接處理
  ├─ json.loads(result["stdout"])
  ├─ 解析 findings
  ├─ 學習優化（telemetry）
  └─ 生成報告
```

**標準 CLI 輸出**（簡化版，AI 已驗證可用）：
```python
# 標準 CLI 模式（參考 function_xss/__main__.py）
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--action", choices=["scan", "exploit", "verify"])
    
    args = parser.parse_args()
    
    # 執行業務邏輯
    result = module.execute(args)
    
    # 輸出簡單 JSON（AI 可直接處理）
    print(json.dumps({
        "status": "completed|failed|timeout",
        "module": "module_name",
        "target": args.target,
        "findings": [...],  # 簡單結構即可
        "metadata": {...}   # 可選
    }, ensure_ascii=False))
```

**關鍵發現**：
- ❌ 不需要複雜的 `FeatureResult` 格式（20+ 欄位）
- ✅ AI 已經直接處理簡單 JSON
- ✅ 學習優化通過 `telemetry` 數據實現

### 2. 歸檔未使用組件（✅ 已驗證）

**歸檔到 `_archive/`**：
- ✅ `service_backbone/dispatcher_base.py`
- ✅ `cognitive_core/dispatcher.py`
- ✅ `service_backbone/coordination/core_service_coordinator.py`
- ✅ `integration/coordinators/base_coordinator.py`
- ✅ `integration/coordinators/dual_loop_coordinator.py`

**原因**：
- AI 已通過 `subprocess` 直接調用 CLI
- 不需要中間消息路由層
- 不需要服務協調層

### 3. 完善模組 CLI

**缺少 CLI 入口的模組**：
- ❌ `features/function_sqli/` - 需要 `__main__.py`
- ❌ `features/function_crypto/` - 需要 `__main__.py`
- ❌ `features/function_info_leak/` - 需要 `__main__.py`

**已有 CLI 的模組**：
- ✅ `features/function_xss/__main__.py`
- ✅ `features/function_ssrf/__main__.py`
- ✅ `features/function_idor/__main__.py`
- ✅ `features/function_bizlogic/__main__.py`

### 4. 擴展 Internal Exploration CLI

**當前狀態**：
- 📄 `classification_data.json` - 276 Flows（僅 AI 模組）

**擴展方向**：
```json
{
  "metadata": {
    "total_flows": 400,  // 目標：新增 ~120 flows
    "module_distribution": {
      "cognitive_core": 68,
      "service_backbone": 54,
      "task_planning": 36,
      "internal_exploration": 66,
      "core_capabilities": 25,
      // 新增：
      "features": 80,      // 攻擊功能模組
      "scan": 50,          // 掃描引擎模組
      "integration": 21    // 整合模組
    }
  },
  "flows": [
    // 現有 276 個 flows...
    
    // 新增 Features 模組 flows
    {
      "id": 277,
      "module": "features",
      "path": ["function_xss/xss_scanner.py"],
      "func_names": ["XSSScanner.scan"],
      "description": "XSS 漏洞掃描（Reflected）",
      "cli_command": "python -m features.function_xss --type reflected --url {target}"
    },
    {
      "id": 278,
      "module": "features",
      "path": ["function_sqli/sqli_scanner.py"],
      "func_names": ["SQLiScanner.scan"],
      "description": "SQL 注入掃描",
      "cli_command": "python -m features.function_sqli --url {target}"
    }
    // ...
  ]
}
```

---

## 📅 實施計劃

### Phase 1: 檔案整理（1天）

#### 1.1 歸檔未使用組件（✅ 驗證：實際未被調用）

**實際驗證結果**：
- ✅ `BaseDispatcher` - 設計但**從未被 CLI 使用**
- ✅ `Integration Coordinators` - 設計但**從未被 AI 調用**（grep 搜尋確認）
- ⚠️ `OastDispatcher` - SSRF 外部服務客戶端（**保留**，不是模組間通訊）

```bash
# 移動未使用的 Dispatcher
_archive/07_configuration_archive/dispatcher/
├── dispatcher_base.py          # 從未被 CLI 使用
├── cognitive_dispatcher.py     # 從未被調用
├── exploration_dispatcher.py   # 從未被調用
└── planning_dispatcher.py      # 從未被調用

# 移動未使用的 Coordinator
_archive/07_configuration_archive/coordinator/
├── base_coordinator.py         # 從未被調用（grep 確認）
├── xss_coordinator.py          # 從未被調用
└── integration/coordinators/   # 整個目錄未使用

# ⚠️ 不移動（實際有用）
services/features/function_ssrf/oast_dispatcher.py  # SSRF 外部服務客戶端
```

**保留**：
- `CapabilityOrchestrator` - AI 決策核心（**已實際使用**）
- `ExecutionPlanner` - 任務規劃核心（**已實際使用**）
- `OastDispatcher` - SSRF 帶外測試（**已實際使用**）
- `CoreServiceCoordinator` - FastAPI 服務管理（**已實際使用**）

#### 1.2 建立歸檔文檔

```markdown
_archive/07_configuration_archive/DISPATCHER_COORDINATOR_ARCHIVE.md

# Dispatcher/Coordinator 架構歸檔

## 歸檔日期
2026-01-11

## 歸檔原因
AIVA 實際採用了更簡潔的 CLI + JSON 架構：
- AI 直接通過 subprocess 調用模組 CLI
- 不需要中間消息路由層（Dispatcher）
- 不需要服務協調層（Coordinator）

## 架構演進
舊架構: AI → Orchestrator → Coordinator → Dispatcher → 模組
新架構: AI → ExecutionPlanner → AI → subprocess CLI → 模組

## 保留的核心組件
- CapabilityOrchestrator: AI 決策與執行
- ExecutionPlanner: 任務規劃
- MultiEngineCoordinator: 掃描引擎實際協調
```

### Phase 2: 補齊模組 CLI（2-3天）

#### 2.1 建立標準 CLI 模板

```python
# templates/module_cli_template.py
"""標準模組 CLI 模板"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from typing import Any, Dict

async def main():
    parser = argparse.ArgumentParser(description="AIVA {MODULE_NAME}")
    
    # 標準參數
    parser.add_argument("--target", required=True, help="目標 URL/IP")
    parser.add_argument("--action", 
                       choices=["scan", "exploit", "verify"], 
                       default="scan",
                       help="執行動作")
    parser.add_argument("--timeout", type=int, default=30, help="超時秒數")
    
    # 模組特定參數
    # parser.add_argument("--custom-param", help="自定義參數")
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    try:
        # 執行業務邏輯
        from .{module_main} import {MainClass}
        
        executor = {MainClass}()
        findings = await executor.execute(
            target=args.target,
            action=args.action,
            timeout=args.timeout
        )
        
        # 構建標準輸出
        result = {
            "status": "completed",
            "module": "{module_name}",
            "target": args.target,
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "timestamp": datetime.now().isoformat(),
            "findings": findings,
            "metadata": {
                "action": args.action,
                "timeout": args.timeout
            }
        }
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0)
        
    except Exception as e:
        error_result = {
            "status": "failed",
            "module": "{module_name}",
            "target": args.target,
            "execution_time": (datetime.now() - start_time).total_seconds(),
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
        
        print(json.dumps(error_result, ensure_ascii=False, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2.2 為缺失模組創建 CLI

**優先級 P0**（必須）：
1. `features/function_sqli/__main__.py` - SQL 注入掃描
2. `features/function_crypto/__main__.py` - 加密漏洞檢測
3. `features/function_info_leak/__main__.py` - 信息洩漏檢測

**優先級 P1**（重要）：
4. `scan/python_engine/__main__.py` - Python 掃描引擎統一入口
5. `scan/rust_engine/cli.rs` - Rust 掃描引擎 JSON 輸出

#### 2.3 驗證現有 CLI

**測試清單**：
```bash
# XSS 模組
python -m features.function_xss --url https://example.com --type reflected
# 預期輸出: {"status": "completed", "findings": [...]}

# SSRF 模組
python -m features.function_ssrf --url https://example.com
# 預期輸出: {"status": "completed", "findings": [...]}

# 商業邏輯模組
python -m features.function_bizlogic --url https://example.com
# 預期輸出: {"status": "completed", "findings": [...]}
```

### Phase 3: 擴展 Classification Data（2天）

#### 3.1 生成 Features 模組 Flows

**執行分析**：
```bash
cd services/core/aiva_core/internal_exploration/python_tools

# 分析 Features 模組
python aiva_exploration_pipeline.py --target features

# 生成結果會自動合併到 classification_data.json
```

#### 3.2 手動補充 CLI 信息

```python
# 為每個 Flow 添加 CLI 命令
{
  "id": 277,
  "module": "features",
  "path": ["function_xss/xss_scanner.py"],
  "func_names": ["XSSScanner.scan_reflected"],
  "description": "Reflected XSS 漏洞掃描",
  "capability": {
    "category": "vulnerability_detection",
    "subcategory": "xss",
    "risk_level": "high"
  },
  "cli_command": "python -m features.function_xss --type reflected --url {target} --param {param}",
  "expected_output": {
    "format": "json",
    "schema": {
      "status": "str",
      "findings": "list",
      "execution_time": "float"
    }
  }
}
```

#### 3.3 更新 CLI 執行器

```python
# aiva_cli_implementation.py 已支援動態導入
# 只需確保 classification_data.json 格式正確
```

### Phase 4: 整合測試（1-2天）

#### 4.1 端到端測試

```python
# test_e2e_cli_architecture.py
import asyncio
import json
from cognitive_core.capability_orchestrator import CapabilityOrchestrator
from task_planning.planner.execution_planner import ExecutionPlanner

async def test_xss_scan():
    """測試 XSS 掃描完整流程"""
    
    # 1. AI 發出需求
    orchestrator = CapabilityOrchestrator()
    requirement = TaskRequirement(
        task_type="xss_scan",
        target="https://example.com",
        objectives=["find_reflected_xss"]
    )
    
    # 2. 任務編排生成計劃
    plan = await orchestrator.plan(requirement)
    
    assert plan.cli_commands  # 確保生成了 CLI 命令
    
    # 3. AI 執行計劃（subprocess）
    result = await orchestrator.execute(plan)
    
    # 4. 驗證返回格式
    assert result.success
    for cmd, output in result.command_outputs.items():
        data = json.loads(output["stdout"])
        assert "status" in data
        assert "findings" in data
        assert data["module"] == "xss"
    
    print("✅ XSS Scan E2E Test Passed")

async def test_multi_module():
    """測試多模組協同"""
    
    orchestrator = CapabilityOrchestrator()
    requirement = TaskRequirement(
        task_type="comprehensive_scan",
        target="https://example.com",
        objectives=["xss", "sqli", "ssrf"]
    )
    
    plan = await orchestrator.plan(requirement)
    result = await orchestrator.execute(plan)
    
    # 驗證多個模組都執行了
    assert len(result.completed_commands) >= 3
    
    print("✅ Multi-Module E2E Test Passed")

if __name__ == "__main__":
    asyncio.run(test_xss_scan())
    asyncio.run(test_multi_module())
```

#### 4.2 性能測試

```python
# test_cli_performance.py
async def test_subprocess_overhead():
    """測試 subprocess 開銷"""
    
    # 測試 1000 次 CLI 調用
    import time
    start = time.time()
    
    for i in range(1000):
        await process_manager.run_command_with_telemetry(
            cmd=["python", "-m", "features.function_xss", "--url", "https://example.com"],
            timeout=5
        )
    
    duration = time.time() - start
    avg = duration / 1000
    
    print(f"平均每次調用: {avg:.3f}s")
    assert avg < 0.2  # 確保平均開銷 < 200ms
```

#### 4.3 JSON Schema 驗證

```python
# test_json_output_schema.py
import jsonschema

STANDARD_SCHEMA = {
    "type": "object",
    "required": ["status", "module", "target", "findings"],
    "properties": {
        "status": {"type": "string", "enum": ["completed", "failed", "timeout"]},
        "module": {"type": "string"},
        "target": {"type": "string"},
        "execution_time": {"type": "number"},
        "findings": {"type": "array"},
        "metadata": {"type": "object"}
    }
}

def test_all_modules_output():
    """驗證所有模組輸出符合標準"""
    
    modules = [
        "features.function_xss",
        "features.function_sqli",
        "features.function_ssrf",
        # ...
    ]
    
    for module in modules:
        result = subprocess.run(
            ["python", "-m", module, "--url", "https://example.com"],
            capture_output=True
        )
        
        data = json.loads(result.stdout)
        
        # 驗證 Schema
        jsonschema.validate(instance=data, schema=STANDARD_SCHEMA)
        
        print(f"✅ {module} output schema valid")
```

### Phase 5: 文檔更新（1天）

#### 5.1 更新架構文檔

```markdown
docs/core_architecture/CLI_ARCHITECTURE.md

# AIVA CLI 架構設計

## 架構原則
1. AI 主導決策
2. 任務編排生成計劃
3. subprocess + JSON 通訊
4. 標準化輸出格式

## 模組 CLI 規範
所有模組必須實現 `__main__.py`：
- 接收標準參數（--target, --action）
- 輸出標準 JSON 格式
- 包含錯誤處理
- 支持超時機制

## 調用流程
AI → ExecutionPlanner → AI → subprocess CLI → 模組 → JSON → AI
```

#### 5.2 建立開發指南

```markdown
docs/development/MODULE_CLI_GUIDE.md

# 模組 CLI 開發指南

## 快速開始
1. 複製 CLI 模板
2. 實現業務邏輯
3. 輸出標準 JSON
4. 添加到 classification_data.json

## JSON 輸出規範
{
  "status": "completed",  // required
  "module": "module_name",  // required
  "target": "...",  // required
  "findings": [...],  // required
  "execution_time": 1.23,  // recommended
  "metadata": {...}  // optional
}

## 測試清單
- [ ] 正常情況輸出
- [ ] 錯誤情況輸出
- [ ] 超時處理
- [ ] JSON Schema 驗證
```

---

## 📊 預期成果

### 量化指標

| 指標 | 當前 | 目標 | 改進 |
|------|------|------|------|
| 可用 Flows | 276 | 400+ | +45% |
| 模組 CLI 覆蓋率 | 57% (4/7) | 100% (7/7) | +43% |
| 冗餘代碼行數 | ~3000 | 0 | -100% |
| 架構複雜度 | 3層 | 2層 | -33% |
| 平均調用延遲 | - | <200ms | - |

### 質量指標

**架構清晰度**：
- ✅ 去除未使用的 Dispatcher/Coordinator
- ✅ 統一 CLI 接口標準
- ✅ 明確的數據流向

**可維護性**：
- ✅ 單一 JSON 配置文件管理所有 Flows
- ✅ 標準化的模組開發模式
- ✅ 自動化測試覆蓋

**可擴展性**：
- ✅ 新增模組只需實現 `__main__.py`
- ✅ 自動註冊到 classification_data.json
- ✅ AI 自動發現和調用

---

## 🚨 風險與緩解

### 風險 1: 現有功能破壞

**風險等級**: 🟡 中等

**緩解措施**：
1. 先歸檔，不直接刪除
2. 保留所有測試用例
3. 分階段遷移
4. 回滾計劃：恢復 Git commit

### 風險 2: CLI 開銷過大

**風險等級**: 🟡 中等

**緩解措施**：
1. 實施性能測試（Phase 4.2）
2. 如需優化，考慮：
   - 進程池複用
   - gRPC 混合模式（高頻調用）
3. 監控實際調用延遲

### 風險 3: JSON 格式不一致

**風險等級**: 🟢 低

**緩解措施**：
1. 嚴格的 Schema 驗證（Phase 4.3）
2. 自動化測試覆蓋
3. 開發模板和文檔

---

## 📝 驗收標準

### Phase 1 驗收
- [ ] Dispatcher/Coordinator 已歸檔到 `_archive/`
- [ ] 歸檔文檔已建立
- [ ] Git commit 記錄清晰

### Phase 2 驗收
- [ ] 所有 Features 模組都有 `__main__.py`
- [ ] 所有模組輸出符合標準 JSON 格式
- [ ] 手動測試通過

### Phase 3 驗收
- [ ] `classification_data.json` 包含 400+ Flows
- [ ] 所有 Flow 都有 `cli_command` 定義
- [ ] `aiva_cli_implementation.py` 能執行新 Flows

### Phase 4 驗收
- [ ] E2E 測試通過
- [ ] 平均 CLI 調用 < 200ms
- [ ] JSON Schema 驗證 100% 通過

### Phase 5 驗收
- [ ] 架構文檔已更新
- [ ] 開發指南已建立
- [ ] 遷移說明已完成

---

## 🎯 下一步行動

### 立即執行（今天）

1. **確認計劃**
   - [ ] Review 本計劃
   - [ ] 確認優先級
   - [ ] 分配資源

2. **啟動 Phase 1**
   - [ ] 建立 `_archive/07_configuration_archive/dispatcher/`
   - [ ] 建立 `_archive/07_configuration_archive/coordinator/`
   - [ ] 移動 Dispatcher 相關檔案
   - [ ] 移動 Coordinator 相關檔案
   - [ ] 建立歸檔文檔

### 本周完成

- [ ] Phase 1: 檔案整理
- [ ] Phase 2: 補齊 3 個優先模組 CLI
- [ ] Phase 3: 開始擴展 classification_data.json

### 兩周完成

- [ ] Phase 4: 整合測試
- [ ] Phase 5: 文檔更新
- [ ] 全面驗收

---

## 📞 聯絡與支持

**技術負責人**: AI Assistant  
**執行團隊**: AIVA Development Team  
**預計完工**: 2026年1月20日

**進度追蹤**: 本計劃文件會持續更新 ✅/❌ 狀態
