#!/usr/bin/env python3
"""
模組連接打通方案
直接修改三個單向模組，讓它們能夠發送訊息到其他模組
"""

import json
from pathlib import Path

def main():
    base_path = Path(r"C:\D\fold7\AIVA-git")
    
    print("\n" + "="*70)
    print("🔧 打通模組連接方案")
    print("   讓 cognitive_core、internal_exploration、task_planning 能發送訊息")
    print("="*70)
    
    print("\n## 現況分析\n")
    
    print("📊 缺失的連接（需要打通）:")
    print("─" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        目前的連接狀態                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐      ❌ 無出站      ┌─────────────────┐       │
│  │ cognitive_core  │ ─────────────────→ │ task_planning   │       │
│  │ (124 入站)      │                     │ (48 入站)       │       │
│  └────────▲────────┘                     └────────▲────────┘       │
│           │                                       │                 │
│           │ 只有入站                              │ 只有入站        │
│           │                                       │                 │
│  ┌────────┴────────┐      ❌ 無出站      ┌────────┴────────┐       │
│  │ external_       │ ←────────────────── │ core_           │       │
│  │ learning        │                     │ capabilities    │       │
│  │ (98出,79入)     │                     │ (1出,129入)     │       │
│  └────────┬────────┘                     └────────┬────────┘       │
│           │                                       │                 │
│           │                 ❌ 無出站              │                 │
│           ▼                                       ▼                 │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                internal_exploration (201 入站)               │   │
│  │                        ❌ 無出站                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                service_backbone (5出,157入)                   │   │
│  │                    ✅ 有雙向連接                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("\n## 打通方案\n")
    
    print("="*70)
    print("🎯 方案核心：使用現有的 2 種通信機制")
    print("="*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                      兩種通信機制                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📡 方式一：MessageBroker (RabbitMQ)                                 │
│  ─────────────────────────────────────                              │
│  • 用途：異步消息、事件驅動、跨語言                                    │
│  • 現有類：service_backbone.messaging.message_broker                 │
│  • 交換機：aiva.tasks, aiva.results, aiva.events, aiva.feedback      │
│  • 範例：await broker.publish_message("aiva.tasks", routing_key, msg) │
│                                                                      │
│  🖥️  方式二：CLI 命令 (subprocess)                                    │
│  ─────────────────────────────────                                   │
│  • 用途：同步執行、跨語言調用 (Python/Rust/Go/TS)                     │
│  • 現有類：task_planning.command_builder                             │
│  • 執行器：core_capabilities.manifests.flow_executor                 │
│  • 範例：subprocess.run(["python", "-m", "module", "--args"])        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    print("\n" + "="*70)
    print("📝 實施計劃：為三個模組添加發送能力")
    print("="*70)
    
    # ===============================================
    # cognitive_core 打通方案
    # ===============================================
    print("\n" + "─"*70)
    print("1️⃣  打通 cognitive_core（讓它能主動調用其他模組）")
    print("─"*70)
    
    print("""
📁 需要新增/修改的文件:

  cognitive_core/
  ├── dispatcher.py         ← 新增：統一發送器
  └── decision/
      └── execution_orchestrator.py  ← 修改：添加發送邏輯

🔧 dispatcher.py 設計:
```python
\"\"\"Cognitive Core Dispatcher - 認知核心發送器

讓 cognitive_core 能夠主動發送訊息到其他模組
\"\"\"
import asyncio
import subprocess
from typing import Any, Dict

from ..service_backbone.messaging.message_broker import MessageBroker
from ..task_planning.command_builder import CommandBuilder


class CognitiveDispatcher:
    \"\"\"認知核心統一發送器
    
    支援兩種發送方式：
    1. 異步消息 (MessageBroker) - 適合事件通知、非阻塞操作
    2. CLI 命令 (subprocess)    - 適合同步執行、跨語言調用
    \"\"\"
    
    def __init__(self):
        self._broker = None
        self._command_builder = None
    
    @property
    def broker(self):
        if self._broker is None:
            self._broker = MessageBroker()
        return self._broker
    
    # ========== 異步消息方式 ==========
    
    async def request_plan(self, objective: str, context: Dict = None):
        \"\"\"請求 task_planning 生成計劃（異步消息）\"\"\"
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="plan.request.cognitive",
            message={
                "action": "generate_plan",
                "objective": objective,
                "context": context or {},
                "source_module": "cognitive_core"
            }
        )
    
    async def execute_capability(self, capability_id: str, params: Dict):
        \"\"\"請求 core_capabilities 執行能力（異步消息）\"\"\"
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="capability.execute",
            message={
                "action": "execute",
                "capability_id": capability_id,
                "params": params,
                "source_module": "cognitive_core"
            }
        )
    
    async def trigger_learning(self, learning_data: Dict):
        \"\"\"請求 external_learning 進行學習（異步消息）\"\"\"
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key="learning.trigger",
            message={
                "action": "train",
                "data": learning_data,
                "source_module": "cognitive_core"
            }
        )
    
    # ========== CLI 命令方式 ==========
    
    def call_task_planning_sync(self, action: str, **kwargs):
        \"\"\"同步調用 task_planning（CLI 方式）\"\"\"
        cmd = [
            "python", "-m",
            "services.core.aiva_core.task_planning.command_router",
            "--action", action,
            "--params", json.dumps(kwargs)
        ]
        return subprocess.run(cmd, capture_output=True, text=True)
    
    def call_core_capabilities_sync(self, capability: str, **kwargs):
        \"\"\"同步調用 core_capabilities（CLI 方式）\"\"\"
        cmd = [
            "python", "-m",
            "services.core.aiva_core.core_capabilities.cli.aiva_cli",
            capability,
            "--params", json.dumps(kwargs)
        ]
        return subprocess.run(cmd, capture_output=True, text=True)
```
""")
    
    # ===============================================
    # internal_exploration 打通方案
    # ===============================================
    print("\n" + "─"*70)
    print("2️⃣  打通 internal_exploration（讓它能主動調用其他模組）")
    print("─"*70)
    
    print("""
📁 需要新增/修改的文件:

  internal_exploration/
  ├── dispatcher.py         ← 新增：統一發送器
  └── python_tools/
      └── aiva_flow_analyzer.py  ← 修改：分析完成後發送

🔧 dispatcher.py 設計:
```python
\"\"\"Internal Exploration Dispatcher - 內部探索發送器

讓 internal_exploration 能夠主動發送分析結果到其他模組
\"\"\"
import asyncio
import subprocess
from typing import Any, Dict

from ..service_backbone.messaging.message_broker import MessageBroker


class ExplorationDispatcher:
    \"\"\"內部探索統一發送器
    
    主要功能：
    1. 分析完成後 → 通知 external_learning 進行訓練
    2. 發現問題後 → 通知 cognitive_core 進行決策
    3. 產生報告後 → 存儲到 service_backbone
    \"\"\"
    
    def __init__(self):
        self._broker = None
    
    @property
    def broker(self):
        if self._broker is None:
            self._broker = MessageBroker()
        return self._broker
    
    # ========== 異步消息方式 ==========
    
    async def notify_analysis_complete(self, analysis_result: Dict):
        \"\"\"通知分析完成（給 external_learning）\"\"\"
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key="analysis.complete",
            message={
                "action": "analysis_complete",
                "result": analysis_result,
                "source_module": "internal_exploration"
            }
        )
    
    async def request_decision(self, issue: Dict):
        \"\"\"請求 cognitive_core 進行決策\"\"\"
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="decision.request",
            message={
                "action": "decide",
                "issue": issue,
                "source_module": "internal_exploration"
            }
        )
    
    async def store_report(self, report: Dict):
        \"\"\"存儲報告到 service_backbone\"\"\"
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.events",
            routing_key="report.store",
            message={
                "action": "store",
                "report": report,
                "source_module": "internal_exploration"
            }
        )
    
    # ========== CLI 命令方式 ==========
    
    def trigger_training_sync(self, training_data: Dict):
        \"\"\"同步觸發訓練（CLI 方式）\"\"\"
        cmd = [
            "python", "-m",
            "services.core.aiva_core.external_learning.training.training_orchestrator",
            "--data", json.dumps(training_data)
        ]
        return subprocess.run(cmd, capture_output=True, text=True)
```
""")
    
    # ===============================================
    # task_planning 打通方案
    # ===============================================
    print("\n" + "─"*70)
    print("3️⃣  打通 task_planning（讓它能主動調用其他模組）")
    print("─"*70)
    
    print("""
📁 需要新增/修改的文件:

  task_planning/
  ├── dispatcher.py         ← 新增：統一發送器
  └── executor/
      └── plan_executor.py  ← 修改：執行時發送到 core_capabilities

🔧 dispatcher.py 設計:
```python
\"\"\"Task Planning Dispatcher - 任務規劃發送器

讓 task_planning 能夠主動調用執行層和決策層
\"\"\"
import asyncio
import subprocess
from typing import Any, Dict

from ..service_backbone.messaging.message_broker import MessageBroker


class PlanningDispatcher:
    \"\"\"任務規劃統一發送器
    
    主要功能：
    1. 規劃完成後 → 調用 core_capabilities 執行
    2. 需要確認時 → 請求 cognitive_core 決策
    3. 需要資源時 → 查詢 service_backbone
    \"\"\"
    
    def __init__(self):
        self._broker = None
    
    @property
    def broker(self):
        if self._broker is None:
            self._broker = MessageBroker()
        return self._broker
    
    # ========== 異步消息方式 ==========
    
    async def execute_plan_step(self, step: Dict, capability_id: str):
        \"\"\"執行計劃步驟（發送到 core_capabilities）\"\"\"
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="capability.execute",
            message={
                "action": "execute_step",
                "step": step,
                "capability_id": capability_id,
                "source_module": "task_planning"
            }
        )
    
    async def confirm_decision(self, plan: Dict, question: str):
        \"\"\"請求 cognitive_core 確認決策\"\"\"
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="decision.confirm",
            message={
                "action": "confirm",
                "plan": plan,
                "question": question,
                "source_module": "task_planning"
            }
        )
    
    async def query_resource(self, resource_type: str):
        \"\"\"查詢 service_backbone 資源狀態\"\"\"
        await self.broker.connect()
        await self.broker.publish_message(
            exchange_name="aiva.tasks",
            routing_key="resource.query",
            message={
                "action": "query",
                "resource_type": resource_type,
                "source_module": "task_planning"
            }
        )
    
    # ========== CLI 命令方式 ==========
    
    def execute_attack_sync(self, attack_type: str, target: str, **kwargs):
        \"\"\"同步執行攻擊（CLI 方式，調用 core_capabilities）\"\"\"
        cmd = [
            "python", "-m",
            "services.core.aiva_core.core_capabilities.attack.attack_executor",
            "--type", attack_type,
            "--target", target,
            "--params", json.dumps(kwargs)
        ]
        return subprocess.run(cmd, capture_output=True, text=True)
    
    def execute_scan_sync(self, scan_type: str, target: str, **kwargs):
        \"\"\"同步執行掃描（CLI 方式）\"\"\"
        cmd = [
            "python", "-m",
            "services.core.aiva_core.core_capabilities.ingestion.scan_module_interface",
            "--type", scan_type,
            "--target", target,
            "--params", json.dumps(kwargs)
        ]
        return subprocess.run(cmd, capture_output=True, text=True)
```
""")
    
    print("\n" + "="*70)
    print("📋 打通後的連接圖")
    print("="*70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        打通後的連接狀態                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐    ✅ 新增出站     ┌─────────────────┐        │
│  │ cognitive_core  │ ─────────────────→ │ task_planning   │        │
│  │ + dispatcher.py │ ←───────────────── │ + dispatcher.py │        │
│  └────────┬────────┘    ✅ 新增出站     └────────┬────────┘        │
│           │                                       │                  │
│           │ ✅ 雙向                               │ ✅ 雙向          │
│           ▼                                       ▼                  │
│  ┌────────┴────────┐                     ┌────────┴────────┐        │
│  │ external_       │ ←─────────────────→ │ core_           │        │
│  │ learning        │                     │ capabilities    │        │
│  └────────┬────────┘                     └────────┬────────┘        │
│           │                                       │                  │
│           │ ✅ 雙向                               │ ✅ 雙向          │
│           ▼                                       ▼                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                internal_exploration                          │    │
│  │                     + dispatcher.py                          │    │
│  │                       ✅ 新增出站                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                service_backbone (消息總線)                    │    │
│  │                    MessageBroker                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

通信方式選擇指南:
┌───────────────────┬──────────────────┬──────────────────────────────┐
│ 場景              │ 推薦方式          │ 原因                          │
├───────────────────┼──────────────────┼──────────────────────────────┤
│ 事件通知          │ MessageBroker    │ 異步、不阻塞、可廣播          │
│ 需要返回值        │ CLI subprocess   │ 同步等待結果                  │
│ 跨語言調用        │ CLI subprocess   │ Rust/Go/TS 都支持命令行       │
│ 長時間任務        │ MessageBroker    │ 異步處理、可監控進度          │
│ 緊急決策          │ CLI subprocess   │ 立即執行、立即返回            │
└───────────────────┴──────────────────┴──────────────────────────────┘
""")
    
    print("\n" + "="*70)
    print("🔧 實施步驟")
    print("="*70)
    
    print("""
### 步驟 1: 創建三個 dispatcher.py 文件

```bash
# 創建文件
touch services/core/aiva_core/cognitive_core/dispatcher.py
touch services/core/aiva_core/internal_exploration/dispatcher.py
touch services/core/aiva_core/task_planning/dispatcher.py
```

### 步驟 2: 在現有腳本中使用 dispatcher

修改 cognitive_core/decision/execution_orchestrator.py:
```python
from ..dispatcher import CognitiveDispatcher

class ExecutionOrchestrator:
    def __init__(self):
        self.dispatcher = CognitiveDispatcher()
    
    async def execute_decision(self, decision):
        # 決策完成後，發送到 task_planning
        await self.dispatcher.request_plan(
            objective=decision.objective,
            context=decision.context
        )
```

### 步驟 3: 添加消息監聽器（接收方）

每個模組需要監聽來自其他模組的消息：
```python
# cognitive_core/listener.py
class CognitiveListener:
    async def start(self):
        await broker.subscribe(
            queue_name="cognitive_core.tasks",
            routing_keys=["decision.*"],
            exchange_name="aiva.tasks",
            callback=self.handle_message
        )
```

### 步驟 4: 更新 flows 配置

在 latest_classification.json 中添加新的 flows：
```json
{
  "id": 841,
  "path": ["dispatcher", "plan_generator"],
  "full_path": [
    "services/core/aiva_core/cognitive_core/dispatcher.py",
    "services/core/aiva_core/task_planning/planner/plan_generator.py"
  ],
  "start": "cognitive_core",
  "end": "task_planning",
  "direction": "outbound"
}
```
""")
    
    print("\n" + "="*70)
    print("📊 預期效果")
    print("="*70)
    
    print("""
| 指標 | 打通前 | 打通後 | 變化 |
|------|--------|--------|------|
| 單向模組 | 3 個 | 0 個 | ✅ -100% |
| cognitive_core 出站 | 0 | 4+ | ✅ 新增 |
| internal_exploration 出站 | 0 | 3+ | ✅ 新增 |
| task_planning 出站 | 0 | 4+ | ✅ 新增 |
| 連接密度 | 30% | 50%+ | ✅ +67% |
| 新增 flows | 0 | 50-80 | 📈 增加 |
""")
    
    print("\n" + "="*70)
    print("💡 其他建議（來自最佳實踐）")
    print("="*70)
    
    print("""
1. **使用 asyncio.create_task** 進行非阻塞發送
   - 發送消息後不等待結果，繼續執行
   - 適合事件通知和日誌記錄

2. **使用 subprocess.run 進行同步調用**
   - 需要立即獲得結果時使用
   - 設置 timeout 避免無限等待
   - 使用 capture_output=True 獲取輸出

3. **統一的消息格式**
   ```python
   message = {
       "action": "...",
       "source_module": "...",
       "timestamp": datetime.now().isoformat(),
       "correlation_id": uuid4(),
       "payload": {...}
   }
   ```

4. **錯誤處理**
   - MessageBroker: 使用 retry 和 dead letter queue
   - subprocess: 檢查 returncode，處理 timeout

5. **監控和日誌**
   - 記錄所有跨模組調用
   - 追蹤消息的流向和處理時間
   - 設置告警閾值
""")
    
    # 保存報告
    output_file = base_path / "docs" / "MODULE_CONNECTION_IMPLEMENTATION_PLAN.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 模組連接打通實施計劃\n\n")
        f.write("**日期**: 2026-01-01\n\n")
        
        f.write("## 目標\n\n")
        f.write("讓 cognitive_core、internal_exploration、task_planning 三個單向模組能夠**主動發送訊息**到其他模組。\n\n")
        
        f.write("## 通信機制\n\n")
        f.write("### 1. MessageBroker (RabbitMQ)\n\n")
        f.write("- **用途**: 異步消息、事件驅動、跨語言\n")
        f.write("- **交換機**: `aiva.tasks`, `aiva.results`, `aiva.events`, `aiva.feedback`\n")
        f.write("- **適合場景**: 事件通知、長時間任務、廣播消息\n\n")
        
        f.write("### 2. CLI 命令 (subprocess)\n\n")
        f.write("- **用途**: 同步執行、跨語言調用\n")
        f.write("- **支援**: Python, Rust, Go, TypeScript, Docker\n")
        f.write("- **適合場景**: 需要返回值、緊急決策、命令行工具\n\n")
        
        f.write("## 實施計劃\n\n")
        
        f.write("### 新增文件\n\n")
        f.write("```\n")
        f.write("services/core/aiva_core/\n")
        f.write("├── cognitive_core/\n")
        f.write("│   ├── dispatcher.py          ← 新增\n")
        f.write("│   └── listener.py            ← 新增\n")
        f.write("├── internal_exploration/\n")
        f.write("│   ├── dispatcher.py          ← 新增\n")
        f.write("│   └── listener.py            ← 新增\n")
        f.write("└── task_planning/\n")
        f.write("    ├── dispatcher.py          ← 新增\n")
        f.write("    └── listener.py            ← 新增\n")
        f.write("```\n\n")
        
        f.write("### Dispatcher 統一接口\n\n")
        f.write("```python\n")
        f.write("class BaseDispatcher:\n")
        f.write("    \"\"\"基礎發送器\"\"\"\n")
        f.write("    \n")
        f.write("    # 異步消息方式\n")
        f.write("    async def send_message(self, exchange, routing_key, message):\n")
        f.write("        await self.broker.publish_message(exchange, routing_key, message)\n")
        f.write("    \n")
        f.write("    # CLI 同步方式\n")
        f.write("    def call_cli(self, module, action, params):\n")
        f.write("        cmd = [\"python\", \"-m\", module, action, \"--params\", json.dumps(params)]\n")
        f.write("        return subprocess.run(cmd, capture_output=True, text=True, timeout=30)\n")
        f.write("```\n\n")
        
        f.write("## 預期效果\n\n")
        f.write("| 指標 | 打通前 | 打通後 |\n")
        f.write("|------|--------|--------|\n")
        f.write("| 單向模組 | 3 | 0 |\n")
        f.write("| cognitive_core 出站 | 0 | 4+ |\n")
        f.write("| internal_exploration 出站 | 0 | 3+ |\n")
        f.write("| task_planning 出站 | 0 | 4+ |\n")
        f.write("| 連接密度 | 30% | 50%+ |\n")
    
    print(f"\n✅ 實施計劃已保存至: {output_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
