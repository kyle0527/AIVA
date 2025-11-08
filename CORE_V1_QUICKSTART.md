# AIVA Core v1 快速開始指南

> ⚡ 5 分鐘上手 AIVA Core v1 輕量級本機核心

---

## 🚀 快速驗證

### 1️⃣ 列出所有能力
```bash
python -m cli_generated.aiva_cli list-caps
```

**預期輸出**:
```
build_graph           build simple call graph from AST summaries
echo                  echo text back
index_repo            index files under a root folder
parse_ast             parse python files and extract simple summaries
render_report         render a markdown report from artifacts
```

### 2️⃣ 執行掃描流程
```bash
python -m cli_generated.aiva_cli scan --target .
```

**預期結果**:
```json
{
  "run_id": "50d700b8-bd7c-4d13-a281-f5f867bf7fa0",
  "nodes": {
    "index": {"ok": true, "error": null},
    "ast": {"ok": true, "error": null},
    "graph": {"ok": true, "error": null},
    "report": {"ok": true, "error": null}
  },
  "ok": true
}
```

### 3️⃣ 查看掃描報告
```bash
# Windows
Get-Content reports\report_*.md | Select-Object -First 30

# Linux/Mac
cat reports/report_*.md | head -30
```

---

## 📦 核心組件

### AivaCore 主類
```python
from services.core.aiva_core_v1 import AivaCore

core = AivaCore()
core.list_caps()           # 列出能力
plan = core.plan(flow)     # 規劃流程
await core.exec(plan)      # 執行流程
```

### 5 個內建能力

| 能力 | 用途 | 輸入 | 輸出 |
|-----|------|------|------|
| `echo` | 測試用回顯 | `text: str` | `{echo: text}` |
| `index_repo` | 檔案索引 | `root: str` | `{files: [paths]}` |
| `parse_ast` | AST 解析 | `files: [paths]` | `{asts: [summaries]}` |
| `build_graph` | 呼叫圖建構 | `asts: [summaries]` | `{graph: {nodes, edges}}` |
| `render_report` | 報告生成 | `artifacts: dict` | `{report: path}` |

---

## 🔧 Python API 使用

### 基本使用
```python
import asyncio
from services.core.aiva_core_v1 import AivaCore

async def main():
    core = AivaCore()
    
    # 規劃流程
    plan = core.plan("config/flows/scan_minimal.yaml", target=".")
    print(f"Run ID: {plan.run_id}")
    print(f"Nodes: {[n.id for n in plan.nodes]}")
    
    # 執行流程
    summary = await core.exec(plan)
    print(f"Success: {summary['ok']}")
    print(f"Results: {summary['nodes']}")

asyncio.run(main())
```

### 直接呼叫能力
```python
from services.core.aiva_core_v1 import AivaCore

core = AivaCore()

# 呼叫 echo 能力
result = core.registry.call("echo", {"text": "Hello AIVA!"})
print(result)  # {"echo": "Hello AIVA!"}

# 呼叫 index_repo 能力
result = core.registry.call("index_repo", {"root": "./services"})
print(f"Found {len(result['files'])} files")
```

### 自訂能力
```python
from services.core.aiva_core_v1 import AivaCore

def my_detector(args):
    """自訂檢測能力"""
    target = args.get("target")
    # 執行檢測邏輯
    findings = []
    # ... 檢測代碼 ...
    return {
        "ok": True,
        "findings": findings,
        "metrics": {"scanned": 100}
    }

# 註冊能力
core = AivaCore()
core.registry.register(
    "my_detector",
    my_detector,
    desc="自訂漏洞檢測器"
)

# 使用能力
result = core.registry.call("my_detector", {"target": "example.com"})
```

---

## 📋 自訂流程

### 建立流程檔案
```yaml
# config/flows/my_scan.yaml
nodes:
  - id: scan
    cap: my_detector          # 使用自訂能力
    args:
      target: "{{target}}"
    
  - id: index
    cap: index_repo
    args:
      root: "{{target}}"
    
  - id: report
    cap: render_report
    needs: [scan, index]      # 依賴前兩個節點
    args:
      format: markdown

policy:
  retry: 1
  risk_cap: "L0,L1"           # 只允許低風險操作
```

### 執行自訂流程
```python
import asyncio
from services.core.aiva_core_v1 import AivaCore

async def run_custom_flow():
    core = AivaCore()
    
    # 規劃流程（傳入變數）
    plan = core.plan(
        "config/flows/my_scan.yaml",
        target="./services/features"
    )
    
    # 執行
    summary = await core.exec(plan)
    return summary

result = asyncio.run(run_custom_flow())
print(result)
```

---

## 🗂️ 產物結構

### 執行產物位置
```
data/run/{run_id}/
├── plan.json              # 執行計劃
├── summary.json           # 執行摘要
└── nodes/                 # 各節點產物
    ├── index.json         # 索引結果
    ├── ast.json           # AST 結果
    ├── graph.json         # 圖結構
    └── report.json        # 報告元數據

reports/
└── report_{hash}.md       # 最終報告
```

### 讀取產物
```python
import json

run_id = "50d700b8-bd7c-4d13-a281-f5f867bf7fa0"

# 讀取執行摘要
with open(f"data/run/{run_id}/summary.json") as f:
    summary = json.load(f)
    print(summary)

# 讀取特定節點產物
with open(f"data/run/{run_id}/nodes/index.json") as f:
    index_data = json.load(f)
    print(f"Indexed {len(index_data['artifacts']['files'])} files")
```

---

## 🔍 整合既有功能

### 整合 SQLi 檢測
```python
# services/features/function_sqli/__init__.py
from .detector.sqli_detector import SqliDetector

def register_capabilities(registry):
    """註冊 SQLi 檢測能力"""
    detector = SqliDetector()
    
    async def sqli_detect(args):
        target = args.get("target")
        params = args.get("params", {})
        results = await detector.detect_sqli(target, params)
        return {
            "ok": True,
            "findings": results,
            "metrics": {"checked": len(results)}
        }
    
    registry.register(
        "sqli_detect",
        sqli_detect,
        desc="SQL 注入檢測 (基於既有引擎)"
    )
```

### 使用整合的能力
```yaml
# config/flows/security_scan.yaml
nodes:
  - id: sqli_check
    cap: sqli_detect
    args:
      target: "{{target}}"
      params:
        timeout: 30
        payload_level: 3
  
  - id: report
    cap: render_report
    needs: [sqli_check]
```

---

## 📊 監控與日誌

### 事件日誌
```python
from services.core.aiva_core_v1.events import EventStore

events = EventStore()

# 記錄事件
events.log("node_start", {
    "node_id": "scan",
    "timestamp": time.time()
})

# 匯出日誌
events.export("logs/aiva_core/events.json")
```

### 查看日誌
```bash
# 查看最近的事件
tail -f logs/aiva_core/events.log

# 分析執行時間
python -c "
import json
with open('data/run/{run_id}/summary.json') as f:
    data = json.load(f)
    for node, result in data['nodes'].items():
        duration = result['ended_at'] - result['started_at']
        print(f'{node}: {duration:.2f}s')
"
```

---

## 🛡️ 風險管控

### 風險等級
- **L0**: 安全（只讀操作）
- **L1**: 低風險（本機寫入）
- **L2**: 中風險（網路請求）
- **L3**: 高風險（執行程式碼、系統操作）

### 設定風險政策
```yaml
policy:
  risk_cap: "L0,L1"    # 只允許 L0 和 L1 操作
  retry: 2              # 失敗重試 2 次
```

### 自訂風險檢查
```python
from services.core.aiva_core_v1.guard import Guard

guard = Guard()

# 檢查節點風險
allowed, reason = guard.check_risk(node, policy)
if not allowed:
    print(f"Blocked: {reason}")
```

---

## 🔗 與原有系統協同

### 並行使用兩個核心
```python
# 原有 AI 決策核心
from services.core.aiva_core.ai_engine import BioNeuronCore
ai_core = BioNeuronCore()

# Core v1 輕量核心
from services.core.aiva_core_v1 import AivaCore
core_v1 = AivaCore()

# 分工協作
# 1. Core v1 執行掃描和分析
plan = core_v1.plan("config/flows/scan_minimal.yaml", target=".")
scan_result = await core_v1.exec(plan)

# 2. AI 核心基於掃描結果決策攻擊路徑
decision = ai_core.decide(scan_result)

# 3. 執行攻擊（由原有系統處理）
attack_result = await ai_core.execute_attack(decision)
```

---

## 📚 常用命令速查

```bash
# 列出能力
python -m cli_generated.aiva_cli list-caps

# 掃描當前目錄
python -m cli_generated.aiva_cli scan --target .

# 掃描指定目錄
python -m cli_generated.aiva_cli scan --target ./services/features

# 查看最新報告
cat reports/report_*.md | head -50

# 查看產物
ls -la data/run

# 查看事件日誌
tail -f logs/aiva_core/events.log
```

---

## ⚠️ 常見問題

### Q: 為什麼顯示 "No module named 'services.features.base'"？
**A**: 這是正常的警告，表示高價值功能模組尚未安裝。Core v1 的基礎功能不受影響。

### Q: 如何安裝 PyYAML？
**A**: 執行 `pip install pyyaml`，可選但建議安裝。

### Q: 產物儲存在哪裡？
**A**: 
- 執行數據: `data/run/{run_id}/`
- 報告: `reports/`
- 日誌: `logs/aiva_core/`

### Q: 如何清理舊產物？
**A**: 
```bash
# 刪除舊執行記錄（保留最近 10 個）
python -c "
import os, shutil
from pathlib import Path
runs = sorted(Path('data/run').iterdir(), key=os.path.getmtime, reverse=True)
for old in runs[10:]:
    shutil.rmtree(old)
"
```

---

## 🎯 下一步

1. **探索既有功能**: 查看 `services/features/` 下的檢測模組
2. **整合自訂能力**: 將既有檢測器註冊為 Core v1 能力
3. **設計新流程**: 建立符合需求的自訂掃描流程
4. **擴展 CLI**: 在 `cli_generated/aiva_cli/__main__.py` 新增命令

---

**開始使用 AIVA Core v1！** 🚀
