# AIVA AI 能力選擇與組合機制報告

> **版本**: v1.0 | **日期**: 2026-01-28 | **狀態**: ✅ 架構確認

---

## 📋 執行摘要

本報告說明 AIVA 系統中 **AI 如何知道要使用哪個能力**，以及**能力如何搭配組合**的完整機制。

### 核心結論

✅ **AI 使用的是已經建立好的 CLI 模組**，不會動態生成新能力  
✅ **AI 的工作是決定「如何搭配使用」這些現成的能力**  
✅ **組合方式只有三種**：順序執行、並行執行、條件分支

---

## 🏗️ 一、整體架構概覽

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AIVA 能力選擇架構                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      已建立的 CLI 能力庫                         │   │
│   │                     (641 flows / 20+ 模組)                       │   │
│   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │   │
│   │  │  XSS   │ │  SQLi  │ │  SSRF  │ │  IDOR  │ │ Recon  │  ...   │   │
│   │  │__main__│ │__main__│ │__main__│ │__main__│ │__main__│        │   │
│   │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                              ▲                                          │
│                              │ subprocess + CLI                         │
│                              │                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      AI 決策層                                   │   │
│   │                                                                  │   │
│   │   決策內容：                                                      │   │
│   │   • 選哪個能力？        → "xss", "sqli", "ssrf"                  │   │
│   │   • 什麼順序執行？      → 先偵察 → 再測試 → 最後利用             │   │
│   │   • 是否並行？          → XSS 和 SQLi 可同時測試                  │   │
│   │   • 根據結果調整？      → 發現注入點 → 深入測試                   │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 二、AI 能力選擇的三層機制

### 2.1 第一層：數據來源 (SystemSelfExplorer)

**職責**：掃描系統有哪些現成的能力可用

**數據來源**：`classification_data.json` (641 個 flows)

```python
# 位置: services/core/aiva_core/internal_exploration/system_self_explorer.py

class SystemSelfExplorer:
    """系統自我探索器 - 知道系統有什麼能力"""
    
    # 關鍵字映射：將 flow 路徑映射到能力類型
    CAPABILITY_KEYWORDS = {
        "sqli": ["sqli", "sql_injection"],
        "xss": ["xss", "cross_site_scripting"],
        "ssrf": ["ssrf", "server_side_request"],
        "rce": ["rce", "remote_code_execution"],
        "csrf": ["csrf", "cross_site_request"],
        "xxe": ["xxe", "xml_external_entity"],
        # ... 共 15+ 種能力類型
    }
    
    async def get_available_attacks(self) -> dict[str, SystemCapability]:
        """返回所有可用的攻擊能力"""
        # 結果: {"sqli": SystemCapability(...), "xss": SystemCapability(...)}
```

**輸出結構**：
```python
@dataclass
class SystemCapability:
    capability_type: str   # "xss", "sqli", "ssrf"
    flow_ids: list[int]    # [1, 2, 3, ...] 對應的 flow ID
    module_path: str       # "function_xss" 模組路徑
    status: str            # "available" 可用狀態
    confidence: float      # 0.8 - 1.0 可信度
```

---

### 2.2 第二層：決策層 (EnhancedDecisionAgent)

**職責**：根據上下文決定使用哪些能力、如何組合

**決策依據**：

| 因素 | 說明 | 範例 |
|------|------|------|
| **用戶限制** | 允許/禁止的能力清單 | `allowed: ["xss", "sqli"]` |
| **RAG 檢索** | 相似案例的歷史經驗 | "類似目標上次用 SQLi 成功" |
| **規則引擎** | 預設的決策規則 | "發現注入點 → 深入測試" |
| **神經網路** | 5M 模型的環境評估 | 環境特徵向量 → 能力評分 |

```python
# 位置: services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py

class EnhancedDecisionAgent:
    """增強決策代理 - 決定用什麼、怎麼組合"""
    
    # 決策規則
    decision_rules = [
        {
            "name": "sql_injection_found",
            "condition": lambda ctx: "sql_injection" in ctx.discovered_vulns,
            "action": "EXPLOIT_SQL_INJECTION",
            "description": "發現 SQL 注入，深入測試"
        },
        {
            "name": "web_service_detected",
            "condition": lambda ctx: any("http" in str(tool).lower() for tool in ctx.available_tools),
            "action": "WEB_ATTACK",
            "description": "檢測到 Web 服務，執行 Web 攻擊"
        },
        # ...
    ]
```

---

### 2.3 第三層：執行層 (Dispatcher + CommandBuilder)

**職責**：將決策轉為 CLI 命令並執行

```python
# 位置: services/core/aiva_core/task_planning/dispatcher.py

class PlanningDispatcher:
    """任務分發器 - 執行 CLI 命令"""
    
    def execute_attack_sync(self, attack_type: str, target: str, **kwargs):
        """同步執行攻擊（CLI 方式）"""
        cmd = [
            "python", "-m",
            "services.core.aiva_core.core_capabilities.attack.attack_executor",
            "--type", attack_type,
            "--target", target,
            "--params", json.dumps(kwargs)
        ]
        return subprocess.run(cmd, capture_output=True, text=True)
```

---

## 🔄 三、能力組合方式（只有三種）

### 3.1 順序執行 (Sequential)

**場景**：前一步的結果影響下一步的決策

```
偵察階段 → 測試階段 → 利用階段
   │           │          │
   ▼           ▼          ▼
 recon      xss/sqli   exploit
```

**程式碼**：
```python
async def execute_plan(self, plan: Dict, parallel: bool = False):
    if not parallel:
        # 順序執行步驟
        for step in plan.get("steps", []):
            result = await self.execute_plan_step(step, step.get("capability_id"))
            # 可根據 result 調整後續步驟
```

**實際範例**：
```
1. 先執行 port_scan → 發現 80/443 開放
2. 再執行 web_crawl → 收集 URL 和參數
3. 最後執行 xss_scan → 針對收集的參數測試
```

---

### 3.2 並行執行 (Parallel)

**場景**：多個能力相互獨立，可同時執行

```
        ┌─→ xss_scan ──┐
        │              │
目標 ──┼─→ sqli_scan ─┼──→ 匯總結果
        │              │
        └─→ csrf_scan ─┘
```

**程式碼**：
```python
async def execute_plan(self, plan: Dict, parallel: bool = True):
    if parallel:
        # 並行執行所有步驟
        tasks = [
            self.execute_plan_step(step, step.get("capability_id"))
            for step in plan.get("steps", [])
        ]
        results = await asyncio.gather(*tasks)
```

**實際範例**：
```
同時執行:
├── XSS 掃描 (5 threads)
├── SQLi 掃描 (5 threads)
└── SSRF 掃描 (5 threads)
↓
匯總所有發現的漏洞
```

---

### 3.3 條件分支 (Conditional)

**場景**：根據前一步結果決定下一步

```
執行 recon
    │
    ├─→ 發現 SQL 錯誤 → 執行 sqli_exploit
    │
    ├─→ 發現 XSS 反射 → 執行 xss_stored_test
    │
    └─→ 無發現 → 執行 fuzzing
```

**程式碼**：
```python
# 規則引擎決策
for rule in decision_rules:
    if rule["condition"](context):
        return rule["action"]  # 根據條件選擇下一步
```

**實際範例**：
```python
if "sql_injection" in discovered_vulns:
    # 發現 SQLi → 深入利用
    execute_attack("sqli_exploit", target)
elif "xss" in discovered_vulns:
    # 發現 XSS → 測試儲存型
    execute_attack("xss_stored", target)
else:
    # 無發現 → 換策略
    execute_attack("fuzzing", target)
```

---

## 📊 四、完整決策流程圖

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AI 能力選擇完整流程                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  輸入: 目標 URL + 用戶限制                                               │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: 查詢可用能力                                             │   │
│  │ SystemSelfExplorer.get_available_attacks()                       │   │
│  │                                                                  │   │
│  │ 結果: {"sqli": ✅, "xss": ✅, "ssrf": ✅, "rce": ❌ ...}         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 2: 過濾用戶限制                                             │   │
│  │                                                                  │   │
│  │ allowed_capabilities: ["xss", "sqli"]                            │   │
│  │ forbidden_capabilities: ["ddos"]                                 │   │
│  │                                                                  │   │
│  │ 結果: ["xss", "sqli"] ← 只保留允許的                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 3: RAG 檢索相似案例                                         │   │
│  │                                                                  │   │
│  │ 查詢: "類似目標的歷史測試經驗"                                     │   │
│  │ 結果: "上次對類似目標，先用 recon → 再用 sqli 成功率 80%"          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 4: 決定組合策略                                             │   │
│  │                                                                  │   │
│  │ 生成計劃:                                                        │   │
│  │   Phase 1: recon (順序)                                          │   │
│  │   Phase 2: xss + sqli (並行)                                     │   │
│  │   Phase 3: 根據結果決定 (條件分支)                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Step 5: 執行 CLI 命令                                            │   │
│  │                                                                  │   │
│  │ python -m services.features.function_xss reflected --url ...     │   │
│  │ python -m services.features.function_sqli inject --url ...       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                             │
│           ▼                                                             │
│  輸出: JSON 結果 → 學習反饋                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✅ 五、核心確認事項

### 5.1 AI 使用「現成的 CLI」

| 項目 | 確認 |
|------|------|
| AI 是否創建新的攻擊工具？ | ❌ 否，使用已建立的 20+ 模組 |
| AI 是否修改工具程式碼？ | ❌ 否，只調用 CLI 並傳入參數 |
| AI 是否決定工具的實現方式？ | ❌ 否，實現方式已固定在各模組中 |

**AI 的工作範圍**：
```
✅ 選擇用哪個工具 (xss / sqli / ssrf)
✅ 決定執行順序 (先偵察 → 再測試)
✅ 決定是否並行 (xss 和 sqli 同時跑)
✅ 根據結果調整 (發現漏洞 → 深入測試)
✅ 傳入參數 (target, timeout, depth)
```

---

### 5.2 組合方式「只有這些」

| 組合方式 | 說明 | 使用時機 |
|----------|------|----------|
| **順序執行** | A → B → C | 前一步結果影響後一步 |
| **並行執行** | A + B + C 同時 | 多個獨立任務 |
| **條件分支** | if A then B else C | 根據結果動態調整 |

這就像「樂高積木」：
- **積木塊** = 各個 CLI 模組 (XSS, SQLi, SSRF...)
- **AI 的工作** = 決定怎麼組合這些積木
- **組合規則** = 順序、並行、條件分支

---

### 5.3 符合實際使用情況

```
實際滲透測試流程：

1. 先用 Recon 收集資訊        ← 順序
   │
   ▼
2. 同時跑 XSS + SQLi + SSRF   ← 並行
   │
   ├─→ 發現 SQLi 漏洞
   │      │
   │      ▼
   │   3. 深入 SQLi 利用       ← 條件分支
   │
   └─→ 無發現
          │
          ▼
       3. 嘗試 Fuzzing         ← 條件分支
```

---

## 📈 六、數據統計

| 項目 | 數量 |
|------|------|
| **總 Flows** | 641 |
| **功能模組** | 20+ |
| **能力類型** | 15+ (sqli, xss, ssrf...) |
| **掃描引擎** | 4 (Python, Rust, Go, TypeScript) |
| **組合方式** | 3 (順序、並行、條件) |

---

## 🔗 七、關鍵檔案參考

| 檔案 | 功能 | 行數 |
|------|------|------|
| [system_self_explorer.py](services/core/aiva_core/internal_exploration/system_self_explorer.py) | 掃描可用能力 | 369 |
| [enhanced_decision_agent.py](services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py) | AI 決策邏輯 | 2725 |
| [dispatcher.py](services/core/aiva_core/task_planning/dispatcher.py) | CLI 執行分發 | 450 |
| [classification_data.json](features_classification/classification_data.json) | 能力定義數據 | 57935 |

---

## 📝 八、總結

```
┌─────────────────────────────────────────────────────────────┐
│                     AIVA AI 能力機制總結                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🔧 工具層: 已建立好的 CLI 模組 (641 flows / 20+ 模組)       │
│            ↑                                                │
│            │ 只負責調用，不負責創建                          │
│            │                                                │
│  🧠 AI 層:  決定「選哪個」+「怎麼組合」                       │
│            │                                                │
│            │ 組合方式:                                       │
│            │ • 順序執行: A → B → C                          │
│            │ • 並行執行: A + B + C                          │
│            │ • 條件分支: if A then B else C                 │
│            │                                                │
│  📊 結果:   符合實際滲透測試流程                             │
│            先偵察 → 同時測試 → 根據結果深入                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**報告生成時間**: 2026-01-28  
**確認狀態**: ✅ 架構理解正確
