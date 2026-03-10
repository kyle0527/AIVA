# 🔧 AttackCoordinator 重構計劃

## 📑 目錄

- [📊 當前問題分析](#-當前問題分析)
  - [❌ 問題 1: `detect_vulnerabilities()` 直接執行](#-問題-1-detect_vulnerabilities-直接執行)
  - [❌ 問題 2: `scan_with_multi_engine()` 直接調用](#-問題-2-scan_with_multi_engine-直接調用)
  - [✅ 正確範例: `execute_attack()`](#-正確範例-execute_attack)
- [🎯 重構方案](#-重構方案)
  - [方案 A: 最小改動（推薦）](#方案-a-最小改動推薦)
    - [Step 1: 新增純規劃方法](#step-1-新增純規劃方法)
    - [Step 2: 保留舊方法，內部調用新方法](#step-2-保留舊方法內部調用新方法)
    - [Step 3: 創建執行器（Integration 層）](#step-3-創建執行器integration-層)
  - [方案 B: 完整重構（長期目標）](#方案-b-完整重構長期目標)
    - [重構步驟](#重構步驟)
- [📋 Schema 定義](#-schema-定義)
  - [VulnDetectionPlan](#vulndetectionplan)
- [🎯 實施優先級](#-實施優先級)
- [✅ 驗收標準](#-驗收標準)
  - [1. 代碼分離度](#1-代碼分離度)
  - [2. 接口標準化](#2-接口標準化)
  - [3. 向後兼容](#3-向後兼容)
- [🔗 相關修改](#-相關修改)
  - [需要同步修改的文件](#需要同步修改的文件)
- [📝 遷移指南（給開發者）](#-遷移指南給開發者)
  - [舊代碼（直接執行）](#舊代碼直接執行)
  - [新代碼（規劃+執行）](#新代碼規劃執行)
- [🚀 立即行動](#-立即行動)

---


**目標**: 將 `attack_coordinator.py` 從「規劃+執行」改為「純規劃器」

---

## 📊 當前問題分析

### ❌ 問題 1: `detect_vulnerabilities()` 直接執行

```python
# 當前代碼 (第 80-150 行)
async def detect_vulnerabilities(self, context):
    for vuln_type in vuln_types:
        module = __import__(module_path)      # ❌ 直接導入
        worker = worker_class()                # ❌ 直接實例化
        result = await worker.process_task()   # ❌ 直接執行
```

**違反原則**: AI 應該只規劃，不應執行

---

### ❌ 問題 2: `scan_with_multi_engine()` 直接調用

```python
# 當前代碼 (第 150-230 行)
async def scan_with_multi_engine(self, context):
    coordinator = MultiEngineCoordinator()    # ❌ 直接實例化
    result = await coordinator.run_scan()     # ❌ 直接執行
```

**違反原則**: 應該返回掃描計劃，由執行層調用

---

### ✅ 正確範例: `execute_attack()`

```python
# 當前代碼 (第 230-320 行)
async def execute_attack(self, context):
    """✅ 這個方法是正確的"""
    executor = AttackExecutor(mode=mode)
    result = await executor.execute_plan_with_ai_analysis(plan, target, ai_analysis)
    return result
```

**為什麼正確**: 
- ✅ 接收已有的 `plan`（由 AI 規劃）
- ✅ 委派給 `AttackExecutor` 執行
- ✅ `AttackCoordinator` 只做調度，不做決策

---

## 🎯 重構方案

### 方案 A: 最小改動（推薦）

**原則**: 保持向後兼容，逐步分離

#### Step 1: 新增純規劃方法

```python
# attack_coordinator.py
class AttackCoordinator:
    
    # ✅ 新增：純規劃方法
    async def plan_vulnerability_detection(
        self, context: dict[str, Any]
    ) -> VulnDetectionPlan:
        """AI 規劃漏洞檢測計劃（不執行）
        
        Returns:
            VulnDetectionPlan: 包含目標、策略、優先級的計劃
        """
        target = context.get("target")
        vuln_types = context.get("vulnerability_types", ["sqli", "xss", "ssrf", "idor"])
        deep_scan = context.get("deep_scan", False)
        
        # ✅ AI 分析：選擇檢測類型
        selected_types = self._select_vuln_types_by_ai(target, vuln_types)
        
        # ✅ AI 決策：評估優先級
        priorities = self._calculate_priorities(selected_types, deep_scan)
        
        # ✅ 返回計劃，不執行
        return VulnDetectionPlan(
            plan_id=f"vuln_detect_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            target=target,
            vuln_types=selected_types,
            priorities=priorities,
            timeout=self._estimate_timeout(selected_types),
            constraints=DecisionConstraints(
                max_concurrent=3,
                risk_level=context.get("risk_level", RiskLevel.LOW)
            )
        )
    
    def _select_vuln_types_by_ai(self, target, vuln_types) -> list[str]:
        """✅ AI 邏輯：根據目標特徵選擇檢測類型"""
        # 例如：分析 target URL 判斷是否為 API、是否有參數等
        if "api" in target.lower():
            return ["sqli", "idor"]  # API 常見漏洞
        if "?" in target:
            return ["sqli", "xss"]   # 有參數，優先注入類
        return vuln_types
    
    def _calculate_priorities(self, vuln_types, deep_scan) -> dict[str, int]:
        """✅ AI 邏輯：計算每種漏洞的優先級"""
        base_priority = 8 if deep_scan else 5
        return {vtype: base_priority for vtype in vuln_types}
    
    def _estimate_timeout(self, vuln_types) -> int:
        """✅ AI 邏輯：估算執行時間"""
        return len(vuln_types) * 60  # 每種類型 60 秒
```

#### Step 2: 保留舊方法，內部調用新方法

```python
    # ⚠️ 保留：向後兼容（標記為 deprecated）
    async def detect_vulnerabilities(self, context: dict[str, Any]) -> dict[str, Any]:
        """檢測漏洞（已棄用，請使用 plan + execute 模式）
        
        此方法保留是為了向後兼容，內部會：
        1. 調用 plan_vulnerability_detection() 規劃
        2. 調用執行層執行計劃
        3. 收集結果
        
        新代碼請使用：
            plan = await coordinator.plan_vulnerability_detection(context)
            result = await executor.execute_detection_plan(plan)
        """
        logger.warning("detect_vulnerabilities() is deprecated, use plan + execute pattern")
        
        # 1. AI 規劃
        plan = await self.plan_vulnerability_detection(context)
        
        # 2. 委派執行（調用 Integration 層）
        from services.integration.executors.vuln_detection_executor import VulnDetectionExecutor
        executor = VulnDetectionExecutor()
        result = await executor.execute_plan(plan)
        
        return result
```

#### Step 3: 創建執行器（Integration 層）

```python
# services/integration/executors/vuln_detection_executor.py
class VulnDetectionExecutor:
    """漏洞檢測執行器（執行層）"""
    
    async def execute_plan(self, plan: VulnDetectionPlan) -> dict[str, Any]:
        """執行 AI 規劃的檢測計劃"""
        results = {
            "plan_id": plan.plan_id,
            "success": True,
            "vulnerabilities_found": [],
            "modules_executed": [],
            "total_findings": 0,
        }
        
        module_map = {
            "sqli": "services.features.function_sqli.worker.SqliWorkerService",
            "xss": "services.features.function_xss.worker.XssWorkerService",
            "ssrf": "services.features.function_ssrf.worker.SsrfWorkerService",
            "idor": "services.features.function_idor.worker.IdorWorkerService",
        }
        
        for vuln_type in plan.vuln_types:
            worker_class = self._load_worker(module_map[vuln_type])
            worker = worker_class()
            
            task = self._build_task(plan, vuln_type)
            detection_result = await worker.process_task(task)
            
            results["vulnerabilities_found"].extend(detection_result.get("findings", []))
            results["modules_executed"].append(vuln_type)
        
        return results
```

---

### 方案 B: 完整重構（長期目標）

**原則**: 完全分離 AI 規劃與執行

#### 重構步驟

1. **移除所有執行代碼**
   ```python
   # 刪除
   - __import__()
   - worker.process_task()
   - coordinator.run_scan()
   ```

2. **只保留規劃方法**
   ```python
   class AttackCoordinator:
       async def plan_scan(self, context) -> ScanPlan
       async def plan_detection(self, context) -> VulnDetectionPlan
       async def plan_attack(self, context) -> AttackPlan
       async def plan_exploitation(self, context) -> ExploitPlan
   ```

3. **所有執行交給 Integration/Features**
   ```
   AI Core → 生成計劃
       ↓
   Integration Executors → 執行計劃
       ↓
   Features Workers → 實際測試
       ↓
   Integration Coordinators → 收集結果
   ```

---

## 📋 Schema 定義

### VulnDetectionPlan

```python
# aiva_common/schemas/ai_plans.py
from pydantic import BaseModel, Field
from typing import List, Dict
from aiva_common.enums import RiskLevel

class VulnDetectionPlan(BaseModel):
    """AI 規劃的漏洞檢測計劃"""
    plan_id: str
    target: str
    vuln_types: List[str] = Field(description="要檢測的漏洞類型")
    priorities: Dict[str, int] = Field(description="每種類型的優先級")
    timeout: int = Field(description="預估執行時間(秒)")
    constraints: DecisionConstraints
    
    class Config:
        json_schema_extra = {
            "example": {
                "plan_id": "vuln_detect_20260110143000",
                "target": "https://example.com/api/users",
                "vuln_types": ["sqli", "idor"],
                "priorities": {"sqli": 8, "idor": 7},
                "timeout": 120,
                "constraints": {
                    "max_concurrent": 3,
                    "risk_level": "low"
                }
            }
        }
```

---

## 🎯 實施優先級

| 步驟 | 工作項 | 優先級 | 預估時間 |
|------|--------|--------|----------|
| 1 | 新增 `plan_vulnerability_detection()` | 🔴 高 | 1 小時 |
| 2 | 創建 `VulnDetectionPlan` schema | 🔴 高 | 30 分鐘 |
| 3 | 創建 `VulnDetectionExecutor` | 🔴 高 | 2 小時 |
| 4 | 修改 `detect_vulnerabilities()` 內部調用 | 🟡 中 | 1 小時 |
| 5 | 測試向後兼容性 | 🔴 高 | 1 小時 |
| 6 | 同樣處理 `scan_with_multi_engine()` | 🟡 中 | 2 小時 |
| 7 | 完整重構（方案 B） | 🟢 低 | 1 天 |

---

## ✅ 驗收標準

### 1. 代碼分離度
```python
# ✅ AttackCoordinator 中不應出現
assert "import" not in plan_method_source
assert "worker.process_task" not in plan_method_source
assert "subprocess.run" not in plan_method_source
```

### 2. 接口標準化
```python
# ✅ 所有規劃方法應返回 Plan 對象
plan = await coordinator.plan_vulnerability_detection(context)
assert isinstance(plan, VulnDetectionPlan)
```

### 3. 向後兼容
```python
# ✅ 舊代碼仍能運行
result = await coordinator.detect_vulnerabilities(context)
assert result["success"] == True
```

---

## 🔗 相關修改

### 需要同步修改的文件

1. **創建 Schema**
   - `aiva_common/schemas/ai_plans.py` - 新建
   
2. **創建執行器**
   - `integration/executors/vuln_detection_executor.py` - 新建
   
3. **修改協調器**
   - `core/task_planning/commander/attack_coordinator.py` - 修改
   
4. **更新測試**
   - `tests/test_attack_coordinator.py` - 修改

---

## 📝 遷移指南（給開發者）

### 舊代碼（直接執行）
```python
# ❌ 舊方式
coordinator = AttackCoordinator(...)
result = await coordinator.detect_vulnerabilities({
    "target": "https://example.com",
    "vulnerability_types": ["sqli", "xss"]
})
```

### 新代碼（規劃+執行）
```python
# ✅ 新方式
# Step 1: AI 規劃
coordinator = AttackCoordinator(...)
plan = await coordinator.plan_vulnerability_detection({
    "target": "https://example.com",
    "vulnerability_types": ["sqli", "xss"]
})

# Step 2: 執行層執行
executor = VulnDetectionExecutor()
result = await executor.execute_plan(plan)

# Step 3: 整合層收集結果
from integration.coordinators.vuln_coordinator import VulnCoordinator
coord = VulnCoordinator()
feedback = await coord.collect_result(result)

# Step 4: AI 分析反饋，決定下一步
next_plan = await coordinator.decide_next_action(feedback)
```

---

## 🚀 立即行動

執行以下命令開始重構：

```bash
# 1. 創建 schema
mkdir -p services/aiva_common/schemas
touch services/aiva_common/schemas/ai_plans.py

# 2. 創建執行器
mkdir -p services/integration/executors
touch services/integration/executors/vuln_detection_executor.py

# 3. 備份原始文件
cp services/core/aiva_core/task_planning/commander/attack_coordinator.py \
   services/core/aiva_core/task_planning/commander/attack_coordinator.py.backup
```
