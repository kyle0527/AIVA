# [階段 8] 分析與學習整合指南

> **版本**: v1.0  
> **日期**: 2026-01-12  
> **階段**: 學習系統 (LearningSystem)  
> **優先級**: ⭐ P0 (最高優先級)

---

## 🎯 核心問題

**問題描述**：目前「外部學習」與「AI 分析」是**分離**的，導致：

1. **分析無學習**: AI 分析執行結果後，不會自動學習
2. **學習無上下文**: 學習系統收到的數據缺少分析上下文
3. **數據流斷裂**: 遙測數據 (HTTP 狀態碼) 未轉換為學習信號

**正確設計**：
```
分析 = 學習  (同步發生，不可分離)
```

---

## 📊 當前架構問題

### 當前實現 (錯誤)

```
┌────────────────────────────────────────────────────────────────┐
│                       當前架構 (分離設計)                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CapabilityOrchestrator.execute()                              │
│    ├─ 執行 CLI 命令                                            │
│    ├─ 收集遙測數據 (status_code: 200/403/500...)               │
│    └─ 返回 ExecutionResult                                     │
│                  │                                             │
│                  │ (數據流中斷)                                │
│                  ▼                                             │
│  ExternalLearningListener (未啟用)                             │
│    └─ 等待 TASK_COMPLETED 事件 ← ❌ 事件從未發送               │
│                  │                                             │
│                  ▼                                             │
│  ExternalLoopConnector.process_execution_result()              │
│    └─ 學習邏輯 (從未被調用)                                     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**問題根源**：
1. `app.py` 中 `ExternalLearningListener` 未啟動 (import 被註釋)
2. 執行完成後未發送 `TASK_COMPLETED` 事件
3. 遙測數據未轉換為學習信號

---

## ✅ 正確架構設計

### 方案 A：整合到 CapabilityOrchestrator (推薦)

```python
┌────────────────────────────────────────────────────────────────┐
│                    正確架構 (整合設計)                           │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  CapabilityOrchestrator.execute()                              │
│    │                                                           │
│    ├─ 1. 執行 CLI 命令                                         │
│    │    └─ AsyncProcessManager.run_command_with_telemetry()   │
│    │                                                           │
│    ├─ 2. 收集遙測數據                                          │
│    │    └─ telemetry = {                                      │
│    │         "http_status_codes": [200, 403, 500],            │
│    │         "triggered_waf": true,                           │
│    │         "response_time": 1.2                             │
│    │       }                                                   │
│    │                                                           │
│    ├─ 3. 分析回應 (同步學習)                                   │
│    │    └─ self._analyze_and_learn(telemetry) ← ⭐ 新增方法    │
│    │         ├─ 200 OK → reward = +1.0 (成功)                 │
│    │         ├─ 403 Forbidden → reward = -0.5 (WAF 阻擋)      │
│    │         ├─ 500 Error → reward = +0.8 (發現漏洞)          │
│    │         └─ 更新經驗緩衝 & 觸發學習                        │
│    │                                                           │
│    └─ 4. 返回結果 (包含學習狀態)                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**優點**：
- ✅ 分析和學習在同一流程中
- ✅ 數據流不中斷
- ✅ 無需事件總線
- ✅ 易於測試和調試

---

## 🔧 實現方案

### 步驟 1：在 CapabilityOrchestrator 中新增學習整合

```python
# capability_orchestrator.py

class CapabilityOrchestrator:
    def __init__(self, rag_kb=None, learning_enabled=True):
        # ... 現有代碼 ...
        
        # 新增：學習系統整合
        self.learning_enabled = learning_enabled
        self._learning_engine = None  # 延遲加載
    
    @property
    def learning_engine(self):
        """延遲加載 ContinuousLearningEngine"""
        if self._learning_engine is None and self.learning_enabled:
            from ..learning_system.learning.continuous_learning import ContinuousLearningEngine
            self._learning_engine = ContinuousLearningEngine()
        return self._learning_engine
    
    async def execute(self, plan: CapabilityPlan) -> ExecutionResult:
        """執行計劃 - 整合分析與學習"""
        
        # ... 現有執行邏輯 ...
        
        for cli_cmd in plan.cli_commands:
            # 1. 執行命令
            result = await process_manager.run_command_with_telemetry(...)
            
            # 2. 收集遙測數據
            telemetry = result.get("telemetry", {})
            
            # 3. 分析並學習 (同步發生) ⭐
            if self.learning_enabled:
                learning_result = await self._analyze_and_learn(
                    command=cli_cmd,
                    telemetry=telemetry,
                    plan_context=plan
                )
                result["learning"] = learning_result
        
        return ExecutionResult(...)
    
    async def _analyze_and_learn(
        self,
        command: str,
        telemetry: dict,
        plan_context: CapabilityPlan
    ) -> dict:
        """分析執行結果並同步學習
        
        這是分析與學習整合的核心方法。
        
        Args:
            command: 執行的 CLI 命令
            telemetry: 遙測數據 (HTTP 狀態碼、WAF 檢測等)
            plan_context: 計劃上下文
        
        Returns:
            學習結果資訊
        """
        # 1. 分析 HTTP 狀態碼
        status_codes = telemetry.get("http_status_codes", [])
        waf_triggered = telemetry.get("triggered_waf", False)
        bypassed = telemetry.get("bypassed_protection", False)
        
        # 2. 計算獎勵 (分析 = 獎勵計算)
        reward = self._calculate_reward(status_codes, waf_triggered, bypassed)
        
        # 3. 構建學習狀態
        state = {
            "plan_id": plan_context.plan_id,
            "command": command,
            "target": plan_context.requirement.target,
            "capabilities": [cap.flow_id for cap in plan_context.selected_capabilities]
        }
        
        # 4. 構建動作
        action = {
            "command": command,
            "parameters": {}  # 從 command 解析參數
        }
        
        # 5. 下一狀態 (基於執行結果)
        next_state = state.copy()
        next_state["last_status_codes"] = status_codes
        next_state["waf_triggered"] = waf_triggered
        
        # 6. 觸發學習 (同步)
        learning_result = await self.learning_engine.process_execution_experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            metadata={
                "telemetry": telemetry,
                "timestamp": datetime.now(UTC).isoformat()
            }
        )
        
        logger.info(
            f"📊 Analysis & Learning: reward={reward:.2f}, "
            f"status_codes={status_codes}, waf={waf_triggered}"
        )
        
        return learning_result
    
    def _calculate_reward(
        self,
        status_codes: list[int],
        waf_triggered: bool,
        bypassed: bool
    ) -> float:
        """計算獎勵值 (基於執行分析)
        
        獎勵規則：
        - 200 OK: +1.0 (成功執行)
        - 403 Forbidden: -0.5 (被 WAF 阻擋，需要調整策略)
        - 500 Internal Error: +0.8 (可能發現漏洞，值得深入)
        - 404 Not Found: -0.2 (目標不存在)
        - Timeout: -0.8 (執行失敗)
        - WAF Triggered: -0.3 (額外懲罰)
        - Bypassed Protection: +0.5 (額外獎勵)
        
        Args:
            status_codes: HTTP 狀態碼列表
            waf_triggered: 是否觸發 WAF
            bypassed: 是否繞過防護
        
        Returns:
            獎勵值 (-1.0 到 +1.5)
        """
        if not status_codes:
            return -0.8  # Timeout 或無回應
        
        # 計算平均獎勵
        rewards = []
        for code in status_codes:
            if code == 200:
                rewards.append(1.0)
            elif code == 403:
                rewards.append(-0.5)
            elif code >= 500:
                rewards.append(0.8)
            elif code == 404:
                rewards.append(-0.2)
            elif code >= 400:
                rewards.append(0.0)
            else:
                rewards.append(0.5)
        
        base_reward = sum(rewards) / len(rewards)
        
        # 調整獎勵
        if waf_triggered:
            base_reward -= 0.3  # WAF 懲罰
        
        if bypassed:
            base_reward += 0.5  # 繞過獎勵
        
        # 限制範圍
        return max(-1.0, min(1.5, base_reward))
```

---

### 步驟 2：擴展 ContinuousLearningEngine

```python
# continuous_learning.py

class ContinuousLearningEngine:
    # ... 現有代碼 ...
    
    async def process_execution_experience(
        self,
        state: dict,
        action: dict,
        reward: float,
        next_state: dict,
        metadata: dict | None = None
    ) -> dict:
        """處理執行經驗 (通用方法)
        
        根據環境自動判斷使用 sandbox 或 production 處理邏輯
        
        Args:
            state: 當前狀態
            action: 執行動作
            reward: 獎勵值
            next_state: 下一狀態
            metadata: 元數據 (包含 telemetry)
        
        Returns:
            學習結果
        """
        # 判斷環境 (基於 metadata 或啟發式)
        is_sandbox = metadata.get("environment") == "sandbox" if metadata else False
        
        if is_sandbox:
            return await self.process_sandbox_experience(
                state, action, reward, next_state, metadata
            )
        else:
            return await self.process_production_experience(
                state, action, reward, next_state, metadata,
                should_learn=True  # 預設學習
            )
```

---

## 📈 數據流示意

```
HTTP Request
    ↓
CapabilityOrchestrator.plan()
    ↓
CapabilityOrchestrator.execute()
    │
    ├─ 執行: aiva scan target --type sqli
    │   └─ 遙測: {status_codes: [200, 403], waf: true}
    │
    ├─ 分析: _calculate_reward()
    │   └─ reward = -0.2  (200=+1.0, 403=-0.5, waf=-0.3)
    │
    └─ 學習: _analyze_and_learn()
        └─ ContinuousLearningEngine.process_execution_experience()
            └─ ExperienceManager.push() ← 存入經驗緩衝
```

---

## ✅ 成功標準

實現完成後，應該能夠：

1. ✅ 執行任何命令後自動觸發學習
2. ✅ HTTP 狀態碼直接轉換為獎勵值
3. ✅ 經驗緩衝持續累積數據
4. ✅ 批次訓練自動觸發
5. ✅ 無需手動發送事件

---

## 🧪 測試策略

```python
async def test_analysis_learning_integration():
    """測試分析與學習整合"""
    
    # 1. 創建測試計劃
    orchestrator = CapabilityOrchestrator(learning_enabled=True)
    plan = create_test_plan()
    
    # 2. 執行 (會自動觸發學習)
    result = await orchestrator.execute(plan)
    
    # 3. 驗證學習發生
    assert "learning" in result.command_outputs[0]
    assert result.command_outputs[0]["learning"]["experience_id"] is not None
    
    # 4. 驗證經驗緩衝
    exp_manager = orchestrator.learning_engine.experience_manager
    assert exp_manager.size() > 0
```

---

## 🔗 相關指南

- [階段 5] 能力權重計算 (`stage5_capability_weighting.md`)
- [階段 7] 遙測數據收集 (`stage7_telemetry_collection.md`)
- [階段 8] 獎勵函數設計 (`stage8_reward_function.md`)
- [階段 8] 持續學習引擎 (`stage8_continuous_learning.md`)

---

## 📝 討論記錄

| 日期 | 討論內容 | 決策 |
|------|----------|------|
| 2026-01-12 | 確認外部學習與分析應整合 | 採用方案 A：整合到 CapabilityOrchestrator |
| 2026-01-12 | 討論獎勵函數設計 | 建立基礎規則：200=+1.0, 403=-0.5, 500=+0.8 |

---

**下一步**: 實現 `_analyze_and_learn()` 方法並測試
