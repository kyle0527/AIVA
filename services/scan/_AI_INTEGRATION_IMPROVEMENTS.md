# Scan 模組 AI 整合改進建議

## 📊 當前狀態評估

### ✅ 已完成的 AI 整合

1. **命令中心對接** - 完成
   - `command_handler.py` 實現 AI 命令處理器
   - 支援 SCAN_PHASE0、SCAN_PHASE1、SCAN_COMPREHENSIVE
   - 使用 `AICommandCenter` 統一調度

2. **適配器模式** - 完成
   - `MultiEngineCoordinator` 使用適配器模式
   - 統一四引擎接口(Python/TypeScript/Rust/Go)
   - 易於擴展和測試

3. **數據合約規範** - 完成
   - 使用 `aiva_common` 統一 Schema
   - Phase0CompletedPayload, Phase1CompletedPayload
   - 標準化的資產和漏洞數據

## ⚠️ 缺失的 AI 整合環節

### 1. **缺少 AI 決策反饋循環**

#### 問題
```python
# command_handler.py - _handle_comprehensive()
# 簡單的決策邏輯（實際應該由 Core AI 決策）
needs_phase1 = (
    len(phase0_data.assets) > 0 or
    phase0_data.summary.urls_found > 5 or
    phase0_data.summary.forms_found > 0
)
```

**問題**: 硬編碼決策規則,沒有使用 AI 決策代理

#### 建議修復
```python
# 調用 AI 決策代理
from services.core.aiva_core.cognitive_core.decision import EnhancedDecisionAgent

decision_context = DecisionContext()
decision_context.discovered_vulns = [...]
decision_context.target_info = {"type": "web", "value": target_url}
decision_context.previous_results = [phase0_result]

decision_agent = EnhancedDecisionAgent()
decision = decision_agent.make_decision(decision_context)

# 根據 AI 決策執行 Phase 1
if decision.action == "RUN_PHASE1":
    selected_engines = decision.params.get("engines", ["python"])
    await self._handle_phase1(...)
```

### 2. **缺少經驗學習機制**

#### 問題
Scan 模組執行掃描後沒有將結果反饋給 `ExperienceManager`

#### 建議修復
```python
# 在 command_handler.py 新增方法
async def _record_scan_experience(
    self,
    scan_id: str,
    phase0_result: Phase0CompletedPayload,
    phase1_result: Optional[Phase1CompletedPayload],
    success: bool,
    execution_time: float
):
    """記錄掃描經驗供 AI 學習"""
    try:
        from services.aiva_common.ai import get_default_experience_manager
        
        experience_manager = get_default_experience_manager()
        
        experience_data = {
            "scan_id": scan_id,
            "targets": [str(t) for t in phase0_result.targets],
            "assets_found": len(phase0_result.assets),
            "technologies": list(phase0_result.fingerprints.technologies.keys()),
            "execution_time": execution_time,
            "success": success,
        }
        
        if phase1_result:
            experience_data.update({
                "engines_used": phase1_result.engines_used,
                "total_assets": len(phase1_result.assets),
            })
        
        await experience_manager.record_experience(
            task_type="scan",
            experience_data=experience_data
        )
        
    except Exception as e:
        self.logger.warning(f"Failed to record experience: {e}")
```

### 3. **缺少能力評估反饋**

#### 問題
四引擎執行掃描後沒有評估各引擎的實際能力

#### 建議修復
```python
# 在 multi_engine_coordinator.py 新增
async def _evaluate_engine_performance(
    self,
    engine_name: str,
    assets_found: int,
    execution_time: float,
    success: bool
):
    """評估引擎性能並反饋給 AI"""
    try:
        from services.aiva_common.ai import get_default_capability_evaluator
        
        evaluator = get_default_capability_evaluator()
        
        evidence = {
            "engine": engine_name,
            "assets_found": assets_found,
            "execution_time": execution_time,
            "success": success,
            "timestamp": datetime.now(UTC).isoformat()
        }
        
        await evaluator.record_capability_evidence(
            capability_id=f"scan_{engine_name}",
            evidence=evidence
        )
        
    except Exception as e:
        self.logger.debug(f"Capability evaluation skipped: {e}")
```

### 4. **缺少 RAG 知識庫查詢**

#### 問題
掃描前沒有查詢知識庫獲取已知的目標資訊

#### 建議修復
```python
# 在 command_handler.py 新增
async def _query_target_knowledge(
    self,
    target_url: str
) -> Dict[str, Any]:
    """從 RAG 知識庫查詢目標的歷史資訊"""
    try:
        from services.core.aiva_core.cognitive_core.rag import get_rag_agent
        
        rag_agent = get_rag_agent()
        
        # 查詢歷史掃描記錄
        query = f"Previous scan results for {target_url}"
        results = await rag_agent.query(query, top_k=5)
        
        return {
            "has_history": len(results) > 0,
            "previous_scans": results,
            "recommended_engines": self._extract_recommended_engines(results)
        }
        
    except Exception as e:
        self.logger.debug(f"RAG query skipped: {e}")
        return {"has_history": False}

# 在 _handle_phase0 使用
knowledge = await self._query_target_knowledge(target_url)
if knowledge["has_history"]:
    self.logger.info(f"Found {len(knowledge['previous_scans'])} historical scans")
```

### 5. **缺少抗幻覺驗證**

#### 問題
AI 決策掃描策略時沒有驗證合理性

#### 建議修復
```python
# 在 command_handler.py 新增
async def _validate_scan_decision(
    self,
    decision: Decision,
    context: DecisionContext
) -> bool:
    """驗證 AI 掃描決策的合理性"""
    try:
        from services.core.aiva_core.cognitive_core.anti_hallucination import (
            AntiHallucinationModule
        )
        
        validator = AntiHallucinationModule()
        
        # 構造攻擊計劃（掃描步驟）
        attack_plan = {
            "name": f"scan_{context.target_info['value']}",
            "steps": [
                {
                    "action": decision.action,
                    "description": decision.reasoning,
                    "parameters": decision.params
                }
            ]
        }
        
        validated_plan = validator.validate_attack_plan(attack_plan)
        
        if len(validated_plan["steps"]) == 0:
            self.logger.warning("AI decision rejected by anti-hallucination")
            return False
            
        return True
        
    except Exception as e:
        self.logger.debug(f"Validation skipped: {e}")
        return True  # 驗證失敗時預設允許
```

## 🎯 優先級實施順序

### 第一階段: 基礎整合(高優先級)

1. **經驗學習** - 在所有掃描完成後記錄經驗
2. **能力評估** - 記錄各引擎的實際表現
3. **RAG 查詢** - 掃描前查詢歷史資訊

### 第二階段: 深度整合(中優先級)

4. **AI 決策代理** - 用 AI 決策取代硬編碼規則
5. **抗幻覺驗證** - 驗證 AI 決策的合理性

### 第三階段: 完整閉環(低優先級)

6. **雙閉環優化** - 整合外部學習的訓練反饋
7. **自適應策略** - 根據學習結果動態調整掃描策略

## 📝 實施檢查清單

- [ ] `command_handler.py` 新增 `_record_scan_experience()`
- [ ] `command_handler.py` 新增 `_query_target_knowledge()`
- [ ] `multi_engine_coordinator.py` 新增 `_evaluate_engine_performance()`
- [ ] `command_handler.py` 替換硬編碼決策為 AI 決策代理
- [ ] `command_handler.py` 新增 `_validate_scan_decision()`
- [ ] 所有引擎適配器在掃描完成後調用能力評估
- [ ] 測試完整的 AI 反饋循環

## 🔄 完整 AI 整合流程

```
1. 用戶請求掃描 → AI Commander
   ↓
2. AI Commander → CommandCenter.execute(SCAN_PHASE0)
   ↓
3. ScanCommandHandler._handle_phase0()
   ├─ _query_target_knowledge()      # 查詢 RAG
   └─ MultiEngineCoordinator.execute_phase0()
      ↓
4. Phase 0 完成 → _record_scan_experience()  # 記錄經驗
   ↓
5. AI Decision Agent.make_decision()    # AI 決策
   ├─ _validate_scan_decision()        # 抗幻覺驗證
   └─ 決定是否執行 Phase 1
      ↓
6. ScanCommandHandler._handle_phase1()
   └─ MultiEngineCoordinator.execute_phase1()
      ├─ Python Adapter.scan()
      ├─ TypeScript Adapter.scan()
      ├─ Rust Adapter.scan()
      └─ Go Adapter.scan()
         ↓
7. 各引擎完成 → _evaluate_engine_performance()  # 評估能力
   ↓
8. Phase 1 完成 → _record_scan_experience()     # 記錄完整經驗
   ↓
9. Integration 模組處理結果 → 反饋給 External Learning
   ↓
10. External Learning 訓練模型 → 優化下次決策
```

## 🚀 快速開始

最小可行實現(MVP):

```python
# 在 command_handler.py 的 _handle_phase0 末尾添加
await self._record_scan_experience(
    scan_id=phase0_payload.scan_id,
    phase0_result=phase0_result,
    phase1_result=None,
    success=True,
    execution_time=execution_time
)
```

這就能立即開始積累掃描經驗,供 AI 學習改進! ✨

## 📚 相關文檔

- [BioNeuron 模型 AI 核心大腦](../../docs/architecture/BioNeuron_模型_AI核心大腦.md)
- [AI 模組整合架構](../../docs/reports/AI_MODULE_INTEGRATION_ARCHITECTURE.md)
- [數據合約規範](../aiva_common/schemas/README.md)
