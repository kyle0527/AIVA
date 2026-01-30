# Result Coordinator 整合規劃

> ⚠️ **未來規劃文檔**  
> 📌 **當前狀態**：未實現，這是一份設計提案  
> 🎯 **目的**：探討如何在 AI 內部增強結果分析能力  
> ✅ **現況**：AI 目前直接處理 CLI 的簡單 JSON 輸出（已驗證可行）  
> 🔮 **時機**：當需要更複雜的分析、學習、優化能力時再考慮實施

---

## 📋 概述

將 Coordinator 功能整合進 AI 的 `cognitive_core` 作為內部模組，負責處理和分析功能模組的 CLI 輸出結果。

**注意**：這是針對未來需求的設計方案，不是當前架構的描述。

## 🎯 設計目標

### 核心原則
1. **零侵入 CLI** - 功能模組的 CLI 不需要修改
2. **內部模組化** - Coordinator 作為 AI 的子模組存在
3. **漸進增強** - 可選開關，不影響現有流程
4. **與學習系統集成** - 為 AI 學習提供結構化數據

### 核心價值
- ✅ 統一結果分析標準
- ✅ 性能追蹤和優化建議
- ✅ 跨功能模式識別
- ✅ 為 AI 學習提供高質量輸入

## 🏗️ 架構設計

### 目錄結構
```
services/core/aiva_core/cognitive_core/
├── capability_orchestrator.py        # 主編排器（已存在）
├── result_coordinators/              # 新增：結果協調器目錄
│   ├── __init__.py                   # 導出主要類
│   ├── base_coordinator.py           # 基礎協調器
│   ├── xss_coordinator.py            # XSS 專用協調器
│   ├── sqli_coordinator.py           # SQL 注入專用協調器
│   ├── ssrf_coordinator.py           # SSRF 專用協調器
│   ├── coordinator_factory.py        # 協調器工廠
│   └── models.py                     # 數據模型
└── learning_system/                  # 學習系統（已存在）
    └── experience_manager.py         # 接收 Coordinator 分析結果
```

### 數據流架構

```
┌─────────────────────────────────────────────────────────────┐
│              CapabilityOrchestrator (AI 主控)                │
│                                                               │
│  1. 生成 CLI 命令                                              │
│  2. 執行 subprocess                                           │
│  3. 獲得原始 JSON 輸出                                         │
│                          ↓                                    │
│  ┌────────────────────────────────────────────────┐          │
│  │       ResultCoordinator (新增內部模組)          │          │
│  │                                                 │          │
│  │  • 接收原始 CLI 輸出                             │          │
│  │  • 分析和增強數據                                │          │
│  │  • 生成優化建議                                  │          │
│  │  • 提取學習信號                                  │          │
│  └────────────────────────────────────────────────┘          │
│                          ↓                                    │
│  4. 返回增強結果給 AI                                          │
│  5. 傳遞學習數據給 ExperienceManager                           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📦 核心組件設計

### 1. BaseCoordinator (基礎協調器)

```python
# result_coordinators/base_coordinator.py

class CoordinatedResult(BaseModel):
    """協調器增強的結果"""
    # 原始數據
    raw_output: Dict[str, Any]
    
    # 增強分析
    analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # 性能指標
    performance: Dict[str, Any] = Field(default_factory=dict)
    
    # 質量評分
    quality_score: float = Field(ge=0.0, le=1.0)
    
    # 優化建議
    recommendations: List[str] = Field(default_factory=list)
    
    # 學習信號（供 AI 學習系統使用）
    learning_signals: Dict[str, Any] = Field(default_factory=dict)


class BaseCoordinator(ABC):
    """結果協調器基類
    
    職責：
    1. 接收功能模組的 CLI 原始輸出
    2. 分析和增強結果
    3. 提取學習信號
    4. 生成優化建議
    """
    
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.logger = get_logger(f"coordinator.{feature_name}")
        
        # 歷史數據（用於對比分析）
        self.historical_results: List[CoordinatedResult] = []
        
    async def coordinate(self, raw_output: Dict[str, Any]) -> CoordinatedResult:
        """協調處理主流程"""
        try:
            # 1. 驗證輸入
            validated = self._validate_input(raw_output)
            
            # 2. 分析結果
            analysis = await self._analyze_result(validated)
            
            # 3. 計算性能指標
            performance = self._calculate_performance(validated)
            
            # 4. 評估質量
            quality_score = self._evaluate_quality(validated, analysis)
            
            # 5. 生成建議
            recommendations = self._generate_recommendations(
                validated, analysis, performance
            )
            
            # 6. 提取學習信號
            learning_signals = self._extract_learning_signals(
                validated, analysis, performance
            )
            
            result = CoordinatedResult(
                raw_output=raw_output,
                analysis=analysis,
                performance=performance,
                quality_score=quality_score,
                recommendations=recommendations,
                learning_signals=learning_signals
            )
            
            # 記錄歷史
            self.historical_results.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Coordination failed: {e}")
            # 降級處理：返回原始數據
            return CoordinatedResult(
                raw_output=raw_output,
                quality_score=0.5,
                recommendations=["協調處理失敗，使用原始結果"]
            )
    
    @abstractmethod
    async def _analyze_result(self, validated: Dict) -> Dict[str, Any]:
        """功能特定的結果分析"""
        pass
    
    def _validate_input(self, raw_output: Dict) -> Dict:
        """驗證和標準化輸入"""
        return raw_output
    
    def _calculate_performance(self, validated: Dict) -> Dict[str, Any]:
        """計算性能指標（基於可用數據）"""
        return {
            "findings_count": len(validated.get("findings", [])),
            "success": validated.get("vulnerable", False),
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    def _evaluate_quality(self, validated: Dict, analysis: Dict) -> float:
        """評估結果質量"""
        # 基礎評分邏輯
        score = 0.5
        
        if validated.get("findings"):
            score += 0.3
            
        if analysis.get("high_confidence_findings"):
            score += 0.2
            
        return min(score, 1.0)
    
    def _generate_recommendations(
        self, validated: Dict, analysis: Dict, performance: Dict
    ) -> List[str]:
        """生成優化建議"""
        recommendations = []
        
        # 基於歷史對比
        if len(self.historical_results) > 5:
            avg_quality = sum(r.quality_score for r in self.historical_results[-5:]) / 5
            if performance.get("quality_score", 0.5) < avg_quality:
                recommendations.append("本次掃描質量低於近期平均水平")
        
        return recommendations
    
    def _extract_learning_signals(
        self, validated: Dict, analysis: Dict, performance: Dict
    ) -> Dict[str, Any]:
        """提取 AI 學習信號"""
        return {
            "feature": self.feature_name,
            "success": validated.get("vulnerable", False),
            "quality": self._evaluate_quality(validated, analysis),
            "patterns": analysis.get("patterns", []),
            "timestamp": datetime.now(UTC).isoformat()
        }
```

### 2. XSSCoordinator (XSS 專用協調器)

```python
# result_coordinators/xss_coordinator.py

class XSSCoordinator(BaseCoordinator):
    """XSS 專用協調器"""
    
    def __init__(self):
        super().__init__(feature_name="xss")
        
        # XSS 特定配置
        self.payload_categories = {
            "script_tag": ["<script>", "</script>"],
            "event_handler": ["onerror=", "onload=", "onclick="],
            "svg_vector": ["<svg", "onload"],
            "img_vector": ["<img", "onerror"]
        }
    
    async def _analyze_result(self, validated: Dict) -> Dict[str, Any]:
        """XSS 特定分析"""
        findings = validated.get("findings", [])
        
        analysis = {
            "total_findings": len(findings),
            "by_category": self._categorize_payloads(findings),
            "high_confidence": self._filter_high_confidence(findings),
            "potential_false_positives": self._detect_false_positives(findings),
            "csp_detected": self._check_csp(validated),
            "waf_detected": self._check_waf(validated)
        }
        
        return analysis
    
    def _categorize_payloads(self, findings: List[Dict]) -> Dict[str, int]:
        """將 payload 分類統計"""
        categories = {cat: 0 for cat in self.payload_categories}
        
        for finding in findings:
            payload = finding.get("payload", "").lower()
            for cat, keywords in self.payload_categories.items():
                if any(kw.lower() in payload for kw in keywords):
                    categories[cat] += 1
                    break
        
        return categories
    
    def _filter_high_confidence(self, findings: List[Dict]) -> List[Dict]:
        """過濾高置信度發現"""
        return [
            f for f in findings 
            if f.get("vulnerable", False) and f.get("status") == 200
        ]
    
    def _detect_false_positives(self, findings: List[Dict]) -> List[Dict]:
        """檢測潛在誤報"""
        potential_fp = []
        
        for finding in findings:
            evidence = finding.get("evidence", "")
            
            # 檢查是否被編碼
            if "&lt;" in evidence or "&gt;" in evidence:
                potential_fp.append(finding)
            # 檢查是否在註釋中
            elif "<!--" in evidence and "-->" in evidence:
                potential_fp.append(finding)
        
        return potential_fp
    
    def _check_csp(self, validated: Dict) -> bool:
        """檢測是否存在 CSP 保護"""
        # 基於 evidence 或 headers 檢測
        for finding in validated.get("findings", []):
            evidence = finding.get("evidence", "").lower()
            if "content-security-policy" in evidence:
                return True
        return False
    
    def _check_waf(self, validated: Dict) -> bool:
        """檢測是否觸發 WAF"""
        for finding in validated.get("findings", []):
            status = finding.get("status", 200)
            if status in [403, 406, 429]:
                return True
        return False
    
    def _generate_recommendations(
        self, validated: Dict, analysis: Dict, performance: Dict
    ) -> List[str]:
        """XSS 特定建議"""
        recommendations = super()._generate_recommendations(
            validated, analysis, performance
        )
        
        # XSS 特定建議
        if analysis.get("csp_detected"):
            recommendations.append("檢測到 CSP，建議測試 CSP bypass 技術")
        
        if analysis.get("waf_detected"):
            recommendations.append("檢測到 WAF，建議使用混淆 payload")
        
        by_category = analysis.get("by_category", {})
        if by_category.get("script_tag", 0) == 0:
            recommendations.append("未測試 <script> 標籤，建議補充測試")
        
        if len(analysis.get("potential_false_positives", [])) > 0:
            recommendations.append(
                f"發現 {len(analysis['potential_false_positives'])} 個潛在誤報，需人工驗證"
            )
        
        return recommendations
```

### 3. CoordinatorFactory (協調器工廠)

```python
# result_coordinators/coordinator_factory.py

class CoordinatorFactory:
    """協調器工廠"""
    
    _coordinators: Dict[str, BaseCoordinator] = {}
    
    @classmethod
    def get_coordinator(cls, feature_name: str) -> Optional[BaseCoordinator]:
        """獲取或創建協調器"""
        if feature_name not in cls._coordinators:
            coordinator = cls._create_coordinator(feature_name)
            if coordinator:
                cls._coordinators[feature_name] = coordinator
        
        return cls._coordinators.get(feature_name)
    
    @classmethod
    def _create_coordinator(cls, feature_name: str) -> Optional[BaseCoordinator]:
        """創建協調器實例"""
        coordinator_map = {
            "xss": XSSCoordinator,
            "sqli": SQLiCoordinator,
            "ssrf": SSRFCoordinator,
            # 更多協調器...
        }
        
        coordinator_class = coordinator_map.get(feature_name)
        if coordinator_class:
            return coordinator_class()
        
        logger.warning(f"No coordinator for {feature_name}, using base")
        return None
```

### 4. CapabilityOrchestrator 集成

```python
# capability_orchestrator.py (修改)

class CapabilityOrchestrator:
    
    def __init__(self, ...):
        # ... 現有初始化 ...
        
        # 新增：結果協調器支援
        from .result_coordinators import CoordinatorFactory
        self.coordinator_factory = CoordinatorFactory
        self.use_coordinators = True  # 可配置開關
    
    async def execute(self, plan: CapabilityPlan) -> ExecutionResult:
        """執行計劃（增強版）"""
        # ... 現有執行邏輯 ...
        
        # 新增：處理每個命令的輸出
        for cli_cmd in plan.cli_commands:
            result = await process_manager.run_command_with_telemetry(...)
            
            # 解析原始輸出
            try:
                raw_output = json.loads(result["stdout"])
            except:
                raw_output = {"error": "Invalid JSON"}
            
            # 可選：通過協調器增強
            if self.use_coordinators:
                feature_name = self._extract_feature_name(cli_cmd)
                coordinator = self.coordinator_factory.get_coordinator(feature_name)
                
                if coordinator:
                    coordinated = await coordinator.coordinate(raw_output)
                    
                    # 保存增強結果
                    command_outputs[cli_cmd] = {
                        "stdout": result["stdout"],
                        "raw": raw_output,
                        "coordinated": coordinated.dict(),
                        "exit_code": result["exit_code"],
                        # ... 其他字段
                    }
                    
                    # 傳遞學習信號給學習系統
                    if self.experience_manager:
                        await self._feed_learning_signals(coordinated)
                else:
                    # 無協調器，使用原始結果
                    command_outputs[cli_cmd] = {
                        "stdout": result["stdout"],
                        "raw": raw_output,
                        "exit_code": result["exit_code"]
                    }
            else:
                # 不使用協調器
                command_outputs[cli_cmd] = {...}
        
        # ... 返回執行結果 ...
    
    def _extract_feature_name(self, cli_cmd: str) -> str:
        """從 CLI 命令提取功能名稱"""
        # python -m function_xss ... -> xss
        if "function_xss" in cli_cmd:
            return "xss"
        elif "function_sqli" in cli_cmd:
            return "sqli"
        elif "function_ssrf" in cli_cmd:
            return "ssrf"
        # ... 更多映射
        return "unknown"
    
    async def _feed_learning_signals(self, coordinated: CoordinatedResult):
        """將協調器的學習信號傳遞給學習系統"""
        if self.experience_manager:
            await self.experience_manager.record_experience(
                feature=coordinated.learning_signals.get("feature"),
                success=coordinated.learning_signals.get("success"),
                quality=coordinated.learning_signals.get("quality"),
                patterns=coordinated.learning_signals.get("patterns"),
                metadata=coordinated.learning_signals
            )
```

## 🔧 配置支援

### 配置檔案
```yaml
# config/coordinator_config.yaml

result_coordinators:
  enabled: true  # 總開關
  
  features:
    xss:
      enabled: true
      quality_threshold: 0.7
      false_positive_detection: true
    
    sqli:
      enabled: true
      quality_threshold: 0.8
    
    ssrf:
      enabled: true
  
  learning:
    feed_to_experience_manager: true
    min_quality_for_learning: 0.6
  
  performance:
    cache_results: true
    max_history: 100
```

## 📊 數據模型

### 增強的 ExecutionResult

```python
class ExecutionResult(BaseModel):
    # ... 現有字段 ...
    
    # 新增：協調器增強數據
    coordinated_results: Dict[str, CoordinatedResult] = Field(
        default_factory=dict,
        description="協調器增強的結果映射 {cmd: coordinated_result}"
    )
    
    # 新增：聚合分析
    aggregate_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="跨功能聚合分析"
    )
```

## 🚀 實施步驟

### Phase 1: 基礎架構 (第 1-2 天)
- [x] 創建 `result_coordinators/` 目錄
- [ ] 實現 `BaseCoordinator` 基類
- [ ] 實現 `CoordinatorFactory` 工廠
- [ ] 實現數據模型 `models.py`
- [ ] 添加配置支援

### Phase 2: XSS 協調器 (第 3 天)
- [ ] 實現 `XSSCoordinator`
- [ ] 添加 payload 分類邏輯
- [ ] 添加誤報檢測
- [ ] 添加 CSP/WAF 檢測
- [ ] 測試與現有 XSS CLI 集成

### Phase 3: AI 集成 (第 4 天)
- [ ] 修改 `CapabilityOrchestrator.execute()`
- [ ] 添加協調器調用邏輯
- [ ] 集成 `ExperienceManager`
- [ ] 測試完整流程

### Phase 4: 更多協調器 (第 5-6 天)
- [ ] 實現 `SQLiCoordinator`
- [ ] 實現 `SSRFCoordinator`
- [ ] 實現 `IDORCoordinator`
- [ ] 跨模組模式識別

### Phase 5: 優化和測試 (第 7 天)
- [ ] 性能優化
- [ ] 錯誤處理完善
- [ ] 單元測試
- [ ] 集成測試
- [ ] 文檔完善

## 🎯 成功指標

1. **功能指標**
   - ✅ 所有功能模組的 CLI 輸出能被協調器處理
   - ✅ 協調器能生成有價值的優化建議
   - ✅ 學習信號能正確傳遞給學習系統

2. **性能指標**
   - ✅ 協調器處理延遲 < 100ms
   - ✅ 不影響現有命令執行流程
   - ✅ 可選開關能正常工作

3. **質量指標**
   - ✅ 能檢測出至少 80% 的誤報
   - ✅ 優化建議準確率 > 70%
   - ✅ 學習信號質量提升 AI 決策效果

## 💡 設計優勢

### vs 原始設計（獨立 Coordinator）
| 項目 | 原始設計 | 新設計 |
|------|---------|--------|
| 位置 | `services/integration/coordinators/` | `cognitive_core/result_coordinators/` |
| 調用方 | 需要外部調用 | AI 內部調用 |
| CLI 改動 | 需要輸出 20+ 字段 | **不需要改動** |
| 耦合度 | 與 AI 分離 | 與 AI 緊密集成 |
| 學習集成 | 需要額外橋接 | 直接傳遞給 ExperienceManager |
| 複雜度 | 548 行基類 | ~150 行基類 |

### 核心優勢
1. ✅ **零侵入** - CLI 完全不用改
2. ✅ **內聚性高** - 作為 AI 的一部分，邏輯自然
3. ✅ **學習閉環** - 直接集成學習系統
4. ✅ **漸進式** - 可選開關，不破壞現有流程
5. ✅ **可擴展** - 工廠模式易於添加新協調器

## 📝 示例：完整流程

```python
# 1. 用戶發起任務
requirement = TaskRequirement(
    task_id="task_001",
    task_type="scan",
    target="https://example.com",
    objectives=["find_xss"]
)

# 2. AI 生成計劃
orchestrator = CapabilityOrchestrator()
plan = await orchestrator.plan(requirement)

# 3. 執行計劃（內部使用協調器）
result = await orchestrator.execute(plan)

# 4. 查看增強結果
for cmd, output in result.command_outputs.items():
    if "coordinated" in output:
        coordinated = output["coordinated"]
        print(f"質量評分: {coordinated['quality_score']}")
        print(f"建議: {coordinated['recommendations']}")
        print(f"分析: {coordinated['analysis']}")

# 5. AI 自動學習（背後自動完成）
# ExperienceManager 已經收到學習信號並更新模型
```

## 🔒 風險控制

### 降級策略
- 協調器異常時，自動降級為原始輸出
- 配置開關可隨時關閉協調器
- 不影響關鍵路徑（命令執行）

### 兼容性
- 完全向後兼容現有流程
- CLI 不需要任何修改
- 可選功能，默認可關閉

## 📚 參考資料

- [原始 Coordinator 設計](../../integration/coordinators/base_coordinator.py)
- [CapabilityOrchestrator 現有實現](./capability_orchestrator.py)
- [ExperienceManager 接口](./learning_system/experience_manager.py)
- [AIVA CLI 架構文檔](../../../../CROSS_LANGUAGE_CLI_DESIGN.md)

---

**狀態**: 規劃完成，待實施
**預計工期**: 7 天
**風險等級**: 低（零侵入，可降級）
