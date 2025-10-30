---
Created: 2025-10-30
Last Modified: 2025-10-30
Document Type: Report
---

# 📚 AIVA 統一 API 文檔

> **📋 文檔類型**: API 使用指南 - 架構統一後版本  
> **🎯 版本**: v5.0 架構統一版  
> **📅 更新日期**: 2025年10月29日  
> **✅ 狀態**: 反映架構修復後的最新API使用方式

---

## 📋 目錄

- [🏗️ 架構統一後的API變更](#️-架構統一後的api變更)
- [🧠 AI 組件統一API](#-ai-組件統一api)
- [🌍 跨語言API使用](#-跨語言api使用)
- [⚡ 性能優化API](#-性能優化api)
- [📊 數據結構標準API](#-數據結構標準api)
- [🔧 實用工具API](#-實用工具api)
- [📝 使用範例](#-使用範例)

---

## 🏗️ 架構統一後的API變更

### ❌ 舊版API (已廢棄)
```python
# ❌ 不要再使用這些導入
from services.core.aiva_core.learning.capability_evaluator import *
from services.core.aiva_core.learning.experience_manager import *
```

### ✅ 新版統一API (推薦)
```python
# ✅ 統一AI組件API
from services.aiva_common.ai.capability_evaluator import AIVACapabilityEvaluator
from services.aiva_common.ai.experience_manager import AIVAExperienceManager
from services.aiva_common.ai.performance_config import PerformanceOptimizer

# ✅ 統一數據結構API
from services.aiva_common.schemas.base import MessageHeader
from services.aiva_common.schemas.ai import ExperienceSample, CapabilityInfo
from services.aiva_common.enums import ModuleName, ProgrammingLanguage

# ✅ 工廠函數API (推薦)
from services.aiva_common.ai import (
    get_capability_evaluator,
    get_experience_manager,
    get_performance_optimizer
)
```

---

## 🧠 AI 組件統一API

### 1. **CapabilityEvaluator API**

#### 基本使用
```python
from services.aiva_common.ai.capability_evaluator import AIVACapabilityEvaluator

# 創建實例
evaluator = AIVACapabilityEvaluator()

# 或使用工廠函數 (推薦)
evaluator = get_capability_evaluator()
```

#### 核心方法
```python
class AIVACapabilityEvaluator:
    async def evaluate_capability(
        self, 
        capability_info: CapabilityInfo,
        evidence: Dict[str, Any]
    ) -> CapabilityScorecard:
        """評估組件能力
        
        Args:
            capability_info: 能力基本信息
            evidence: 評估證據數據
            
        Returns:
            CapabilityScorecard: 詳細評估結果
        """
    
    async def collect_evidence(
        self,
        target_component: str,
        evaluation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """收集評估證據"""
    
    async def benchmark_performance(
        self,
        capability_id: str,
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """性能基準測試"""
    
    async def continuous_monitoring(
        self,
        capability_id: str,
        monitoring_config: Dict[str, Any]
    ) -> None:
        """持續監控設置"""
```

### 2. **ExperienceManager API**

#### 基本使用
```python
from services.aiva_common.ai.experience_manager import AIVAExperienceManager

# 創建實例
manager = AIVAExperienceManager()

# 或使用工廠函數 (推薦)
manager = get_experience_manager()
```

#### 核心方法
```python
class AIVAExperienceManager:
    async def store_experience(
        self,
        experience: ExperienceSample
    ) -> str:
        """存儲經驗樣本
        
        Args:
            experience: 經驗樣本數據
            
        Returns:
            str: 存儲ID
        """
    
    async def retrieve_experiences(
        self,
        query_params: Dict[str, Any],
        limit: int = 100
    ) -> List[ExperienceSample]:
        """檢索相關經驗"""
        
    async def evaluate_experience_quality(
        self,
        experience: ExperienceSample
    ) -> Dict[str, Any]:
        """評估經驗質量"""
        
    async def update_experience_weights(
        self,
        experience_id: str,
        performance_feedback: Dict[str, float]
    ) -> None:
        """更新經驗權重"""
```

---

## 🌍 跨語言API使用

### TypeScript API
```typescript
// TypeScript 完整對應實現
import { 
    AIVACapabilityEvaluator,
    AIVAExperienceManager,
    PerformanceConfig 
} from './aiva_common_ts';

import {
    ExperienceSample,
    CapabilityInfo,
    CapabilityScorecard
} from './aiva_common_ts/schemas';

// 使用方式與Python完全一致
const evaluator = new AIVACapabilityEvaluator();
const manager = new AIVAExperienceManager();

// 異步方法調用
const result = await evaluator.evaluateCapability(capabilityInfo, evidence);
const experiences = await manager.retrieveExperiences(queryParams);
```

### Go API (部分可用)
```go
// Go 模組使用 (3/4 完成)
package main

import (
    "github.com/aiva/function_cspm_go"
    "github.com/aiva/function_sca_go" 
    "github.com/aiva/function_ssrf_go"
)

// 高並發掃描
result := function_ssrf_go.ScanTarget(target)
cspmResult := function_cspm_go.CheckCompliance(config)
```

### Rust API (部分可用)  
```rust
// Rust 高性能模組 (2/3 完成)
use aiva_sast_rust::StaticAnalyzer;
use aiva_info_gatherer_rust::InfoCollector;

// 高性能靜態分析
let analyzer = StaticAnalyzer::new();
let results = analyzer.analyze_code(source_code).await?;
```

---

## ⚡ 性能優化API

### PerformanceOptimizer
```python
from services.aiva_common.ai.performance_config import (
    PerformanceOptimizer,
    CapabilityEvaluatorOptimizer,
    ExperienceManagerOptimizer
)

# 全局性能優化器
optimizer = PerformanceOptimizer(environment="production")

# 組件特定優化器
capability_optimizer = CapabilityEvaluatorOptimizer()
experience_optimizer = ExperienceManagerOptimizer()

# 應用優化配置
await optimizer.optimize_all_components()
await capability_optimizer.apply_caching_strategy()
await experience_optimizer.enable_batch_processing()
```

### 性能監控API
```python
from services.aiva_common.ai.performance_config import performance_monitor

# 性能監控裝飾器
@performance_monitor(operation="capability_evaluation")
async def evaluate_with_monitoring(capability_info):
    # 自動記錄性能指標
    return await evaluator.evaluate_capability(capability_info)

# 批處理優化
@batch_optimizer(batch_size=50, timeout=30)
async def batch_evaluate_capabilities(capability_list):
    # 自動批處理優化
    results = []
    for capability in capability_list:
        result = await evaluate_capability(capability)
        results.append(result)
    return results
```

---

## 📊 數據結構標準API

### 核心數據結構
```python
from services.aiva_common.schemas.base import MessageHeader
from services.aiva_common.schemas.ai import ExperienceSample, CapabilityInfo
from services.aiva_common.enums import ModuleName, ProgrammingLanguage

# MessageHeader - 統一消息頭
header = MessageHeader(
    message_id="ai_eval_001",
    trace_id="trace_123",
    source_module=ModuleName.AI_ENGINE,
    # timestamp 和 version 自動設置
)

# ExperienceSample - 經驗樣本
sample = ExperienceSample(
    sample_id="sample_001",
    session_id="session_123", 
    plan_id="plan_456",
    state_before={"context": "initial"},
    action_taken={"command": "scan"},
    state_after={"result": "completed"},
    reward=0.85
)

# CapabilityInfo - 能力信息
capability = CapabilityInfo(
    id="cap_001",
    language=ProgrammingLanguage.PYTHON,
    entrypoint="main.py",
    topic="vulnerability_scanning"
)
```

### 安全的數據創建
```python
from services.aiva_common.schemas.factory import SafeSchemaFactory

# 推薦的統一工廠模式
header = SafeSchemaFactory.create_message_header(
    message_id="ai_scan_001",
    source=ModuleName.SCAN_ENGINE
)

sample = SafeSchemaFactory.create_experience_sample(
    sample_id="exp_001",
    session_id="sess_001",
    action_data={"scan_type": "deep"},
    reward=0.75
)
```

---

## 🔧 實用工具API

### 環境檢查工具
```python
from services.aiva_common.utils.environment_checker import check_aiva_environment

# 全面環境檢查
env_status = await check_aiva_environment()
print(f"環境狀態: {env_status['overall_health']}%")
print(f"AI組件狀態: {env_status['ai_components']['status']}")
print(f"Schema合規性: {env_status['schema_compliance']}")
```

### 配置管理工具
```python
from services.aiva_common.config.manager import AIVAConfigManager

# 統一配置管理
config_manager = AIVAConfigManager()

# 獲取環境特定配置  
config = config_manager.get_config(
    component="capability_evaluator",
    environment="production"
)

# 動態配置更新
await config_manager.update_config(
    component="experience_manager", 
    updates={"batch_size": 100, "cache_ttl": 3600}
)
```

---

## 📝 使用範例

### 完整工作流程範例
```python
import asyncio
from services.aiva_common.ai import (
    get_capability_evaluator,
    get_experience_manager,
    get_performance_optimizer
)
from services.aiva_common.schemas.ai import CapabilityInfo, ExperienceSample
from services.aiva_common.enums import ProgrammingLanguage

async def main():
    # 1. 初始化組件 (使用工廠函數)
    evaluator = get_capability_evaluator()
    manager = get_experience_manager()
    optimizer = get_performance_optimizer()
    
    # 2. 應用性能優化
    await optimizer.optimize_all_components()
    
    # 3. 創建能力信息
    capability = CapabilityInfo(
        id="python_scanner",
        language=ProgrammingLanguage.PYTHON,
        entrypoint="scanner.py",
        topic="security_scanning"
    )
    
    # 4. 評估能力
    evidence = {"scan_results": [1, 2, 3], "performance": 0.95}
    scorecard = await evaluator.evaluate_capability(capability, evidence)
    
    # 5. 存儲經驗
    experience = ExperienceSample(
        sample_id="exp_001",
        session_id="sess_001", 
        plan_id="plan_001",
        state_before={"target": "app.com"},
        action_taken={"scan": "deep"},
        state_after={"findings": 5},
        reward=0.8
    )
    
    experience_id = await manager.store_experience(experience)
    
    # 6. 檢索相關經驗
    similar_experiences = await manager.retrieve_experiences(
        query_params={"topic": "security_scanning", "min_reward": 0.7},
        limit=10
    )
    
    print(f"✅ 能力評估完成: {scorecard.overall_score}")
    print(f"✅ 經驗存儲ID: {experience_id}")
    print(f"✅ 找到相似經驗: {len(similar_experiences)} 個")

# 運行示例
if __name__ == "__main__":
    asyncio.run(main())
```

### TypeScript 對應範例
```typescript
import {
    AIVACapabilityEvaluator,
    AIVAExperienceManager,
    PerformanceConfig
} from './aiva_common_ts';

import {
    CapabilityInfo,
    ExperienceSample,
    ProgrammingLanguage
} from './aiva_common_ts/schemas';

async function main() {
    // 1. 初始化組件
    const evaluator = new AIVACapabilityEvaluator();
    const manager = new AIVAExperienceManager();
    
    // 2. 創建能力信息
    const capability: CapabilityInfo = {
        id: "typescript_scanner",
        language: ProgrammingLanguage.TYPESCRIPT,
        entrypoint: "scanner.ts",
        topic: "security_scanning"
    };
    
    // 3. 評估和存儲 (與Python API完全一致)
    const evidence = { scan_results: [1, 2, 3], performance: 0.95 };
    const scorecard = await evaluator.evaluateCapability(capability, evidence);
    
    console.log(`✅ TypeScript 能力評估完成: ${scorecard.overall_score}`);
}

main().catch(console.error);
```

---

## 🚨 重要遷移指南

### 從舊API遷移
1. **更新所有導入語句**
   ```python
   # ❌ 舊版
   from services.core.aiva_core.learning import *
   
   # ✅ 新版
   from services.aiva_common.ai import *
   ```

2. **使用工廠函數** (推薦)
   ```python
   # ✅ 推薦方式
   evaluator = get_capability_evaluator()
   manager = get_experience_manager()
   ```

3. **遵循新的數據結構**
   ```python
   # ✅ 使用統一的Schema
   from services.aiva_common.schemas.ai import ExperienceSample
   from services.aiva_common.enums import ModuleName
   ```

### 性能優化建議
- 🚀 **優先使用工廠函數**: 自動應用性能優化
- 📊 **啟用批處理**: 對大量數據處理使用批處理API  
- 💾 **配置緩存**: 在生產環境啟用多層緩存
- 📈 **性能監控**: 使用內置性能監控裝飾器

---

**📝 文檔資訊**
- **版本**: v5.0 架構統一版
- **維護狀態**: ✅ 持續更新
- **相關文檔**: `ARCHITECTURE_UNIFICATION_COMPLETION_REPORT.md`, `AIVA_COMPREHENSIVE_GUIDE.md`
- **技術支援**: 完整API支援，包含錯誤處理和最佳實踐