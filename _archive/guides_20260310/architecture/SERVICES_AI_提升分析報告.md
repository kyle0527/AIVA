# AIVA Services AI 提升分析報告

**日期**: 2026年1月11日  
**分析範圍**: `services/` 目錄，特別是 AI 核心模組  
**版本**: v1.0

---

## 📊 目錄

- [現況概覽](#現況概覽)
- [AI 架構分析](#ai-架構分析)
- [問題診斷](#問題診斷)
- [改進方案](#改進方案)
- [實施計畫](#實施計畫)

---

## 🔍 現況概覽

### 目錄結構

```
services/
├── core/                          # AI 核心引擎
│   └── aiva_core/
│       ├── cognitive_core/        # 認知核心 (決策中樞)
│       │   ├── decision/          # 決策代理
│       │   ├── neural/            # 神經網路 (5M 模型)
│       │   ├── rag/               # RAG 知識庫
│       │   ├── learning_system/   # 經驗學習
│       │   ├── internal_loop_connector.py   # 內閉環
│       │   └── external_loop_connector.py   # 外閉環
│       ├── core_capabilities/     # 核心能力
│       └── task_planning/         # 任務規劃
├── features/                      # 功能模組 (攻擊)
│   └── features_ready/            # 已完成功能
│       ├── function_sqli/         # SQL 注入
│       ├── function_xss/          # XSS
│       ├── function_ssrf/         # SSRF
│       └── function_idor/         # IDOR
├── scan/                          # 掃描引擎
│   ├── rust_engine/               # Rust 高性能
│   ├── python_engine/             # Python 檢測
│   └── coordinators/              # 協調器 (🔴 缺失)
└── integration/                   # 整合層
```

### 已實現 AI 組件

| 組件 | 檔案 | 行數 | 狀態 |
|------|------|------|------|
| **EnhancedDecisionAgent** | `decision/enhanced_decision_agent.py` | 2231 | ✅ 完整 |
| **RealDecisionEngine** | `neural/real_neural_core.py` | 1077 | ✅ 完整 |
| **RAGEngine** | `rag/rag_engine.py` | 588 | ✅ 完整 |
| **InternalLoopConnector** | `internal_loop_connector.py` | 2036 | ✅ 完整 |
| **ExternalLoopConnector** | `external_loop_connector.py` | 447 | ✅ 完整 |
| **CapabilityOrchestrator** | `capability_orchestrator.py` | 1118 | ✅ 完整 |
| **ExperienceManager** | `learning_system/experience_manager.py` | 646 | ✅ 完整 |
| **AttackCoordinator** | `task_planning/commander/attack_coordinator.py` | 596 | ⚠️ 依賴缺失 |
| **MultiEngineCoordinator** | `scan/coordinators/` | 0 | 🔴 未實現 |

---

## 🧠 AI 架構分析

### 1. 決策流程 (現況)

```
用戶輸入
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  EnhancedDecisionAgent (決策代理)                                     │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │
│  │ 5M Neural │  │ RAG 檢索  │  │ 規則引擎  │  │ 經驗學習  │        │
│  │   (50%)   │  │   (+5%)   │  │   (20%)   │  │   (30%)   │        │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘        │
│        └───────────────┴───────────────┴─────────────┘              │
│                              │                                       │
│                    _ensemble_decision (加權融合)                     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                         HighLevelIntent
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CapabilityOrchestrator (能力編排)                                    │
│  - 查詢 RAG 找相關能力                                               │
│  - 選擇最佳能力組合                                                   │
│  - 生成 CLI 命令序列                                                  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  AttackCoordinator (攻擊協調)                                         │
│  - 調用 features 功能模組                                            │
│  - 🔴 MultiEngineCoordinator (缺失)                                  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
                        功能模組 (SQLi/XSS/SSRF/IDOR)
```

### 2. 5M 神經網路架構

```python
# 現有架構：5M 特化神經網路
RealAICore:
    input_size:     512 維 (語意編碼)
    hidden_layers:  [1600, 1200, 1024, 512]
    output_size:    100 維
    total_params:   ~5M
    
編碼器：SentenceTransformer ('all-MiniLM-L6-v2')
    output_dim:     384 維
    + Bug Bounty Features: 32 維
    = 416 維 → 投影至 512 維
```

### 3. 雙閉環機制

| 閉環 | 連接器 | 功能 | 狀態 |
|------|--------|------|------|
| **內閉環** | InternalLoopConnector | AI 自我認知 (能力發現 → RAG) | ✅ 已實現 |
| **外閉環** | ExternalLoopConnector | 執行學習 (偏差分析 → 訓練) | ✅ 已實現 |

---

## 🔴 問題診斷

### P0 - 關鍵問題

#### 1. MultiEngineCoordinator 未實現
```
位置: services/scan/coordinators/multi_engine_coordinator.py
問題: 檔案不存在，僅有 __init__.py 引用
影響: 
  - AttackCoordinator.coordinate_multilang() 無法執行
  - 多引擎協調完全失效
  - 掃描策略 (fast/balanced/comprehensive) 無法使用
```

#### 2. 神經網路輸出未對齊
```python
# 現況
RealAICore.output_size = 100 維
RealDecisionEngine.action_map = 僅 5 個動作

# 問題
- 100 維輸出只使用了 5 個類別映射
- 95 維浪費
- 無法細粒度決策
```

#### 3. RAG 檢索效能
```python
# 現況
RAGEngine.enhance_attack_plan():
    - 3 次向量搜索 (techniques/experiences/best_practices)
    - 每次 top_k=5
    - 無緩存機制

# 問題
- 冷啟動延遲高
- 重複查詢浪費資源
```

### P1 - 重要問題

#### 4. 經驗學習不完整
```python
# ExperienceManager 缺陷
- reward 計算過於簡化 (固定閾值 0.6)
- 無 TD-error 優先級採樣
- 經驗衰減機制缺失
```

#### 5. 決策權重固定
```python
# EnhancedDecisionAgent._ensemble_decision()
W_NEURAL = 0.5      # 固定
W_EXPERIENCE = 0.3  # 固定
W_RULE = 0.2        # 固定

# 問題
- 無法根據情境動態調整
- 新手模式 vs 專家模式 無差異
```

### P2 - 優化空間

#### 6. 編碼器單一
```python
# 現況
SentenceTransformer('all-MiniLM-L6-v2')  # 384 維

# 建議
- 代碼專用編碼器 (CodeBERT, GraphCodeBERT)
- 安全領域微調模型
```

#### 7. 攻擊類型覆蓋不足
```python
# 現有 features_ready
function_sqli, function_xss, function_ssrf, function_idor

# 缺少
function_xxe, function_lfi, function_rfi, function_ssti
function_deserialization, function_race_condition
```

---

## 🚀 改進方案

### 方案 A: MultiEngineCoordinator 實現 (P0)

```python
# services/scan/coordinators/multi_engine_coordinator.py

"""多引擎協調器 - AI 控制的多語言掃描引擎統一入口"""

import asyncio
import subprocess
from typing import Any
from dataclasses import dataclass
from enum import Enum
from aiva_common.utils import get_logger

logger = get_logger(__name__)


class ScanStrategy(str, Enum):
    """掃描策略"""
    FAST = "fast"              # 快速：僅 Rust 引擎
    BALANCED = "balanced"      # 平衡：Rust + Python
    COMPREHENSIVE = "comprehensive"  # 全面：全部引擎
    AGGRESSIVE = "aggressive"  # 激進：並行 + 深度
    SMART = "smart"            # 智能：AI 動態決策


@dataclass
class EngineConfig:
    """引擎配置"""
    name: str
    language: str
    cli_command: str
    timeout: int = 300
    priority: int = 5


class MultiEngineCoordinator:
    """多引擎協調器
    
    統一管理 Rust/Go/TypeScript/Python 掃描引擎
    通過 subprocess + JSON 標準化通訊
    """
    
    # 引擎配置表
    ENGINES = {
        "rust_scanner": EngineConfig(
            name="Rust Scanner",
            language="rust",
            cli_command="services/scan/rust_engine/target/release/rust_scanner",
            timeout=120,
            priority=10  # 最高優先級
        ),
        "python_analyzer": EngineConfig(
            name="Python Analyzer",
            language="python",
            cli_command="python -m services.scan.python_engine.passive_analyzer",
            timeout=180,
            priority=7
        ),
        # 預留 Go/TypeScript 引擎
    }
    
    def __init__(self):
        self._initialized = False
        self._active_scans: dict[str, asyncio.Task] = {}
        
    async def initialize(self) -> None:
        """初始化協調器"""
        # 健康檢查各引擎
        for engine_id, config in self.ENGINES.items():
            available = await self._check_engine_health(config)
            logger.info(f"Engine {config.name}: {'✅' if available else '❌'}")
        
        self._initialized = True
        logger.info("MultiEngineCoordinator initialized")
    
    async def _check_engine_health(self, config: EngineConfig) -> bool:
        """檢查引擎健康狀態"""
        try:
            # 對於 Rust 引擎，檢查二進制是否存在
            if config.language == "rust":
                import os
                return os.path.exists(config.cli_command)
            return True
        except Exception:
            return False
    
    async def execute_strategy_fast(
        self, scan_id: str, targets: list[str], max_depth: int = 3
    ) -> dict[str, Any]:
        """快速掃描策略"""
        return await self._execute_scan(
            scan_id=scan_id,
            targets=targets,
            engines=["rust_scanner"],
            max_depth=max_depth
        )
    
    async def execute_strategy_balanced(
        self, scan_id: str, targets: list[str], max_depth: int = 3
    ) -> dict[str, Any]:
        """平衡掃描策略"""
        return await self._execute_scan(
            scan_id=scan_id,
            targets=targets,
            engines=["rust_scanner", "python_analyzer"],
            max_depth=max_depth
        )
    
    async def execute_strategy_comprehensive(
        self, scan_id: str, targets: list[str], max_depth: int = 5
    ) -> dict[str, Any]:
        """全面掃描策略"""
        return await self._execute_scan(
            scan_id=scan_id,
            targets=targets,
            engines=list(self.ENGINES.keys()),
            max_depth=max_depth
        )
    
    async def execute_strategy_aggressive(
        self, scan_id: str, targets: list[str], max_depth: int = 10
    ) -> dict[str, Any]:
        """激進掃描策略（並行）"""
        tasks = []
        for engine_id in self.ENGINES.keys():
            task = self._execute_single_engine(
                scan_id=f"{scan_id}_{engine_id}",
                targets=targets,
                engine_id=engine_id,
                max_depth=max_depth
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._merge_results(results)
    
    async def execute_strategy_smart(
        self, scan_id: str, targets: list[str], max_depth: int = 5
    ) -> dict[str, Any]:
        """智能掃描策略（AI 決策）"""
        # Phase 1: 快速掃描探測
        initial_result = await self.execute_strategy_fast(scan_id, targets, max_depth=2)
        
        # Phase 2: 根據初步結果決定深度
        if initial_result.get("findings_count", 0) > 0:
            # 發現漏洞，進行深度掃描
            return await self.execute_strategy_comprehensive(scan_id, targets, max_depth)
        else:
            # 無發現，使用平衡策略
            return await self.execute_strategy_balanced(scan_id, targets, max_depth)
    
    async def _execute_scan(
        self,
        scan_id: str,
        targets: list[str],
        engines: list[str],
        max_depth: int
    ) -> dict[str, Any]:
        """執行掃描"""
        results = {
            "scan_id": scan_id,
            "targets": targets,
            "engines_used": engines,
            "findings": [],
            "errors": [],
            "duration": 0.0
        }
        
        import time
        start_time = time.time()
        
        for engine_id in engines:
            engine_result = await self._execute_single_engine(
                scan_id, targets, engine_id, max_depth
            )
            
            if engine_result.get("success"):
                results["findings"].extend(engine_result.get("findings", []))
            else:
                results["errors"].append(engine_result.get("error"))
        
        results["duration"] = time.time() - start_time
        results["findings_count"] = len(results["findings"])
        
        return results
    
    async def _execute_single_engine(
        self,
        scan_id: str,
        targets: list[str],
        engine_id: str,
        max_depth: int
    ) -> dict[str, Any]:
        """執行單一引擎"""
        config = self.ENGINES.get(engine_id)
        if not config:
            return {"success": False, "error": f"Unknown engine: {engine_id}"}
        
        try:
            # 構建 CLI 命令
            cmd = [
                config.cli_command,
                "--targets", ",".join(targets),
                "--depth", str(max_depth),
                "--output-format", "json"
            ]
            
            # 執行 subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=config.timeout
            )
            
            if process.returncode == 0:
                import json
                output = json.loads(stdout.decode())
                return {"success": True, "findings": output.get("findings", [])}
            else:
                return {"success": False, "error": stderr.decode()}
                
        except asyncio.TimeoutError:
            return {"success": False, "error": f"Timeout after {config.timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _merge_results(self, results: list) -> dict[str, Any]:
        """合併多引擎結果"""
        merged = {
            "findings": [],
            "errors": [],
            "engines_completed": 0
        }
        
        for result in results:
            if isinstance(result, Exception):
                merged["errors"].append(str(result))
            elif isinstance(result, dict):
                merged["findings"].extend(result.get("findings", []))
                if result.get("success"):
                    merged["engines_completed"] += 1
        
        merged["findings_count"] = len(merged["findings"])
        return merged
```

### 方案 B: 神經網路輸出優化 (P0)

```python
# 修改 enhanced_decision_agent.py

# 擴展動作映射（從 5 個增加到 25 個）
ACTION_MAP_EXTENDED = {
    # 掃描動作 (0-9)
    "reconnaissance": "SCAN_RECON",
    "port_scan": "SCAN_PORTS",
    "service_detection": "SCAN_SERVICES",
    "web_crawl": "SCAN_WEB",
    "api_discovery": "SCAN_API",
    
    # 漏洞檢測動作 (10-19)
    "sql_injection": "TEST_SQLI",
    "cross_site_scripting": "TEST_XSS",
    "server_side_request_forgery": "TEST_SSRF",
    "idor": "TEST_IDOR",
    "xxe": "TEST_XXE",
    "lfi": "TEST_LFI",
    "rfi": "TEST_RFI",
    "ssti": "TEST_SSTI",
    "deserialization": "TEST_DESERIAL",
    "race_condition": "TEST_RACE",
    
    # 攻擊動作 (20-29)
    "exploit_sqli": "EXPLOIT_SQLI",
    "exploit_xss": "EXPLOIT_XSS",
    "exploit_ssrf": "EXPLOIT_SSRF",
    "file_upload_bypass": "EXPLOIT_UPLOAD",
    "auth_bypass": "EXPLOIT_AUTH",
    
    # 控制動作 (30-34)
    "stop_operation": "STOP",
    "change_strategy": "CHANGE_STRATEGY",
    "require_confirmation": "REQUIRE_CONFIRM",
    "escalate_privilege": "ESCALATE",
    "persist_access": "PERSIST"
}

# 修改 RealAICore 輸出維度
# output_size: 100 → 35 (精確匹配動作數)
```

### 方案 C: 動態決策權重 (P1)

```python
# 新增: adaptive_weight_manager.py

class AdaptiveWeightManager:
    """自適應權重管理器
    
    根據任務上下文和歷史效能動態調整決策權重
    """
    
    def __init__(self):
        # 基礎權重
        self.base_weights = {
            "neural": 0.5,
            "experience": 0.3,
            "rule": 0.2
        }
        
        # 情境調整因子
        self.context_factors = {
            "high_risk": {"neural": -0.1, "rule": +0.2},
            "low_experience": {"neural": +0.1, "experience": -0.1},
            "time_critical": {"neural": +0.15, "experience": -0.1},
            "known_target": {"experience": +0.15, "neural": -0.1}
        }
    
    def get_weights(self, context: DecisionContext) -> dict[str, float]:
        """根據上下文計算動態權重"""
        weights = self.base_weights.copy()
        
        # 高風險情境
        if context.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            self._apply_factor(weights, "high_risk")
        
        # 經驗不足情境
        if context.attempts_without_success >= 3:
            self._apply_factor(weights, "low_experience")
        
        # 時間緊迫情境
        if context.time_constraints and context.time_constraints < 300:
            self._apply_factor(weights, "time_critical")
        
        # 歸一化
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def _apply_factor(self, weights: dict, factor_name: str) -> None:
        """應用調整因子"""
        factors = self.context_factors.get(factor_name, {})
        for key, delta in factors.items():
            if key in weights:
                weights[key] = max(0.05, min(0.8, weights[key] + delta))
```

### 方案 D: RAG 效能優化 (P1)

```python
# 修改 rag_engine.py

from functools import lru_cache
from typing import Tuple
import hashlib


class CachedRAGEngine(RAGEngine):
    """帶緩存的 RAG 引擎"""
    
    def __init__(self, knowledge_base, cache_size: int = 1000):
        super().__init__(knowledge_base)
        self._cache_size = cache_size
        self._query_cache: dict[str, Tuple[float, Any]] = {}
    
    def _cache_key(self, query: str, query_type: str) -> str:
        """生成緩存鍵"""
        return hashlib.md5(f"{query_type}:{query}".encode()).hexdigest()
    
    async def search_with_cache(
        self, 
        query: str, 
        query_type: str,
        top_k: int = 5,
        cache_ttl: float = 300.0  # 5 分鐘
    ) -> list:
        """帶緩存的搜索"""
        import time
        
        cache_key = self._cache_key(query, query_type)
        
        # 檢查緩存
        if cache_key in self._query_cache:
            timestamp, result = self._query_cache[cache_key]
            if time.time() - timestamp < cache_ttl:
                return result
        
        # 執行實際搜索
        result = await self.knowledge_base.search(
            query=f"{query_type} {query}",
            top_k=top_k
        )
        
        # 更新緩存
        self._query_cache[cache_key] = (time.time(), result)
        
        # 清理過期緩存
        self._cleanup_cache(cache_ttl)
        
        return result
    
    def _cleanup_cache(self, ttl: float) -> None:
        """清理過期緩存"""
        import time
        current = time.time()
        expired = [
            k for k, (ts, _) in self._query_cache.items()
            if current - ts > ttl
        ]
        for k in expired:
            del self._query_cache[k]
    
    async def enhance_attack_plan_optimized(
        self,
        target,
        objective: str
    ) -> dict:
        """優化版攻擊計畫增強（批量查詢）"""
        query = f"{objective} {target.target_type} {target.target_url}"
        
        # 批量並行查詢（3 個查詢同時執行）
        import asyncio
        
        techniques_task = self.search_with_cache(query, "attack_technique", 5)
        experiences_task = self.search_with_cache(query, "experience success", 5)
        practices_task = self.search_with_cache(query, "best_practice", 3)
        
        techniques, experiences, practices = await asyncio.gather(
            techniques_task, experiences_task, practices_task
        )
        
        return {
            "similar_techniques": techniques,
            "successful_experiences": experiences,
            "best_practices": practices
        }
```

### 方案 E: 優先級經驗採樣 (P1)

```python
# 修改 experience_manager.py

import numpy as np


class PrioritizedExperienceManager(ExperienceManager):
    """優先級經驗管理器
    
    實現 Prioritized Experience Replay (PER)
    基於 TD-error 進行優先採樣
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        super().__init__(capacity)
        
        # PER 參數
        self.alpha = alpha  # 優先級指數 (0=均勻, 1=完全優先)
        self.beta = beta    # 重要性採樣指數
        
        # 優先級樹（簡化版：使用列表）
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        self._position = 0
    
    def push(
        self,
        state: dict,
        action: dict,
        next_state: dict,
        reward: float,
        metadata: dict = None,
        td_error: float = None,  # 新增: TD-error
        environment: str = None
    ) -> str:
        """保存經驗並設置優先級"""
        
        exp_id = super().push(state, action, next_state, reward, metadata, environment)
        
        # 設置優先級（基於 TD-error 或 reward）
        priority = abs(td_error) if td_error else abs(reward - 0.5) + 0.01
        self.priorities[self._position] = priority ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        
        self._position = (self._position + 1) % self.capacity
        
        return exp_id
    
    def prioritized_sample(self, batch_size: int) -> Tuple[list, np.ndarray, np.ndarray]:
        """優先級採樣
        
        Returns:
            (samples, weights, indices)
        """
        n = len(self.memory)
        if n == 0:
            return [], np.array([]), np.array([])
        
        # 計算採樣概率
        priorities = self.priorities[:n]
        probs = priorities / priorities.sum()
        
        # 採樣索引
        indices = np.random.choice(n, size=min(batch_size, n), p=probs, replace=False)
        
        # 重要性採樣權重
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # 歸一化
        
        samples = [self.memory[i] for i in indices]
        
        return samples, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """更新優先級"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + 0.01
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)
```

---

## 📅 實施計畫

### Phase 1: 緊急修復 (1-2 天)

| 優先級 | 任務 | 檔案 | 預估時間 |
|--------|------|------|----------|
| P0.1 | 實現 MultiEngineCoordinator | `scan/coordinators/multi_engine_coordinator.py` | 4 小時 |
| P0.2 | 修復 AttackCoordinator 依賴 | `task_planning/commander/attack_coordinator.py` | 1 小時 |
| P0.3 | 測試掃描流程 | - | 2 小時 |

### Phase 2: 核心優化 (3-5 天)

| 優先級 | 任務 | 檔案 | 預估時間 |
|--------|------|------|----------|
| P1.1 | 擴展動作映射 | `decision/enhanced_decision_agent.py` | 3 小時 |
| P1.2 | 調整神經網路輸出 | `neural/real_neural_core.py` | 4 小時 |
| P1.3 | 實現動態權重 | `decision/adaptive_weight_manager.py` (新) | 4 小時 |
| P1.4 | RAG 緩存優化 | `rag/rag_engine.py` | 3 小時 |
| P1.5 | 優先級經驗採樣 | `learning_system/experience_manager.py` | 3 小時 |

### Phase 3: 功能擴展 (1-2 週)

| 優先級 | 任務 | 目錄 | 預估時間 |
|--------|------|------|----------|
| P2.1 | 新增 XXE 模組 | `features/features_ready/function_xxe/` | 2 天 |
| P2.2 | 新增 LFI/RFI 模組 | `features/features_ready/function_lfi/` | 2 天 |
| P2.3 | 代碼專用編碼器 | `neural/code_encoder.py` (新) | 3 天 |
| P2.4 | 效能基準測試 | `tests/benchmarks/` | 1 天 |

---

## 📊 預期效益

### 量化指標

| 指標 | 現況 | 目標 | 提升 |
|------|------|------|------|
| 掃描引擎覆蓋 | 0/4 (缺 Coordinator) | 4/4 | +100% |
| 動作類型支援 | 5 種 | 35 種 | +600% |
| RAG 查詢延遲 | ~500ms | ~100ms | -80% |
| 經驗採樣品質 | 均勻 | 優先級 | +30% 訓練效率 |
| 決策適應性 | 固定權重 | 動態權重 | +20% 準確率 |

### 質化改進

1. **AI 決策精度**: 更細粒度的動作空間，更精確的攻擊選擇
2. **多引擎協同**: 真正實現 Rust/Go/Python 引擎的統一調度
3. **經驗學習**: 基於 TD-error 的優先採樣，加速收斂
4. **系統穩定性**: 消除 MultiEngineCoordinator 缺失導致的錯誤

---

## ✅ 檢查清單

### 立即執行
- [ ] 創建 `services/scan/coordinators/multi_engine_coordinator.py`
- [ ] 驗證 Rust 引擎二進制是否存在
- [ ] 測試 `AttackCoordinator.coordinate_multilang()`

### 短期優化
- [ ] 擴展 `ACTION_MAP_EXTENDED`
- [ ] 實現 `AdaptiveWeightManager`
- [ ] 添加 RAG 查詢緩存
- [ ] 實現 `PrioritizedExperienceManager`

### 中期擴展
- [ ] 新增 XXE/LFI/RFI 功能模組
- [ ] 整合 CodeBERT 編碼器
- [ ] 完善效能監控

---

**報告完成**: ✅  
**下一步**: 根據 Phase 1 優先級開始實施
