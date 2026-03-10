# AIVA 雙閉環架構合規性分析報告

**分析日期**: 2025-11-28  
**分析範圍**: 內部閉環 + 外部閉環 aiva_common v2.0 合規性  
**目標**: 評估雙閉環組件對 aiva_common 標準的遵循程度並提供修復方案

---

## 📊 執行摘要

### 當前狀態

**🔍 合規性審查結果**:

| 組件 | 組件完整度 | aiva_common 合規性 | 關鍵問題數 | 狀態 |
|------|----------|-------------------|----------|------|
| **internal_loop_connector.py** | 95% | 30% | 5 | 🔴 不合規 |
| **external_loop_connector.py** | 90% | 30% | 5 | 🔴 不合規 |
| **相關整合組件** | 85% | 40% | 3 | 🟡 部分合規 |

**結論**: 🔴 **雙閉環組件功能完整但未遵循 aiva_common v2.0 標準，需要全面重構以符合規範**

### 關鍵發現

**✅ 已實現的優點**:
- 雙閉環架構設計完整（內部自省 + 外部學習）
- 數據流向清晰（exploration → analysis → RAG → decision）
- 延遲加載模式正確（lazy loading）
- 基礎功能可運行（95% 組件完整度）

**❌ 主要違規項目**:
1. ❌ **日誌系統**: 使用標準 `logging` 而非統一 `aiva_common.utils.logging.get_logger`
2. ❌ **數據驗證**: 無 Pydantic 模型，直接使用 dict
3. ❌ **命令模式**: 未整合 AICommand/AICommandResult 架構
4. ❌ **錯誤處理**: 使用基本 try-except，未使用統一錯誤處理
5. ❌ **類型註解**: 部分缺失或不完整

---

## 🔄 雙閉環架構回顧

### 設計理念（符合 AIVA v2.0）

```
┌──────────────────────────────────────────────────────────┐
│         AIVA 雙閉環自我優化系統 (v2.0 架構)                 │
└──────────────────────────────────────────────────────────┘

  內部閉環 (Know Thyself)          外部閉環 (Learn from Battle)
  ═══════════════════════          ════════════════════════════
  
  SystemSelfExplorer               Task Execution
       ↓                                 ↓
  CapabilityAnalyzer               Result Analysis
       ↓                                 ↓
  InternalLoopConnector            ExternalLoopConnector
       ↓                                 ↓
  RAG Knowledge Base               Deviation Analyzer
       ↓                                 ↓
  Self-Awareness Query             Model Trainer
       ↓                                 ↓
  ═══════════════════════════════════════════════════════════
                    ↓
         AI Decision Center (統一指揮)
                    ↓
         AICommand/AICommandResult 架構
                    ↓
         優化方案生成與執行
```

### 應有的數據流（v2.0 標準）

**內部閉環應有流程**:
```python
# 應該使用 AICommand 架構
from aiva_common.ai import AICommand, AICommandResult
from aiva_common.utils.logging import get_logger
from aiva_common.schemas.capability import CapabilityInfo

command = AICommand(
    command_type="sync_capabilities",
    parameters={"force_refresh": False}
)

result: AICommandResult = await internal_loop_connector.execute(command)
# result.data 包含 Pydantic 驗證的 CapabilityInfo 列表
```

**外部閉環應有流程**:
```python
# 應該使用 AICommand 架構
from aiva_common.ai import AICommand, AICommandResult
from aiva_common.schemas.ai import ExperienceSample

command = AICommand(
    command_type="process_execution_result",
    parameters={
        "plan": validated_plan,  # Pydantic 模型
        "trace": validated_trace  # Pydantic 模型
    }
)

result: AICommandResult = await external_loop_connector.execute(command)
# result.data 包含 Pydantic 驗證的訓練結果
```

---

## 🚨 違規項目詳細分析

### 1. internal_loop_connector.py 違規清單

#### ❌ 違規 1: 使用標準 logging
**位置**: 行 14-17
```python
# 當前錯誤代碼
import logging
logger = logging.getLogger(__name__)

# 應修正為
from aiva_common.utils.logging import get_logger
logger = get_logger(__name__)
```

**影響**: 
- 日誌格式不統一
- 無法集中配置日誌級別
- 不符合 AIVA Common 統一日誌規範

---

#### ❌ 違規 2: 缺少 Pydantic 模型
**位置**: 行 60-118（sync_capabilities_to_rag 方法）
```python
# 當前錯誤: 直接返回 dict
async def sync_capabilities_to_rag(self, force_refresh: bool = False) -> dict[str, Any]:
    # ...
    return {
        "modules_scanned": len(modules),
        "capabilities_found": len(capabilities),
        # ... 無類型驗證的 dict
    }

# 應修正為使用 Pydantic 模型
from pydantic import BaseModel
from aiva_common.schemas.capability import CapabilityInfo

class InternalLoopSyncResult(BaseModel):
    """內部閉環同步結果"""
    modules_scanned: int
    capabilities_found: int
    capabilities: list[CapabilityInfo]  # 使用 Pydantic 模型
    documents_added: int
    timestamp: datetime
    success: bool
    error: str | None = None

async def sync_capabilities_to_rag(
    self, 
    force_refresh: bool = False
) -> InternalLoopSyncResult:
    # ... 返回經過驗證的 Pydantic 模型
    return InternalLoopSyncResult(
        modules_scanned=len(modules),
        capabilities_found=len(capabilities),
        capabilities=[CapabilityInfo(**cap) for cap in capabilities],
        # ...
    )
```

**影響**:
- 無法在編譯時捕獲類型錯誤
- 數據驗證缺失，可能傳遞無效數據
- 不符合 aiva_common v2.0 "All data validated" 要求

---

#### ❌ 違規 3: 未整合 AICommand 架構
**位置**: 整個類（無 execute 方法）
```python
# 當前錯誤: 直接調用方法
result = await connector.sync_capabilities_to_rag(force_refresh=True)

# 應修正為 AICommand 架構
from aiva_common.ai import AICommand, AICommandResult

class InternalLoopConnector:
    async def execute(self, command: AICommand) -> AICommandResult:
        """統一命令執行入口"""
        if command.command_type == "sync_capabilities":
            result = await self.sync_capabilities_to_rag(
                force_refresh=command.parameters.get("force_refresh", False)
            )
            return AICommandResult(
                command_id=command.command_id,
                success=result.success,
                data=result.model_dump(),
                error=result.error
            )
        else:
            raise ValueError(f"Unknown command: {command.command_type}")

# 新的調用方式
command = AICommand(
    command_type="sync_capabilities",
    parameters={"force_refresh": True}
)
result = await connector.execute(command)
```

**影響**:
- 無法統一管理 AI 組件調用
- 不符合 v2.0 "AI直接指揮架構"
- 無法利用統一的錯誤處理和監控

---

#### ❌ 違規 4: 基本錯誤處理
**位置**: 行 130-140
```python
# 當前錯誤: 基本 try-except
except Exception as e:
    logger.error(f"❌ Internal loop sync failed: {e}", exc_info=True)
    return {
        "success": False,
        "error": str(e)
    }

# 應修正為統一錯誤處理
from aiva_common.error_handling import (
    AIVAError, 
    ErrorType, 
    ErrorSeverity,
    create_error_context
)

except Exception as e:
    error_context = create_error_context(
        error_type=ErrorType.AI_PROCESSING,
        severity=ErrorSeverity.HIGH,
        message="Internal loop sync failed",
        details={"force_refresh": force_refresh},
        exception=e
    )
    logger.error(f"❌ Internal loop sync failed: {error_context}")
    
    raise AIVAError(
        message="Internal loop synchronization failed",
        error_type=ErrorType.AI_PROCESSING,
        severity=ErrorSeverity.HIGH,
        context=error_context
    )
```

**影響**:
- 錯誤信息不完整
- 無法統一監控和告警
- 不符合 aiva_common 統一錯誤處理規範

---

#### ❌ 違規 5: 缺少完整類型註解
**位置**: 多處
```python
# 當前錯誤: 使用 Any
def _convert_to_documents(self, capabilities: list[dict]) -> list[dict]:
    # ... dict 沒有類型安全

# 應修正為完整類型
from aiva_common.schemas.capability import CapabilityInfo
from aiva_common.schemas.ai import RAGDocument

def _convert_to_documents(
    self, 
    capabilities: list[CapabilityInfo]
) -> list[RAGDocument]:
    # ... 完整類型註解
```

**影響**:
- IDE 無法提供完整的類型提示
- 無法利用 mypy/pyright 進行靜態檢查
- 不符合 aiva_common "Full annotation" 要求

---

### 2. external_loop_connector.py 違規清單

#### ❌ 違規 1: 使用標準 logging
**位置**: 行 14-17
```python
# 當前錯誤代碼（與 internal_loop_connector 相同）
import logging
logger = logging.getLogger(__name__)

# 應修正為
from aiva_common.utils.logging import get_logger
logger = get_logger(__name__)
```

---

#### ❌ 違規 2: 缺少 Pydantic 模型
**位置**: 行 66-136（process_execution_result 方法）
```python
# 當前錯誤: 直接使用 dict
async def process_execution_result(
    self,
    plan: dict[str, Any],  # ❌ 應該是 Pydantic 模型
    trace: list[dict[str, Any]]  # ❌ 應該是 Pydantic 模型
) -> dict[str, Any]:  # ❌ 應該返回 Pydantic 模型
    # ...
    return {
        "deviations_found": len(deviations),
        # ... 無類型驗證
    }

# 應修正為
from pydantic import BaseModel
from aiva_common.schemas.ai import ExecutionPlan, ExecutionTrace

class DeviationRecord(BaseModel):
    """偏差記錄"""
    type: Literal["incomplete_execution", "execution_failures", "slow_execution"]
    severity: Literal["high", "medium", "low"]
    score: float
    details: dict[str, Any]

class ExternalLoopProcessResult(BaseModel):
    """外部閉環處理結果"""
    deviations_found: int
    deviations_significant: bool
    deviations: list[DeviationRecord]
    training_triggered: bool
    weights_updated: bool
    new_weights_version: str | None
    timestamp: datetime
    success: bool
    error: str | None = None

async def process_execution_result(
    self,
    plan: ExecutionPlan,  # ✅ Pydantic 模型
    trace: list[ExecutionTrace]  # ✅ Pydantic 模型
) -> ExternalLoopProcessResult:  # ✅ 返回 Pydantic 模型
    # ...
    return ExternalLoopProcessResult(
        deviations_found=len(deviations),
        deviations=deviations,
        # ... 完整驗證
    )
```

---

#### ❌ 違規 3: 未整合 AICommand 架構
**位置**: 整個類（與 internal_loop_connector 相同問題）
```python
# 應添加統一執行入口
async def execute(self, command: AICommand) -> AICommandResult:
    """統一命令執行入口"""
    if command.command_type == "process_execution_result":
        plan = ExecutionPlan(**command.parameters["plan"])
        trace = [ExecutionTrace(**t) for t in command.parameters["trace"]]
        result = await self.process_execution_result(plan, trace)
        return AICommandResult(
            command_id=command.command_id,
            success=result.success,
            data=result.model_dump(),
            error=result.error
        )
    else:
        raise ValueError(f"Unknown command: {command.command_type}")
```

---

#### ❌ 違規 4: 基本錯誤處理
**位置**: 行 133-147（與 internal_loop_connector 相同問題）
```python
# 當前錯誤: 基本 try-except
except Exception as e:
    logger.error(f"❌ External loop processing failed: {e}", exc_info=True)
    return {
        "success": False,
        "error": str(e)
    }

# 應使用統一錯誤處理（同 internal_loop_connector 修復方案）
```

---

#### ❌ 違規 5: 缺少完整類型註解
**位置**: 行 149-189（_analyze_deviations 方法）
```python
# 當前錯誤
def _analyze_deviations(
    self,
    plan: dict[str, Any],  # ❌
    trace: list[dict[str, Any]]  # ❌
) -> list[dict[str, Any]]:  # ❌
    # ...

# 應修正為
def _analyze_deviations(
    self,
    plan: ExecutionPlan,  # ✅
    trace: list[ExecutionTrace]  # ✅
) -> list[DeviationRecord]:  # ✅
    # ...
```

---

## 🔧 修復方案

### 階段 1: 建立 Pydantic 模型（優先級: P0）

**新文件**: `services/aiva_common/schemas/dual_loop.py`

```python
"""
雙閉環架構專用 Schema

定義內部閉環和外部閉環使用的 Pydantic 模型
"""

from datetime import datetime, UTC
from typing import Any, Literal
from pydantic import BaseModel, Field

# ==================== 內部閉環 Schema ====================

class ModuleCapability(BaseModel):
    """模組能力記錄"""
    module: str = Field(..., description="模組路徑")
    name: str = Field(..., description="能力名稱")
    parameters: list[dict[str, Any]] = Field(..., description="參數列表")
    return_type: str | None = Field(None, description="返回類型")
    description: str | None = Field(None, description="描述")
    complexity: int = Field(..., description="複雜度評分", ge=1, le=5)
    health_score: float = Field(..., description="健康分數", ge=0, le=1)

class InternalLoopSyncResult(BaseModel):
    """內部閉環同步結果"""
    modules_scanned: int = Field(..., description="掃描的模組數", ge=0)
    capabilities_found: int = Field(..., description="發現的能力數", ge=0)
    capabilities: list[ModuleCapability] = Field(..., description="能力列表")
    documents_added: int = Field(..., description="添加的文檔數", ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    success: bool = Field(..., description="是否成功")
    error: str | None = Field(None, description="錯誤信息")

# ==================== 外部閉環 Schema ====================

class ExecutionPlan(BaseModel):
    """執行計劃 (AST)"""
    plan_id: str = Field(..., description="計劃ID")
    steps: list[dict[str, Any]] = Field(..., description="計劃步驟")
    expected_duration: float | None = Field(None, description="預期耗時(秒)")
    metadata: dict[str, Any] | None = Field(None, description="元數據")

class ExecutionTrace(BaseModel):
    """執行軌跡記錄"""
    step_id: str = Field(..., description="步驟ID")
    status: Literal["success", "failed", "skipped"] = Field(..., description="狀態")
    duration: float = Field(..., description="耗時(秒)", ge=0)
    output: Any | None = Field(None, description="輸出")
    error: str | None = Field(None, description="錯誤信息")

class DeviationRecord(BaseModel):
    """偏差記錄"""
    type: Literal[
        "incomplete_execution",
        "execution_failures",
        "slow_execution",
        "unexpected_output"
    ] = Field(..., description="偏差類型")
    severity: Literal["high", "medium", "low"] = Field(..., description="嚴重程度")
    score: float = Field(..., description="偏差分數", ge=0)
    details: dict[str, Any] = Field(..., description="詳細信息")

class ExternalLoopProcessResult(BaseModel):
    """外部閉環處理結果"""
    deviations_found: int = Field(..., description="發現的偏差數", ge=0)
    deviations_significant: bool = Field(..., description="是否顯著")
    deviations: list[DeviationRecord] = Field(..., description="偏差列表")
    training_triggered: bool = Field(..., description="是否觸發訓練")
    weights_updated: bool = Field(..., description="權重是否更新")
    new_weights_version: str | None = Field(None, description="新權重版本")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    success: bool = Field(..., description="是否成功")
    error: str | None = Field(None, description="錯誤信息")

# ==================== AICommand 整合 ====================

class DualLoopCommand(BaseModel):
    """雙閉環專用命令"""
    command_type: Literal[
        "sync_capabilities",
        "process_execution_result",
        "query_capability",
        "analyze_deviation"
    ] = Field(..., description="命令類型")
    parameters: dict[str, Any] = Field(..., description="命令參數")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
```

---

### 階段 2: 修復 internal_loop_connector.py（優先級: P0）

**完整修復代碼**:

```python
"""Internal Loop Connector - 內部閉環連接器 (v2.0 合規版本)

將系統能力探索結果同步到 RAG 知識庫，實現 AI 的自我認知

數據流：
internal_exploration (能力掃描) → InternalLoopConnector → RAG (知識注入) → AI Self-Awareness

遵循 aiva_common v2.0 規範:
✅ 使用統一的日誌記錄 (get_logger)
✅ 使用 Pydantic 模型進行數據驗證
✅ 整合 AICommand/AICommandResult 架構
✅ 使用統一的錯誤處理
✅ 完整的類型註解
"""

from datetime import datetime, UTC
from typing import Any
from uuid import uuid4

# ✅ 修復 1: 使用統一日誌
from aiva_common.utils.logging import get_logger

# ✅ 修復 2: 引入 Pydantic 模型
from aiva_common.schemas.dual_loop import (
    ModuleCapability,
    InternalLoopSyncResult,
    DualLoopCommand
)

# ✅ 修復 3: 引入 AICommand 架構
from aiva_common.ai import AICommand, AICommandResult

# ✅ 修復 4: 使用統一錯誤處理
from aiva_common.error_handling import (
    AIVAError,
    ErrorType,
    ErrorSeverity,
    create_error_context
)

logger = get_logger(__name__)


class InternalLoopConnector:
    """內部閉環連接器 (v2.0 合規版本)
    
    職責：
    1. 接收系統能力掃描結果（ModuleCapability 列表）
    2. 轉換為 RAG 知識文檔格式
    3. 注入到 RAG 知識庫
    4. 提供自我認知查詢能力
    
    這是 AI 自我優化雙重閉環中「對內認知閉環」的關鍵組件
    """
    
    def __init__(self):
        """初始化內部閉環連接器"""
        self._module_explorer = None
        self._capability_analyzer = None
        self._rag_engine = None
        
        logger.info("InternalLoopConnector initialized (v2.0 compliant)")
    
    @property
    def rag_engine(self):
        """延遲加載 RAG Engine"""
        if self._rag_engine is None:
            from ..rag_engine.rag import RAGEngine
            self._rag_engine = RAGEngine()
        return self._rag_engine
    
    @property
    def module_explorer(self):
        """延遲加載 ModuleExplorer"""
        if self._module_explorer is None:
            from ..internal_exploration.module_explorer import ModuleExplorer
            self._module_explorer = ModuleExplorer()
        return self._module_explorer
    
    @property
    def capability_analyzer(self):
        """延遲加載 CapabilityAnalyzer"""
        if self._capability_analyzer is None:
            from ..internal_exploration.capability_analyzer import CapabilityAnalyzer
            self._capability_analyzer = CapabilityAnalyzer()
        return self._capability_analyzer
    
    # ✅ 修復 3: 添加 AICommand 統一執行入口
    async def execute(self, command: AICommand) -> AICommandResult:
        """統一命令執行入口（AICommand 架構）
        
        Args:
            command: AI 命令對象
            
        Returns:
            命令執行結果
        """
        try:
            if command.command_type == "sync_capabilities":
                force_refresh = command.parameters.get("force_refresh", False)
                result = await self.sync_capabilities_to_rag(force_refresh)
                
                return AICommandResult(
                    command_id=command.command_id,
                    success=result.success,
                    data=result.model_dump(),
                    error=result.error
                )
            
            elif command.command_type == "query_capability":
                query = command.parameters.get("query", "")
                results = await self._query_from_rag(query)
                
                return AICommandResult(
                    command_id=command.command_id,
                    success=True,
                    data={"results": results}
                )
            
            else:
                raise ValueError(f"Unknown command type: {command.command_type}")
                
        except Exception as e:
            error_context = create_error_context(
                error_type=ErrorType.AI_PROCESSING,
                severity=ErrorSeverity.HIGH,
                message=f"Internal loop command failed: {command.command_type}",
                details={"command": command.model_dump()},
                exception=e
            )
            logger.error(f"❌ Command execution failed: {error_context}")
            
            return AICommandResult(
                command_id=command.command_id,
                success=False,
                error=str(e)
            )
    
    # ✅ 修復 2: 使用 Pydantic 返回類型
    async def sync_capabilities_to_rag(
        self, 
        force_refresh: bool = False
    ) -> InternalLoopSyncResult:
        """同步能力到 RAG 知識庫 (v2.0 合規版本)
        
        Args:
            force_refresh: 是否強制刷新（清空舊數據）
            
        Returns:
            同步結果（Pydantic 模型）
        """
        logger.info("🔄 Starting internal loop synchronization...")
        
        try:
            # 步驟 1: 掃描模組
            logger.info("  Step 1: Scanning modules...")
            modules = await self.module_explorer.explore_all_modules()
            
            # 步驟 2: 分析能力
            logger.info("  Step 2: Analyzing capabilities...")
            capabilities_raw = await self.capability_analyzer.analyze_capabilities(modules)
            
            # ✅ 修復 2: 轉換為 Pydantic 模型
            capabilities = [
                ModuleCapability(**cap) 
                for cap in capabilities_raw
            ]
            
            # 步驟 3: 轉換為文檔
            logger.info("  Step 3: Converting to documents...")
            documents = self._convert_to_documents(capabilities)
            
            # 步驟 4: 注入 RAG
            logger.info("  Step 4: Injecting to RAG...")
            documents_added = await self._inject_to_rag(documents, force_refresh)
            
            # ✅ 修復 2: 返回 Pydantic 模型
            result = InternalLoopSyncResult(
                modules_scanned=len(modules),
                capabilities_found=len(capabilities),
                capabilities=capabilities,
                documents_added=documents_added,
                timestamp=datetime.now(UTC),
                success=True
            )
            
            logger.info(f"✅ Internal loop sync completed: {result.model_dump()}")
            return result
            
        except Exception as e:
            # ✅ 修復 4: 使用統一錯誤處理
            error_context = create_error_context(
                error_type=ErrorType.AI_PROCESSING,
                severity=ErrorSeverity.HIGH,
                message="Internal loop sync failed",
                details={"force_refresh": force_refresh},
                exception=e
            )
            logger.error(f"❌ Internal loop sync failed: {error_context}")
            
            # ✅ 修復 2: 返回 Pydantic 模型（錯誤情況）
            return InternalLoopSyncResult(
                modules_scanned=0,
                capabilities_found=0,
                capabilities=[],
                documents_added=0,
                timestamp=datetime.now(UTC),
                success=False,
                error=str(e)
            )
    
    # ✅ 修復 5: 完整類型註解
    def _convert_to_documents(
        self, 
        capabilities: list[ModuleCapability]
    ) -> list[dict[str, Any]]:
        """將能力轉換為 RAG 文檔格式
        
        Args:
            capabilities: 能力列表（Pydantic 模型）
            
        Returns:
            RAG 文檔列表
        """
        documents = []
        
        for cap in capabilities:
            # 構建可讀的能力描述
            params_str = ", ".join(
                f"{p['name']}: {p.get('annotation', 'Any')}" 
                for p in cap.parameters
            )
            
            content_parts = [
                f"# Capability: {cap.name}",
                f"\nModule: {cap.module}",
                f"Function: {cap.name}({params_str})",
            ]
            
            if cap.return_type:
                content_parts.append(f"Returns: {cap.return_type}")
            
            if cap.description:
                content_parts.append(f"\nDescription: {cap.description}")
            
            content_parts.extend([
                f"\nComplexity: {cap.complexity}/5",
                f"Health Score: {cap.health_score:.2f}"
            ])
            
            documents.append({
                "id": f"cap-{cap.module}-{cap.name}",
                "content": "\n".join(content_parts),
                "metadata": {
                    "type": "capability",
                    "module": cap.module,
                    "function": cap.name,
                    "complexity": cap.complexity,
                    "health_score": cap.health_score
                }
            })
        
        return documents
    
    async def _inject_to_rag(
        self, 
        documents: list[dict[str, Any]], 
        force_refresh: bool = False
    ) -> int:
        """注入文檔到 RAG
        
        Args:
            documents: 文檔列表
            force_refresh: 是否強制刷新
            
        Returns:
            添加的文檔數
        """
        if force_refresh:
            await self.rag_engine.clear_collection("capabilities")
        
        await self.rag_engine.add_documents(
            collection="capabilities",
            documents=documents
        )
        
        return len(documents)
    
    async def _query_from_rag(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """從 RAG 查詢能力
        
        Args:
            query: 查詢字符串
            top_k: 返回結果數
            
        Returns:
            查詢結果列表
        """
        results = await self.rag_engine.query(
            collection="capabilities",
            query=query,
            top_k=top_k
        )
        return results
```

---

### 階段 3: 修復 external_loop_connector.py（優先級: P0）

**修復要點**（與 internal_loop_connector 類似）:
1. ✅ 替換 `import logging` 為 `from aiva_common.utils.logging import get_logger`
2. ✅ 添加 Pydantic 模型：`ExecutionPlan`, `ExecutionTrace`, `DeviationRecord`, `ExternalLoopProcessResult`
3. ✅ 添加 `async def execute(self, command: AICommand) -> AICommandResult` 方法
4. ✅ 使用統一錯誤處理 `create_error_context` 和 `AIVAError`
5. ✅ 完整類型註解

**完整修復代碼**（由於長度限制，僅列出核心結構）:

```python
"""External Loop Connector - 外部閉環連接器 (v2.0 合規版本)"""

from aiva_common.utils.logging import get_logger
from aiva_common.schemas.dual_loop import (
    ExecutionPlan,
    ExecutionTrace,
    DeviationRecord,
    ExternalLoopProcessResult
)
from aiva_common.ai import AICommand, AICommandResult
from aiva_common.error_handling import (
    AIVAError,
    ErrorType,
    ErrorSeverity,
    create_error_context
)

logger = get_logger(__name__)

class ExternalLoopConnector:
    """外部閉環連接器 (v2.0 合規版本)"""
    
    async def execute(self, command: AICommand) -> AICommandResult:
        """統一命令執行入口"""
        # ... 實現 AICommand 架構
    
    async def process_execution_result(
        self,
        plan: ExecutionPlan,  # ✅ Pydantic 模型
        trace: list[ExecutionTrace]  # ✅ Pydantic 模型
    ) -> ExternalLoopProcessResult:  # ✅ Pydantic 返回
        """處理執行結果並觸發學習循環 (v2.0 合規版本)"""
        # ... 使用 Pydantic 模型和統一錯誤處理
    
    def _analyze_deviations(
        self,
        plan: ExecutionPlan,  # ✅ 完整類型
        trace: list[ExecutionTrace]  # ✅ 完整類型
    ) -> list[DeviationRecord]:  # ✅ 完整類型
        """分析執行偏差 (v2.0 合規版本)"""
        # ... 返回 Pydantic 模型列表
```

---

### 階段 4: 更新整合組件（優先級: P1）

**需要更新的文件**:
1. `services/integration/coordinators/*_coordinator.py`
   - 更新為使用 `AICommand` 調用雙閉環連接器
   - 使用 Pydantic 模型處理數據

2. `services/core/aiva_core/cognitive_core/ai_capability_query.py`
   - 更新為使用 `AICommand` 架構
   - 返回 Pydantic 模型

3. `aiva_cli.py`
   - 更新 CLI 命令以使用新的 Pydantic 模型
   - 添加類型提示

---

## ✅ 驗證計劃

### 單元測試（優先級: P0）

**新文件**: `tests/test_dual_loop_compliance.py`

```python
"""雙閉環合規性測試"""

import pytest
from aiva_common.ai import AICommand
from aiva_common.schemas.dual_loop import (
    InternalLoopSyncResult,
    ExternalLoopProcessResult,
    ModuleCapability,
    ExecutionPlan,
    ExecutionTrace
)

class TestInternalLoopCompliance:
    """內部閉環合規性測試"""
    
    @pytest.mark.asyncio
    async def test_sync_capabilities_returns_pydantic(self):
        """測試 sync_capabilities_to_rag 返回 Pydantic 模型"""
        from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
        
        connector = InternalLoopConnector()
        result = await connector.sync_capabilities_to_rag(force_refresh=False)
        
        # ✅ 驗證返回 Pydantic 模型
        assert isinstance(result, InternalLoopSyncResult)
        assert hasattr(result, 'modules_scanned')
        assert hasattr(result, 'capabilities')
        assert all(isinstance(cap, ModuleCapability) for cap in result.capabilities)
    
    @pytest.mark.asyncio
    async def test_execute_command_pattern(self):
        """測試 AICommand 架構"""
        from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
        
        connector = InternalLoopConnector()
        command = AICommand(
            command_type="sync_capabilities",
            parameters={"force_refresh": True}
        )
        
        result = await connector.execute(command)
        
        # ✅ 驗證返回 AICommandResult
        assert result.success is True or result.success is False
        assert result.command_id == command.command_id

class TestExternalLoopCompliance:
    """外部閉環合規性測試"""
    
    @pytest.mark.asyncio
    async def test_process_execution_accepts_pydantic(self):
        """測試 process_execution_result 接受 Pydantic 模型"""
        from services.core.aiva_core.cognitive_core.external_loop_connector import ExternalLoopConnector
        
        connector = ExternalLoopConnector()
        
        # ✅ 使用 Pydantic 模型
        plan = ExecutionPlan(
            plan_id="test-plan",
            steps=[{"action": "scan", "target": "localhost"}],
            expected_duration=10.0
        )
        
        trace = [
            ExecutionTrace(
                step_id="step-1",
                status="success",
                duration=5.0,
                output="Scan completed"
            )
        ]
        
        result = await connector.process_execution_result(plan, trace)
        
        # ✅ 驗證返回 Pydantic 模型
        assert isinstance(result, ExternalLoopProcessResult)
        assert result.deviations_found >= 0
```

---

### 整合測試（優先級: P1）

**新文件**: `tests/test_dual_loop_integration.py`

```python
"""雙閉環端到端整合測試"""

import pytest
from aiva_common.ai import AICommand

class TestDualLoopIntegration:
    """雙閉環端到端測試"""
    
    @pytest.mark.asyncio
    async def test_full_internal_loop_cycle(self):
        """測試完整內部閉環循環"""
        from services.core.aiva_core.cognitive_core.internal_loop_connector import InternalLoopConnector
        
        connector = InternalLoopConnector()
        
        # 步驟 1: 同步能力
        command = AICommand(
            command_type="sync_capabilities",
            parameters={"force_refresh": True}
        )
        sync_result = await connector.execute(command)
        assert sync_result.success is True
        
        # 步驟 2: 查詢能力
        query_command = AICommand(
            command_type="query_capability",
            parameters={"query": "scan vulnerability"}
        )
        query_result = await connector.execute(query_command)
        assert query_result.success is True
        assert len(query_result.data["results"]) > 0
    
    @pytest.mark.asyncio
    async def test_full_external_loop_cycle(self):
        """測試完整外部閉環循環"""
        from services.core.aiva_core.cognitive_core.external_loop_connector import ExternalLoopConnector
        from aiva_common.schemas.dual_loop import ExecutionPlan, ExecutionTrace
        
        connector = ExternalLoopConnector()
        
        # 構造執行結果
        plan = ExecutionPlan(
            plan_id="attack-plan-1",
            steps=[
                {"action": "scan", "target": "example.com"},
                {"action": "exploit", "vulnerability": "XSS"}
            ],
            expected_duration=30.0
        )
        
        trace = [
            ExecutionTrace(
                step_id="step-1",
                status="success",
                duration=10.0,
                output="Scan found 3 issues"
            ),
            ExecutionTrace(
                step_id="step-2",
                status="failed",
                duration=5.0,
                error="Exploit blocked by WAF"
            )
        ]
        
        # 執行外部閉環
        command = AICommand(
            command_type="process_execution_result",
            parameters={
                "plan": plan.model_dump(),
                "trace": [t.model_dump() for t in trace]
            }
        )
        
        result = await connector.execute(command)
        assert result.success is True
        
        # 驗證偏差分析
        process_result = ExternalLoopProcessResult(**result.data)
        assert process_result.deviations_found > 0
        assert process_result.deviations_significant is True
```

---

## 📊 修復進度追蹤

### 完成標準

| 任務 | 文件 | 狀態 | 完成度 |
|-----|------|------|--------|
| **建立 Pydantic 模型** | services/aiva_common/schemas/dual_loop.py | 🔴 待完成 | 0% |
| **修復內部閉環連接器** | services/core/aiva_core/cognitive_core/internal_loop_connector.py | 🔴 待完成 | 0% |
| **修復外部閉環連接器** | services/core/aiva_core/cognitive_core/external_loop_connector.py | 🔴 待完成 | 0% |
| **單元測試** | tests/test_dual_loop_compliance.py | 🔴 待完成 | 0% |
| **整合測試** | tests/test_dual_loop_integration.py | 🔴 待完成 | 0% |
| **更新文檔** | README.md, AI_SELF_OPTIMIZATION_DUAL_LOOP_DESIGN.md | 🔴 待完成 | 0% |

---

## 🎯 預期效果

### 修復前 vs 修復後

| 項目 | 修復前 | 修復後 |
|-----|-------|--------|
| **日誌系統** | ❌ 標準 logging | ✅ 統一 get_logger |
| **數據驗證** | ❌ 無驗證 dict | ✅ Pydantic 完整驗證 |
| **命令架構** | ❌ 直接方法調用 | ✅ AICommand 統一入口 |
| **錯誤處理** | ❌ 基本 try-except | ✅ 統一錯誤處理 |
| **類型安全** | ❌ 部分註解 | ✅ 完整類型註解 |
| **合規性評分** | 🔴 30% | 🟢 100% |

### 架構優勢

修復後將獲得以下優勢：

1. **類型安全**: 
   - IDE 提供完整類型提示
   - mypy/pyright 靜態檢查
   - 運行時數據驗證

2. **統一管理**:
   - 所有 AI 組件通過 AICommand 調用
   - 集中式錯誤處理和監控
   - 統一日誌格式

3. **可擴展性**:
   - 新增命令只需添加 command_type
   - Pydantic 模型自動生成 JSON Schema
   - 支持 OpenAPI 文檔生成

4. **可測試性**:
   - Pydantic 模型易於 mock
   - AICommand 支持錄制/回放
   - 端到端測試覆蓋完整

---

## 📝 實施計劃

### Week 1: 基礎架構

- [ ] Day 1-2: 建立 `dual_loop.py` Pydantic 模型
- [ ] Day 3-4: 修復 `internal_loop_connector.py`
- [ ] Day 5-6: 修復 `external_loop_connector.py`
- [ ] Day 7: 編寫單元測試

### Week 2: 整合與測試

- [ ] Day 1-2: 更新整合組件（coordinators）
- [ ] Day 3-4: 編寫整合測試
- [ ] Day 5-6: 端到端測試驗證
- [ ] Day 7: 文檔更新和評審

### Week 3: 優化與部署

- [ ] Day 1-2: 性能優化
- [ ] Day 3-4: 安全審查
- [ ] Day 5-6: 生產環境部署
- [ ] Day 7: 監控和調優

---

## 🔗 相關文檔

- [aiva_common README](../services/aiva_common/README.md) - v2.0 標準參考
- [DUAL_LOOP_FEASIBILITY_ANALYSIS](DUAL_LOOP_FEASIBILITY_ANALYSIS.md) - 可行性分析
- [AI_INTEGRATION_COMPLETION_REPORT](../AI_INTEGRATION_COMPLETION_REPORT.md) - 整合報告

---

**報告生成時間**: 2025-11-28 12:45:00  
**報告狀態**: 完整分析完成，待實施修復  
**下一步行動**: 開始建立 dual_loop.py Pydantic 模型
