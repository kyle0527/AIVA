"""
Flow Executor Adapter - CLIDecisionEngine 與 FlowExecutor 的橋接層

將 CLIDecisionEngine 選出的攻擊流程轉換為 FlowExecutor 可執行的格式。
這是 RAG 決策層與實際執行層的連接橋樑。

使用流程:
    1. CLIDecisionEngine 根據掃描結果推薦 AttackFlow
    2. FlowExecutorAdapter 將 AttackFlow 轉換為執行參數
    3. 調用 external FlowExecutor (aiva_external_executor.py) 執行攻擊
    4. 返回執行結果給 AI Commander

Architecture:
    AI Commander (掃描結果)
         ↓
    CLIDecisionEngine (推薦攻擊流程)
         ↓
    FlowExecutorAdapter (參數轉換) ← 本模組
         ↓
    External FlowExecutor (執行攻擊)
         ↓
    返回結果
"""

from dataclasses import dataclass
from typing import Any

from aiva_common.utils import get_logger

from .cli_decision_engine import AttackCapability, AttackFlow, CLIDecisionEngine

logger = get_logger(__name__)

@dataclass
class ExecutionConfig:
    """執行配置"""
    timeout: int = 30
    max_retries: int = 3
    verbose: bool = False

@dataclass
class ExecutionResult:
    """執行結果"""
    success: bool
    flow_id: int
    module: str
    output: Any
    error: str | None = None
    execution_time: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "flow_id": self.flow_id,
            "module": self.module,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
        }

class FlowExecutorAdapter:
    """Flow Executor 適配器
    
    連接 CLIDecisionEngine 和實際的攻擊執行器。
    負責：
    1. 參數轉換（AttackFlow → FlowExecutor 參數）
    2. 執行協調（調用正確的執行器）
    3. 結果處理（標準化輸出格式）
    """
    
    def __init__(
        self,
        decision_engine: CLIDecisionEngine | None = None,
        external_executor: Any | None = None
    ):
        """初始化適配器
        
        Args:
            decision_engine: CLI 決策引擎（如未提供則創建新實例）
            external_executor: 外部執行器（aiva_external_executor.FlowExecutor）
        """
        self.decision_engine = decision_engine or CLIDecisionEngine()
        self.external_executor = external_executor
        
        # 如果沒有提供執行器，嘗試導入
        if self.external_executor is None:
            self._initialize_external_executor()
        
        logger.info("✅ FlowExecutorAdapter 已初始化")
    
    def _initialize_external_executor(self) -> None:
        """初始化外部執行器"""
        try:
            # 導入 external exploration 的 FlowExecutor
            # 路徑: services/core/aiva_core/internal_exploration/aiva_external_executor.py
            from services.core.aiva_core.internal_exploration.aiva_external_executor import (
                MultiLangExecutor,
            )
            
            self.external_executor = MultiLangExecutor()
            logger.info("✅ 已載入 External FlowExecutor")
        except ImportError as e:
            logger.warning(f"⚠️ 無法載入 External FlowExecutor: {e}")
            logger.info("將在執行時動態載入")
    
    def execute_recommended_flows(
        self,
        scan_context: dict[str, Any],
        top_k: int = 3,
        config: ExecutionConfig | None = None
    ) -> list[ExecutionResult]:
        """根據掃描上下文推薦並執行攻擊流程
        
        這是主要的對外接口，完整流程：
        1. 使用 CLIDecisionEngine 推薦攻擊流程
        2. 轉換參數
        3. 執行攻擊
        4. 返回結果
        
        Args:
            scan_context: 掃描上下文
                {
                    "vulnerability_type": "XSS",
                    "target_info": {
                        "url": "https://example.com/search",
                        "parameter": "q",
                        "method": "GET"
                    },
                    "scan_results": {...}
                }
            top_k: 推薦前 K 個流程
            config: 執行配置
        
        Returns:
            執行結果列表
        """
        config = config or ExecutionConfig()
        
        # 1. 使用決策引擎推薦流程
        recommended_flows = self.decision_engine.recommend_flows(
            scan_context=scan_context,
            top_k=top_k
        )
        
        if not recommended_flows:
            logger.warning("⚠️ 沒有找到合適的攻擊流程")
            return []
        
        logger.info(f"📋 推薦了 {len(recommended_flows)} 個攻擊流程")
        
        # 2. 執行每個推薦的流程
        results = []
        for flow in recommended_flows:
            result = self.execute_flow(
                flow=flow,
                target_context=scan_context.get('target_info', {}),
                config=config
            )
            results.append(result)
        
        return results
    
    def execute_flow(
        self,
        flow: AttackFlow,
        target_context: dict[str, Any],
        config: ExecutionConfig | None = None
    ) -> ExecutionResult:
        """執行單個攻擊流程
        
        Args:
            flow: 要執行的攻擊流程
            target_context: 目標上下文（URL、參數等）
            config: 執行配置
        
        Returns:
            執行結果
        """
        config = config or ExecutionConfig()
        
        logger.info(f"🚀 執行 Flow {flow.flow_id}: {flow.entry_point}")
        
        # 檢查執行器
        if self.external_executor is None:
            logger.error("❌ External FlowExecutor 未初始化")
            return ExecutionResult(
                success=False,
                flow_id=flow.flow_id,
                module=flow.module,
                output=None,
                error="External FlowExecutor 未初始化"
            )
        
        # 轉換參數並執行
        try:
            # 構建執行參數
            execution_params = self._build_execution_params(flow, target_context)
            
            # 調用 External FlowExecutor
            # 注意：這裡需要根據實際的 FlowExecutor API 調整
            output = self.external_executor.execute_flow(
                flow_id=flow.flow_id,
                params=execution_params,
                timeout=config.timeout
            )
            
            logger.info(f"✅ Flow {flow.flow_id} 執行成功")
            return ExecutionResult(
                success=True,
                flow_id=flow.flow_id,
                module=flow.module,
                output=output
            )
        
        except Exception as e:
            logger.error(f"❌ Flow {flow.flow_id} 執行失敗: {e}")
            return ExecutionResult(
                success=False,
                flow_id=flow.flow_id,
                module=flow.module,
                output=None,
                error=str(e)
            )
    
    def _build_execution_params(
        self,
        flow: AttackFlow,
        target_context: dict[str, Any]
    ) -> dict[str, Any]:
        """構建執行參數
        
        將 AttackFlow 和 target_context 轉換為 FlowExecutor 需要的格式。
        
        Args:
            flow: 攻擊流程
            target_context: 目標上下文
        
        Returns:
            執行參數字典
        """
        params = {
            "module": flow.module,
            "entry_point": flow.entry_point,
            "file_name": flow.file_name,
            "target": target_context,
            "metadata": {
                "use_case": flow.use_case,
                "language": flow.language,
                "full_path": flow.full_path
            }
        }
        
        return params
    
    def search_and_execute(
        self,
        capability: AttackCapability,
        target_context: dict[str, Any],
        keywords: list[str] | None = None,
        limit: int = 3,
        config: ExecutionConfig | None = None
    ) -> list[ExecutionResult]:
        """搜索並執行指定能力的攻擊流程
        
        更直接的接口：指定攻擊類型 + 關鍵字，直接執行。
        
        Args:
            capability: 攻擊能力（如 AttackCapability.XSS）
            target_context: 目標上下文
            keywords: 搜索關鍵字
            limit: 最大執行數量
            config: 執行配置
        
        Returns:
            執行結果列表
        """
        config = config or ExecutionConfig()
        
        # 搜索流程
        flows = self.decision_engine.search_flows(
            capability=capability,
            keywords=keywords,
            only_operable=True,
            limit=limit
        )
        
        if not flows:
            logger.warning(f"⚠️ 未找到 {capability.value} 相關的可操作流程")
            return []
        
        # 執行所有匹配的流程
        results = []
        for flow in flows:
            result = self.execute_flow(flow, target_context, config)
            results.append(result)
        
        return results

# ==================== 使用範例 ====================

def example_usage():
    """使用範例"""
    # 1. 初始化適配器
    adapter = FlowExecutorAdapter()
    
    # 2. 模擬掃描結果上下文
    scan_context = {
        "vulnerability_type": "XSS",
        "target_info": {
            "url": "https://testphp.vulnweb.com/search.php",
            "parameter": "searchFor",
            "method": "GET"
        },
        "scan_results": {
            "confidence": "high",
            "severity": "medium"
        }
    }
    
    config = ExecutionConfig(verbose=True)
    results = adapter.execute_recommended_flows(
        scan_context=scan_context,
        top_k=3,
        config=config
    )
    
    # 4. 顯示結果
    print(f"\n執行了 {len(results)} 個攻擊流程:")
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} Flow {result.flow_id} ({result.module})")
        if result.error:
            print(f"   錯誤: {result.error}")
    
    # 5. 直接搜索並執行特定能力
    print("\n\n直接執行 SQLi 攻擊:")
    sqli_results = adapter.search_and_execute(
        capability=AttackCapability.SQLI,
        target_context={
            "url": "https://testphp.vulnweb.com/listproducts.php",
            "parameter": "cat",
            "method": "GET"
        },
        keywords=["注入", "檢測"],
        limit=2,
        config=config
    )
    
    for result in sqli_results:
        print(f"  Flow {result.flow_id}: {result.success}")

if __name__ == "__main__":
    example_usage()
