"""
AIDecisionCore - AI 決策核心

雙閉環架構的大腦，負責：
1. 整合內部探索 (SystemSelfExplorer)
2. 整合 RAG 知識庫 (RAGTrigger)
3. 生成智能掃描策略

實現日期: 2026-01-20
更新日期: 2026-02-02 - 移除未使用的 ExternalFeedbackLoop
"""

from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from aiva_common.utils import get_logger

if TYPE_CHECKING:
    from aiva_core.internal_exploration.system_self_explorer import SystemSelfExplorer
    from aiva_core.cognitive_core.learning_system.rag_trigger import RAGTrigger

logger = get_logger(__name__)

@dataclass
class UserConstraints:
    """用戶限制條件"""
    target: str                          # 目標網址
    allowed_capabilities: list[str]      # 允許的能力，如 ["sqli", "xss"]
    forbidden_capabilities: list[str]    # 禁止的能力，如 ["dos", "rce"]
    time_limit: int = 300                # 時間限制 (秒)
    max_depth: int = 3                   # 最大爬取深度
    rate_limit: int = 10                 # 請求速率限制 (req/s)
    scope_include: list[str] = field(default_factory=list)  # 包含路徑
    scope_exclude: list[str] = field(default_factory=list)  # 排除路徑

@dataclass
class ScanStrategy:
    """掃描策略"""
    strategy_name: str                   # 策略名稱: fast, balanced, smart
    selected_engines: list[str]          # 選中的引擎
    selected_capabilities: list[str]     # 選中的能力
    flow_ids: list[int]                  # 要執行的 flow IDs
    priority_targets: list[str]          # 優先目標
    attack_payloads: dict = field(default_factory=dict)  # 攻擊 payload
    reasoning: str = ""                  # 決策理由

class AIDecisionCore:
    """AI 決策核心
    
    整合多個數據源進行智能決策
    
    使用示例:
        decision_core = AIDecisionCore()
        await decision_core.initialize()
        
        constraints = UserConstraints(
            target="http://test.com",
            allowed_capabilities=["xss", "sqli"],
            forbidden_capabilities=["dos"]
        )
        
        strategy = await decision_core.decide_scan_strategy(constraints)
        print(strategy.selected_capabilities)  # ["xss", "sqli"]
        print(strategy.flow_ids)  # [11, 97, 98]
    """
    
    def __init__(self):
        """初始化 AI 決策核心"""
        self._initialized = False
        self.system_explorer: Optional['SystemSelfExplorer'] = None
        self.rag_trigger: Optional['RAGTrigger'] = None
    
    async def initialize(self) -> None:
        """初始化決策核心，加載依賴組件"""
        logger.info("🧠 正在初始化 AIDecisionCore...")
        
        # 1. 初始化內部探索器 (必須)
        try:
            from aiva_core.internal_exploration.system_self_explorer import SystemSelfExplorer
            self.system_explorer = SystemSelfExplorer()
            self.system_explorer.initialize()  # 改為同步調用
            logger.info("   ✓ SystemSelfExplorer 已加載")
        except Exception as e:
            logger.error(f"   ✗ SystemSelfExplorer 加載失敗: {e}")
            raise
        
        # 2. 初始化 RAG 觸發器（必需組件）
        from aiva_core.cognitive_core.learning_system.rag_trigger import RAGTrigger
        self.rag_trigger = RAGTrigger()
        logger.info("   ✓ RAGTrigger 已加載")
        
        self._initialized = True
        logger.info("✅ AIDecisionCore 初始化完成")
    
    async def decide_scan_strategy(
        self, 
        constraints: UserConstraints,
        use_rag: bool = True
    ) -> ScanStrategy:
        """決定掃描策略
        
        Args:
            constraints: 用戶限制條件
            use_rag: 是否使用 RAG 搜索建議
            
        Returns:
            ScanStrategy: 掃描策略
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"🤔 正在為目標 {constraints.target} 生成掃描策略...")
        
        # ===== 步驟 1: 查詢內部能力 =====
        logger.info("   [1/3] 查詢系統可用能力...")
        assert self.system_explorer is not None, "system_explorer 未初始化"
        available_attacks = self.system_explorer.get_available_attacks()
        available_engines = self.system_explorer.get_available_engines()
        
        logger.info(f"        系統可用攻擊: {list(available_attacks.keys())}")
        logger.info(f"        系統可用引擎: {list(available_engines.keys())}")
        
        # ===== 步驟 2: 過濾符合用戶限制的能力 =====
        logger.info("   [2/3] 過濾符合用戶限制的能力...")
        selected_capabilities = self._filter_capabilities(
            available_attacks,
            constraints.allowed_capabilities,
            constraints.forbidden_capabilities
        )
        
        logger.info(f"        選中能力: {selected_capabilities}")
        
        # ===== 步驟 3: RAG 搜索建議 (可選) =====
        rag_suggestions = {}
        if use_rag and self.rag_trigger:
            logger.info("   [3/3] RAG 搜索相關建議...")
            try:
                rag_suggestions = await self._search_rag_suggestions(
                    constraints.target,
                    selected_capabilities
                )
                logger.info(f"        RAG 建議: {len(rag_suggestions)} 條")
            except Exception as e:
                logger.warning(f"        RAG 搜索失敗: {e}")
        else:
            logger.info("   [3/3] 跳過 RAG 搜索")
        
        # ===== 步驟 4: 生成策略 =====
        strategy = await self._generate_strategy(
            constraints,
            selected_capabilities,
            available_attacks,
            available_engines,
            rag_suggestions
        )
        
        logger.info(f"✅ 策略生成完成: {strategy.strategy_name}")
        logger.info(f"   • 選中引擎: {strategy.selected_engines}")
        logger.info(f"   • 選中能力: {strategy.selected_capabilities}")
        logger.info(f"   • Flow IDs: {strategy.flow_ids}")
        
        return strategy
    
    def _filter_capabilities(
        self,
        available_attacks: dict,
        allowed: list[str],
        forbidden: list[str]
    ) -> list[str]:
        """過濾能力列表"""
        selected = []
        
        for cap_type in available_attacks.keys():
            # 檢查是否在允許列表中
            if allowed and cap_type not in allowed:
                logger.debug(f"        跳過 {cap_type}: 不在允許列表中")
                continue
            
            # 檢查是否在禁止列表中
            if forbidden and cap_type in forbidden:
                logger.debug(f"        跳過 {cap_type}: 在禁止列表中")
                continue
            
            selected.append(cap_type)
        
        return selected
    
    async def _search_rag_suggestions(
        self,
        target: str,
        capabilities: list[str]
    ) -> dict:
        """使用 RAG 搜索建議 (可選)"""
        if not self.rag_trigger:
            return {}
        
        try:
            # 構造搜索查詢
            query = f"Target: {target}, Testing: {', '.join(capabilities)}"
            
            # 搜索相似案例
            results = await self.rag_trigger.search(
                query=query,
                mode="hybrid",
                top_k=3
            )
            
            return {
                "similar_cases": results.get("internal_results", []),
                "external_suggestions": results.get("external_results", [])
            }
        except Exception as e:
            logger.warning(f"RAG 搜索錯誤: {e}")
            return {}
    
    async def _generate_strategy(
        self,
        constraints: UserConstraints,
        selected_capabilities: list[str],
        available_attacks: dict,
        available_engines: dict,
        rag_suggestions: dict
    ) -> ScanStrategy:
        """生成最終策略"""
        
        # 收集所有相關的 flow IDs
        flow_ids = []
        for cap_type in selected_capabilities:
            cap_info = available_attacks.get(cap_type)
            if cap_info:
                flow_ids.extend(cap_info.flow_ids)
        
        # 選擇引擎 (基於簡單規則)
        selected_engines = []
        
        # 規則 1: 總是使用 Python 引擎 (最通用)
        if "python_engine" in available_engines:
            selected_engines.append("python_engine")
        
        # 規則 2: 如果需要快速掃描，添加 Rust 引擎
        if "rust_engine" in available_engines:
            selected_engines.append("rust_engine")
        
        # 生成決策理由
        reasoning_parts = [
            f"目標: {constraints.target}",
            f"用戶允許的能力: {constraints.allowed_capabilities or '全部'}",
            f"用戶禁止的能力: {constraints.forbidden_capabilities or '無'}",
            f"系統匹配的能力: {selected_capabilities}",
            f"計劃執行 {len(flow_ids)} 個 flows"
        ]
        
        if rag_suggestions:
            reasoning_parts.append(f"RAG 提供 {len(rag_suggestions.get('similar_cases', []))} 條建議")
        
        reasoning = "\n".join(reasoning_parts)
        
        return ScanStrategy(
            strategy_name="smart",
            selected_engines=selected_engines,
            selected_capabilities=selected_capabilities,
            flow_ids=flow_ids,
            priority_targets=[constraints.target],
            reasoning=reasoning
        )
    
    async def get_flow_execution_order(self, strategy: ScanStrategy) -> list[dict]:
        """獲取 flow 執行順序
        
        Returns:
            [
                {"flow_id": 11, "capability": "sqli", "priority": 1},
                {"flow_id": 97, "capability": "xss", "priority": 2},
            ]
        """
        execution_plan = []
        
        assert self.system_explorer is not None, "system_explorer 未初始化"
        available_attacks = self.system_explorer.get_available_attacks()
        
        for cap_type in strategy.selected_capabilities:
            cap_info = available_attacks.get(cap_type)
            if cap_info:
                for flow_id in cap_info.flow_ids:
                    execution_plan.append({
                        "flow_id": flow_id,
                        "capability": cap_type,
                        "priority": 1  # 簡化版本，所有優先級相同
                    })
        
        return execution_plan
    
    async def generate_attack_plan(
        self, 
        strategy: ScanStrategy,
        constraints: UserConstraints
    ) -> dict:
        """生成符合 UnifiedExecutor 格式的攻擊計劃
        
        將 AI 決策轉換為 task_planning 可執行的計劃
        
        Returns:
            AttackPlan 格式的字典
        """
        from datetime import datetime, timezone
        from uuid import uuid4
        
        execution_order = await self.get_flow_execution_order(strategy)
        
        # 構建攻擊步驟
        steps = []
        for i, task in enumerate(execution_order, 1):
            steps.append({
                "step_id": i,
                "flow_id": task["flow_id"],
                "capability": task["capability"],
                "action": "execute_flow",
                "params": {
                    "target": constraints.target,
                },
                "expected_output": "vulnerability_report",
                "timeout": 60
            })
        
        # 構建完整計劃
        plan = {
            "plan_id": str(uuid4()),
            "target": {
                "url": constraints.target,
                "metadata": {
                    "allowed_capabilities": constraints.allowed_capabilities,
                    "forbidden_capabilities": constraints.forbidden_capabilities,
                    "time_limit": constraints.time_limit
                }
            },
            "objective": f"Scan {constraints.target} with capabilities: {strategy.selected_capabilities}",
            "steps": steps,
            "metadata": {
                "strategy_name": strategy.strategy_name,
                "selected_engines": strategy.selected_engines,
                "reasoning": strategy.reasoning,
                "created_by": "AIDecisionCore"
            },
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        return plan

# ==================== 便利函數 ====================

async def quick_decision(
    target: str,
    allowed: Optional[list[str]] = None,
    forbidden: Optional[list[str]] = None
) -> ScanStrategy:
    """快速決策 (便利函數)
    
    Args:
        target: 目標網址
        allowed: 允許的能力，None 表示全部
        forbidden: 禁止的能力，默認為空
    
    Returns:
        ScanStrategy
    """
    core = AIDecisionCore()
    await core.initialize()
    
    constraints = UserConstraints(
        target=target,
        allowed_capabilities=allowed or [],
        forbidden_capabilities=forbidden or []
    )
    
    return await core.decide_scan_strategy(constraints)

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("=" * 60)
        print("AIDecisionCore 測試".center(60))
        print("=" * 60)
        
        # 創建決策核心
        core = AIDecisionCore()
        await core.initialize()
        
        # 測試場景 1: 只允許 XSS 和 SQLi
        print("\n【測試 1】只允許 XSS 和 SQLi")
        constraints1 = UserConstraints(
            target="http://test.com",
            allowed_capabilities=["xss", "sqli"],
            forbidden_capabilities=["dos", "rce"]
        )
        
        strategy1 = await core.decide_scan_strategy(constraints1)
        print(f"策略: {strategy1.strategy_name}")
        print(f"選中能力: {strategy1.selected_capabilities}")
        print(f"Flow IDs: {strategy1.flow_ids}")
        print(f"\n決策理由:\n{strategy1.reasoning}")
        
        # 測試場景 2: 允許所有攻擊
        print("\n" + "=" * 60)
        print("\n【測試 2】允許所有攻擊")
        constraints2 = UserConstraints(
            target="http://vulnerable-site.com",
            allowed_capabilities=[],
            forbidden_capabilities=[]
        )
        
        strategy2 = await core.decide_scan_strategy(constraints2, use_rag=False)
        print(f"策略: {strategy2.strategy_name}")
        print(f"選中能力: {strategy2.selected_capabilities}")
        print(f"Flow IDs 數量: {len(strategy2.flow_ids)}")
        
        print("\n" + "=" * 60)
    
    asyncio.run(main())
