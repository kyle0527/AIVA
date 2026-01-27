"""
規劃執行協調器 - 連接 AI 決策和任務規劃系統

將 AI 決策生成的掃描策略轉換為任務規劃系統可執行的計劃，
然後通過 UnifiedExecutor 執行。

架構:
    UserInput → AIDecisionCore → PlanningExecutionCoordinator → UnifiedExecutor → CLI Flows
    
實現日期: 2026-01-20
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime, UTC

# 設置正確的 Python 路徑
PROJECT_ROOT = Path(__file__).parent.parent
SERVICES_PATH = PROJECT_ROOT / "services"
sys.path.insert(0, str(SERVICES_PATH / "core"))
sys.path.insert(0, str(SERVICES_PATH))

from aiva_common.utils import get_logger

# 使用完整導入路徑避免模組衝突
import services.core.aiva_core.cognitive_core.ai_decision_core as ai_decision
import services.core.aiva_core.internal_exploration.system_self_explorer as system_explorer

AIDecisionCore = ai_decision.AIDecisionCore
UserConstraints = ai_decision.UserConstraints
SystemSelfExplorer = system_explorer.SystemSelfExplorer

logger = get_logger(__name__)


class PlanningExecutionCoordinator:
    """規劃執行協調器
    
    連接 AI 決策層和任務規劃執行層
    """
    
    def __init__(self):
        self.decision_core = None
        self.unified_executor = None
        self.cli_executor = None
        self._initialized = False
    
    async def initialize(self):
        """初始化所有組件"""
        logger.info("🔧 正在初始化 PlanningExecutionCoordinator...")
        
        # 1. 初始化 AI 決策核心
        self.decision_core = AIDecisionCore()
        await self.decision_core.initialize()
        logger.info("   ✓ AIDecisionCore 已加載")
        
        # 2. 初始化 UnifiedExecutor (可選)
        try:
            from aiva_core.task_planning.unified_executor import UnifiedAttackExecutor
            self.unified_executor = UnifiedAttackExecutor()
            logger.info("   ✓ UnifiedExecutor 已加載")
        except Exception as e:
            logger.warning(f"   ⚠ UnifiedExecutor 不可用: {e}")
            self.unified_executor = None
        
        # 3. 初始化統一執行器控制層 (整合內外執行器)
        try:
            from aiva_core.internal_exploration.unified_executor_controller import UnifiedExecutorController
            self.unified_controller = UnifiedExecutorController()
            self.unified_controller.initialize()
            logger.info("   ✓ UnifiedExecutorController 已加載")
        except Exception as e:
            logger.warning(f"   ⚠ UnifiedExecutorController 不可用: {e}")
            self.unified_controller = None
        
        self._initialized = True
        logger.info("✅ PlanningExecutionCoordinator 初始化完成")
    
    async def execute_with_constraints(
        self,
        target: str,
        allowed_capabilities: Optional[list[str]] = None,
        forbidden_capabilities: Optional[list[str]] = None,
        time_limit: int = 300,
        max_depth: int = 3,
        dry_run: bool = False
    ) -> dict:
        """根據用戶限制執行掃描
        
        完整流程:
        1. AI 決策生成策略
        2. 轉換為攻擊計劃
        3. 通過規劃系統執行
        4. 返回結果
        
        Args:
            target: 目標網址
            allowed_capabilities: 允許的能力
            forbidden_capabilities: 禁止的能力
            time_limit: 時間限制
            max_depth: 爬取深度
            dry_run: 是否只預覽不執行
            
        Returns:
            執行結果字典
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"{'🔍 預覽模式' if dry_run else '⚡ 執行模式'}: {target}")
        
        # ===== 步驟 1: AI 決策 =====
        logger.info("\n[1/4] 🧠 AI 決策生成掃描策略...")
        
        constraints = UserConstraints(
            target=target,
            allowed_capabilities=allowed_capabilities or [],
            forbidden_capabilities=forbidden_capabilities or [],
            time_limit=time_limit,
            max_depth=max_depth
        )
        
        strategy = await self.decision_core.decide_scan_strategy(
            constraints,
            use_rag=True,
            use_history=False
        )
        
        logger.info(f"   ✓ 策略: {strategy.strategy_name}")
        logger.info(f"   ✓ 選中能力: {strategy.selected_capabilities}")
        logger.info(f"   ✓ Flow 數量: {len(strategy.flow_ids)}")
        
        # ===== 步驟 2: 生成攻擊計劃 =====
        logger.info("\n[2/4] 📋 生成攻擊計劃...")
        
        attack_plan = await self.decision_core.generate_attack_plan(
            strategy, constraints
        )
        
        logger.info(f"   ✓ 計劃 ID: {attack_plan['plan_id']}")
        logger.info(f"   ✓ 步驟數: {len(attack_plan['steps'])}")
        
        if dry_run:
            logger.info("\n[3/4] 🔍 預覽模式 - 顯示執行計劃")
            return {
                "mode": "dry_run",
                "strategy": strategy,
                "attack_plan": attack_plan,
                "preview": self._format_plan_preview(attack_plan)
            }
        
        # ===== 步驟 3: 執行計劃 =====
        logger.info("\n[3/4] ⚡ 執行攻擊計劃...")
        
        execution_results = await self._execute_plan(attack_plan)
        
        # ===== 步驟 4: 生成報告 =====
        logger.info("\n[4/4] 📊 生成執行報告...")
        
        report = self._generate_report(
            strategy=strategy,
            attack_plan=attack_plan,
            execution_results=execution_results
        )
        
        logger.info("✅ 執行完成")
        
        return report
    
    async def _execute_plan(self, attack_plan: dict) -> list[dict]:
        """執行攻擊計劃
        
        優先順序:
        1. UnifiedExecutor (如果可用)
        2. 直接調用 CLI Executor
        """
        results = []
        
        if self.unified_executor:
            # 方案 1: 通過 UnifiedExecutor
            logger.info("   使用 UnifiedExecutor 執行")
            try:
                # TODO: 調用 unified_executor.execute_attack_plan()
                logger.warning("   UnifiedExecutor 整合待實現")
                results = await self._execute_via_cli(attack_plan)
            except Exception as e:
                logger.error(f"   UnifiedExecutor 執行失敗: {e}")
                results = await self._execute_via_cli(attack_plan)
        else:
            # 方案 2: 直接調用 CLI
            logger.info("   直接使用 CLI Executor 執行")
            results = await self._execute_via_cli(attack_plan)
        
        return results
    
    async def _execute_via_cli(self, attack_plan: dict) -> list[dict]:
        """通過 CLI 執行計劃"""
        if not self.cli_executor:
            logger.error("   CLI Executor 不可用")
            return []
        
        results = []
        steps = attack_plan.get('steps', [])
        
        for i, step in enumerate(steps, 1):
            flow_id = step['flow_id']
            capability = step['capability']
            
            logger.info(f"   [{i}/{len(steps)}] 執行 Flow {flow_id} ({capability})...")
            
            try:
                # 執行 flow (dry_run=True 為預覽)
                result = self.cli_executor.execute_flow(
                    flow_id=flow_id,
                    dry_run=False  # 實際執行
                )
                
                results.append({
                    "step_id": step['step_id'],
                    "flow_id": flow_id,
                    "capability": capability,
                    "success": True,
                    "output": result if result else "No output",
                    "error": None
                })
                
                logger.info(f"      ✓ 完成")
                
            except Exception as e:
                logger.error(f"      ✗ 失敗: {e}")
                results.append({
                    "step_id": step['step_id'],
                    "flow_id": flow_id,
                    "capability": capability,
                    "success": False,
                    "output": None,
                    "error": str(e)
                })
        
        return results
    
    def _format_plan_preview(self, attack_plan: dict) -> str:
        """格式化計劃預覽"""
        lines = []
        lines.append("=" * 70)
        lines.append("攻擊計劃預覽".center(70))
        lines.append("=" * 70)
        
        lines.append(f"\n計劃 ID: {attack_plan['plan_id']}")
        lines.append(f"目標: {attack_plan['target']['url']}")
        lines.append(f"目的: {attack_plan['objective']}")
        
        lines.append(f"\n執行步驟 ({len(attack_plan['steps'])} 個):")
        for step in attack_plan['steps']:
            lines.append(
                f"  {step['step_id']}. Flow {step['flow_id']:3d} - "
                f"{step['capability']:15s} (timeout: {step['timeout']}s)"
            )
        
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)
    
    def _generate_report(
        self,
        strategy: dict,
        attack_plan: dict,
        execution_results: list[dict]
    ) -> dict:
        """生成執行報告"""
        successful = sum(1 for r in execution_results if r['success'])
        failed = len(execution_results) - successful
        
        return {
            "status": "completed",
            "summary": {
                "target": attack_plan['target']['url'],
                "strategy": strategy.strategy_name,
                "total_tasks": len(execution_results),
                "successful": successful,
                "failed": failed,
                "success_rate": successful / len(execution_results) if execution_results else 0
            },
            "strategy_info": {
                "selected_capabilities": strategy.selected_capabilities,
                "selected_engines": strategy.selected_engines,
                "reasoning": strategy.reasoning
            },
            "execution_details": execution_results,
            "attack_plan": attack_plan,
            "timestamp": datetime.now(UTC).isoformat()
        }


async def main_interactive():
    """互動模式入口"""
    print("=" * 70)
    print("AIVA 規劃執行系統 - 互動模式".center(70))
    print("=" * 70)
    
    coordinator = PlanningExecutionCoordinator()
    await coordinator.initialize()
    
    # 顯示系統能力
    explorer = coordinator.decision_core.system_explorer
    summary = await explorer.get_system_summary()
    
    print(f"\n📊 系統狀態:")
    print(f"   總 Flows: {summary['total_flows']}")
    print(f"   總能力: {summary['total_capabilities']}")
    print(f"   攻擊能力: {', '.join(summary['attack_capabilities'][:10])}")
    
    # 獲取用戶輸入
    print("\n" + "=" * 70)
    print("請輸入掃描參數".center(70))
    print("=" * 70)
    
    target = input("\n目標網址 (例: http://test.com): ").strip()
    if not target:
        print("❌ 目標網址不能為空")
        return
    
    print("\n允許的能力 (逗號分隔，Enter=全部):")
    allowed_input = input("允許: ").strip()
    allowed = [x.strip() for x in allowed_input.split(',')] if allowed_input else None
    
    print("\n禁止的能力 (逗號分隔，Enter=無):")
    forbidden_input = input("禁止: ").strip()
    forbidden = [x.strip() for x in forbidden_input.split(',')] if forbidden_input else None
    
    print("\n執行模式:")
    print("  1. 預覽模式 (dry-run)")
    print("  2. 實際執行")
    mode = input("選擇 (1/2): ").strip()
    dry_run = mode == "1"
    
    # 執行
    print("\n" + "=" * 70)
    result = await coordinator.execute_with_constraints(
        target=target,
        allowed_capabilities=allowed,
        forbidden_capabilities=forbidden,
        dry_run=dry_run
    )
    
    # 顯示結果
    if dry_run:
        print("\n" + result['preview'])
    else:
        print("\n📊 執行報告:")
        print(f"   狀態: {result['status']}")
        print(f"   總任務: {result['summary']['total_tasks']}")
        print(f"   成功: {result['summary']['successful']}")
        print(f"   失敗: {result['summary']['failed']}")
        print(f"   成功率: {result['summary']['success_rate']:.2%}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AIVA 規劃執行協調器")
    parser.add_argument('--target', type=str, help='目標網址')
    parser.add_argument('--allow', type=str, help='允許的能力 (逗號分隔)')
    parser.add_argument('--forbid', type=str, help='禁止的能力 (逗號分隔)')
    parser.add_argument('--dry-run', action='store_true', help='預覽模式')
    
    args = parser.parse_args()
    
    async def main():
        if args.target:
            # 命令列模式
            coordinator = PlanningExecutionCoordinator()
            await coordinator.initialize()
            
            allowed = [x.strip() for x in args.allow.split(',')] if args.allow else None
            forbidden = [x.strip() for x in args.forbid.split(',')] if args.forbid else None
            
            result = await coordinator.execute_with_constraints(
                target=args.target,
                allowed_capabilities=allowed,
                forbidden_capabilities=forbidden,
                dry_run=args.dry_run
            )
            
            print("\n" + "=" * 70)
            if args.dry_run:
                print(result['preview'])
            else:
                import json
                print(json.dumps(result['summary'], indent=2, ensure_ascii=False))
        else:
            # 互動模式
            await main_interactive()
    
    asyncio.run(main())
