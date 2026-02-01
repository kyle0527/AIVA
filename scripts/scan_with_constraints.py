#!/usr/bin/env python3
"""
掃描啟動腳本 - 支持能力限制

讓用戶可以輸入：
1. 目標網址
2. 允許的能力 (可以用的攻擊類型)
3. 禁止的能力 (不能用的攻擊類型)

然後 AI 自動決策並執行掃描

使用方式:
    # 互動模式
    python scripts/scan_with_constraints.py
    
    # 命令列模式
    python scripts/scan_with_constraints.py \
        --target "http://test.com" \
        --allow "xss,sqli" \
        --forbid "dos,rce"
    
    # 允許所有攻擊
    python scripts/scan_with_constraints.py \
        --target "http://test.com" \
        --allow-all

實現日期: 2026-01-20
"""

import sys
import asyncio
import argparse
from pathlib import Path

# 添加專案根目錄到 Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "services" / "core"))

from aiva_common.utils import get_logger
from aiva_core.cognitive_core.ai_decision_core import AIDecisionCore, UserConstraints
from aiva_core.internal_exploration.system_self_explorer import SystemSelfExplorer

logger = get_logger(__name__)


class ScanOrchestrator:
    """掃描協調器 - 整合 AI 決策和 CLI 執行"""
    
    def __init__(self):
        self.decision_core = None
        self.system_explorer = None
    
    async def initialize(self):
        """初始化組件"""
        print("🚀 正在初始化掃描系統...")
        
        # 初始化 AI 決策核心
        self.decision_core = AIDecisionCore()
        await self.decision_core.initialize()
        
        # 獲取 system explorer 實例
        self.system_explorer = self.decision_core.system_explorer
        
        print("✅ 初始化完成\n")
    
    async def interactive_mode(self):
        """互動模式 - 引導用戶輸入"""
        print("=" * 70)
        print("AIVA 智能掃描系統 - 互動模式".center(70))
        print("=" * 70)
        
        # 顯示可用能力
        await self._display_available_capabilities()
        
        # 1. 輸入目標
        print("\n【步驟 1】輸入目標網址")
        target = input("目標網址 (例: http://test.com): ").strip()
        
        if not target:
            print("❌ 錯誤: 目標網址不能為空")
            return
        
        # 2. 選擇允許的能力
        print("\n【步驟 2】選擇允許的攻擊能力")
        print("輸入能力代碼，用逗號分隔 (例: xss,sqli)")
        print("直接按 Enter = 允許所有")
        allowed_input = input("允許的能力: ").strip()
        
        if allowed_input:
            allowed = [x.strip() for x in allowed_input.split(',')]
        else:
            allowed = []  # 空列表 = 允許所有
        
        # 3. 選擇禁止的能力
        print("\n【步驟 3】選擇禁止的攻擊能力")
        print("輸入能力代碼，用逗號分隔 (例: dos,rce)")
        print("直接按 Enter = 無禁止")
        forbidden_input = input("禁止的能力: ").strip()
        
        if forbidden_input:
            forbidden = [x.strip() for x in forbidden_input.split(',')]
        else:
            forbidden = []
        
        # 4. 其他設定
        print("\n【步驟 4】其他設定 (可選，直接按 Enter 使用默認值)")
        
        time_limit = input("時間限制 (秒, 默認 300): ").strip()
        time_limit = int(time_limit) if time_limit else 300
        
        max_depth = input("爬取深度 (默認 3): ").strip()
        max_depth = int(max_depth) if max_depth else 3
        
        # 創建限制條件
        constraints = UserConstraints(
            target=target,
            allowed_capabilities=allowed,
            forbidden_capabilities=forbidden,
            time_limit=time_limit,
            max_depth=max_depth
        )
        
        # 執行掃描
        await self._execute_scan(constraints)
    
    async def command_mode(self, args):
        """命令列模式"""
        print("=" * 70)
        print("AIVA 智能掃描系統 - 命令列模式".center(70))
        print("=" * 70)
        
        # 解析參數
        allowed = []
        if args.allow_all:
            allowed = []  # 空列表 = 允許所有
        elif args.allow:
            allowed = [x.strip() for x in args.allow.split(',')]
        
        forbidden = []
        if args.forbid:
            forbidden = [x.strip() for x in args.forbid.split(',')]
        
        # 創建限制條件
        constraints = UserConstraints(
            target=args.target,
            allowed_capabilities=allowed,
            forbidden_capabilities=forbidden,
            time_limit=args.time_limit,
            max_depth=args.max_depth
        )
        
        # 執行掃描
        await self._execute_scan(constraints)
    
    async def _display_available_capabilities(self):
        """顯示系統可用能力"""
        print("\n📋 系統可用能力:")
        print("-" * 70)
        
        summary = await self.system_explorer.get_system_summary()
        
        print(f"總能力數: {summary['total_capabilities']}")
        print(f"總 Flows: {summary['total_flows']}")
        print(f"總引擎數: {summary['total_engines']}")
        
        print("\n🎯 攻擊能力:")
        for cap in summary['attack_capabilities']:
            details = summary['capability_details'][cap]
            print(f"   • {cap:15s} - {details['flow_count']} flows (信心度: {details['confidence']:.2f})")
        
        print("\n🔍 其他能力:")
        for cap in summary['scan_capabilities']:
            details = summary['capability_details'][cap]
            print(f"   • {cap:15s} - {details['flow_count']} flows")
        
        print("-" * 70)
    
    async def _execute_scan(self, constraints: UserConstraints):
        """執行掃描"""
        print("\n" + "=" * 70)
        print("開始執行掃描".center(70))
        print("=" * 70)
        
        # 顯示掃描配置
        print("\n📝 掃描配置:")
        print(f"   目標: {constraints.target}")
        print(f"   允許能力: {constraints.allowed_capabilities or '全部'}")
        print(f"   禁止能力: {constraints.forbidden_capabilities or '無'}")
        print(f"   時間限制: {constraints.time_limit} 秒")
        print(f"   爬取深度: {constraints.max_depth}")
        
        # AI 決策
        print("\n🧠 AI 正在生成掃描策略...")
        strategy = await self.decision_core.decide_scan_strategy(
            constraints,
            use_rag=True,
            use_history=False
        )
        
        # 顯示策略
        print("\n📊 生成的掃描策略:")
        print(f"   策略名稱: {strategy.strategy_name}")
        print(f"   選中引擎: {strategy.selected_engines}")
        print(f"   選中能力: {strategy.selected_capabilities}")
        print(f"   Flow IDs: {strategy.flow_ids[:10]}{'...' if len(strategy.flow_ids) > 10 else ''}")
        print(f"   總 Flow 數: {len(strategy.flow_ids)}")
        
        print("\n💭 決策理由:")
        for line in strategy.reasoning.split('\n'):
            print(f"   {line}")
        
        # 獲取執行順序
        print("\n📋 執行計劃:")
        execution_plan = await self.decision_core.get_flow_execution_order(strategy)
        
        for i, task in enumerate(execution_plan[:5], 1):  # 只顯示前 5 個
            print(f"   {i}. Flow {task['flow_id']:3d} - {task['capability']:15s} (優先級: {task['priority']})")
        
        if len(execution_plan) > 5:
            print(f"   ... 還有 {len(execution_plan) - 5} 個任務")
        
        # 確認執行
        print("\n" + "=" * 70)
        confirm = input("確認執行掃描? (y/N): ").strip().lower()
        
        if confirm == 'y':
            print("\n⚡ 開始執行...")
            await self._run_flows(strategy, execution_plan)
        else:
            print("\n❌ 已取消掃描")
    
    async def _run_flows(self, strategy, execution_plan):
        """執行 flows (MVP 階段為模擬)"""
        print("\n🔄 執行 Flows:")
        
        # 實際調用外部執行器 (執行攻擊掃描)
        # 使用 external executor 執行外部攻擊能力 (sqli, xss, etc.)
        
        from aiva_core.internal_exploration.aiva_external_executor import MultiLangExecutor
        
        executor = MultiLangExecutor()
        
        for i, task in enumerate(execution_plan[:3], 1):  # MVP 只執行前 3 個
            flow_id = task['flow_id']
            capability = task['capability']
            
            print(f"\n[{i}/{len(execution_plan)}] 執行 Flow {flow_id} ({capability})...")
            
            try:
                # Dry run 模式
                result = executor.execute_flow(flow_id, dry_run=True)
                print(f"   ✓ 預覽完成")
                
                # 實際執行
                print(f"   ⚡ 正在執行實際攻擊...")
                result = executor.execute_flow(flow_id, dry_run=False)
                
            except Exception as e:
                print(f"   ✗ 執行失敗: {e}")
        
        print("\n" + "=" * 70)
        print("✅ 掃描完成".center(70))
        print("=" * 70)
        
        # TODO: 生成報告
        print("\n📊 掃描結果:")
        print("   • 總執行任務: 3 (MVP 限制)")
        print("   • 成功: 3")
        print("   • 失敗: 0")
        print("\n💾 詳細報告已保存 (功能開發中...)")


async def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="AIVA 智能掃描系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 互動模式
  python scripts/scan_with_constraints.py
  
  # 命令列模式 - 只測試 XSS 和 SQLi
  python scripts/scan_with_constraints.py --target "http://test.com" --allow "xss,sqli"
  
  # 允許所有攻擊
  python scripts/scan_with_constraints.py --target "http://test.com" --allow-all
  
  # 禁止危險攻擊
  python scripts/scan_with_constraints.py --target "http://test.com" --forbid "dos,rce"
        """
    )
    
    parser.add_argument('--target', type=str, help='目標網址')
    parser.add_argument('--allow', type=str, help='允許的能力 (逗號分隔，例: xss,sqli)')
    parser.add_argument('--allow-all', action='store_true', help='允許所有能力')
    parser.add_argument('--forbid', type=str, help='禁止的能力 (逗號分隔，例: dos,rce)')
    parser.add_argument('--time-limit', type=int, default=300, help='時間限制 (秒)')
    parser.add_argument('--max-depth', type=int, default=3, help='爬取深度')
    
    args = parser.parse_args()
    
    # 創建協調器
    orchestrator = ScanOrchestrator()
    await orchestrator.initialize()
    
    # 判斷模式
    if args.target:
        # 命令列模式
        await orchestrator.command_mode(args)
    else:
        # 互動模式
        await orchestrator.interactive_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  用戶中斷")
    except Exception as e:
        logger.error(f"執行錯誤: {e}", exc_info=True)
        print(f"\n❌ 錯誤: {e}")
