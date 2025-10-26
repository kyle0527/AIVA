"""
AIVA 核心模組整合範例

展示對話助理、技能圖、能力評估器三大組件的協同工作
"""

import asyncio
from datetime import datetime
from typing import Dict, Any

# 修正導入路徑
import sys
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from services.core.aiva_core.dialog.assistant import dialog_assistant
    from services.core.aiva_core.decision.skill_graph import skill_graph
    from services.core.aiva_core.learning.capability_evaluator import capability_evaluator
    from services.integration.capability import CapabilityRegistry
except ImportError as e:
    print(f"⚠️ 導入錯誤: {e}")
    print("請確保 AIVA 項目結構正確且已安裝相關依賴")
    
    # 創建模擬對象以便演示繼續進行
    class MockComponent:
        async def process_user_input(self, input_text): 
            return {"intent": "mock", "message": "模擬回應", "executable": False}
        async def initialize(self): pass
        def get_graph_statistics(self): return {"nodes": 0, "edges": 0}
        async def get_recommendations(self, cap_id, limit=5): return []
        async def find_execution_path(self, start, goal, max_path_length=5): return []
        async def start_evaluation_session(self, capability_id, inputs): return "mock_session"
        async def end_evaluation_session(self, session_id, outputs, success, **kwargs): pass
        async def add_user_feedback(self, session_id, feedback_score, user_feedback=None): pass
        def get_evaluation_statistics(self): return {"total_sessions": 0, "success_rate": 0.0, "sessions_with_feedback": 0, "average_feedback_score": 0.0, "active_training_plans": 0}
        async def get_capability_insights(self, limit=10, days=30): return []
        async def rebuild_if_needed(self): pass
    
    class MockRegistry:
        async def list_capabilities(self, limit=None): return []
    
    dialog_assistant = MockComponent()
    skill_graph = MockComponent()
    capability_evaluator = MockComponent()
    CapabilityRegistry = MockRegistry


async def demo_dialog_assistant():
    """演示對話助理功能"""
    print("🎯 對話助理演示")
    print("=" * 50)
    
    # 模擬用戶對話
    test_inputs = [
        "現在系統會什麼？",
        "幫我跑 https://example.com 的掃描",
        "產生 CLI 指令",
        "系統狀況如何？",
        "解釋 SQL 注入掃描"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 用戶: {user_input}")
        
        try:
            response = await dialog_assistant.process_user_input(user_input)
            
            print(f"🤖 助理: {response['message'][:200]}...")
            print(f"📊 意圖: {response['intent']}")
            print(f"🔧 可執行: {response['executable']}")
            
            if response.get('data'):
                print(f"📈 數據: {list(response['data'].keys())}")
        
        except Exception as e:
            print(f"❌ 錯誤: {e}")
        
        print("-" * 30)


async def demo_skill_graph():
    """演示技能圖功能"""
    print("\n🧠 技能圖演示")
    print("=" * 50)
    
    try:
        # 初始化技能圖
        print("🔧 初始化技能圖...")
        await skill_graph.initialize()
        
        # 獲取圖統計信息
        stats = skill_graph.get_graph_statistics()
        print(f"📊 圖統計:")
        print(f"  節點數: {stats.get('nodes', 0)}")
        print(f"  邊數: {stats.get('edges', 0)}")
        print(f"  密度: {stats.get('density', 0):.3f}")
        print(f"  連通性: {stats.get('is_connected', False)}")
        
        # 模擬能力推薦
        capabilities = await CapabilityRegistry().list_capabilities(limit=3)
        if capabilities:
            capability_id = capabilities[0].id
            print(f"\n🎯 為能力 {capability_id} 獲取推薦:")
            
            recommendations = await skill_graph.get_recommendations(capability_id, limit=3)
            
            for i, (rec_id, score, reason) in enumerate(recommendations, 1):
                print(f"  {i}. {rec_id} (分數: {score:.3f}) - {reason}")
        
        # 尋找執行路徑
        if len(capabilities) >= 2:
            start_cap = capabilities[0].id
            goal = "scanning"
            
            print(f"\n🛤️ 尋找從 {start_cap} 到 {goal} 的執行路徑:")
            paths = await skill_graph.find_execution_path(start_cap, goal, max_path_length=3)
            
            for i, path in enumerate(paths[:2], 1):
                print(f"  路徑 {i}: {path.description}")
                print(f"    成功機率: {path.success_probability:.1%}")
                print(f"    預估時間: {path.estimated_time:.1f}秒")
    
    except Exception as e:
        print(f"❌ 技能圖演示錯誤: {e}")


async def demo_capability_evaluator():
    """演示能力評估器功能"""
    print("\n📈 能力評估器演示")
    print("=" * 50)
    
    try:
        # 模擬評估會話
        print("🔧 開始評估會話...")
        
        session_id = await capability_evaluator.start_evaluation_session(
            capability_id="test_capability",
            inputs={"url": "https://example.com", "timeout": 30}
        )
        
        print(f"📋 會話ID: {session_id}")
        
        # 模擬會話結束
        await asyncio.sleep(0.1)  # 模擬執行時間
        
        await capability_evaluator.end_evaluation_session(
            session_id=session_id,
            outputs={"status": "completed", "findings": 3},
            success=True,
            execution_time_ms=1500.0,
            memory_usage_mb=25.6
        )
        
        # 添加用戶反饋
        await capability_evaluator.add_user_feedback(
            session_id=session_id,
            feedback_score=4.5,
            user_feedback="掃描結果很準確，速度也很快"
        )
        
        print("✅ 評估會話完成")
        
        # 獲取評估統計
        stats = capability_evaluator.get_evaluation_statistics()
        print(f"\n📊 評估統計:")
        print(f"  總會話數: {stats['total_sessions']}")
        print(f"  成功率: {stats['success_rate']:.1%}")
        print(f"  有反饋會話: {stats['sessions_with_feedback']}")
        print(f"  平均反饋分數: {stats['average_feedback_score']:.1f}/5")
        print(f"  活躍訓練計劃: {stats['active_training_plans']}")
        
        # 嘗試分析能力洞察
        capabilities = await CapabilityRegistry().list_capabilities(limit=2)
        if capabilities:
            print(f"\n🔍 分析能力洞察...")
            
            insights = await capability_evaluator.get_capability_insights(limit=2, days=30)
            
            for insight in insights:
                print(f"\n📋 能力: {insight.name}")
                print(f"  成功率: {insight.success_rate:.1%}")
                print(f"  平均執行時間: {insight.avg_execution_time:.0f}ms")  
                print(f"  信心分數: {insight.confidence_score:.1%}")
                print(f"  改進趨勢: {insight.improvement_trend:+.3f}")
                
                if insight.optimization_suggestions:
                    print(f"  優化建議: {insight.optimization_suggestions[0]}")
                
                if insight.training_recommendations:
                    print(f"  訓練推薦: {insight.training_recommendations[0]}")
    
    except Exception as e:
        print(f"❌ 能力評估器演示錯誤: {e}")


async def demo_integrated_workflow():
    """演示三大組件的整合工作流"""
    print("\n🔄 整合工作流演示")
    print("=" * 50)
    
    try:
        # 1. 用戶透過對話助理詢問系統能力
        print("1️⃣ 用戶詢問系統能力...")
        response = await dialog_assistant.process_user_input("現在系統會什麼？")
        
        if response.get('data') and response['data'].get('capabilities'):
            capabilities = response['data']['capabilities'][:2]
            print(f"   發現 {len(capabilities)} 個能力")
            
            # 2. 技能圖分析能力關係
            print("\n2️⃣ 技能圖分析能力關係...")
            await skill_graph.rebuild_if_needed()
            
            if capabilities:
                cap_id = capabilities[0]['id']
                recommendations = await skill_graph.get_recommendations(cap_id, limit=2)
                print(f"   為 {cap_id} 找到 {len(recommendations)} 個推薦")
            
            # 3. 能力評估器提供性能洞察
            print("\n3️⃣ 能力評估器分析性能...")
            insights = await capability_evaluator.get_capability_insights(limit=2, days=30)
            print(f"   生成 {len(insights)} 個能力洞察")
            
            # 4. 整合結果並回應用戶
            print("\n4️⃣ 整合結果...")
            recommendations_count = 0
            try:
                if capabilities:
                    cap_id = capabilities[0]['id']
                    recommendations = await skill_graph.get_recommendations(cap_id, limit=2)
                    recommendations_count = len(recommendations)
            except:
                recommendations_count = 0
            
            integrated_response = {
                "available_capabilities": len(capabilities),
                "skill_recommendations": recommendations_count,
                "performance_insights": len(insights),
                "system_health": "良好" if insights and all(i.confidence_score > 0.5 for i in insights) else "需要改進"
            }
            
            print(f"   整合回應: {integrated_response}")
        
        print("\n✅ 整合工作流完成")
    
    except Exception as e:
        print(f"❌ 整合工作流錯誤: {e}")


async def main():
    """主演示函數"""
    print("🚀 AIVA 核心模組整合演示")
    print("=" * 70)
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    try:
        # 按順序演示各組件
        await demo_dialog_assistant()
        await asyncio.sleep(1)
        
        await demo_skill_graph()
        await asyncio.sleep(1)
        
        await demo_capability_evaluator()
        await asyncio.sleep(1)
        
        await demo_integrated_workflow()
        
        print("\n" + "=" * 70)
        print("🎉 所有演示完成！")
        print("=" * 70)
    
    except Exception as e:
        print(f"❌ 演示過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 執行演示
    asyncio.run(main())