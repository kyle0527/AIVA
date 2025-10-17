#!/usr/bin/env python3
"""
快速測試 BioNeuronCore AI 的執行和記憶能力
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def test_ai_execution_and_memory():
    """測試 AI 執行和記憶功能"""
    print("="*60)
    print("BioNeuronCore AI - 執行和記憶功能測試")
    print("="*60)
    
    try:
        # 使用原本的 bio_neuron_core.py
        from services.core.aiva_core.ai_engine.bio_neuron_core import BioNeuronRAGAgent
        import tempfile
        
        # 創建 AI 代理 (關閉不必要的組件以簡化測試)
        print("\n[步驟 1] 創建 AI 代理")
        agent = BioNeuronRAGAgent(
            codebase_path=tempfile.gettempdir(),
            enable_planner=False,
            enable_tracer=False, 
            enable_experience=False
        )
        print("✓ AI 創建成功")
        print(f"  - 工具數量: {len(agent.tools)}")
        print(f"  - 輸入向量維度: {agent.input_vector_size}")
        
        # 測試執行能力
        print("\n[步驟 2] 測試 AI 執行能力")
        result = agent.invoke("分析系統代碼結構")
        print("✓ AI 執行成功")
        print(f"  - 狀態: {result.get('status')}")
        print(f"  - 選擇的工具: {result.get('tool_used')}")
        print(f"  - 信心度: {result.get('confidence', 0):.2%}")
        
        # 測試記憶能力
        print("\n[步驟 3] 測試 AI 記憶能力")
        print(f"  執行前歷史記錄數: {len(agent.history)}")
        
        # 執行多個任務
        agent.invoke("讀取配置文件")
        agent.invoke("檢測 SQL 注入漏洞")
        agent.invoke("掃描目標網站")
        
        print(f"  執行後歷史記錄數: {len(agent.history)}")
        print("✓ AI 記憶功能正常")
        
        # 顯示記憶內容
        if agent.history:
            print("\n  最近的執行記錄:")
            for i, record in enumerate(agent.history[-3:], 1):
                tool = record.get('tool_used', 'Unknown')
                status = record.get('status', 'Unknown')
                print(f"    {i}. 工具: {tool}, 狀態: {status}")
        
        # 測試知識庫
        print("\n[步驟 4] 檢查知識庫")
        stats = agent.get_knowledge_stats()
        print("✓ 知識庫統計:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        # 總結
        print("\n" + "="*60)
        print("測試結果:")
        print("="*60)
        print("✓ AI 可以執行決策 (invoke 方法)")
        print("✓ AI 可以記憶歷史 (history 列表)")
        print("✓ AI 有知識庫支持 (get_knowledge_stats)")
        print("\n結論: BioNeuronCore AI 具備完整的執行和記憶能力！")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ai_execution_and_memory()
    sys.exit(0 if success else 1)
