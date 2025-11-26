#!/usr/bin/env python3
"""
AIVA核心與能力整合測試
"""
from aiva_capability_orchestrator import AIVACapabilityOrchestrator

def test_integration():
    """測試完整的核心與能力整合"""
    print("🔍 AIVA核心與能力整合測試")
    print("=" * 40)
    
    # 創建編排器實例
    orchestrator = AIVACapabilityOrchestrator()
    print(f"🤖 AI核心狀態: {'已載入' if orchestrator.ai_core else '未載入'}")
    
    if orchestrator.ai_core:
        print(f"📊 模型參數: {orchestrator.ai_core.total_params:,}")
        print(f"🔧 模型類型: {'5M特化神經網路' if orchestrator.ai_core.use_5m_model else '基礎模型'}")
    
    # 測試能力執行
    print("\n🚀 執行綜合分析...")
    results, features, decision = orchestrator.execute_comprehensive_analysis('https://test.example.com')
    
    print(f"📈 能力執行結果: {len(results)}個能力")
    for result in results:
        print(f"   - {result.capability_type}: {result.status} (信心度: {result.confidence:.3f})")
    
    print(f"🧠 特徵向量: {len(features)}維")
    print(f"⚡ AI決策: {'成功' if decision else '失敗'}")
    
    if decision:
        print(f"🎯 決策詳情:")
        print(f"   - 類別: {decision['primary_decision']['class']}")
        print(f"   - 信心度: {decision['primary_decision']['confidence']:.4f}")
        print(f"   - 備選動作數: {len(decision['alternative_actions'])}")
    
    print("\n✅ 整合測試完成!")
    return True

if __name__ == "__main__":
    test_integration()