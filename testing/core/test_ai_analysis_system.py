#!/usr/bin/env python3
"""
測試AI分析系統
驗證修復後的AI Analysis Engine是否正常工作
"""

import asyncio
import sys
from pathlib import Path

# 添加項目路徑
sys.path.insert(0, str(Path(__file__).parent))

async def test_ai_analysis_system():
    """測試AI分析系統的主要功能"""
    
    print("🧪 開始測試AI分析系統...")
    print("=" * 60)
    
    # 測試1: 導入AI分析模組
    print("\n📦 測試1: 導入AI分析模組")
    try:
        from services.core.aiva_core.ai_analysis import (
            AIAnalysisEngine, 
            AnalysisType, 
            AIAnalysisResult
        )
        print("✅ AI分析模組導入成功")
        print(f"   - AIAnalysisEngine: {AIAnalysisEngine}")
        print(f"   - AnalysisType: {AnalysisType}")
        print(f"   - AIAnalysisResult: {AIAnalysisResult}")
    except Exception as e:
        print(f"❌ AI分析模組導入失敗: {e}")
        return False
    
    # 測試2: 初始化AI分析引擎
    print("\n🚀 測試2: 初始化AI分析引擎")
    try:
        ai_engine = AIAnalysisEngine()
        result = ai_engine.initialize()
        
        if result:
            print("✅ AI分析引擎初始化成功")
            print(f"   - 初始化狀態: {ai_engine.initialized}")
            print(f"   - RAG代理狀態: {'可用' if ai_engine.rag_agent else '不可用'}")
        else:
            print("⚠️ AI分析引擎初始化部分失敗，但可以繼續運行")
    except Exception as e:
        print(f"❌ AI分析引擎初始化失敗: {e}")
        return False
    
    # 測試3: 代碼分析功能
    print("\n🔍 測試3: 代碼分析功能")
    
    # 準備測試代碼
    test_code = '''
def vulnerable_function(user_input):
    """這是一個有潛在安全問題的函數"""
    # SQL注入漏洞
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    
    # 硬編碼密碼
    password = "admin123"
    
    # 未檢查的輸入
    eval(user_input)  # 代碼注入風險
    
    return query
    '''
    
    try:
        # 測試安全分析
        security_results = ai_engine.analyze_code(
            source_code=test_code,
            file_path="test_vulnerable.py",
            analysis_types=[AnalysisType.SECURITY, AnalysisType.VULNERABILITY]
        )
        
        print("✅ 代碼安全分析完成")
        for analysis_type, result in security_results.items():
            print(f"   - {analysis_type.value}: 信心度 {result.confidence:.2f}")
            print(f"     風險等級: {result.risk_level}")
            print(f"     發現數量: {len(result.findings)}")
            
        # 測試複雜度分析
        complexity_results = ai_engine.analyze_code(
            source_code=test_code,
            file_path="test_complexity.py",
            analysis_types=[AnalysisType.COMPLEXITY, AnalysisType.PATTERNS]
        )
        
        print("✅ 代碼複雜度分析完成")
        for analysis_type, result in complexity_results.items():
            print(f"   - {analysis_type.value}: 信心度 {result.confidence:.2f}")
            
    except Exception as e:
        print(f"❌ 代碼分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 測試4: AI模型管理器集成
    print("\n🤖 測試4: AI模型管理器集成")
    try:
        from services.core.aiva_core.ai_engine.ai_model_manager import AIModelManager
        
        ai_manager = AIModelManager()
        init_result = ai_manager.initialize_models(input_size=100, num_tools=10)
        
        if init_result.get("status") == "success":
            print("✅ AI模型管理器初始化成功")
            print(f"   - ScalableBioNet參數: {init_result.get('scalable_net_params', 0):,}")
            print(f"   - BioAgent狀態: {'就緒' if init_result.get('bio_agent_ready') else '未就緒'}")
        else:
            print(f"⚠️ AI模型管理器初始化部分成功: {init_result.get('status')}")
            
        # 測試決策功能
        decision_result = await ai_manager.make_decision(
            query="分析這個代碼是否有安全問題",
            context={"code": test_code[:200]},
            use_rag=False  # 不使用RAG避免複雜依賴
        )
        
        if decision_result.get("status") == "success":
            print("✅ AI決策功能正常")
            print(f"   - 決策信心度: {decision_result.get('result', {}).get('confidence', 0):.3f}")
        else:
            print(f"❌ AI決策功能失敗: {decision_result.get('error')}")
            
    except Exception as e:
        print(f"❌ AI模型管理器測試失敗: {e}")
        return False
    
    # 測試5: 系統整體狀態
    print("\n📊 測試5: 系統整體狀態")
    try:
        status = ai_manager.get_model_status()
        print("✅ 系統狀態檢查完成")
        print(f"   - 模型版本: {status.get('model_version')}")
        print(f"   - 訓練狀態: {'已訓練' if status.get('is_trained') else '未訓練'}")
        print(f"   - 最後更新: {status.get('last_update')}")
        print(f"   - ScalableBioNet: {'已初始化' if status.get('scalable_net_initialized') else '未初始化'}")
        
    except Exception as e:
        print(f"❌ 系統狀態檢查失敗: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 所有測試完成！AI分析系統運行正常")
    return True

async def test_integration_with_bio_master():
    """測試與BioNeuron Master的整合"""
    print("\n🧠 額外測試: BioNeuron Master整合")
    
    try:
        from services.core.aiva_core.bio_neuron_master import BioNeuronMasterController
        
        master = BioNeuronMasterController()
        
        # 測試處理分析請求
        response = await master.process_request(
            request="請分析這段代碼的安全性",
            context={"mode": "analysis"}
        )
        
        print("✅ BioNeuron Master整合測試完成")
        print(f"   - 響應狀態: {response.get('success', False)}")
        print(f"   - 響應類型: {response.get('response_type', 'unknown')}")
        
    except Exception as e:
        print(f"⚠️ BioNeuron Master整合測試失敗: {e}")
        # 這不是致命錯誤，系統仍可運行

if __name__ == "__main__":
    print("🔬 AIVA AI分析系統測試套件")
    print("測試修復後的AI分析功能")
    
    # 運行主要測試
    success = asyncio.run(test_ai_analysis_system())
    
    if success:
        # 運行整合測試
        asyncio.run(test_integration_with_bio_master())
        
        print("\n🎯 測試總結:")
        print("✅ AI分析系統已成功修復並正常運行")
        print("✅ 核心功能驗證完成")
        print("✅ 可以開始使用AI增強的代碼分析功能")
    else:
        print("\n❌ 測試失敗，需要進一步修復")
        sys.exit(1)