#!/usr/bin/env python3
"""
AIVA AI 系統靶場實測腳本

測試完整的 AI 核心系統，包括：
- 統一的 AI 模型管理器
- 性能優化的神經網路
- 訓練系統整合
- 實際決策場景模擬
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# 添加路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'core'))

async def test_scenario_1_basic_initialization():
    """場景 1: 基本系統初始化測試"""
    print("🎯 場景 1: 基本系統初始化測試")
    print("=" * 50)
    
    try:
        from aiva_core.ai_engine import AIModelManager, PerformanceConfig
        
        # 創建性能配置
        config = PerformanceConfig(
            max_concurrent_tasks=10,
            batch_size=16,
            prediction_cache_size=500,
            use_quantized_weights=True
        )
        
        # 初始化 AI 模型管理器
        manager = AIModelManager(
            model_dir=Path("./test_models")
        )
        
        # 初始化模型
        init_result = await manager.initialize_models(
            input_size=64,
            num_tools=8
        )
        
        print(f"✅ 初始化結果: {init_result['status']}")
        if init_result['status'] == 'success':
            print(f"📊 ScalableBioNet 參數: {init_result.get('scalable_net_params', 'N/A')}")
            print(f"🧠 RAG Agent 就緒: {init_result.get('bio_agent_ready', False)}")
            print(f"📅 模型版本: {init_result.get('model_version', 'N/A')}")
        else:
            print(f"❌ 初始化失敗: {init_result.get('error', 'Unknown error')}")
        
        # 獲取模型狀態
        status = await manager.get_model_status()
        print("\n📋 模型狀態:")
        for key, value in status.items():
            print(f"  • {key}: {value}")
        
        return True, manager
        
    except Exception as e:
        print(f"❌ 場景 1 失敗: {e}")
        import traceback
        print(traceback.format_exc())
        return False, None


async def test_scenario_2_decision_making(manager):
    """場景 2: AI 決策能力測試"""
    print("\n🎯 場景 2: AI 決策能力測試")
    print("=" * 50)
    
    try:
        # 測試場景列表
        test_queries = [
            "分析這個系統的安全漏洞",
            "執行滲透測試計畫",
            "檢測網路異常行為",
            "生成安全評估報告",
            "建議防護措施"
        ]
        
        decision_results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 測試 {i}: {query}")
            
            context = {
                "test_id": i,
                "environment": "test_lab",
                "timestamp": time.time()
            }
            
            # 執行決策
            start_time = time.time()
            result = await manager.make_decision(query, context, use_rag=False)
            decision_time = time.time() - start_time
            
            if result['status'] == 'success':
                confidence = result['result'].get('confidence', 0.0)
                print(f"  ✅ 決策成功 (耗時: {decision_time:.3f}s, 信心度: {confidence:.2f})")
                decision_results.append({
                    "query": query,
                    "success": True,
                    "confidence": confidence,
                    "time": decision_time
                })
            else:
                print(f"  ❌ 決策失敗: {result.get('error', 'Unknown error')}")
                decision_results.append({
                    "query": query,
                    "success": False,
                    "time": decision_time
                })
        
        # 統計結果
        success_count = sum(1 for r in decision_results if r['success'])
        avg_time = sum(r['time'] for r in decision_results) / len(decision_results)
        avg_confidence = sum(r.get('confidence', 0) for r in decision_results if r['success']) / max(success_count, 1)
        
        print(f"\n📊 決策測試統計:")
        print(f"  • 成功率: {success_count}/{len(test_queries)} ({success_count/len(test_queries)*100:.1f}%)")
        print(f"  • 平均耗時: {avg_time:.3f}s")
        print(f"  • 平均信心度: {avg_confidence:.2f}")
        
        return success_count == len(test_queries)
        
    except Exception as e:
        print(f"❌ 場景 2 失敗: {e}")
        return False


async def test_scenario_3_performance_optimization(manager):
    """場景 3: 性能優化測試"""
    print("\n🎯 場景 3: 性能優化測試")
    print("=" * 50)
    
    try:
        from aiva_core.ai_engine import OptimizedScalableBioNet, PerformanceConfig
        import numpy as np
        
        # 創建優化的神經網路
        config = PerformanceConfig(
            max_concurrent_tasks=20,
            batch_size=32,
            prediction_cache_size=1000,
            use_quantized_weights=True
        )
        
        optimized_net = OptimizedScalableBioNet(
            input_size=64,
            num_tools=8,
            config=config
        )
        
        print("🚀 測試批次預測性能...")
        
        # 準備測試數據
        batch_size = 50
        test_inputs = [np.random.randn(64) for _ in range(batch_size)]
        
        # 執行批次預測
        start_time = time.time()
        results = await optimized_net.predict_batch(test_inputs)
        batch_time = time.time() - start_time
        
        print(f"✅ 批次預測完成:")
        print(f"  • 批次大小: {batch_size}")
        print(f"  • 總耗時: {batch_time:.3f}s")
        print(f"  • 單個預測耗時: {batch_time/batch_size*1000:.1f}ms")
        print(f"  • 吞吐量: {batch_size/batch_time:.1f} predictions/s")
        
        # 獲取性能統計
        stats = optimized_net.get_performance_stats()
        print(f"\n📈 性能統計:")
        print(f"  • 總預測次數: {stats['predictions']}")
        print(f"  • 平均預測時間: {stats['avg_prediction_time']*1000:.1f}ms")
        print(f"  • 快取命中率: {stats['spiking_layer_cache']['hit_rate']:.2%}")
        print(f"  • 記憶體使用: {stats['memory_usage']['total_mb']:.1f}MB")
        
        # 測試記憶體優化
        print(f"\n🧹 執行記憶體優化...")
        optimization_result = optimized_net.optimize_memory()
        print(f"✅ 記憶體優化完成:")
        print(f"  • 快取已清空: {optimization_result['caches_cleared']}")
        print(f"  • GC 回收物件: {optimization_result['gc_collected']}")
        print(f"  • 優化後記憶體: {optimization_result['memory_usage']['total_mb']:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 場景 3 失敗: {e}")
        import traceback
        print(traceback.format_exc())
        return False


async def test_scenario_4_training_integration(manager):
    """場景 4: 訓練系統整合測試"""
    print("\n🎯 場景 4: 訓練系統整合測試")
    print("=" * 50)
    
    try:
        from aiva_core.learning import ScalableBioTrainingConfig
        
        # 準備訓練配置
        training_config = ScalableBioTrainingConfig(
            learning_rate=0.001,
            epochs=3,  # 較少的 epochs 以快速測試
            batch_size=16,
            early_stopping_patience=2
        )
        
        print("🎓 開始模型訓練測試...")
        
        # 模擬訓練數據
        training_samples = []
        for i in range(100):
            sample = type('Sample', (), {
                'context': f"test_context_{i}",
                'result': f"test_result_{i}",
                'score': 0.8 + (i % 20) * 0.01  # 模擬不同質量的樣本
            })()
            training_samples.append(sample)
        
        # 執行訓練
        start_time = time.time()
        training_result = await manager.train_models(
            training_data=training_samples,
            config=training_config,
            use_experience_samples=False
        )
        training_time = time.time() - start_time
        
        if training_result['status'] == 'success':
            print(f"✅ 訓練完成 (耗時: {training_time:.1f}s):")
            print(f"  • 模型版本: {training_result['model_version']}")
            print(f"  • 使用樣本: {training_result['samples_used']}")
            print(f"  • 模型路徑: {training_result['model_path']}")
            
            # 訓練結果細節
            training_details = training_result['training_results']
            print(f"  • 最終損失: {training_details['final_loss']:.4f}")
            print(f"  • 最終準確率: {training_details['final_accuracy']:.4f}")
            print(f"  • 訓練輪數: {training_details['epochs_trained']}")
            
            return True
        else:
            print(f"❌ 訓練失敗: {training_result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ 場景 4 失敗: {e}")
        import traceback
        print(traceback.format_exc())
        return False


async def test_scenario_5_stress_test(manager):
    """場景 5: 壓力測試"""
    print("\n🎯 場景 5: 高負載壓力測試")
    print("=" * 50)
    
    try:
        print("⚡ 執行並發決策測試...")
        
        # 創建大量並發決策任務
        concurrent_tasks = 50
        tasks = []
        
        for i in range(concurrent_tasks):
            query = f"安全分析任務 #{i+1}"
            context = {"task_id": i+1, "priority": "high"}
            task = manager.make_decision(query, context, use_rag=False)
            tasks.append(task)
        
        # 執行並發測試
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 分析結果
        successful_results = [r for r in results if not isinstance(r, Exception) and r.get('status') == 'success']
        failed_results = [r for r in results if isinstance(r, Exception) or r.get('status') != 'success']
        
        print(f"✅ 並發測試完成:")
        print(f"  • 總任務數: {concurrent_tasks}")
        print(f"  • 成功任務: {len(successful_results)}")
        print(f"  • 失敗任務: {len(failed_results)}")
        print(f"  • 成功率: {len(successful_results)/concurrent_tasks*100:.1f}%")
        print(f"  • 總耗時: {total_time:.2f}s")
        print(f"  • 平均耗時: {total_time/concurrent_tasks*1000:.1f}ms/task")
        print(f"  • 吞吐量: {concurrent_tasks/total_time:.1f} tasks/s")
        
        # 檢查是否有異常
        if failed_results:
            print(f"\n⚠️  發現 {len(failed_results)} 個失敗任務:")
            for i, result in enumerate(failed_results[:3]):  # 只顯示前3個
                if isinstance(result, Exception):
                    print(f"  • 異常 {i+1}: {type(result).__name__}: {result}")
                else:
                    print(f"  • 失敗 {i+1}: {result.get('error', 'Unknown error')}")
        
        return len(successful_results) / concurrent_tasks >= 0.8  # 80% 成功率為通過標準
        
    except Exception as e:
        print(f"❌ 場景 5 失敗: {e}")
        return False


async def run_complete_test_suite():
    """執行完整的靶場測試套件"""
    print("🚀 AIVA AI 系統靶場實測開始")
    print("=" * 70)
    print("測試範圍: AI 核心統一、性能優化、訓練整合、實戰場景")
    print("=" * 70)
    
    test_results = []
    manager = None
    
    # 場景 1: 基本初始化
    success_1, manager = await test_scenario_1_basic_initialization()
    test_results.append(("基本系統初始化", success_1))
    
    if not success_1 or not manager:
        print("\n❌ 基本初始化失敗，無法繼續後續測試")
        return
    
    # 場景 2: 決策能力
    success_2 = await test_scenario_2_decision_making(manager)
    test_results.append(("AI 決策能力", success_2))
    
    # 場景 3: 性能優化
    success_3 = await test_scenario_3_performance_optimization(manager)
    test_results.append(("性能優化", success_3))
    
    # 場景 4: 訓練整合
    success_4 = await test_scenario_4_training_integration(manager)
    test_results.append(("訓練系統整合", success_4))
    
    # 場景 5: 壓力測試
    success_5 = await test_scenario_5_stress_test(manager)
    test_results.append(("高負載壓力測試", success_5))
    
    # 最終報告
    print("\n" + "=" * 70)
    print("🏁 靶場實測結果總結")
    print("=" * 70)
    
    passed_tests = sum(1 for _, success in test_results if success)
    total_tests = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"{status} {test_name}")
    
    print(f"\n📊 整體測試結果: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("🎉 所有測試通過！AI 系統已準備好進行實戰部署")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  大部分測試通過，系統基本可用，但需要調整部分功能")
    else:
        print("❌ 多個測試失敗，需要進一步調試和優化")
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(run_complete_test_suite())
    except KeyboardInterrupt:
        print("\n⏹️  測試被用戶中斷")
    except Exception as e:
        print(f"\n💥 測試過程發生未預期錯誤: {e}")
        import traceback
        print(traceback.format_exc())