#!/usr/bin/env python3
"""
AIVA 全功能 Worker 實戰測試

實際執行所有 Worker 模組對靶場進行真實測試:
- SSRF Worker: 伺服器端請求偽造檢測
- SQLi Worker: SQL 注入檢測 (5 引擎)
- XSS Worker: 跨站腳本檢測 (Reflected/DOM/Blind)
- IDOR Worker: 不安全直接對象引用檢測
- GraphQL AuthZ Worker: GraphQL 權限繞過檢測
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# 添加路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'services', 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

# 測試配置
TARGET_URL = "http://localhost:3000"
RESULTS = {}

def print_section(title: str):
    """打印分隔段落"""
    print("\n" + "=" * 70)
    print(f"🎯 {title}")
    print("=" * 70)

def print_subsection(title: str):
    """打印子段落"""
    print(f"\n{'─' * 60}")
    print(f"📋 {title}")
    print(f"{'─' * 60}")

async def test_ssrf_worker():
    """測試 SSRF Worker 實際執行"""
    print_section("SSRF Worker 實戰測試")
    
    start_time = time.time()
    
    try:
        from services.aiva_common.schemas import Task, Target
        from services.aiva_common.enums import TaskType, TaskStatus, ScanStrategy
        
        # 導入 SSRF Worker
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'features', 'function_ssrf'))
        from services.features.function_ssrf.worker import SsrfWorkerService
        
        print("✅ SSRF Worker 模組導入成功")
        
        # 創建測試任務
        task = Task(
            task_id="ssrf_test_001",
            scan_id="scan_full_test",
            task_type=TaskType.FUNCTION_SSRF,
            target=Target(
                url=f"{TARGET_URL}/api/fetch",
                method="POST",
                headers={"Content-Type": "application/json"}
            ),
            strategy=ScanStrategy.DEEP,
            priority=1
        )
        
        print(f"✅ 測試任務創建: {task.task_id}")
        print(f"   目標: {task.target.url}")
        print(f"   方法: {task.target.method}")
        
        # 初始化 Worker
        worker = SsrfWorkerService()
        print("✅ SSRF Worker 初始化完成")
        
        # 執行測試
        print("\n🔍 開始 SSRF 檢測...")
        result = await worker.process_task(task)
        
        elapsed = time.time() - start_time
        
        # 輸出結果
        print(f"\n✅ SSRF 測試完成 (耗時: {elapsed:.2f}s)")
        print(f"   發現漏洞數: {len(result.get('findings', []))}")
        
        if result.get('statistics_summary'):
            stats = result['statistics_summary']
            print(f"\n📊 統計數據:")
            print(f"   請求總數: {stats.get('total_requests', 0)}")
            print(f"   成功請求: {stats.get('successful_requests', 0)}")
            print(f"   失敗請求: {stats.get('failed_requests', 0)}")
            print(f"   OAST 探針: {stats.get('oast_probes_sent', 0)}")
            print(f"   OAST 回調: {stats.get('oast_callbacks_received', 0)}")
            
            if stats.get('module_specific'):
                mod = stats['module_specific']
                print(f"   測試向量: {mod.get('total_vectors_tested', 0)}")
                print(f"   內部檢測: {mod.get('internal_detection_tests', 0)}")
                print(f"   OAST 測試: {mod.get('oast_tests', 0)}")
        
        RESULTS['ssrf'] = {
            'status': 'success',
            'elapsed': elapsed,
            'findings': len(result.get('findings', [])),
            'statistics': result.get('statistics_summary', {})
        }
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ SSRF Worker 測試失敗: {e}")
        import traceback
        print(traceback.format_exc())
        
        RESULTS['ssrf'] = {
            'status': 'failed',
            'elapsed': elapsed,
            'error': str(e)
        }
        return False

async def test_sqli_worker():
    """測試 SQLi Worker 實際執行"""
    print_section("SQLi Worker 實戰測試 (5 引擎)")
    
    start_time = time.time()
    
    try:
        from services.aiva_common.schemas import Task, Target
        from services.aiva_common.enums import TaskType, ScanStrategy
        
        # 導入 SQLi Worker
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'features', 'function_sqli'))
        from services.features.function_sqli.worker import SqliWorkerService
        
        print("✅ SQLi Worker 模組導入成功")
        
        # 創建測試任務
        task = Task(
            task_id="sqli_test_001",
            scan_id="scan_full_test",
            task_type=TaskType.FUNCTION_SQLI,
            target=Target(
                url=f"{TARGET_URL}/search?q=test",
                method="GET"
            ),
            strategy=ScanStrategy.DEEP,
            priority=1
        )
        
        print(f"✅ 測試任務創建: {task.task_id}")
        print(f"   目標: {task.target.url}")
        
        # 初始化 Worker
        worker = SqliWorkerService()
        print("✅ SQLi Worker 初始化完成")
        
        # 執行測試
        print("\n🔍 開始 SQL 注入檢測 (5 引擎)...")
        print("   引擎: Error-based, Boolean-based, Time-based, Union-based, OOB")
        result = await worker.process_task_dict(task)
        
        elapsed = time.time() - start_time
        
        # 輸出結果
        print(f"\n✅ SQLi 測試完成 (耗時: {elapsed:.2f}s)")
        print(f"   發現漏洞數: {len(result.get('findings', []))}")
        
        if result.get('statistics_summary'):
            stats = result['statistics_summary']
            print(f"\n📊 統計數據:")
            print(f"   請求總數: {stats.get('total_requests', 0)}")
            print(f"   成功請求: {stats.get('successful_requests', 0)}")
            print(f"   Payload 測試: {stats.get('payloads_tested', 0)}")
            
            if stats.get('module_specific'):
                mod = stats['module_specific']
                print(f"\n🔧 引擎執行狀態:")
                print(f"   Error Detection: {'✅' if mod.get('error_detection_enabled') else '❌'}")
                print(f"   Boolean Detection: {'✅' if mod.get('boolean_detection_enabled') else '❌'}")
                print(f"   Time Detection: {'✅' if mod.get('time_detection_enabled') else '❌'}")
                print(f"   Union Detection: {'✅' if mod.get('union_detection_enabled') else '❌'}")
                print(f"   OOB Detection: {'✅' if mod.get('oob_detection_enabled') else '❌'}")
                print(f"   策略: {mod.get('strategy', 'N/A')}")
        
        RESULTS['sqli'] = {
            'status': 'success',
            'elapsed': elapsed,
            'findings': len(result.get('findings', [])),
            'statistics': result.get('statistics_summary', {})
        }
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ SQLi Worker 測試失敗: {e}")
        import traceback
        print(traceback.format_exc())
        
        RESULTS['sqli'] = {
            'status': 'failed',
            'elapsed': elapsed,
            'error': str(e)
        }
        return False

async def test_xss_worker():
    """測試 XSS Worker 實際執行"""
    print_section("XSS Worker 實戰測試 (Reflected/DOM/Blind)")
    
    start_time = time.time()
    
    try:
        from services.aiva_common.schemas import Task, Target
        from services.aiva_common.enums import TaskType, ScanStrategy
        
        # 導入 XSS Worker
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'features', 'function_xss'))
        from services.features.function_xss.worker import XssWorkerService
        
        print("✅ XSS Worker 模組導入成功")
        
        # 創建測試任務
        task = Task(
            task_id="xss_test_001",
            scan_id="scan_full_test",
            task_type=TaskType.FUNCTION_XSS,
            target=Target(
                url=f"{TARGET_URL}/profile?name=test",
                method="GET"
            ),
            strategy=ScanStrategy.DEEP,
            priority=1
        )
        
        print(f"✅ 測試任務創建: {task.task_id}")
        print(f"   目標: {task.target.url}")
        
        # 初始化 Worker
        worker = XssWorkerService()
        print("✅ XSS Worker 初始化完成")
        
        # 執行測試
        print("\n🔍 開始 XSS 檢測...")
        print("   類型: Reflected XSS, DOM XSS, Blind XSS (OAST)")
        result = await worker.process_task(task)
        
        elapsed = time.time() - start_time
        
        # 輸出結果
        print(f"\n✅ XSS 測試完成 (耗時: {elapsed:.2f}s)")
        print(f"   發現漏洞數: {len(result.get('findings', []))}")
        
        if result.get('statistics_summary'):
            stats = result['statistics_summary']
            print(f"\n📊 統計數據:")
            print(f"   請求總數: {stats.get('total_requests', 0)}")
            print(f"   成功請求: {stats.get('successful_requests', 0)}")
            print(f"   Payload 測試: {stats.get('payloads_tested', 0)}")
            print(f"   OAST 探針: {stats.get('oast_probes_sent', 0)}")
            
            if stats.get('module_specific'):
                mod = stats['module_specific']
                print(f"\n🔧 檢測功能:")
                print(f"   Reflected XSS 測試: {mod.get('reflected_xss_tests', 0)}")
                print(f"   DOM XSS 提升: {mod.get('dom_xss_escalations', 0)}")
                print(f"   Blind XSS: {'✅' if mod.get('blind_xss_enabled') else '❌'}")
                print(f"   DOM 測試: {'✅' if mod.get('dom_testing_enabled') else '❌'}")
                print(f"   Stored XSS: {'✅' if mod.get('stored_xss_tested') else '❌'}")
        
        RESULTS['xss'] = {
            'status': 'success',
            'elapsed': elapsed,
            'findings': len(result.get('findings', [])),
            'statistics': result.get('statistics_summary', {})
        }
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ XSS Worker 測試失敗: {e}")
        import traceback
        print(traceback.format_exc())
        
        RESULTS['xss'] = {
            'status': 'failed',
            'elapsed': elapsed,
            'error': str(e)
        }
        return False

async def test_idor_worker():
    """測試 IDOR Worker 實際執行"""
    print_section("IDOR Worker 實戰測試 (權限檢測)")
    
    start_time = time.time()
    
    try:
        from services.aiva_common.schemas import Task, Target
        from services.aiva_common.enums import TaskType, ScanStrategy
        
        # 導入 IDOR Worker
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'features', 'function_idor'))
        from services.features.function_idor.worker import IdorWorkerService
        
        print("✅ IDOR Worker 模組導入成功")
        
        # 創建測試任務
        task = Task(
            task_id="idor_test_001",
            scan_id="scan_full_test",
            task_type=TaskType.FUNCTION_IDOR,
            target=Target(
                url=f"{TARGET_URL}/api/user/123",
                method="GET"
            ),
            strategy=ScanStrategy.DEEP,
            priority=1
        )
        
        print(f"✅ 測試任務創建: {task.task_id}")
        print(f"   目標: {task.target.url}")
        
        # 初始化 Worker
        worker = IdorWorkerService()
        print("✅ IDOR Worker 初始化完成")
        
        # 執行測試
        print("\n🔍 開始 IDOR 檢測...")
        print("   類型: 水平權限提升 (BOLA) + 垂直權限提升 (BFLA)")
        result = await worker.process_task(task)
        
        elapsed = time.time() - start_time
        
        # 輸出結果
        print(f"\n✅ IDOR 測試完成 (耗時: {elapsed:.2f}s)")
        
        if isinstance(result, dict):
            findings = result.get('findings', [])
            print(f"   發現漏洞數: {len(findings)}")
            
            if result.get('statistics_summary'):
                stats = result['statistics_summary']
                print(f"\n📊 統計數據:")
                print(f"   請求總數: {stats.get('total_requests', 0)}")
                print(f"   成功請求: {stats.get('successful_requests', 0)}")
                
                if stats.get('module_specific'):
                    mod = stats['module_specific']
                    print(f"\n🔧 檢測詳情:")
                    print(f"   水平測試 (BOLA): {mod.get('horizontal_tests', 0)}")
                    print(f"   垂直測試 (BFLA): {mod.get('vertical_tests', 0)}")
                    print(f"   資源 ID 變異: {mod.get('resource_id_mutations', 0)}")
        else:
            print(f"   測試執行完成")
        
        RESULTS['idor'] = {
            'status': 'success',
            'elapsed': elapsed,
            'findings': len(result.get('findings', [])) if isinstance(result, dict) else 0
        }
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ IDOR Worker 測試失敗: {e}")
        import traceback
        print(traceback.format_exc())
        
        RESULTS['idor'] = {
            'status': 'failed',
            'elapsed': elapsed,
            'error': str(e)
        }
        return False

async def test_graphql_authz_worker():
    """測試 GraphQL AuthZ Worker 實際執行"""
    print_section("GraphQL AuthZ Worker 實戰測試")
    
    start_time = time.time()
    
    try:
        from services.aiva_common.schemas import Task, Target
        from services.aiva_common.enums import TaskType, ScanStrategy
        
        # 導入 GraphQL Worker
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'services', 'features', 'graphql_authz'))
        from services.features.graphql_authz.worker import GraphqlAuthzWorkerService
        
        print("✅ GraphQL AuthZ Worker 模組導入成功")
        
        # 創建測試任務
        task = Task(
            task_id="graphql_test_001",
            scan_id="scan_full_test",
            task_type=TaskType.FUNCTION_GRAPHQL_AUTHZ,
            target=Target(
                url=f"{TARGET_URL}/graphql",
                method="POST"
            ),
            strategy=ScanStrategy.DEEP,
            priority=1
        )
        
        print(f"✅ 測試任務創建: {task.task_id}")
        print(f"   目標: {task.target.url}")
        
        # 初始化 Worker
        worker = GraphqlAuthzWorkerService()
        print("✅ GraphQL AuthZ Worker 初始化完成")
        
        # 執行測試
        print("\n🔍 開始 GraphQL 權限繞過檢測...")
        result = await worker.process_task(task)
        
        elapsed = time.time() - start_time
        
        # 輸出結果
        print(f"\n✅ GraphQL 測試完成 (耗時: {elapsed:.2f}s)")
        
        if isinstance(result, dict):
            findings = result.get('findings', [])
            print(f"   發現漏洞數: {len(findings)}")
        
        RESULTS['graphql_authz'] = {
            'status': 'success',
            'elapsed': elapsed,
            'findings': len(result.get('findings', [])) if isinstance(result, dict) else 0
        }
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ GraphQL AuthZ Worker 測試失敗: {e}")
        import traceback
        print(traceback.format_exc())
        
        RESULTS['graphql_authz'] = {
            'status': 'failed',
            'elapsed': elapsed,
            'error': str(e)
        }
        return False

async def test_ai_core_system():
    """測試 AI 核心系統"""
    print_section("AI 核心系統實戰測試")
    
    try:
        from services.core.aiva_core.ai_engine import AIModelManager, PerformanceConfig
        
        # 初始化
        config = PerformanceConfig(
            max_concurrent_tasks=20,
            batch_size=32,
            prediction_cache_size=1000
        )
        
        manager = AIModelManager(model_dir=Path("./test_models"))
        
        print("🔧 初始化 AI 模型...")
        init_result = await manager.initialize_models(input_size=64, num_tools=10)
        
        print(f"✅ 初始化: {init_result['status']}")
        print(f"   ScalableBioNet 參數: {init_result.get('scalable_net_params', 0):,}")
        print(f"   RAG Agent: {'✅' if init_result.get('bio_agent_ready') else '❌'}")
        
        # 決策測試
        print("\n🧠 測試 AI 決策能力...")
        test_queries = [
            "檢測到 SQL 注入漏洞，建議修復方案",
            "分析 SSRF 攻擊模式",
            "評估 XSS 風險等級",
            "生成滲透測試報告"
        ]
        
        success_count = 0
        for i, query in enumerate(test_queries, 1):
            try:
                result = await manager.make_decision(query)
                if result.get('success'):
                    success_count += 1
                    print(f"   ✅ 測試 {i}: {query[:20]}... (信心度: {result.get('confidence', 0):.2f})")
            except:
                print(f"   ❌ 測試 {i}: 失敗")
        
        print(f"\n📊 AI 決策成功率: {success_count}/{len(test_queries)} ({success_count/len(test_queries)*100:.0f}%)")
        
        # 批次測試
        print("\n⚡ 測試批次預測性能...")
        batch_result = await manager.predict_batch([f"query_{i}" for i in range(100)])
        print(f"   ✅ 批次大小: 100")
        print(f"   ✅ 預測完成")
        
        RESULTS['ai_core'] = {
            'status': 'success',
            'decision_success_rate': success_count / len(test_queries),
            'model_params': init_result.get('scalable_net_params', 0)
        }
        
        return True
        
    except Exception as e:
        print(f"❌ AI 核心測試失敗: {e}")
        import traceback
        print(traceback.format_exc())
        
        RESULTS['ai_core'] = {
            'status': 'failed',
            'error': str(e)
        }
        return False

async def generate_final_report():
    """生成最終測試報告"""
    print_section("📊 全功能測試總結報告")
    
    total_tests = len(RESULTS)
    successful_tests = sum(1 for r in RESULTS.values() if r.get('status') == 'success')
    
    print(f"\n總測試模組: {total_tests}")
    print(f"成功模組: {successful_tests}")
    print(f"失敗模組: {total_tests - successful_tests}")
    print(f"成功率: {successful_tests/total_tests*100:.1f}%")
    
    print("\n" + "─" * 70)
    print("各模組詳細結果:")
    print("─" * 70)
    
    for module, result in RESULTS.items():
        status_icon = "✅" if result.get('status') == 'success' else "❌"
        print(f"\n{status_icon} {module.upper()}")
        print(f"   狀態: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'success':
            if 'elapsed' in result:
                print(f"   耗時: {result['elapsed']:.2f}s")
            if 'findings' in result:
                print(f"   發現漏洞: {result['findings']}")
            if 'statistics' in result:
                stats = result['statistics']
                if stats.get('total_requests'):
                    print(f"   請求總數: {stats['total_requests']}")
        else:
            if 'error' in result:
                print(f"   錯誤: {result['error'][:100]}...")
    
    # 保存報告
    report_file = Path("aiva_full_worker_test_report.json")
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'target': TARGET_URL,
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': successful_tests / total_tests,
        'results': RESULTS
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 完整報告已保存: {report_file.absolute()}")
    
    print("\n" + "=" * 70)
    if successful_tests == total_tests:
        print("🎉 所有功能測試通過! AIVA 系統全功能運作正常!")
    elif successful_tests >= total_tests * 0.8:
        print("✅ 大部分功能測試通過! AIVA 系統基本可用!")
    else:
        print("⚠️  部分功能需要調整，請查看詳細報告")
    print("=" * 70)

async def main():
    """主測試流程"""
    print("=" * 70)
    print("🚀 AIVA 全功能 Worker 實戰測試開始")
    print("=" * 70)
    print(f"目標靶場: {TARGET_URL}")
    print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 執行所有測試
    await test_ssrf_worker()
    await asyncio.sleep(1)
    
    await test_sqli_worker()
    await asyncio.sleep(1)
    
    await test_xss_worker()
    await asyncio.sleep(1)
    
    await test_idor_worker()
    await asyncio.sleep(1)
    
    await test_graphql_authz_worker()
    await asyncio.sleep(1)
    
    await test_ai_core_system()
    
    # 生成報告
    await generate_final_report()

if __name__ == "__main__":
    asyncio.run(main())

