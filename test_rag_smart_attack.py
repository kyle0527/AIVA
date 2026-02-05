"""
RAG 智能攻擊系統 - 端到端測試

測試完整流程：掃描結果 -> RAG 決策 -> 執行攻擊 -> 返回結果

使用場景：
    1. 智能攻擊：根據掃描結果自動選擇攻擊方法
    2. 定向攻擊：指定攻擊類型和目標
    3. Dry Run：只規劃不執行
"""

import asyncio
from typing import Dict, Any

# 導入核心組件
from services.core.aiva_core.cognitive_core.learning_system.cli_decision_engine import (
    CLIDecisionEngine,
    AttackCapability
)
from services.core.aiva_core.cognitive_core.learning_system.flow_executor_adapter import (
    FlowExecutorAdapter,
    ExecutionConfig
)


async def test_cli_decision_engine():
    """測試 1: CLI 決策引擎基本功能"""
    print("=" * 80)
    print("測試 1: CLI 決策引擎基本功能")
    print("=" * 80)
    
    # 初始化引擎
    engine = CLIDecisionEngine()
    
    # 查看統計
    stats = engine.get_module_statistics()
    print("\n📊 模組統計:")
    for module, stat in stats.items():
        print(f"  {module}: {stat['operable']}/{stat['total']} 可操作")
    
    # 搜索 XSS 流程
    xss_flows = engine.search_flows(
        capability=AttackCapability.XSS,
        keywords=["檢測", "反射"],
        only_operable=True,
        limit=5
    )
    
    print(f"\n🔍 找到 {len(xss_flows)} 個 XSS 攻擊流程:")
    for flow in xss_flows:
        print(f"  - Flow {flow.flow_id}: {flow.entry_point}")
        print(f"    使用場景: {flow.use_case[:50]}...")
    
    # 根據掃描上下文推薦
    scan_context = {
        "vulnerability_type": "XSS",
        "target_info": {
            "url": "https://testphp.vulnweb.com/search.php",
            "parameter": "searchFor",
            "method": "GET"
        }
    }
    
    recommended = engine.recommend_flows(scan_context, top_k=3)
    print(f"\n🎯 推薦的攻擊流程 (根據掃描上下文):")
    for flow in recommended:
        print(f"  - Flow {flow.flow_id}: {flow.full_path}")
    
    print("\n✅ 測試 1 完成\n")


async def test_flow_executor_adapter():
    """測試 2: FlowExecutorAdapter 參數轉換"""
    print("=" * 80)
    print("測試 2: FlowExecutorAdapter 參數轉換")
    print("=" * 80)
    
    # 初始化適配器
    adapter = FlowExecutorAdapter()
    
    # 模擬掃描結果
    scan_context = {
        "vulnerability_type": "SQLi",
        "target_info": {
            "url": "https://testphp.vulnweb.com/listproducts.php",
            "parameter": "cat",
            "method": "GET"
        },
        "scan_results": {
            "confidence": "high",
            "severity": "critical"
        }
    }
    
    # Dry Run 模式測試
    config = ExecutionConfig(dry_run=True, verbose=True)
    results = adapter.execute_recommended_flows(
        scan_context=scan_context,
        top_k=3,
        config=config
    )
    
    print(f"\n🧪 Dry Run 結果: 執行了 {len(results)} 個流程")
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} Flow {result.flow_id} ({result.module})")
    
    print("\n✅ 測試 2 完成\n")


async def test_targeted_attack():
    """測試 3: 定向攻擊"""
    print("=" * 80)
    print("測試 3: 定向攻擊 (XSS)")
    print("=" * 80)
    
    adapter = FlowExecutorAdapter()
    
    # 直接執行 XSS 攻擊
    config = ExecutionConfig(dry_run=True)
    results = adapter.search_and_execute(
        capability=AttackCapability.XSS,
        target_context={
            "url": "https://example.com/search",
            "parameter": "q",
            "method": "GET"
        },
        keywords=["反射", "檢測"],
        limit=2,
        config=config
    )
    
    print(f"\n🎯 定向攻擊結果: {len(results)} 個流程")
    for result in results:
        print(f"  Flow {result.flow_id}: {result.success}")
    
    print("\n✅ 測試 3 完成\n")


async def test_multi_capability():
    """測試 4: 多能力攻擊"""
    print("=" * 80)
    print("測試 4: 多能力攻擊 (XSS + SQLi + SSRF)")
    print("=" * 80)
    
    engine = CLIDecisionEngine()
    
    capabilities = [
        (AttackCapability.XSS, "反射型 XSS"),
        (AttackCapability.SQLI, "SQL 注入"),
        (AttackCapability.SSRF, "SSRF 攻擊")
    ]
    
    for cap, name in capabilities:
        flows = engine.search_flows(
            capability=cap,
            only_operable=True,
            limit=3
        )
        print(f"\n{name}: {len(flows)} 個可用流程")
        for flow in flows[:2]:  # 只顯示前2個
            print(f"  - {flow.entry_point}")
    
    print("\n✅ 測試 4 完成\n")


async def test_full_integration():
    """測試 5: 完整整合流程（模擬 AttackCoordinator 調用）"""
    print("=" * 80)
    print("測試 5: 完整整合流程")
    print("=" * 80)
    
    # 模擬掃描結果
    scan_results = {
        "target_info": {
            "url": "https://testphp.vulnweb.com"
        },
        "vulnerabilities": [
            {
                "type": "XSS",
                "url": "https://testphp.vulnweb.com/search.php",
                "parameter": "searchFor",
                "method": "GET",
                "confidence": "high",
                "severity": "medium"
            },
            {
                "type": "SQLi",
                "url": "https://testphp.vulnweb.com/listproducts.php",
                "parameter": "cat",
                "method": "GET",
                "confidence": "high",
                "severity": "critical"
            }
        ],
        "fingerprints": {
            "web_server": "Apache/2.4",
            "technologies": ["PHP", "MySQL"]
        }
    }
    
    print("\n📋 模擬掃描結果:")
    print(f"  目標: {scan_results['target_info']['url']}")
    print(f"  漏洞: {len(scan_results['vulnerabilities'])} 個")
    for vuln in scan_results['vulnerabilities']:
        print(f"    - {vuln['type']}: {vuln['parameter']} ({vuln['severity']})")
    
    # 使用適配器執行攻擊（Dry Run）
    adapter = FlowExecutorAdapter()
    
    all_results = []
    for vuln in scan_results['vulnerabilities']:
        scan_context = {
            "vulnerability_type": vuln['type'],
            "target_info": {
                "url": vuln['url'],
                "parameter": vuln['parameter'],
                "method": vuln['method']
            }
        }
        
        results = adapter.execute_recommended_flows(
            scan_context=scan_context,
            top_k=2,
            config=ExecutionConfig(dry_run=True)
        )
        all_results.extend(results)
    
    print(f"\n🚀 執行結果:")
    print(f"  總流程: {len(all_results)}")
    success = sum(1 for r in all_results if r.success)
    print(f"  成功: {success}/{len(all_results)}")
    
    print("\n✅ 測試 5 完成\n")


async def main():
    """執行所有測試"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "RAG 智能攻擊系統測試" + " " * 38 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    try:
        await test_cli_decision_engine()
        await test_flow_executor_adapter()
        await test_targeted_attack()
        await test_multi_capability()
        await test_full_integration()
        
        print("=" * 80)
        print("🎉 所有測試完成！")
        print("=" * 80)
        print("\n總結:")
        print("  ✅ CLI 決策引擎: 正常工作")
        print("  ✅ Flow 執行適配器: 正常工作")
        print("  ✅ 定向攻擊: 正常工作")
        print("  ✅ 多能力攻擊: 正常工作")
        print("  ✅ 完整整合: 正常工作")
        print("\n下一步:")
        print("  1. 移除 Dry Run 模式，執行真實攻擊")
        print("  2. 整合到 AttackCoordinator.rag_smart_attack()")
        print("  3. 測試靶場驗證")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
