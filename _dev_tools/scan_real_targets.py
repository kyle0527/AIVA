"""
真實靶場滲透測試 - 按照 AIVA十大核心能力實戰指南.md 執行
"""
import asyncio
import sys
import os
from pathlib import Path

# 添加項目路徑
sys.path.append(str(Path(__file__).parent))

# 根據 Docker 截圖中的靶場信息
TARGETS = [
    "http://localhost:5432",  # aiva-postgres
    "http://localhost:3003",  # vigilant_shockle (bkimminict)
    "http://localhost:8080",  # laughing_jang (webgoat)
    "http://localhost:3001",  # ecstatic_ritchie (bkimminict)
    "http://localhost:3000",  # juice-shop-live
]

async def full_penetration_test(target: str):
    """
    完整的滲透測試流程 - 按照指南範例
    """
    print(f"\n{'='*80}")
    print(f"🎯 目標: {target}")
    print(f"{'='*80}\n")
    
    try:
        # 導入必要模組
        from services.scan.engines.python_engine.scan_orchestrator import ScanOrchestrator
        from services.scan.engines.python_engine.strategy_controller import StrategyController
        from services.scan.engines.python_engine.sensitive_data_scanner import SensitiveDataScanner
        from services.scan.engines.python_engine.scope_manager import ScopeManager
        # 模組整合: external_learning → cognitive_core/learning_system
        from services.core.aiva_core.cognitive_core.learning_system.experience_manager import ExperienceManager
        from aiva_common.schemas import Phase0StartPayload
        
        print("✅ 模組導入成功")
        
        # Step 1: Phase0 快速偵察
        print("\n🔍 Phase 0: 快速偵察...")
        orchestrator = ScanOrchestrator()
        
        import uuid
        request = Phase0StartPayload(
            scan_id=f"scan_{uuid.uuid4().hex[:10]}",  # 至少10字符
            targets=[target],
            max_depth=3
        )
        
        try:
            result = await orchestrator.execute_phase0(request)
            print(f"✅ URLs 發現: {result.urls_found}")
            print(f"✅ 表單發現: {result.forms_found}")
            print(f"✅ 技術棧: {result.tech_stack}")
            print(f"✅ 是否SPA: {result.is_spa}")
            
            # Step 2: 根據結果調整策略
            print("\n⚙️ 調整掃描策略...")
            controller = StrategyController(strategy="balanced")
            controller.adjust_from_phase0(phase0_summary={
                "urls_found": result.urls_found,
                "forms_found": result.forms_found,
                "endpoints_found": result.endpoints_found,
                "tech_stack": result.tech_stack,
                "is_spa": result.is_spa
            })
            params = controller.get_parameters()
            print(f"✅ Max pages: {params.max_pages}")
            print(f"✅ Dynamic scan: {params.enable_dynamic_scan}")
            
            # Step 3: 敏感資訊檢測
            print("\n🔎 檢測敏感資訊...")
            scanner = SensitiveDataScanner()
            if hasattr(result, 'html_content') and result.html_content:
                matches = scanner.scan_content(
                    content=result.html_content,
                    source_url=target,
                    content_type="html"
                )
                print(f"✅ 發現 {len(matches)} 個敏感資訊")
                for match in matches[:5]:  # 只顯示前5個
                    print(f"   - {match.type}: {match.severity}")
            
            # Step 4: 範圍內URL過濾
            print("\n📋 過濾範圍內 URL...")
            from services.aiva_common.schemas import ScanScope
            scope = ScanScope(
                allowed_hosts=[target.split('//')[1].split(':')[0]],
                include_subdomains=True
            )
            scope_manager = ScopeManager(scope)
            discovered_urls = result.urls_found if hasattr(result, 'urls_found') else []
            if isinstance(discovered_urls, int):
                print(f"✅ 發現 {discovered_urls} 個URL (總數)")
            else:
                in_scope_urls = scope_manager.filter_urls(discovered_urls)
                print(f"✅ 發現 {len(in_scope_urls)} 個範圍內URL")
            
            # Step 5: 記錄經驗
            print("\n📚 記錄掃描經驗...")
            experience_manager = ExperienceManager()
            exp_id = experience_manager.push(
                state={"target": target, "phase": "phase0"},
                action={"type": "reconnaissance", "depth": 3},
                next_state={"success": True, "urls_found": result.urls_found},
                reward=0.85,
                metadata={"execution_time": 10}
            )
            print(f"✅ 經驗記錄ID: {exp_id}")
            
            print(f"\n✅ {target} 掃描完成!")
            return True
            
        except Exception as scan_error:
            print(f"⚠️ 掃描過程錯誤: {scan_error}")
            print(f"   這可能是因為目標服務尚未完全啟動或配置問題")
            return False
            
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        print(f"   請確認模組路徑正確")
        return False
    except Exception as e:
        print(f"❌ 執行錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_capabilities():
    """
    測試個別能力 - 按照指南中的單獨使用方式
    """
    print("\n" + "="*80)
    print("🧪 測試個別能力")
    print("="*80)
    
    try:
        # 測試 1: 能力搜索
        print("\n1️⃣ 測試 search_capabilities...")
        from services.core.aiva_core.core_capabilities.capability_registry import get_capability_registry
        
        registry = get_capability_registry()
        xss_capabilities = registry.search_capabilities("xss")
        
        print(f"✅ 找到 {len(xss_capabilities)} 個 XSS 相關能力:")
        for cap in xss_capabilities[:3]:
            print(f"   - {cap.name} ({cap.module})")
        
        # 測試 2: 範圍檢查
        print("\n2️⃣ 測試 is_in_scope...")
        from services.scan.engines.python_engine.scope_manager import ScopeManager
        from services.aiva_common.schemas import ScanScope
        
        scope = ScanScope(
            allowed_hosts=["localhost", "127.0.0.1"],
            include_subdomains=True
        )
        manager = ScopeManager(scope)
        
        test_urls = [
            "http://localhost:3000/page",
            "http://127.0.0.1:3000/api",
            "http://external.com/ads"
        ]
        
        for url in test_urls:
            result = manager.is_in_scope(url)
            print(f"   {url}: {'✅ 在範圍內' if result else '❌ 範圍外'}")
        
        # 測試 3: URL 隊列管理
        print("\n3️⃣ 測試 URL 隊列管理...")
        from services.scan.engines.python_engine.core_crawling_engine.url_queue_manager import URLQueueManager
        
        queue = URLQueueManager(max_depth=3)
        queue.add(url="http://localhost:3000", parent_url=None, depth=0)
        
        added = queue.add_batch(
            urls=["http://localhost:3000/page1", "http://localhost:3000/page2"],
            parent_url="http://localhost:3000",
            depth=1
        )
        
        stats = queue.get_statistics()
        print(f"   ✅ 添加了 {added} 個URL")
        print(f"   ✅ 待處理: {stats['queued_urls']}")
        print(f"   ✅ 已發現: {stats['seen_urls']}")
        
        # 測試 4: 經驗管理
        print("\n4️⃣ 測試經驗管理...")
        # 模組整合: external_learning → cognitive_core/learning_system
        from services.core.aiva_core.cognitive_core.learning_system.experience_manager import ExperienceManager
        
        exp_manager = ExperienceManager()
        exp_id = exp_manager.push(
            state={"target": "test", "action": "scan"},
            action={"type": "test", "payload": "test_payload"},
            next_state={"success": True},
            reward=0.9,
            metadata={"test": True}
        )
        print(f"   ✅ 經驗記錄成功: {exp_id}")
        
        # 嘗試採樣
        try:
            batch = exp_manager.sample(batch_size=1)
            print(f"   ✅ 採樣成功: {len(batch)} 個經驗")
        except Exception as sample_error:
            print(f"   ⚠️ 採樣失敗 (可能資料庫為空): {sample_error}")
        
        print("\n✅ 所有個別能力測試完成!")
        return True
        
    except Exception as e:
        print(f"❌ 能力測試錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主執行流程"""
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     AIVA 真實靶場滲透測試                                   ║
    ║     按照: AIVA十大核心能力實戰指南.md                        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # 先測試個別能力
    print("\n📌 階段 1: 測試個別能力")
    await test_individual_capabilities()
    
    # 再對靶場進行完整測試
    print("\n\n📌 階段 2: 對靶場進行完整滲透測試")
    
    success_count = 0
    for target in TARGETS:
        result = await full_penetration_test(target)
        if result:
            success_count += 1
        await asyncio.sleep(2)  # 避免請求過快
    
    print(f"\n{'='*80}")
    print(f"🎉 測試完成!")
    print(f"✅ 成功: {success_count}/{len(TARGETS)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    asyncio.run(main())
