"""
最小化協調器測試 - 直接驗證協調器能否調用各引擎
"""
import asyncio
import sys
from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator


async def main():
    print("="*80)
    print("Test 1: Initialize Coordinator")
    print("="*80)
    
    coordinator = MultiEngineCoordinator()
    await coordinator.initialize()
    
    print(f"\nAvailable engines: {[e.value for e in coordinator.available_engines]}")
    
    # Test 2: Execute Phase 1 with Go engine
    print("\n" + "="*80)
    print("Test 2: Execute Phase 1 with Go Engine")
    print("="*80)
    
    result = await coordinator.execute_phase1(
        scan_id="scan_test_go",  # 必須以 scan_ 開頭
        targets=["http://localhost:3000"],
        selected_engines=["go"],
        max_depth=3,
        max_urls=100
    )
    
    print(f"\nGo Engine Result:")
    print(f"  - Status: {result.status}")
    print(f"  - Assets: {len(result.assets)}")
    print(f"  - URLs: {result.summary.urls_found}")
    print(f"  - Forms: {result.summary.forms_found}")
    print(f"  - APIs: {result.summary.apis_found}")
    
    if result.assets:
        print(f"\n  First 3 assets:")
        for i, asset in enumerate(result.assets[:3], 1):
            print(f"    {i}. [{asset.type}] {asset.value[:60]}")
    
    # Test 3: Execute Phase 1 with Rust engine
    print("\n" + "="*80)
    print("Test 3: Execute Phase 1 with Rust Engine")
    print("="*80)
    
    result = await coordinator.execute_phase1(
        scan_id="test_rust",
        targets=["http://localhost:3000"],
        selected_engines=["rust"],
        max_depth=3,
        max_urls=100
    )
    
    print(f"\nRust Engine Result:")
    print(f"  - Status: {result.status}")
    print(f"  - Assets: {len(result.assets)}")
    print(f"  - URLs: {result.summary.urls_found}")
    print(f"  - Forms: {result.summary.forms_found}")
    print(f"  - APIs: {result.summary.apis_found}")
    
    if result.assets:
        print(f"\n  First 3 assets:")
        for i, asset in enumerate(result.assets[:3], 1):
            print(f"    {i}. [{asset.type}] {asset.value[:60]}")
    
    # Test 4: Execute Phase 1 with TypeScript engine
    print("\n" + "="*80)
    print("Test 4: Execute Phase 1 with TypeScript Engine")
    print("="*80)
    
    result = await coordinator.execute_phase1(
        scan_id="test_typescript",
        targets=["http://localhost:3000"],
        selected_engines=["typescript"],
        max_depth=3,
        max_urls=100
    )
    
    print(f"\nTypeScript Engine Result:")
    print(f"  - Status: {result.status}")
    print(f"  - Assets: {len(result.assets)}")
    print(f"  - URLs: {result.summary.urls_found}")
    print(f"  - Forms: {result.summary.forms_found}")
    print(f"  - APIs: {result.summary.apis_found}")
    
    if result.assets:
        print(f"\n  First 3 assets:")
        for i, asset in enumerate(result.assets[:3], 1):
            print(f"    {i}. [{asset.type}] {asset.value[:60]}")
    
    # Test 5: Execute Phase 1 with Python engine
    print("\n" + "="*80)
    print("Test 5: Execute Phase 1 with Python Engine")
    print("="*80)
    
    result = await coordinator.execute_phase1(
        scan_id="test_python",
        targets=["http://localhost:3000"],
        selected_engines=["python"],
        max_depth=3,
        max_urls=100
    )
    
    print(f"\nPython Engine Result:")
    print(f"  - Status: {result.status}")
    print(f"  - Assets: {len(result.assets)}")
    print(f"  - URLs: {result.summary.urls_found}")
    print(f"  - Forms: {result.summary.forms_found}")
    print(f"  - APIs: {result.summary.apis_found}")
    
    if result.assets:
        print(f"\n  First 3 assets:")
        for i, asset in enumerate(result.assets[:3], 1):
            print(f"    {i}. [{asset.type}] {asset.value[:60]}")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
