"""
協調器驅動引擎驗證腳本
目的: 驗證協調器能否實際驅動各語言引擎,並讓靶場產生反應

測試重點:
1. Phase 0: Rust 引擎是否被協調器成功調用
2. Phase 1 單引擎: Python/Go 引擎是否被成功驅動
3. Phase 1 雙引擎: 是否能同時驅動兩個引擎
4. 靶場反應: 引擎是否真的訪問了靶場 (檢查資產內容)

參考各語言指南:
- Rust: services/scan/engines/rust_engine/USAGE_GUIDE.md
- Go: services/scan/engines/go_engine/USAGE_GUIDE.md
- Python: services/scan/engines/python_engine/README.md
"""

import asyncio
import time
from services.aiva_common.command_center import get_command_center
from services.aiva_common.schemas import AICommand, CommandType
from services.scan.command_handler import ScanCommandHandler


class CoordinatorEngineValidator:
    """協調器引擎驅動驗證器"""
    
    def __init__(self):
        self.command_center = get_command_center()
        self.scan_handler = ScanCommandHandler()
        self.command_center.register_module("scan", self.scan_handler)
        self.target = "http://localhost:3000"  # Juice Shop
        
    async def test_phase0_rust(self):
        """測試 1: Phase 0 協調器驅動 Rust 引擎"""
        print("\n" + "="*80)
        print("測試 1: Phase 0 - 協調器驅動 Rust 引擎")
        print("="*80)
        
        command = AICommand(
            command_id="validate_phase0_rust",
            command_type=CommandType.SCAN_PHASE0,
            target_module="scan",
            payload={
                "scan_id": "scan_validate_phase0",
                "targets": [self.target],
                "max_depth": 3,
                "timeout": 60  # 最小 60 秒
            }
        )
        
        start = time.time()
        result = await self.command_center.execute(command)
        elapsed = time.time() - start
        
        print(f"\n📊 執行結果:")
        print(f"  - 狀態: {result.status}")
        print(f"  - 成功: {result.success}")
        print(f"  - 耗時: {elapsed:.2f}秒")
        
        if result.success:
            metrics = result.metrics
            print(f"\n✅ Rust 引擎被成功驅動:")
            print(f"  - 資產數: {metrics.get('assets_found', 0)}")
            print(f"  - URLs: {metrics.get('urls_found', 0)}")
            print(f"  - APIs: {metrics.get('apis_found', 0)}")
            print(f"  - 表單: {metrics.get('forms_found', 0)}")
            
            # 驗證靶場反應: 檢查是否有實際資產
            if metrics.get('assets_found', 0) > 0:
                print(f"\n🎯 靶場反應: ✅ 發現 {metrics['assets_found']} 個資產")
                
                # 顯示前 3 個資產示例
                if result.result and 'assets' in result.result:
                    assets = result.result['assets'][:3]
                    print(f"\n前 3 個資產示例:")
                    for i, asset in enumerate(assets, 1):
                        asset_type = asset.get('type', 'unknown')
                        asset_value = asset.get('value', 'N/A')[:60]
                        print(f"  {i}. [{asset_type}] {asset_value}")
                
                return True
            else:
                print(f"\n❌ 靶場無反應: 未發現任何資產")
                return False
        else:
            print(f"\n❌ Rust 引擎調用失敗:")
            print(f"  - 錯誤: {result.error}")
            return False
    
    async def test_phase1_python(self):
        """測試 2: Phase 1 協調器驅動 Python 引擎"""
        print("\n" + "="*80)
        print("測試 2: Phase 1 - 協調器驅動 Python 引擎")
        print("="*80)
        
        command = AICommand(
            command_id="validate_phase1_python",
            command_type=CommandType.SCAN_PHASE1,
            target_module="scan",
            payload={
                "scan_id": "scan_validate_python",
                "targets": [self.target],
                "selected_engines": ["python"],  # 僅 Python
                "max_depth": 3,
                "max_urls": 100,
                "timeout": 300  # 最小 300 秒
            }
        )
        
        start = time.time()
        result = await self.command_center.execute(command)
        elapsed = time.time() - start
        
        print(f"\n📊 執行結果:")
        print(f"  - 狀態: {result.status}")
        print(f"  - 成功: {result.success}")
        print(f"  - 耗時: {elapsed:.2f}秒")
        
        if result.success:
            metrics = result.metrics
            print(f"\n✅ Python 引擎被成功驅動:")
            print(f"  - 總資產: {metrics.get('total_assets', 0)}")
            print(f"  - URLs: {metrics.get('urls_found', 0)}")
            print(f"  - 表單: {metrics.get('forms_found', 0)}")
            
            # 驗證靶場反應
            total_assets = metrics.get('total_assets', 0)
            if total_assets > 0:
                print(f"\n🎯 靶場反應: ✅ 發現 {total_assets} 個資產")
                
                # 顯示資產示例
                if result.result and 'assets' in result.result:
                    assets = result.result['assets'][:3]
                    print(f"\n前 3 個資產示例:")
                    for i, asset in enumerate(assets, 1):
                        asset_type = asset.get('type', 'unknown')
                        asset_value = asset.get('value', 'N/A')[:60]
                        print(f"  {i}. [{asset_type}] {asset_value}")
                
                return True
            else:
                print(f"\n❌ 靶場無反應: 未發現任何資產")
                return False
        else:
            print(f"\n❌ Python 引擎調用失敗:")
            print(f"  - 錯誤: {result.error}")
            return False
    
    async def test_phase1_go(self):
        """測試 3: Phase 1 協調器驅動 Go 引擎"""
        print("\n" + "="*80)
        print("測試 3: Phase 1 - 協調器驅動 Go 引擎")
        print("="*80)
        
        command = AICommand(
            command_id="validate_phase1_go",
            command_type=CommandType.SCAN_PHASE1,
            target_module="scan",
            payload={
                "scan_id": "scan_validate_go",
                "targets": [self.target],
                "selected_engines": ["go"],  # 僅 Go
                "timeout": 300  # 最小 300 秒
            }
        )
        
        start = time.time()
        result = await self.command_center.execute(command)
        elapsed = time.time() - start
        
        print(f"\n📊 執行結果:")
        print(f"  - 狀態: {result.status}")
        print(f"  - 成功: {result.success}")
        print(f"  - 耗時: {elapsed:.2f}秒")
        
        if result.success:
            metrics = result.metrics
            print(f"\n✅ Go 引擎被成功驅動:")
            print(f"  - 總資產: {metrics.get('total_assets', 0)}")
            
            # Go 引擎返回的是漏洞測試資產
            total_assets = metrics.get('total_assets', 0)
            if total_assets > 0:
                print(f"\n🎯 靶場反應: ✅ 執行了 {total_assets} 個 SSRF 測試")
                
                # 顯示測試示例
                if result.result and 'assets' in result.result:
                    assets = result.result['assets'][:3]
                    print(f"\n前 3 個 SSRF 測試:")
                    for i, asset in enumerate(assets, 1):
                        asset_type = asset.get('type', 'unknown')
                        asset_value = asset.get('value', 'N/A')[:60]
                        print(f"  {i}. [{asset_type}] {asset_value}")
                
                return True
            else:
                print(f"\n❌ 靶場無反應: 未執行任何測試")
                return False
        else:
            print(f"\n❌ Go 引擎調用失敗:")
            print(f"  - 錯誤: {result.error}")
            return False
    
    async def test_phase1_dual_engines(self):
        """測試 4: Phase 1 協調器同時驅動 Python + Rust"""
        print("\n" + "="*80)
        print("測試 4: Phase 1 - 協調器同時驅動 Python + Rust")
        print("="*80)
        
        command = AICommand(
            command_id="validate_phase1_dual",
            command_type=CommandType.SCAN_PHASE1,
            target_module="scan",
            payload={
                "scan_id": "scan_validate_dual",
                "targets": [self.target],
                "selected_engines": ["python", "rust"],  # 雙引擎
                "max_depth": 3,
                "max_urls": 100,
                "timeout": 300  # 最小 300 秒
            }
        )
        
        start = time.time()
        result = await self.command_center.execute(command)
        elapsed = time.time() - start
        
        print(f"\n📊 執行結果:")
        print(f"  - 狀態: {result.status}")
        print(f"  - 成功: {result.success}")
        print(f"  - 耗時: {elapsed:.2f}秒")
        
        if result.success:
            metrics = result.metrics
            total_assets = metrics.get('total_assets', 0)
            
            print(f"\n✅ 雙引擎被成功驅動:")
            print(f"  - 總資產: {total_assets}")
            
            # 檢查各引擎貢獻
            python_assets = metrics.get('python_assets', 0)
            rust_assets = metrics.get('rust_assets', 0)
            
            if python_assets > 0 or rust_assets > 0:
                print(f"  - Python 貢獻: {python_assets}個")
                print(f"  - Rust 貢獻: {rust_assets}個")
                
                if total_assets > 0:
                    print(f"\n🎯 靶場反應: ✅ 雙引擎合計發現 {total_assets} 個資產")
                    print(f"  - 協調器成功聚合了兩個引擎的結果")
                    return True
                else:
                    print(f"\n⚠️ 各引擎有發現,但聚合後為 0 (可能去重問題)")
                    return False
            else:
                print(f"\n❌ 靶場無反應: 兩個引擎都未發現資產")
                return False
        else:
            print(f"\n❌ 雙引擎調用失敗:")
            print(f"  - 錯誤: {result.error}")
            return False
    
    async def run_all_tests(self):
        """執行所有驗證測試"""
        print("\n" + "="*80)
        print("🚀 開始驗證協調器驅動引擎能力")
        print("="*80)
        print(f"目標靶場: {self.target}")
        print(f"測試時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        
        # 測試 1: Phase 0 Rust
        results['phase0_rust'] = await self.test_phase0_rust()
        await asyncio.sleep(2)
        
        # 測試 2: Phase 1 Python
        results['phase1_python'] = await self.test_phase1_python()
        await asyncio.sleep(2)
        
        # 測試 3: Phase 1 Go
        results['phase1_go'] = await self.test_phase1_go()
        await asyncio.sleep(2)
        
        # 測試 4: Phase 1 雙引擎
        results['phase1_dual'] = await self.test_phase1_dual_engines()
        
        # 總結
        print("\n" + "="*80)
        print("📊 驗證總結")
        print("="*80)
        
        total = len(results)
        passed = sum(1 for v in results.values() if v)
        
        print(f"\n測試結果: {passed}/{total} 通過")
        print(f"\n詳細:")
        print(f"  1. Phase 0 (Rust):      {'✅ 通過' if results['phase0_rust'] else '❌ 失敗'}")
        print(f"  2. Phase 1 (Python):    {'✅ 通過' if results['phase1_python'] else '❌ 失敗'}")
        print(f"  3. Phase 1 (Go):        {'✅ 通過' if results['phase1_go'] else '❌ 失敗'}")
        print(f"  4. Phase 1 (雙引擎):    {'✅ 通過' if results['phase1_dual'] else '❌ 失敗'}")
        
        if passed == total:
            print(f"\n🎉 協調器驗證通過!")
            print(f"   ✅ 協調器能夠成功驅動各語言引擎")
            print(f"   ✅ 各引擎能讓靶場產生實際反應")
            print(f"   ✅ 多引擎協同工作正常")
        else:
            print(f"\n⚠️ 協調器驗證部分失敗 ({total-passed}/{total})")
            print(f"   需要檢查失敗的引擎配置")
        
        return passed == total


async def main():
    """主函數"""
    validator = CoordinatorEngineValidator()
    success = await validator.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
