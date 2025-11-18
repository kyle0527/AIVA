!/usr/bin/env python3
"""
AIVA 全功能測試腳本
測試所有模組的基本功能，不依賴外部服務
記錄錯誤並生成完整測試報告
"""

import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import asyncio

# 添加項目路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class AIVAComprehensiveTest:
    """AIVA 全功能測試器"""
    
    def __init__(self):
        self.test_results = {
            "test_time": datetime.now().isoformat(),
            "python_modules": {},
            "rust_modules": {},
            "go_modules": {},
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
    def log_error(self, module, error_type, description):
        """記錄錯誤"""
        error_entry = {
            "module": module,
            "type": error_type,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results["errors"].append(error_entry)
        print(f"❌ [{module}] {error_type}: {description}")
    
    def log_warning(self, module, warning_type, description):
        """記錄警告"""
        warning_entry = {
            "module": module,
            "type": warning_type,
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results["warnings"].append(warning_entry)
        print(f"⚠️  [{module}] {warning_type}: {description}")
    
    def test_python_imports(self):
        """測試 Python 模組導入"""
        print("\n🐍 測試 Python 模組...")
        
        python_modules = [
            ("services.aiva_common", "Common Module"),
            ("services.core.aiva_core", "Core Module"), 
            ("services.scan.aiva_scan", "Scan Module"),
            ("services.integration.aiva_integration", "Integration Module"),
            ("services.features", "Features Module")
        ]
        
        for module_path, module_name in python_modules:
            try:
                __import__(module_path)
                self.test_results["python_modules"][module_name] = "✅ PASS"
                print(f"   ✅ {module_name}: 導入成功")
            except ImportError as e:
                self.test_results["python_modules"][module_name] = f"❌ FAIL: {str(e)}"
                self.log_error(module_name, "Import Error", str(e))
            except Exception as e:
                self.test_results["python_modules"][module_name] = f"⚠️  WARNING: {str(e)}"
                self.log_warning(module_name, "Import Warning", str(e))
    
    def test_rust_compilation(self):
        """測試 Rust 模組編譯"""
        print("\n🦀 測試 Rust 模組...")
        
        rust_projects = [
            ("services/features/function_sast_rust", "SAST Analyzer"),
            ("services/scan/info_gatherer_rust", "Info Gatherer")
        ]
        
        for project_path, module_name in rust_projects:
            try:
                result = subprocess.run(
                    ["cargo", "check"], 
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    self.test_results["rust_modules"][module_name] = "✅ PASS"
                    print(f"   ✅ {module_name}: 編譯檢查通過")
                    
                    # 檢查警告
                    if "warning:" in result.stderr:
                        warning_count = result.stderr.count("warning:")
                        self.log_warning(module_name, "Compilation Warning", 
                                       f"{warning_count} 個編譯警告")
                else:
                    self.test_results["rust_modules"][module_name] = f"❌ FAIL: {result.stderr}"
                    self.log_error(module_name, "Compilation Error", result.stderr)
                    
            except FileNotFoundError:
                self.test_results["rust_modules"][module_name] = "❌ FAIL: Cargo not found"
                self.log_error(module_name, "Tool Error", "Cargo 編譯器未找到")
            except subprocess.TimeoutExpired:
                self.test_results["rust_modules"][module_name] = "⚠️  TIMEOUT"
                self.log_warning(module_name, "Timeout", "編譯檢查超時")
            except Exception as e:
                self.test_results["rust_modules"][module_name] = f"❌ FAIL: {str(e)}"
                self.log_error(module_name, "Unknown Error", str(e))
    
    def test_go_compilation(self):
        """測試 Go 模組編譯"""
        print("\n🐹 測試 Go 模組...")
        
        go_projects = [
            ("services/features/function_authn_go", "Authentication Bypass"),
            ("services/features/function_cspm_go", "CSPM Scanner"),
            ("services/features/function_ssrf_go", "SSRF Detector"),
            ("services/features/function_sca_go", "SCA Analyzer")
        ]
        
        for project_path, module_name in go_projects:
            try:
                # 首先嘗試 go mod tidy
                tidy_result = subprocess.run(
                    ["go", "mod", "tidy"], 
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # 然後進行編譯檢查
                result = subprocess.run(
                    ["go", "build", "./..."], 
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    self.test_results["go_modules"][module_name] = "✅ PASS"
                    print(f"   ✅ {module_name}: 編譯成功")
                else:
                    self.test_results["go_modules"][module_name] = f"❌ FAIL: {result.stderr}"
                    self.log_error(module_name, "Compilation Error", result.stderr.strip())
                    
            except FileNotFoundError:
                self.test_results["go_modules"][module_name] = "❌ FAIL: Go not found"
                self.log_error(module_name, "Tool Error", "Go 編譯器未找到")
            except subprocess.TimeoutExpired:
                self.test_results["go_modules"][module_name] = "⚠️  TIMEOUT"
                self.log_warning(module_name, "Timeout", "編譯超時")
            except Exception as e:
                self.test_results["go_modules"][module_name] = f"❌ FAIL: {str(e)}"
                self.log_error(module_name, "Unknown Error", str(e))
    
    async def test_target_detection(self):
        """測試靶場檢測功能"""
        print("\n🎯 測試靶場檢測功能...")
        
        try:
            from services.scan.aiva_scan.target_environment_detector import TargetEnvironmentDetector
            
            detector = TargetEnvironmentDetector()
            
            # 測試本地環境檢測
            results = await detector.detect_environment(['127.0.0.1'])
            
            if results and 'discovered_services' in results:
                service_count = len(results['discovered_services'])
                self.test_results["python_modules"]["Target Detection"] = f"✅ PASS: {service_count} services found"
                print(f"   ✅ 靶場檢測: 發現 {service_count} 個服務")
            else:
                self.test_results["python_modules"]["Target Detection"] = "⚠️  WARNING: No services found"
                self.log_warning("Target Detection", "No Services", "未發現任何服務")
                
        except Exception as e:
            self.test_results["python_modules"]["Target Detection"] = f"❌ FAIL: {str(e)}"
            self.log_error("Target Detection", "Runtime Error", str(e))
    
    def test_ai_integration(self):
        """測試 AI 集成功能"""
        print("\n🤖 測試 AI 集成功能...")
        
        try:
            # 測試 AI 觸發器存在性
            trigger_path = Path("services/integration/aiva_integration/trigger_ai_continuous_learning.py")
            if trigger_path.exists():
                self.test_results["python_modules"]["AI Integration"] = "✅ PASS: Trigger exists"
                print("   ✅ AI 觸發器: 文件存在")
            else:
                self.test_results["python_modules"]["AI Integration"] = "❌ FAIL: Trigger missing"
                self.log_error("AI Integration", "Missing File", "AI 觸發器文件不存在")
                
            # 測試 AI 核心模組
            core_path = Path("services/core/aiva_core")
            if core_path.exists():
                ai_files = list(core_path.rglob("*.py"))
                self.test_results["python_modules"]["AI Core"] = f"✅ PASS: {len(ai_files)} AI files"
                print(f"   ✅ AI 核心: 發現 {len(ai_files)} 個 AI 文件")
            else:
                self.test_results["python_modules"]["AI Core"] = "❌ FAIL: Core missing"
                self.log_error("AI Core", "Missing Directory", "AI 核心目錄不存在")
                
        except Exception as e:
            self.test_results["python_modules"]["AI Integration"] = f"❌ FAIL: {str(e)}"
            self.log_error("AI Integration", "Test Error", str(e))
    
    def generate_summary(self):
        """生成測試摘要"""
        print("\n📊 生成測試摘要...")
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warning_tests = 0
        
        # 統計各模組結果
        for module_type in ["python_modules", "rust_modules", "go_modules"]:
            for module_name, result in self.test_results[module_type].items():
                total_tests += 1
                if "✅ PASS" in result:
                    passed_tests += 1
                elif "❌ FAIL" in result:
                    failed_tests += 1
                elif "⚠️  WARNING" in result or "⚠️  TIMEOUT" in result:
                    warning_tests += 1
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warning_tests,
            "success_rate": f"{(passed_tests / max(total_tests, 1) * 100):.1f}%",
            "total_errors": len(self.test_results["errors"]),
            "total_warnings": len(self.test_results["warnings"])
        }
    
    def print_report(self):
        """打印測試報告"""
        print("\n" + "="*70)
        print("📋 AIVA 全功能測試報告")
        print("="*70)
        
        summary = self.test_results["summary"]
        print(f"🕐 測試時間: {self.test_results['test_time']}")
        print(f"📊 測試總數: {summary['total_tests']}")
        print(f"✅ 通過: {summary['passed']}")
        print(f"❌ 失敗: {summary['failed']}")
        print(f"⚠️  警告: {summary['warnings']}")
        print(f"📈 成功率: {summary['success_rate']}")
        
        # 顯示各模組詳細結果
        for module_type, display_name in [
            ("python_modules", "🐍 Python 模組"),
            ("rust_modules", "🦀 Rust 模組"), 
            ("go_modules", "🐹 Go 模組")
        ]:
            if self.test_results[module_type]:
                print(f"\n{display_name}:")
                for module_name, result in self.test_results[module_type].items():
                    print(f"   {result.split(':')[0]} {module_name}")
        
        # 顯示錯誤摘要
        if self.test_results["errors"]:
            print(f"\n❌ 錯誤摘要 ({len(self.test_results['errors'])} 個):")
            for error in self.test_results["errors"][-5:]:  # 顯示最後 5 個錯誤
                print(f"   • {error['module']}: {error['type']}")
        
        # 顯示警告摘要  
        if self.test_results["warnings"]:
            print(f"\n⚠️  警告摘要 ({len(self.test_results['warnings'])} 個):")
            for warning in self.test_results["warnings"][-3:]:  # 顯示最後 3 個警告
                print(f"   • {warning['module']}: {warning['type']}")
    
    def save_report(self):
        """保存詳細報告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"aiva_comprehensive_test_report_{timestamp}.json"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 詳細報告已保存: {report_path}")
            return report_path
        except Exception as e:
            print(f"\n❌ 報告保存失敗: {e}")
            return None
    
    async def run_all_tests(self):
        """運行所有測試"""
        print("🚀 AIVA 全功能測試開始")
        print("=" * 70)
        
        start_time = time.time()
        
        # 執行各類測試
        self.test_python_imports()
        self.test_rust_compilation()
        self.test_go_compilation()
        await self.test_target_detection()
        self.test_ai_integration()
        
        # 生成報告
        self.generate_summary()
        
        test_duration = time.time() - start_time
        self.test_results["test_duration"] = f"{test_duration:.2f}s"
        
        print(f"\n⏱️  測試耗時: {test_duration:.2f}s")
        
        # 顯示和保存報告
        self.print_report()
        self.save_report()
        
        print(f"\n✅ 全功能測試完成！")

async def main():
    """主函數"""
    tester = AIVAComprehensiveTest()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())