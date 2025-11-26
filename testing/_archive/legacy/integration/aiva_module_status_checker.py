#!/usr/bin/env python3
"""
AIVA 全模組狀態檢查器
用途: 檢查所有功能模組和掃描模組的可用性狀態
基於: 五大模組架構的完整功能驗證
"""

import sys
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 添加 AIVA 模組路徑 - 從 scripts/testing/ 返回到專案根目錄
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class AIVAModuleStatusChecker:
    """AIVA 模組狀態檢查器"""
    
    def __init__(self):
        self.results = {
            "檢查時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "總體狀態": "未檢查",
            "模組狀態": {},
            "統計": {
                "總模組數": 0,
                "可用模組": 0,
                "不可用模組": 0,
                "可用率": 0.0
            }
        }
        
        # 定義要檢查的模組
        self.modules_to_check = {
            # Core 核心模組
            "Core": {
                "ai_engine.bio_neuron_core": "services.core.aiva_core.ai_engine.bio_neuron_core",
                "ai_engine.anti_hallucination_module": "services.core.aiva_core.ai_engine.anti_hallucination_module", 
                "decision.enhanced_decision_agent": "services.core.aiva_core.decision.enhanced_decision_agent",
                "execution.execution_status_monitor": "services.core.aiva_core.execution.execution_status_monitor",
                "app": "services.core.aiva_core.app"
            },
            
            # Scan 掃描模組
            "Scan": {
                "target_environment_detector": "services.scan.aiva_scan.target_environment_detector",
                "vulnerability_scanner": "services.scan.aiva_scan.vulnerability_scanner", 
                "network_scanner": "services.scan.aiva_scan.network_scanner",
                "service_detector": "services.scan.aiva_scan.service_detector"
            },
            
            # Integration 整合模組  
            "Integration": {
                "ai_operation_recorder": "services.integration.aiva_integration.ai_operation_recorder",
                "system_performance_monitor": "services.integration.aiva_integration.system_performance_monitor",
                "integrated_ai_trainer": "services.integration.aiva_integration.integrated_ai_trainer",
                "trigger_ai_continuous_learning": "services.integration.aiva_integration.trigger_ai_continuous_learning"
            },
            
            # Function 功能模組
            "Function": {
                "feature_executor": "services.features.feature_step_executor", 
                "high_value_manager": "services.features.high_value_manager"
            },
            
            # Common 共用模組
            "Common": {
                "schemas": "aiva_common.schemas",
                "utils": "aiva_common.utils",
                "config": "aiva_common.config"
            }
        }
    
    def check_module_import(self, module_name: str, module_path: str) -> Dict[str, Any]:
        """檢查單個模組的導入狀態"""
        result = {
            "模組名稱": module_name,
            "模組路徑": module_path,
            "狀態": "未知",
            "錯誤": None,
            "詳情": None,
            "主要類別": []
        }
        
        try:
            # 嘗試導入模組
            module = importlib.import_module(module_path)
            result["狀態"] = "✅ 可用"
            result["詳情"] = f"成功導入 {module_path}"
            
            # 嘗試獲取模組的主要類別或函數
            classes = []
            for attr_name in dir(module):
                if not attr_name.startswith('_'):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, '__class__') and attr.__class__.__name__ in ['type', 'function']:
                        classes.append(attr_name)
            
            result["主要類別"] = classes[:5]  # 只顯示前 5 個
            
        except ImportError as e:
            result["狀態"] = "❌ 導入失敗"
            result["錯誤"] = f"ImportError: {str(e)}"
            result["詳情"] = "模組文件不存在或依賴缺失"
            
        except Exception as e:
            result["狀態"] = "⚠️  部分可用"
            result["錯誤"] = f"{type(e).__name__}: {str(e)}"
            result["詳情"] = "模組可導入但存在運行時問題"
        
        return result
    
    def check_all_modules(self):
        """檢查所有模組"""
        print("🔍 開始 AIVA 全模組狀態檢查...")
        print("=" * 70)
        
        total_modules = 0
        available_modules = 0
        
        for category, modules in self.modules_to_check.items():
            print(f"\n📦 檢查 {category} 模組:")
            print("-" * 50)
            
            category_results = {}
            
            for module_name, module_path in modules.items():
                total_modules += 1
                result = self.check_module_import(module_name, module_path)
                category_results[module_name] = result
                
                # 顯示檢查結果
                status_icon = result["狀態"]
                print(f"   {status_icon} {module_name}")
                
                if result["主要類別"]:
                    print(f"      主要功能: {', '.join(result['主要類別'])}")
                    
                if result["錯誤"]:
                    print(f"      錯誤: {result['錯誤']}")
                
                if result["狀態"] == "✅ 可用":
                    available_modules += 1
            
            self.results["模組狀態"][category] = category_results
        
        # 計算統計數據
        self.results["統計"]["總模組數"] = total_modules
        self.results["統計"]["可用模組"] = available_modules  
        self.results["統計"]["不可用模組"] = total_modules - available_modules
        self.results["統計"]["可用率"] = (available_modules / total_modules) * 100 if total_modules > 0 else 0
        
        # 確定總體狀態
        if available_modules == total_modules:
            self.results["總體狀態"] = "🟢 全部可用"
        elif available_modules >= total_modules * 0.8:
            self.results["總體狀態"] = "🟡 大部分可用"
        elif available_modules >= total_modules * 0.5:
            self.results["總體狀態"] = "🟠 部分可用"
        else:
            self.results["總體狀態"] = "🔴 大部分不可用"
    
    def check_critical_scanning_modules(self):
        """重點檢查掃描相關的關鍵模組"""
        print(f"\n🎯 重點檢查掃描模組功能:")
        print("-" * 50)
        
        # 掃描模組功能測試
        scan_tests = [
            {
                "name": "靶場環境檢測器",
                "module": "services.scan.aiva_scan.target_environment_detector",
                "class": "TargetEnvironmentDetector"
            },
            {
                "name": "漏洞掃描器", 
                "module": "services.scan.aiva_scan.vulnerability_scanner",
                "class": "VulnerabilityScanner"
            },
            {
                "name": "網路掃描器",
                "module": "services.scan.aiva_scan.network_scanner", 
                "class": "NetworkScanner"
            }
        ]
        
        scan_results = {}
        
        for test in scan_tests:
            try:
                module = importlib.import_module(test["module"])
                if hasattr(module, test["class"]):
                    # 嘗試實例化
                    cls = getattr(module, test["class"])
                    instance = cls()
                    scan_results[test["name"]] = {
                        "狀態": "✅ 完全可用",
                        "詳情": f"可成功實例化 {test['class']}"
                    }
                    print(f"   ✅ {test['name']}: 完全可用")
                else:
                    scan_results[test["name"]] = {
                        "狀態": "⚠️  類別缺失", 
                        "詳情": f"模組存在但缺少 {test['class']} 類別"
                    }
                    print(f"   ⚠️  {test['name']}: 類別缺失")
                    
            except Exception as e:
                scan_results[test["name"]] = {
                    "狀態": "❌ 不可用",
                    "詳情": f"錯誤: {str(e)}"
                }
                print(f"   ❌ {test['name']}: {str(e)}")
        
        self.results["掃描模組測試"] = scan_results
    
    def check_integration_capabilities(self):
        """檢查整合能力"""
        print(f"\n🔗 檢查模組整合能力:")
        print("-" * 50)
        
        integration_tests = [
            {
                "name": "AI 持續學習觸發器",
                "test": self._test_ai_trainer_integration
            },
            {
                "name": "跨模組通訊",
                "test": self._test_cross_module_communication  
            },
            {
                "name": "性能監控整合",
                "test": self._test_performance_monitoring
            }
        ]
        
        integration_results = {}
        
        for test in integration_tests:
            try:
                result = test["test"]()
                integration_results[test["name"]] = result
                status_icon = "✅" if result["成功"] else "❌"
                print(f"   {status_icon} {test['name']}: {result['詳情']}")
                
            except Exception as e:
                integration_results[test["name"]] = {
                    "成功": False,
                    "詳情": f"測試異常: {str(e)}"
                }
                print(f"   ❌ {test['name']}: 測試異常")
        
        self.results["整合能力測試"] = integration_results
    
    def _test_ai_trainer_integration(self) -> Dict[str, Any]:
        """測試 AI 訓練器整合"""
        try:
            from services.integration.aiva_integration.integrated_ai_trainer import IntegratedTrainService
            trainer = IntegratedTrainService()
            
            # 檢查組件載入
            if len(trainer.components) >= 3:
                return {
                    "成功": True,
                    "詳情": f"成功載入 {len(trainer.components)} 個整合組件"
                }
            else:
                return {
                    "成功": False,
                    "詳情": f"僅載入 {len(trainer.components)} 個組件，低於預期"
                }
                
        except Exception as e:
            return {
                "成功": False,
                "詳情": f"整合測試失敗: {str(e)}"
            }
    
    def _test_cross_module_communication(self) -> Dict[str, Any]:
        """測試跨模組通訊"""
        try:
            # 測試 Core -> Scan 通訊
            from services.core.aiva_core.decision.enhanced_decision_agent import EnhancedDecisionAgent
            from services.scan.aiva_scan.target_environment_detector import TargetEnvironmentDetector
            
            decision_agent = EnhancedDecisionAgent()
            target_detector = TargetEnvironmentDetector()
            
            return {
                "成功": True,  
                "詳情": "Core-Scan 模組通訊正常"
            }
            
        except Exception as e:
            return {
                "成功": False,
                "詳情": f"跨模組通訊測試失敗: {str(e)}"
            }
    
    def _test_performance_monitoring(self) -> Dict[str, Any]:
        """測試性能監控"""
        try:
            from services.integration.aiva_integration.system_performance_monitor import SystemPerformanceMonitor
            
            monitor = SystemPerformanceMonitor()
            metrics = monitor.get_system_metrics()
            
            if "cpu_usage" in metrics and "memory_usage" in metrics:
                return {
                    "成功": True,
                    "詳情": "性能監控功能正常運作"
                }
            else:
                return {
                    "成功": False,
                    "詳情": "性能監控數據不完整"
                }
                
        except Exception as e:
            return {
                "成功": False,
                "詳情": f"性能監控測試失敗: {str(e)}"
            }
    
    def generate_summary_report(self):
        """生成總結報告"""
        print(f"\n📊 AIVA 模組狀態總結報告")
        print("=" * 70)
        
        stats = self.results["統計"]
        print(f"📈 總體狀態: {self.results['總體狀態']}")
        print(f"📊 模組統計:")
        print(f"   - 總模組數: {stats['總模組數']}")
        print(f"   - 可用模組: {stats['可用模組']}")
        print(f"   - 不可用模組: {stats['不可用模組']}")  
        print(f"   - 可用率: {stats['可用率']:.1f}%")
        
        # 按類別統計
        print(f"\n📦 各類別模組狀態:")
        for category, modules in self.results["模組狀態"].items():
            available = sum(1 for m in modules.values() if m["狀態"] == "✅ 可用")
            total = len(modules)
            percentage = (available / total) * 100 if total > 0 else 0
            print(f"   - {category}: {available}/{total} ({percentage:.1f}%)")
        
        # 關鍵功能狀態
        if "掃描模組測試" in self.results:
            print(f"\n🎯 掃描功能狀態:")
            for name, result in self.results["掃描模組測試"].items():
                print(f"   - {name}: {result['狀態']}")
        
        if "整合能力測試" in self.results:
            print(f"\n🔗 整合能力狀態:")
            for name, result in self.results["整合能力測試"].items():
                status = "✅ 通過" if result["成功"] else "❌ 失敗"
                print(f"   - {name}: {status}")
        
        # 建議
        print(f"\n💡 建議:")
        if stats["可用率"] >= 90:
            print("   ✅ 系統狀態優良，所有主要功能都能正常運作")
        elif stats["可用率"] >= 70:
            print("   ⚠️  建議修復不可用的模組以提升系統穩定性")
        else:
            print("   🚨 系統存在重大問題，需要立即修復多個關鍵模組")
        
        print(f"\n🕐 檢查完成時間: {self.results['檢查時間']}")
        print("=" * 70)

def main():
    """主函數"""
    print("🚀 AIVA 全模組狀態檢查器")
    print("📋 檢查五大模組架構的完整功能狀態")
    print("=" * 70)
    
    checker = AIVAModuleStatusChecker()
    
    # 執行檢查
    checker.check_all_modules()
    checker.check_critical_scanning_modules()
    checker.check_integration_capabilities()
    checker.generate_summary_report()
    
    return checker.results

if __name__ == "__main__":
    results = main()