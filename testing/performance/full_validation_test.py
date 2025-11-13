#!/usr/bin/env python3
"""
AIVA 系統全功能驗證腳本
驗證使用者指南和README中的所有功能
"""

import asyncio
import json
import time
import requests
import subprocess
import sys
from pathlib import Path

# 設置專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

class AIVAValidator:
    def __init__(self):
        self.launcher_process = None
        self.results = {}
        
    async def start_aiva_service(self):
        """啟動AIVA核心服務"""
        print("🚀 第一步：啟動AIVA核心服務...")
        
        try:
            # 啟動launcher作為背景進程
            self.launcher_process = subprocess.Popen(
                [sys.executable, "aiva_launcher.py", "--mode", "core_only"],
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待服務啟動
            print("⏳ 等待服務啟動...")
            await asyncio.sleep(5)
            
            # 檢查進程是否還在運行
            if self.launcher_process.poll() is None:
                print("✅ 核心服務啟動成功")
                self.results["service_startup"] = "SUCCESS"
                return True
            else:
                stdout, stderr = self.launcher_process.communicate()
                print(f"❌ 服務啟動失敗: {stderr}")
                self.results["service_startup"] = "FAILED"
                return False
                
        except Exception as e:
            print(f"❌ 啟動服務時發生錯誤: {e}")
            self.results["service_startup"] = f"ERROR: {e}"
            return False
    
    def test_health_check(self):
        """測試健康檢查端點"""
        print("\n🏥 第二步：驗證服務健康狀態...")
        
        try:
            response = requests.get("http://localhost:8001/health", timeout=10)
            health_data = response.json()
            
            print("✅ 健康檢查成功:")
            print(json.dumps(health_data, ensure_ascii=False, indent=2))
            
            # 驗證預期的健康檢查格式
            expected_keys = ["status", "service", "components"]
            if all(key in health_data for key in expected_keys):
                print("✅ 健康檢查格式正確")
                self.results["health_check"] = "SUCCESS"
                self.results["health_data"] = health_data
                return True
            else:
                print("⚠️ 健康檢查格式不完整")
                self.results["health_check"] = "INCOMPLETE"
                return False
                
        except Exception as e:
            print(f"❌ 健康檢查失敗: {e}")
            self.results["health_check"] = f"FAILED: {e}"
            return False
    
    async def test_ai_dialog(self):
        """測試AI對話助手"""
        print("\n🤖 第三步：測試AI對話助手...")
        
        try:
            from services.core.aiva_core.dialog.assistant import AIVADialogAssistant
            
            assistant = AIVADialogAssistant()
            
            # 測試基本對話
            test_queries = [
                "現在系統會什麼？",
                "系統狀況如何？",
                "你好，AIVA！"
            ]
            
            dialog_results = {}
            
            for query in test_queries:
                print(f"  📝 測試查詢: {query}")
                response = await assistant.process_user_input(query)
                dialog_results[query] = {
                    "intent": response.get("intent"),
                    "executable": response.get("executable"),
                    "message_length": len(response.get("message", ""))
                }
                print(f"    ✅ 意圖: {response.get('intent')}")
                print(f"    ✅ 可執行: {response.get('executable')}")
                
            self.results["ai_dialog"] = "SUCCESS"
            self.results["dialog_details"] = dialog_results
            print("✅ AI對話助手測試完成")
            return True
            
        except Exception as e:
            print(f"❌ AI對話助手測試失敗: {e}")
            self.results["ai_dialog"] = f"FAILED: {e}"
            return False
    
    async def test_capability_discovery(self):
        """測試能力發現系統"""
        print("\n🔍 第四步：驗證能力發現系統...")
        
        try:
            from services.integration.capability.registry import global_registry
            
            # 觸發能力發現
            discovered = await global_registry.discover_capabilities()
            print(f"  📊 發現能力: {len(discovered)} 個")
            
            # 獲取統計信息
            stats = await global_registry.get_capability_stats()
            print(f"  📈 能力統計:")
            for key, value in stats.items():
                print(f"    - {key}: {value}")
            
            # 驗證是否符合指南中的預期
            total_capabilities = stats.get("total_capabilities", 0)
            language_dist = stats.get("by_language", {})
            
            if total_capabilities >= 10:
                print("✅ 能力數量符合預期 (>= 10)")
                self.results["capability_discovery"] = "SUCCESS"
                self.results["capability_stats"] = stats
                return True
            else:
                print(f"⚠️ 能力數量不足: {total_capabilities}")
                self.results["capability_discovery"] = "INSUFFICIENT"
                return False
                
        except Exception as e:
            print(f"❌ 能力發現測試失敗: {e}")
            self.results["capability_discovery"] = f"FAILED: {e}"
            return False
    
    def test_documentation_accuracy(self):
        """驗證文檔準確性"""
        print("\n📚 第五步：驗證指南準確性...")
        
        accuracy_score = 0
        total_checks = 0
        
        # 檢查服務啟動是否符合指南描述
        if self.results.get("service_startup") == "SUCCESS":
            print("✅ 服務啟動符合使用者指南描述")
            accuracy_score += 1
        total_checks += 1
        
        # 檢查健康檢查是否符合README描述
        if self.results.get("health_check") == "SUCCESS":
            print("✅ 健康檢查符合README描述")
            accuracy_score += 1
        total_checks += 1
        
        # 檢查AI對話是否符合指南描述
        if self.results.get("ai_dialog") == "SUCCESS":
            print("✅ AI對話功能符合使用者指南描述")
            accuracy_score += 1
        total_checks += 1
        
        # 檢查能力發現是否符合文檔描述
        if self.results.get("capability_discovery") == "SUCCESS":
            stats = self.results.get("capability_stats", {})
            expected_langs = ["python", "go", "rust"]
            actual_langs = list(stats.get("by_language", {}).keys())
            
            if all(lang in actual_langs for lang in expected_langs):
                print("✅ 跨語言支持符合文檔描述")
                accuracy_score += 1
        total_checks += 1
        
        accuracy_percentage = (accuracy_score / total_checks) * 100
        print(f"\n📊 文檔準確性評分: {accuracy_score}/{total_checks} ({accuracy_percentage:.1f}%)")
        
        self.results["documentation_accuracy"] = {
            "score": accuracy_score,
            "total": total_checks,
            "percentage": accuracy_percentage
        }
        
        return accuracy_percentage >= 80
    
    async def test_target_scanning(self, target_url="http://httpbin.org"):
        """測試實際靶場掃描功能"""
        print(f"\n🎯 第六步：實際靶場驗證 - 目標: {target_url}")
        
        try:
            from services.core.aiva_core.dialog.assistant import AIVADialogAssistant
            
            assistant = AIVADialogAssistant()
            
            # 測試掃描相關查詢
            scan_queries = [
                f"幫我測試這個網站 {target_url}",
                "解釋 SQL 注入掃描功能",
                "比較 Python 和 Go 版本的 SSRF 差異",
                "產生可執行的 CLI 指令"
            ]
            
            scan_results = {}
            
            for query in scan_queries:
                print(f"  🎯 測試掃描查詢: {query}")
                try:
                    response = await assistant.process_user_input(query)
                    scan_results[query] = {
                        "intent": response.get("intent"),
                        "executable": response.get("executable"),
                        "success": True
                    }
                    print(f"    ✅ 處理成功，意圖: {response.get('intent')}")
                except Exception as e:
                    scan_results[query] = {
                        "error": str(e),
                        "success": False
                    }
                    print(f"    ❌ 處理失敗: {e}")
            
            successful_queries = sum(1 for result in scan_results.values() if result.get("success"))
            success_rate = (successful_queries / len(scan_queries)) * 100
            
            print(f"  📊 掃描查詢成功率: {successful_queries}/{len(scan_queries)} ({success_rate:.1f}%)")
            
            self.results["target_scanning"] = {
                "success_rate": success_rate,
                "results": scan_results
            }
            
            return success_rate >= 75
            
        except Exception as e:
            print(f"❌ 靶場驗證失敗: {e}")
            self.results["target_scanning"] = f"FAILED: {e}"
            return False
    
    def cleanup(self):
        """清理資源"""
        print("\n🧹 清理資源...")
        if self.launcher_process and self.launcher_process.poll() is None:
            try:
                self.launcher_process.terminate()
                self.launcher_process.wait(timeout=5)
                print("✅ 服務已正常停止")
            except:
                self.launcher_process.kill()
                print("⚠️ 服務已強制停止")
    
    def generate_report(self):
        """生成驗證報告"""
        print("\n" + "="*50)
        print("📋 AIVA 系統全功能驗證報告")
        print("="*50)
        
        total_tests = len([k for k in self.results.keys() if not k.endswith("_data") and not k.endswith("_details") and not k.endswith("_stats")])
        passed_tests = len([k for k, v in self.results.items() if v == "SUCCESS" and not k.endswith("_data") and not k.endswith("_details") and not k.endswith("_stats")])
        
        print(f"📊 總體結果: {passed_tests}/{total_tests} 項測試通過")
        print(f"✅ 系統就緒度: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📝 詳細結果:")
        for key, value in self.results.items():
            if not key.endswith("_data") and not key.endswith("_details") and not key.endswith("_stats"):
                status = "✅ 通過" if value == "SUCCESS" else "❌ 失敗" if "FAILED" in str(value) else "⚠️ 部分成功"
                print(f"  - {key}: {status}")
        
        # 特殊報告
        if "documentation_accuracy" in self.results:
            doc_acc = self.results["documentation_accuracy"]
            print(f"\n📚 文檔準確性: {doc_acc['percentage']:.1f}%")
        
        if "target_scanning" in self.results and isinstance(self.results["target_scanning"], dict):
            scan_rate = self.results["target_scanning"]["success_rate"]
            print(f"🎯 靶場測試成功率: {scan_rate:.1f}%")
        
        # 能力統計
        if "capability_stats" in self.results:
            stats = self.results["capability_stats"]
            print(f"\n🔍 系統能力統計:")
            print(f"  - 總能力數: {stats.get('total_capabilities', 0)}")
            print(f"  - 語言分布: {stats.get('by_language', {})}")
        
        print("\n" + "="*50)
        
        return (passed_tests/total_tests)*100 >= 80

async def main():
    """主函數"""
    validator = AIVAValidator()
    
    try:
        print("🎉 開始AIVA系統全功能驗證")
        print("=" * 50)
        
        # 執行所有驗證步驟
        service_ok = await validator.start_aiva_service()
        if not service_ok:
            print("❌ 服務啟動失敗，終止驗證")
            return
        
        health_ok = validator.test_health_check()
        dialog_ok = await validator.test_ai_dialog()
        capability_ok = await validator.test_capability_discovery()
        doc_ok = validator.test_documentation_accuracy()
        target_ok = await validator.test_target_scanning()
        
        # 生成最終報告
        overall_success = validator.generate_report()
        
        if overall_success:
            print("🎉 驗證完成：AIVA系統完全就緒！")
            print("✅ 使用者指南和README準確無誤")
            print("🚀 可以開始進行實際安全測試")
        else:
            print("⚠️ 驗證完成：發現一些問題需要修正")
        
    except KeyboardInterrupt:
        print("\n⚠️ 驗證被用戶中斷")
    except Exception as e:
        print(f"\n❌ 驗證過程中發生錯誤: {e}")
    finally:
        validator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())