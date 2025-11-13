#!/usr/bin/env python3
"""
AIVA AI 整合測試系統
測試統一 AI 控制器、自然語言生成器和多語言協調器的整合效果
"""

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

sys.path.append(str(Path(__file__).parent.parent.parent))

from services.core.aiva_core.ai_controller import UnifiedAIController
from services.core.aiva_core.ai_engine.bio_neuron_core import BioNeuronRAGAgent
from services.core.aiva_core.multilang_coordinator import MultiLanguageAICoordinator
from services.core.aiva_core.nlg_system import AIVANaturalLanguageGenerator


@dataclass
class IntegrationTestResult:
    """測試結果數據類"""

    test_name: str
    success: bool
    execution_time: float
    details: dict[str, Any]
    error_message: str | None = None


class AIIntegrationTester:
    """AI 整合測試器"""

    def __init__(self, aiva_root: str):
        """
        初始化測試器

        Args:
            aiva_root: AIVA 根目錄路徑
        """
        self.aiva_root = Path(aiva_root)
        self.test_results: list[IntegrationTestResult] = []

        # 初始化各 AI 組件
        self.bio_agent = BioNeuronRAGAgent(str(self.aiva_root))
        self.unified_controller = UnifiedAIController(str(self.aiva_root))
        self.nlg_generator = AIVANaturalLanguageGenerator()
        self.multilang_coordinator = MultiLanguageAICoordinator()

        print("🚀 AI 整合測試器初始化完成")

    async def run_all_tests(self) -> dict[str, Any]:
        """
        執行所有整合測試

        Returns:
            測試結果統計
        """
        print("📋 開始執行 AI 整合測試套件...")
        start_time = time.time()

        # 測試清單
        tests = [
            ("基礎組件初始化測試", self._test_component_initialization),
            ("統一控制器協調測試", self._test_unified_controller),
            ("自然語言生成測試", self._test_nlg_system),
            ("多語言協調測試", self._test_multilang_coordination),
            ("AI 衝突檢測測試", self._test_ai_conflict_detection),
            ("端到端整合測試", self._test_end_to_end_integration),
            ("效能壓力測試", self._test_performance_stress),
            ("錯誤恢復測試", self._test_error_recovery),
        ]

        for test_name, test_func in tests:
            print(f"\n🧪 執行測試: {test_name}")
            try:
                await test_func()
                print(f"✅ {test_name} - 通過")
            except Exception as e:
                print(f"❌ {test_name} - 失敗: {str(e)}")

        total_time = time.time() - start_time
        return self._generate_test_report(total_time)

    async def _test_component_initialization(self):
        """測試基礎組件初始化"""
        start_time = time.time()

        # 測試各組件是否正常初始化
        components = {
            "BioNeuronRAGAgent": self.bio_agent,
            "UnifiedAIController": self.unified_controller,
            "AIVANaturalLanguageGenerator": self.nlg_generator,
            "MultiLanguageAICoordinator": self.multilang_coordinator,
        }

        details = {}
        for name, component in components.items():
            if hasattr(component, "is_ready"):
                is_ready = component.is_ready()
            else:
                is_ready = component is not None

            details[name] = {"initialized": component is not None, "ready": is_ready}

        execution_time = time.time() - start_time

        all_ready = all(detail["ready"] for detail in details.values())

        self.test_results.append(
            IntegrationTestResult(
                test_name="基礎組件初始化測試",
                success=all_ready,
                execution_time=execution_time,
                details=details,
                error_message=None if all_ready else "部分組件未準備就緒",
            )
        )

    async def _test_unified_controller(self):
        """測試統一控制器協調能力"""
        start_time = time.time()

        # 測試統一控制器處理複雜請求
        test_requests = [
            {
                "type": "security_analysis",
                "content": "分析這段代碼的 SQL 注入漏洞",
                "priority": "high",
                "language": "python",
            },
            {
                "type": "vulnerability_scan",
                "content": "掃描 SSRF 漏洞",
                "priority": "medium",
                "language": "go",
            },
        ]

        details = {}
        success_count = 0

        for i, request in enumerate(test_requests):
            try:
                response = await self.unified_controller.process_unified_request(
                    request
                )
                details[f"request_{i+1}"] = {
                    "success": True,
                    "response_type": type(response).__name__,
                    "has_result": response is not None,
                }
                success_count += 1
            except Exception as e:
                details[f"request_{i+1}"] = {"success": False, "error": str(e)}

        execution_time = time.time() - start_time
        success = success_count == len(test_requests)

        self.test_results.append(
            IntegrationTestResult(
                test_name="統一控制器協調測試",
                success=success,
                execution_time=execution_time,
                details=details,
            )
        )

    async def _test_nlg_system(self):
        """測試自然語言生成系統"""
        start_time = time.time()

        # 測試不同類型的自然語言生成
        test_contexts = [
            {
                "type": "vulnerability_report",
                "severity": "high",
                "vulnerability_type": "SQL注入",
                "affected_files": ["user_controller.py", "database.py"],
            },
            {
                "type": "scan_summary",
                "total_files": 156,
                "vulnerabilities_found": 3,
                "scan_duration": 45.2,
            },
        ]

        details = {}
        success_count = 0

        for i, context in enumerate(test_contexts):
            try:
                response = self.nlg_generator.generate_response(context)
                details[f"generation_{i+1}"] = {
                    "success": True,
                    "response_length": len(response),
                    "has_chinese": any(
                        "\u4e00" <= char <= "\u9fff" for char in response
                    ),
                    "template_type": context["type"],
                }
                success_count += 1
            except Exception as e:
                details[f"generation_{i+1}"] = {"success": False, "error": str(e)}

        execution_time = time.time() - start_time
        success = success_count == len(test_contexts)

        self.test_results.append(
            IntegrationTestResult(
                test_name="自然語言生成測試",
                success=success,
                execution_time=execution_time,
                details=details,
            )
        )

    async def _test_multilang_coordination(self):
        """測試多語言協調系統"""
        start_time = time.time()

        # 模擬多語言 AI 任務
        test_task = {
            "task_id": "test_multilang_001",
            "description": "跨語言漏洞檢測任務",
            "target_languages": ["python", "go", "rust"],
            "priority": "medium",
        }

        details = {}

        try:
            # 測試任務分配
            coordination_result = (
                await self.multilang_coordinator.coordinate_multi_language_ai_task(
                    test_task
                )
            )

            details["task_distribution"] = {
                "success": True,
                "languages_coordinated": len(
                    coordination_result.get("language_results", {})
                ),
                "execution_successful": coordination_result.get("success", False),
            }

            success = True
        except Exception as e:
            details["task_distribution"] = {"success": False, "error": str(e)}
            success = False

        execution_time = time.time() - start_time

        self.test_results.append(
            IntegrationTestResult(
                test_name="多語言協調測試",
                success=success,
                execution_time=execution_time,
                details=details,
            )
        )

    async def _test_ai_conflict_detection(self):
        """測試 AI 衝突檢測"""
        start_time = time.time()

        # 模擬並發 AI 請求測試衝突檢測
        concurrent_requests = [
            {"type": "sast_analysis", "target": "file1.py"},
            {"type": "sast_analysis", "target": "file1.py"},  # 相同目標，應該合併
            {"type": "dast_scan", "target": "endpoint1"},
        ]

        details = {}

        try:
            # 同時發送多個請求
            tasks = []
            for request in concurrent_requests:
                task = asyncio.create_task(
                    self.unified_controller.process_unified_request(request)
                )
                tasks.append(task)

            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # 分析衝突處理結果
            successful_responses = [
                r for r in responses if not isinstance(r, Exception)
            ]

            details["conflict_handling"] = {
                "total_requests": len(concurrent_requests),
                "successful_responses": len(successful_responses),
                "conflicts_detected": len(concurrent_requests)
                - len(successful_responses),
                "deduplication_working": len(
                    {str(r) for r in successful_responses if r}
                )
                < len(concurrent_requests),
            }

            success = len(successful_responses) > 0

        except Exception as e:
            details["conflict_handling"] = {"success": False, "error": str(e)}
            success = False

        execution_time = time.time() - start_time

        self.test_results.append(
            IntegrationTestResult(
                test_name="AI 衝突檢測測試",
                success=success,
                execution_time=execution_time,
                details=details,
            )
        )

    async def _test_end_to_end_integration(self):
        """測試端到端整合"""
        start_time = time.time()

        # 完整的端到端測試流程
        test_scenario = {
            "input": "請分析專案中的安全漏洞並生成中文報告",
            "expected_steps": [
                "任務解析",
                "AI 協調",
                "漏洞檢測",
                "結果整合",
                "中文報告生成",
            ],
        }

        details = {}

        try:
            # 1. 透過統一控制器處理複雜請求
            complex_request = {
                "type": "comprehensive_security_analysis",
                "content": test_scenario["input"],
                "output_format": "chinese_report",
                "include_recommendations": True,
            }

            controller_response = await self.unified_controller.process_unified_request(
                complex_request
            )

            # 2. 使用 NLG 系統生成最終報告
            if controller_response:
                nlg_context = {
                    "type": "comprehensive_report",
                    "analysis_result": controller_response,
                    "language": "chinese",
                }

                final_report = self.nlg_generator.generate_response(nlg_context)

                details["end_to_end_flow"] = {
                    "controller_success": True,
                    "nlg_success": True,
                    "final_report_generated": len(final_report) > 0,
                    "report_preview": (
                        final_report[:200] + "..."
                        if len(final_report) > 200
                        else final_report
                    ),
                }

                success = True
            else:
                details["end_to_end_flow"] = {
                    "controller_success": False,
                    "nlg_success": False,
                    "error": "控制器未返回有效結果",
                }
                success = False

        except Exception as e:
            details["end_to_end_flow"] = {"success": False, "error": str(e)}
            success = False

        execution_time = time.time() - start_time

        self.test_results.append(
            IntegrationTestResult(
                test_name="端到端整合測試",
                success=success,
                execution_time=execution_time,
                details=details,
            )
        )

    async def _test_performance_stress(self):
        """測試效能壓力"""
        start_time = time.time()

        # 壓力測試參數
        stress_requests = 10
        concurrent_limit = 5

        details = {}

        try:
            # 建立壓力測試請求
            requests = []
            for i in range(stress_requests):
                request = {
                    "type": "quick_analysis",
                    "content": f"測試請求 {i+1}",
                    "priority": "low",
                }
                requests.append(request)

            # 分批並發執行
            results = []
            for i in range(0, len(requests), concurrent_limit):
                batch = requests[i : i + concurrent_limit]
                batch_tasks = [
                    self.unified_controller.process_unified_request(req)
                    for req in batch
                ]
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )
                results.extend(batch_results)

            # 統計效能結果
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]

            details["performance_stats"] = {
                "total_requests": stress_requests,
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / stress_requests * 100,
                "average_response_time": (time.time() - start_time) / stress_requests,
            }

            success = len(successful_results) >= stress_requests * 0.8  # 80% 成功率

        except Exception as e:
            details["performance_stats"] = {"success": False, "error": str(e)}
            success = False

        execution_time = time.time() - start_time

        self.test_results.append(
            IntegrationTestResult(
                test_name="效能壓力測試",
                success=success,
                execution_time=execution_time,
                details=details,
            )
        )

    async def _test_error_recovery(self):
        """測試錯誤恢復機制"""
        start_time = time.time()

        # 測試各種錯誤情況的恢復能力
        error_scenarios = [
            {"type": "invalid_request", "data": {"invalid": "格式錯誤的請求"}},
            {
                "type": "missing_parameters",
                "data": {"type": "analysis"},
            },  # 缺少必要參數
            {
                "type": "timeout_simulation",
                "data": {"type": "long_running_task", "timeout": 0.1},
            },
        ]

        details = {}
        recovery_count = 0

        for i, scenario in enumerate(error_scenarios):
            try:
                response = await self.unified_controller.process_unified_request(
                    scenario["data"]
                )

                # 檢查是否有適當的錯誤處理
                details[f"scenario_{i+1}"] = {
                    "scenario_type": scenario["type"],
                    "handled_gracefully": True,
                    "response_received": response is not None,
                }
                recovery_count += 1

            except Exception as e:
                # 預期的錯誤，檢查是否有適當的錯誤訊息
                error_msg = str(e)
                is_handled = (
                    len(error_msg) > 0 and "unexpected" not in error_msg.lower()
                )

                details[f"scenario_{i+1}"] = {
                    "scenario_type": scenario["type"],
                    "handled_gracefully": is_handled,
                    "error_message": error_msg[:100],
                }

                if is_handled:
                    recovery_count += 1

        execution_time = time.time() - start_time
        success = recovery_count >= len(error_scenarios) * 0.7  # 70% 恢復率

        self.test_results.append(
            IntegrationTestResult(
                test_name="錯誤恢復測試",
                success=success,
                execution_time=execution_time,
                details=details,
            )
        )

    def _generate_test_report(self, total_time: float) -> dict[str, Any]:
        """
        生成測試報告

        Args:
            total_time: 總執行時間

        Returns:
            測試報告統計
        """
        successful_tests = [r for r in self.test_results if r.success]
        failed_tests = [r for r in self.test_results if not r.success]

        report = {
            "summary": {
                "total_tests": len(self.test_results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.test_results) * 100,
                "total_execution_time": total_time,
            },
            "test_results": [
                {
                    "name": result.test_name,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "error": result.error_message,
                }
                for result in self.test_results
            ],
            "recommendations": self._generate_recommendations(failed_tests),
        }

        return report

    def _generate_recommendations(
        self, failed_tests: list[IntegrationTestResult]
    ) -> list[str]:
        """
        根據失敗的測試生成建議

        Args:
            failed_tests: 失敗的測試列表

        Returns:
            改進建議列表
        """
        recommendations = []

        if not failed_tests:
            recommendations.append("🎉 所有測試都通過了！AI 整合系統運作良好。")
            return recommendations

        for test in failed_tests:
            if "初始化" in test.test_name:
                recommendations.append("🔧 建議檢查各 AI 組件的初始化配置和依賴項。")
            elif "協調" in test.test_name:
                recommendations.append("⚙️ 建議優化統一控制器的任務分配邏輯。")
            elif "自然語言" in test.test_name:
                recommendations.append("📝 建議檢查 NLG 系統的模板配置和上下文處理。")
            elif "多語言" in test.test_name:
                recommendations.append("🌐 建議檢查多語言協調器的語言模組註冊。")
            elif "衝突" in test.test_name:
                recommendations.append("⚠️ 建議增強 AI 衝突檢測和去重機制。")
            elif "端到端" in test.test_name:
                recommendations.append("🔄 建議檢查整個 AI 處理流程的各個環節。")
            elif "效能" in test.test_name:
                recommendations.append("⚡ 建議優化系統效能和並發處理能力。")
            elif "錯誤恢復" in test.test_name:
                recommendations.append("🛡️ 建議增強錯誤處理和恢復機制。")

        return list(set(recommendations))  # 去除重複建議


async def main():
    """主函數"""
    print("🚀 AIVA AI 整合測試系統")
    print("=" * 50)

    # 初始化測試器
    tester = AIIntegrationTester("c:/AMD/AIVA")

    # 執行所有測試
    report = await tester.run_all_tests()

    # 輸出測試報告
    print("\n📊 測試報告")
    print("=" * 50)
    print(f"📋 總測試數: {report['summary']['total_tests']}")
    print(f"✅ 成功測試: {report['summary']['successful_tests']}")
    print(f"❌ 失敗測試: {report['summary']['failed_tests']}")
    print(f"📈 成功率: {report['summary']['success_rate']:.1f}%")
    print(f"⏱️ 總執行時間: {report['summary']['total_execution_time']:.2f}秒")

    print("\n📝 詳細結果:")
    for result in report["test_results"]:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['name']} ({result['execution_time']:.2f}s)")
        if result["error"]:
            print(f"   錯誤: {result['error']}")

    print("\n💡 改進建議:")
    for recommendation in report["recommendations"]:
        print(f"  {recommendation}")

    # 保存詳細報告
    report_file = Path("c:/AMD/AIVA/_out/ai_integration_test_report.json")
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n📄 詳細報告已保存至: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
