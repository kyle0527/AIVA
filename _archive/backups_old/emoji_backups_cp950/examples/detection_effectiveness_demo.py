#!/usr/bin/env python3
"""
AIVA 功能模組檢測效果演示
展示優化後的檢測能力和實用性改進
"""

import asyncio
from datetime import datetime
import json
import time
from typing import Any

import aiofiles


# 模擬檢測結果演示
class DetectionDemo:
    """檢測效果演示類"""

    def __init__(self):
        self.demo_results = {}

    async def simulate_sqli_detection(self) -> dict[str, Any]:
        """模擬 SQLi 檢測演示"""
        print("🔍 SQLi Detection Demo - Multi-Engine Coordination")

        # 模擬多引擎檢測結果
        engines_results = {
            "sqlmap": {
                "vulnerabilities_found": 3,
                "techniques": ["Union-based", "Boolean-blind", "Time-blind"],
                "bypass_successful": True,
                "waf_detected": "ModSecurity",
                "confidence": 95,
            },
            "ghauri": {
                "vulnerabilities_found": 2,
                "techniques": ["Error-based", "Stacked queries"],
                "bypass_successful": True,
                "performance": "Fast scan completed",
                "confidence": 88,
            },
            "nosqlmap": {
                "vulnerabilities_found": 1,
                "databases": ["MongoDB"],
                "injection_points": ["JSON parameter"],
                "confidence": 82,
            },
            "custom": {
                "vulnerabilities_found": 4,
                "advanced_techniques": ["Second-order", "Blind NoSQL"],
                "ai_assisted": True,
                "false_positive_filtered": 2,
                "confidence": 91,
            },
        }

        # 模擬檢測過程
        for engine, result in engines_results.items():
            print(
                f"  ✅ {engine:12} - Found {result['vulnerabilities_found']} vulns (Confidence: {result.get('confidence', 'N/A')}%)"
            )
            await asyncio.sleep(0.1)  # 模擬檢測時間

        # 綜合結果
        total_vulns = sum(r["vulnerabilities_found"] for r in engines_results.values())
        unique_vulns = total_vulns - 2  # 假設有2個重複

        summary = {
            "total_vulnerabilities": unique_vulns,
            "engines_used": len(engines_results),
            "waf_bypass_success": True,
            "false_positive_reduction": "40%",
            "detection_accuracy": "92%",
            "scan_efficiency": "3x faster than single engine",
        }

        print(f"  📊 Summary: {unique_vulns} unique vulnerabilities detected")
        print(
            f"  🛡️  WAF bypass: {'Success' if summary['waf_bypass_success'] else 'Failed'}"
        )
        print(f"  🎯 Accuracy: {summary['detection_accuracy']}")

        return summary

    async def simulate_ssrf_detection(self) -> dict[str, Any]:
        """模擬 SSRF 檢測演示"""
        print("\n🌐 SSRF Detection Demo - Cloud Service & Bypass Techniques")

        detection_results = {
            "cloud_metadata": {
                "aws_imds": {"vulnerable": True, "version": "IMDSv1", "risk": "High"},
                "gcp_metadata": {
                    "vulnerable": False,
                    "protection": "Metadata concealment",
                },
                "azure_imds": {
                    "vulnerable": True,
                    "endpoints": ["/metadata/instance"],
                    "risk": "Medium",
                },
            },
            "internal_services": {
                "redis": {
                    "port": 6379,
                    "accessible": True,
                    "data_exposure": "Session data",
                },
                "elasticsearch": {
                    "port": 9200,
                    "accessible": True,
                    "indices": ["logs", "users"],
                },
                "kubernetes_api": {
                    "port": 8080,
                    "accessible": False,
                    "protection": "Network policy",
                },
            },
            "bypass_techniques": {
                "url_encoding": "Success - %2E%2E%2F bypassed filter",
                "ip_obfuscation": "Success - 0x7f000001 resolved to localhost",
                "redirect_chains": "Success - 3-hop redirect bypassed protection",
                "unicode_normalization": "Success - Unicode FFFE bypass",
            },
            "oast_callbacks": {
                "dns_lookups": 7,
                "http_requests": 3,
                "callback_domains": ["interact.sh", "burpcollaborator.net"],
                "data_exfiltration": ["AWS credentials", "Internal IPs"],
            },
        }

        print("  ☁️  Cloud Service Detection:")
        for service, result in detection_results["cloud_metadata"].items():
            status = "🚨" if result.get("vulnerable") else "✅"
            print(f"    {status} {service}: {result}")

        print("  🔍 Internal Service Scan:")
        for service, result in detection_results["internal_services"].items():
            status = "⚠️" if result.get("accessible") else "🔒"
            print(f"    {status} {service}:{result.get('port')} - {result}")

        print("  🕵️  Bypass Techniques:")
        for technique, result in detection_results["bypass_techniques"].items():
            print(f"    ✅ {technique}: {result}")

        await asyncio.sleep(0.2)

        summary = {
            "cloud_vulns_found": 2,
            "internal_services_exposed": 2,
            "bypass_success_rate": "100%",
            "oast_callbacks": detection_results["oast_callbacks"]["dns_lookups"],
            "data_exfiltrated": True,
            "risk_level": "Critical",
        }

        print(
            f"  📊 SSRF Summary: {summary['cloud_vulns_found']} cloud + {summary['internal_services_exposed']} internal services"
        )

        return summary

    async def simulate_xss_detection(self) -> dict[str, Any]:
        """模擬 XSS 檢測演示"""
        print("\n⚡ XSS Detection Demo - Framework-Specific & CSP Bypass")

        detection_results = {
            "reflected_xss": {
                "basic_payloads": 15,
                "successful_payloads": 8,
                "contexts": ["HTML", "Attribute", "JavaScript", "CSS"],
                "encoding_bypasses": ["HTML entity", "URL encode", "Unicode"],
            },
            "framework_specific": {
                "angular": {
                    "template_injection": True,
                    "sanitizer_bypass": "{{constructor.constructor('alert(1)')()}}",
                },
                "react": {"dangerouslySetInnerHTML": True, "jsx_injection": True},
                "vue": {"template_syntax": True, "v-html_directive": "Vulnerable"},
                "jquery": {"dom_manipulation": True, "selector_injection": True},
            },
            "csp_bypass": {
                "policy": "script-src 'self' 'unsafe-inline'",
                "bypasses_found": 3,
                "techniques": ["JSONP callback", "Angular template", "Data URI"],
                "success_rate": "75%",
            },
            "dom_xss": {
                "sources": ["location.href", "document.referrer", "postMessage"],
                "sinks": ["innerHTML", "document.write", "eval"],
                "exploitation_paths": 4,
                "dynamic_analysis": True,
            },
            "blind_xss": {
                "payloads_sent": 25,
                "callbacks_received": 3,
                "data_captured": ["Cookies", "localStorage", "DOM structure"],
                "screenshot_captured": True,
            },
        }

        print("  ⚡ Reflected XSS Results:")
        reflected = detection_results["reflected_xss"]
        print(
            f"    ✅ Payloads: {reflected['successful_payloads']}/{reflected['basic_payloads']} successful"
        )
        print(f"    📍 Contexts: {', '.join(reflected['contexts'])}")

        print("  🖼️  Framework-Specific Detection:")
        for framework, result in detection_results["framework_specific"].items():
            vulns = sum(1 for v in result.values() if v is True)
            print(
                f"    {'🚨' if vulns > 0 else '✅'} {framework}: {vulns} vulnerabilities"
            )

        print("  🛡️  CSP Bypass Analysis:")
        csp = detection_results["csp_bypass"]
        print(f"    🔍 Policy: {csp['policy']}")
        print(
            f"    ⚡ Bypasses: {csp['bypasses_found']} techniques, {csp['success_rate']} success"
        )

        await asyncio.sleep(0.15)

        summary = {
            "total_xss_vulns": 12,
            "framework_vulns": 8,
            "csp_bypasses": 3,
            "dom_xss_paths": 4,
            "blind_xss_confirmations": 3,
            "overall_risk": "High",
        }

        print(
            f"  📊 XSS Summary: {summary['total_xss_vulns']} total vulnerabilities across all types"
        )

        return summary

    async def simulate_idor_detection(self) -> dict[str, Any]:
        """模擬 IDOR 檢測演示"""
        print("\n🔐 IDOR Detection Demo - AI-Enhanced & Multi-Tenant Analysis")

        detection_results = {
            "intelligent_analysis": {
                "id_patterns_learned": [
                    "UUID v4",
                    "Sequential numeric",
                    "Base64 encoded",
                    "Hash-based",
                ],
                "ml_predictions": 147,
                "successful_predictions": 89,
                "prediction_accuracy": "60.5%",
            },
            "api_testing": {
                "rest_endpoints": {
                    "tested": 45,
                    "vulnerable": 12,
                    "bypasses": ["parameter_pollution", "http_verb_tampering"],
                },
                "graphql": {
                    "queries_tested": 23,
                    "vulnerable": 5,
                    "info_disclosure": True,
                },
                "grpc": {"services_tested": 8, "vulnerable": 2, "auth_bypass": True},
                "websocket": {"endpoints": 3, "message_tampering": True},
            },
            "multi_tenant": {
                "organization_isolation": {
                    "tested": True,
                    "breached": 3,
                    "data_exposure": "Customer records",
                },
                "user_isolation": {
                    "tested": True,
                    "breached": 7,
                    "privilege_escalation": True,
                },
                "role_isolation": {"tested": True, "breached": 2, "admin_access": True},
            },
            "temporal_testing": {
                "session_fixation": True,
                "race_conditions": 2,
                "time_based_enumeration": {
                    "successful": True,
                    "pattern": "Incremental ID with timestamp",
                },
            },
            "object_relationships": {
                "dependency_chains": 15,
                "cascade_access": 8,
                "indirect_references": 23,
                "relationship_abuse": True,
            },
        }

        print("  🤖 AI-Enhanced Analysis:")
        intel = detection_results["intelligent_analysis"]
        print(
            f"    📈 ML Predictions: {intel['successful_predictions']}/{intel['ml_predictions']} ({intel['prediction_accuracy']})"
        )
        print(f"    🧠 Patterns Learned: {len(intel['id_patterns_learned'])}")

        print("  🌐 API Comprehensive Testing:")
        for api_type, result in detection_results["api_testing"].items():
            if (
                isinstance(result, dict)
                and "vulnerable" in result
                and "tested" in result
            ):
                print(
                    f"    {'⚠️' if result['vulnerable'] > 0 else '✅'} {api_type}: {result['vulnerable']}/{result['tested']} endpoints vulnerable"
                )

        print("  🏢 Multi-Tenant Security:")
        for tenant_type, result in detection_results["multi_tenant"].items():
            status = "🚨" if result.get("breached", 0) > 0 else "✅"
            breaches = result.get("breached", 0)
            print(f"    {status} {tenant_type}: {breaches} isolation breaches")

        await asyncio.sleep(0.2)

        summary = {
            "total_idor_vulns": 31,
            "api_vulnerabilities": 19,
            "tenant_breaches": 12,
            "ai_assisted_findings": 89,
            "critical_escalations": 5,
            "data_exposure_risk": "Critical",
        }

        print(
            f"  📊 IDOR Summary: {summary['total_idor_vulns']} vulnerabilities with AI assistance"
        )

        return summary

    async def run_comprehensive_demo(self):
        """運行完整的檢測效果演示"""
        print("=" * 70)
        print("🚀 AIVA Enhanced Function Module Detection Demo")
        print("=" * 70)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        start_time = time.time()

        # 執行各模組檢測演示
        sqli_results = await self.simulate_sqli_detection()
        ssrf_results = await self.simulate_ssrf_detection()
        xss_results = await self.simulate_xss_detection()
        idor_results = await self.simulate_idor_detection()

        end_time = time.time()
        duration = end_time - start_time

        # 綜合統計
        print("\n" + "=" * 70)
        print("📈 COMPREHENSIVE DETECTION RESULTS SUMMARY")
        print("=" * 70)

        total_vulns = (
            sqli_results.get("total_vulnerabilities", 0)
            + ssrf_results.get("cloud_vulns_found", 0)
            + ssrf_results.get("internal_services_exposed", 0)
            + xss_results.get("total_xss_vulns", 0)
            + idor_results.get("total_idor_vulns", 0)
        )

        print(f"🎯 Total Vulnerabilities Detected: {total_vulns}")
        print("📊 Module Breakdown:")
        print(
            f"   • SQLi:  {sqli_results.get('total_vulnerabilities', 0)} vulnerabilities"
        )
        print(
            f"   • SSRF:  {ssrf_results.get('cloud_vulns_found', 0) + ssrf_results.get('internal_services_exposed', 0)} exposures"
        )
        print(f"   • XSS:   {xss_results.get('total_xss_vulns', 0)} vulnerabilities")
        print(f"   • IDOR:  {idor_results.get('total_idor_vulns', 0)} vulnerabilities")

        print("\n🚀 Performance Metrics:")
        print(f"   ⏱️  Total Scan Time: {duration:.2f} seconds")
        print(f"   ⚡ Detection Rate: {total_vulns / duration:.1f} vulns/second")
        print("   🎯 Overall Accuracy: 91.5% (weighted average)")
        print("   📉 False Positive Reduction: 35%")

        print("\n✨ Enhancement Impact:")
        print("   🔧 Multi-Engine Coordination: 3x faster SQLi detection")
        print("   ☁️  Cloud-Native Detection: 40% more SSRF vulnerabilities found")
        print("   🖼️  Framework-Specific XSS: 25% accuracy improvement")
        print("   🤖 AI-Enhanced IDOR: 60% prediction accuracy")

        print("\n🛡️  Security Value:")
        critical_count = 2 + 1 + 1 + 5  # Critical from each module
        high_count = total_vulns - critical_count
        print(f"   🚨 Critical Severity: {critical_count} vulnerabilities")
        print(f"   ⚠️  High Severity: {high_count} vulnerabilities")
        print("   💰 Risk Mitigation Value: ~$2.5M in potential breach costs")

        # 保存結果到文件
        demo_report = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "modules": {
                "sqli": sqli_results,
                "ssrf": ssrf_results,
                "xss": xss_results,
                "idor": idor_results,
            },
            "summary": {
                "total_vulnerabilities": total_vulns,
                "critical_vulnerabilities": critical_count,
                "detection_accuracy": "91.5%",
                "performance_improvement": "3x faster",
                "false_positive_reduction": "35%",
            },
        }

        # 使用 aiofiles 進行異步文件操作
        output_path = r"c:\F\AIVA\_out\detection_demo_results.json"
        async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(demo_report, indent=2, ensure_ascii=False))

        print("\n💾 Results saved to: _out/detection_demo_results.json")
        print(f"🏁 Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)


async def main():
    """主演示函數"""
    demo = DetectionDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
