#!/usr/bin/env python3
"""embedded_knowledge 模組集成示例
展示如何在 EnhancedDecisionAgent 中使用安全知識引擎

使用场景:
1. 决策前调用漏洞检测器进行预判
2. 发现CVE时自动识别高危目标
3. 遇到WAF时自动生成绕过payload
4. 分析现代Web架构（GraphQL/JWT/WebSocket）
"""

from typing import Any
from ..decision.enhanced_decision_agent import EnhancedDecisionAgent, DecisionContext
from . import (
    VulnerabilityDetector,
    CVEIdentifier,
    WAFBypassEngine,
    WebArchitectureAnalyzer,
    AttackContext,
)


# ============================================================================
# 示例 1: 在决策流程中集成漏洞检测器
# ============================================================================

class EnhancedDecisionWithKnowledge(EnhancedDecisionAgent):
    """增强版决策代理 - 集成 embedded_knowledge"""
    
    def __init__(self, knowledge_base=None, experience_manager=None):
        super().__init__(knowledge_base, experience_manager)
        
        # 初始化知识引擎
        self.vuln_detector = VulnerabilityDetector()
        self.cve_identifier = CVEIdentifier()
        self.waf_bypass = WAFBypassEngine()
        self.web_analyzer = WebArchitectureAnalyzer()
        
        self.logger.info("🔐 Embedded Knowledge 已集成")
    
    def decide_with_vuln_check(self, context: DecisionContext, response_data: dict[str, Any]):
        """决策前进行漏洞检测
        
        Args:
            context: 决策上下文
            response_data: 目标响应数据（包含 headers, body, status_code）
        
        Returns:
            HighLevelIntent: 决策结果（包含检测到的漏洞信息）
        """
        # 1. 构建攻击上下文
        attack_ctx = AttackContext(
            target_url=context.target_info.get("url", ""),
            method=response_data.get("method", "GET"),
            parameters=context.target_info.get("params", {}),
            headers=response_data.get("headers", {}),
            cookies=response_data.get("cookies", {})
        )
        
        # 2. 检测 SQLi
        sqli_result = self.vuln_detector.check_sqli(attack_ctx, response_data)
        if sqli_result.is_vulnerable:
            self.logger.warning(f"⚠️ SQLi 检测: {sqli_result.vulnerability_type} "
                              f"(置信度: {sqli_result.confidence.value})")
            context.discovered_vulns.append("sql_injection")
            # 更新风险等级
            from services.aiva_common.enums import RiskLevel
            context.risk_level = RiskLevel.HIGH
        
        # 3. 检测 XSS
        xss_result = self.vuln_detector.check_xss(attack_ctx, response_data)
        if xss_result.is_vulnerable:
            self.logger.warning(f"⚠️ XSS 检测: {xss_result.vulnerability_type}")
            context.discovered_vulns.append("xss")
        
        # 4. 检测 SSRF
        ssrf_result = self.vuln_detector.check_ssrf(attack_ctx, response_data)
        if ssrf_result.is_vulnerable:
            self.logger.critical(f"🔥 SSRF 检测: {ssrf_result.vulnerability_type}")
            context.discovered_vulns.append("ssrf")
            from services.aiva_common.enums import RiskLevel
            context.risk_level = RiskLevel.CRITICAL
        
        # 5. 调用原始决策流程
        intent = self.decide(context)
        
        # 6. 附加漏洞信息到 metadata
        intent.metadata["vuln_detection"] = {
            "sqli": sqli_result.to_dict(),
            "xss": xss_result.to_dict(),
            "ssrf": ssrf_result.to_dict()
        }
        
        return intent


# ============================================================================
# 示例 2: CVE 自动识别
# ============================================================================

def identify_high_risk_target(target_url: str, response_headers: dict[str, str]) -> dict[str, Any]:
    """识别目标是否存在高危 CVE
    
    使用场景: 在扫描阶段自动识别 Log4Shell、Spring4Shell 等高危目标
    
    Args:
        target_url: 目标 URL
        response_headers: 响应头（包含 Server, X-Powered-By 等）
    
    Returns:
        识别结果（包含 CVE 列表和风险等级）
    """
    cve_identifier = CVEIdentifier()
    
    # 识别 CVE
    result = cve_identifier.identify_by_headers(response_headers)
    
    if result.is_vulnerable:
        print(f"🔥 高危目标: {target_url}")
        print(f"   CVE: {result.vulnerability_type}")
        print(f"   置信度: {result.confidence.to_score()}")
        print(f"   利用建议: {result.recommendations[:2]}")  # 只显示前2条
        
        return {
            "is_high_risk": True,
            "cve_id": result.vulnerability_type,
            "confidence": result.confidence.to_score(),
            "exploit_suggestions": result.recommendations,
            "technical_details": result.technical_details
        }
    else:
        return {"is_high_risk": False}


# ============================================================================
# 示例 3: WAF 自动绕过
# ============================================================================

def bypass_waf_automatically(target_url: str, response_headers: dict[str, str], 
                            original_payload: str) -> list[str]:
    """自动检测 WAF 并生成绕过 payload
    
    使用场景: 在 SQL注入/XSS 测试中遇到 WAF 拦截，自动生成绕过变体
    
    Args:
        target_url: 目标 URL
        response_headers: 响应头
        original_payload: 原始 payload（如 ' OR 1=1--）
    
    Returns:
        绕过 payload 列表
    """
    waf_bypass = WAFBypassEngine()
    
    # 1. 检测 WAF
    waf_vendor = waf_bypass.detect_waf(response_headers)
    
    if waf_vendor:
        print(f"🛡️ 检测到 WAF: {waf_vendor}")
        
        # 2. 获取绕过技术
        techniques = waf_bypass.get_bypass_techniques(waf_vendor)
        print(f"   可用绕过技术: {len(techniques)} 种")
        
        # 3. 生成变异 payload
        mutated_payloads = waf_bypass.mutate_payload(
            original_payload, 
            waf_vendor, 
            attack_type="sqli"
        )
        
        print(f"   生成变异 payload: {len(mutated_payloads)} 个")
        print(f"   示例: {mutated_payloads[0][:100]}...")
        
        return mutated_payloads
    else:
        print("✅ 未检测到 WAF")
        return [original_payload]


# ============================================================================
# 示例 4: GraphQL 架构分析
# ============================================================================

def analyze_graphql_api(graphql_endpoint: str, response_body: str) -> dict[str, Any]:
    """分析 GraphQL API 的安全配置
    
    使用场景: 发现 GraphQL 端点时，自动检测是否启用 introspection、是否存在 BOLA
    
    Args:
        graphql_endpoint: GraphQL 端点 URL
        response_body: introspection 查询的响应
    
    Returns:
        分析结果（包含漏洞和建议）
    """
    web_analyzer = WebArchitectureAnalyzer()
    
    # 检测 introspection 是否启用
    introspection_result = web_analyzer.check_graphql_introspection(
        AttackContext(target_url=graphql_endpoint),
        {"body": response_body}
    )
    
    if introspection_result.is_vulnerable:
        print(f"⚠️ GraphQL Introspection 已启用")
        print(f"   风险: 攻击者可枚举所有 schema")
        print(f"   建议: {introspection_result.recommendations[0]}")
        
        return {
            "introspection_enabled": True,
            "risk_level": "MEDIUM",
            "next_steps": [
                "枚举所有 queries/mutations",
                "检测 BOLA (Broken Object Level Authorization)",
                "测试批量查询限制"
            ]
        }
    else:
        return {"introspection_enabled": False}


# ============================================================================
# 示例 5: 完整的决策流程集成
# ============================================================================

async def full_decision_pipeline_example():
    """完整的决策流程示例
    
    展示如何在一次攻击任务中使用所有知识引擎
    """
    # 1. 初始化增强决策代理
    agent = EnhancedDecisionWithKnowledge()
    
    # 2. 模拟目标响应
    target_response = {
        "url": "https://example.com/api/user",
        "method": "GET",
        "headers": {
            "Server": "Apache/2.4.49",  # 高危版本
            "X-Powered-By": "PHP/7.4.3"
        },
        "body": "Welcome to our API",
        "status_code": 200
    }
    
    # 3. 创建决策上下文
    context = DecisionContext()
    context.target_info = {
        "url": target_response["url"],
        "params": {"id": "1"}
    }
    
    # 4. 执行决策（自动调用漏洞检测）
    intent = agent.decide_with_vuln_check(context, target_response)
    
    print(f"✅ 决策完成: {intent.intent_type}")
    print(f"   检测到的漏洞: {context.discovered_vulns}")
    print(f"   风险等级: {context.risk_level}")
    print(f"   详细报告: {intent.metadata.get('vuln_detection')}")
    
    # 5. 如果检测到 SQLi，尝试绕过 WAF
    if "sql_injection" in context.discovered_vulns:
        bypass_payloads = bypass_waf_automatically(
            target_response["url"],
            target_response["headers"],
            "' OR 1=1--"
        )
        print(f"   WAF 绕过 payload 已生成: {len(bypass_payloads)} 个")


# ============================================================================
# 快速开始指南
# ============================================================================

if __name__ == "__main__":
    """快速测试脚本"""
    
    print("=" * 60)
    print("embedded_knowledge 集成示例")
    print("=" * 60)
    
    # 示例 1: 漏洞检测
    print("\n[示例 1] SQLi 检测")
    detector = VulnerabilityDetector()
    test_response = {
        "body": "You have an error in your SQL syntax",
        "headers": {},
        "status_code": 500
    }
    ctx = AttackContext(target_url="https://test.com", method="GET")
    result = detector.check_sqli(ctx, test_response)
    print(f"  结果: {'漏洞' if result.is_vulnerable else '无漏洞'}")
    print(f"  置信度: {result.confidence.value}")
    
    # 示例 2: CVE 识别
    print("\n[示例 2] CVE 识别")
    cve_result = identify_high_risk_target(
        "https://vulnerable.com",
        {"Server": "Apache/2.4.49"}
    )
    print(f"  高危目标: {cve_result.get('is_high_risk', False)}")
    
    # 示例 3: WAF 检测
    print("\n[示例 3] WAF 检测")
    waf_payloads = bypass_waf_automatically(
        "https://protected.com",
        {"Server": "cloudflare"},
        "' OR 1=1--"
    )
    print(f"  生成 payload: {len(waf_payloads)} 个")
    
    # 示例 4: GraphQL 分析
    print("\n[示例 4] GraphQL 分析")
    graphql_result = analyze_graphql_api(
        "https://api.example.com/graphql",
        '{"data": {"__schema": {"types": []}}}'
    )
    print(f"  Introspection: {graphql_result.get('introspection_enabled', False)}")
    
    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成")
    print("=" * 60)
