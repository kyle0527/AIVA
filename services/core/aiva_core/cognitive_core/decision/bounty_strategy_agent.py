"""Bug Bounty 特化決策代理

從 enhanced_decision_agent.py 拆分而來。
負責 Bug Bounty 外閉環的 Phase 1/2 策略決策。

對應 13 步驟外閉環:
- Step 6: decide_phase1_strategy() — Phase1 深度掃描決策
- Step 9: decide_phase2_targets() — Phase2 目標排序
- Step 11: evaluate_phase2_results() — 結果評估與行動決策

Architecture:
    - 依賴 EnhancedDecisionAgent 的 neural_engine / experience_manager
    - 由 EnhancedDecisionAgent 透過 mixin / 委託調用
    - 符合 aiva_common 數據合約規範

Date: 2026-02-09 (拆分自 enhanced_decision_agent.py)
"""

import logging
from datetime import datetime
from typing import Any, Optional

from aiva_common.utils import get_logger

logger = get_logger(__name__)


class BountyStrategyAgent:
    """Bug Bounty 特化策略決策代理

    負責 Phase 1/2 外閉環的策略決策，包含:
    - Phase1 深度掃描決策 (Step 6)
    - Phase2 目標優先級排序 (Step 9)
    - Phase2 結果評估 (Step 11)
    - WAF/Rate Limit 自適應策略
    """

    def __init__(
        self,
        neural_engine: Any = None,
        experience_manager: Any = None,
        use_neural_decision: bool = False,
    ):
        self.logger = logger
        self.neural_engine = neural_engine
        self.experience_manager = experience_manager
        self.use_neural_decision = use_neural_decision

        # 策略初始化
        self.waf_bypass_strategies = self._initialize_waf_strategies()
        self.rate_limit_profiles = self._initialize_rate_profiles()

    # ═══════════════════════════════════════════════════════════════
    # Phase 1 決策 (Step 6)
    # ═══════════════════════════════════════════════════════════════

    def decide_phase1_strategy(
        self,
        phase0_result: dict[str, Any],
        program_scope: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """步驟 6: AI 決策是否需要 Phase1 深度掃描

        實際 Bug Bounty 場景考量：
        1. Program Scope 限制（必須遵守範圍）
        2. Rate Limiting 和 WAF 檢測（避免被封禁）
        3. 高價值漏洞類型優先（SSRF > SQLi > XSS）
        4. 歷史數據：同類型目標成功率

        OWASP WSTG 對應：
        - 4.1 Information Gathering → Phase0
        - 4.2 Configuration Testing → Phase1 起點
        - 4.7 Input Validation Testing → Phase1 核心

        Args:
            phase0_result: Phase0 偵察結果
            program_scope: Program 範圍限制

        Returns:
            決策結果字典
        """
        self.logger.info("🧠 [Step 6: Phase1 Strategy] 開始 AI 決策...")

        # === 1. 提取 Phase0 情報 ===
        summary = phase0_result.get("summary", {})
        fingerprints = phase0_result.get("fingerprints", {})
        recommendations = phase0_result.get("recommendations", {})
        endpoints = phase0_result.get("endpoints", [])

        urls_found = summary.get("urls_found", 0)
        forms_found = summary.get("forms_found", 0)
        apis_found = summary.get("apis_found", 0)
        subdomains_found = summary.get("subdomains_found", 0)
        waf_detected = fingerprints.get("waf_detected", False)
        waf_vendor = fingerprints.get("waf_vendor")
        technologies = fingerprints.get("technologies", [])

        # === 2. Program Scope 合規檢查 ===
        testing_restrictions: dict[str, Any] = {}
        if program_scope:
            testing_restrictions = program_scope.get("testing_restrictions", {})
            if testing_restrictions.get("no_automated_tools"):
                self.logger.warning("   ⚠️ Program 禁止自動化工具，切換為手動輔助模式")

        # === 3. 高價值目標識別 ===
        high_value_indicators = {
            "api_endpoints": apis_found,
            "file_upload_forms": sum(
                1 for e in endpoints if "upload" in e.get("url", "").lower()
            ),
            "auth_endpoints": sum(
                1
                for e in endpoints
                if any(
                    k in e.get("url", "").lower()
                    for k in ["login", "auth", "oauth", "token", "session"]
                )
            ),
            "admin_panels": sum(
                1
                for e in endpoints
                if any(
                    k in e.get("url", "").lower()
                    for k in ["admin", "dashboard", "manage", "config"]
                )
            ),
            "payment_flows": sum(
                1
                for e in endpoints
                if any(
                    k in e.get("url", "").lower()
                    for k in ["payment", "checkout", "billing", "subscription"]
                )
            ),
            "user_input_heavy": forms_found,
            "graphql_endpoints": sum(
                1 for e in endpoints if "graphql" in e.get("url", "").lower()
            ),
        }

        high_value_score = min(
            1.0,
            (
                high_value_indicators["api_endpoints"] * 0.15
                + high_value_indicators["file_upload_forms"] * 0.2
                + high_value_indicators["auth_endpoints"] * 0.15
                + high_value_indicators["admin_panels"] * 0.2
                + high_value_indicators["payment_flows"] * 0.25
                + high_value_indicators["graphql_endpoints"] * 0.1
            )
            / 5.0,
        )

        self.logger.info(f"   📊 高價值目標評分: {high_value_score:.2f}")
        self.logger.info(
            f"   📊 發現: {apis_found} APIs, {forms_found} Forms, {subdomains_found} Subdomains"
        )

        # === 4. 技術棧風險評估 ===
        tech_risk_multiplier = 1.0
        tech_insights: list[str] = []

        high_risk_techs = {
            "php": 1.3,
            "wordpress": 1.4,
            "struts": 1.5,
            "spring": 1.2,
            "laravel": 1.1,
            "rails": 1.1,
            "node": 1.0,
            "java": 1.2,
            "asp.net": 1.1,
        }

        for tech in technologies:
            tech_lower = tech.lower()
            for risk_tech, multiplier in high_risk_techs.items():
                if risk_tech in tech_lower:
                    tech_risk_multiplier = max(tech_risk_multiplier, multiplier)
                    tech_insights.append(f"{tech} (風險係數 {multiplier})")

        # === 5. 神經網路決策 ===
        ai_confidence = 0.5
        ai_attack_vector = "reconnaissance"
        ai_recommended_focus: list[str] = []

        if self.use_neural_decision and self.neural_engine:
            try:
                neural_context = (
                    f"Program: {program_scope.get('name', 'unknown') if program_scope else 'unknown'} | "
                    f"Scope: {urls_found} URLs, {apis_found} APIs, {forms_found} Forms | "
                    f"Tech: {','.join(technologies[:5])} | "
                    f"WAF: {waf_vendor if waf_detected else 'None'} | "
                    f"HighValue: {high_value_score:.2f} | "
                    f"Upload: {high_value_indicators['file_upload_forms']}, Admin: {high_value_indicators['admin_panels']}"
                )

                ai_result = self.neural_engine.generate_decision(
                    task_description="decide_phase1_strategy",
                    context=neural_context,
                )

                ai_confidence = ai_result.get("confidence", 0.5)
                ai_attack_vector = ai_result.get("attack_vector", "reconnaissance")
                ai_recommended_focus = ai_result.get("recommended_tools", [])

                self.logger.info(
                    f"   🧠 AI 信心度: {ai_confidence:.2f}, 推薦向量: {ai_attack_vector}"
                )

            except Exception as e:
                self.logger.warning(f"   ⚠️ 神經網路決策回退: {e}")

        # === 6. 時間預估 ===
        estimated_time_hours = self._estimate_phase1_time(phase0_result) / 3600.0

        # === 7. 決策邏輯 ===
        need_phase1 = False
        reasoning_parts: list[str] = []
        priority_targets: list[dict[str, Any]] = []

        if high_value_score > 0.5:
            need_phase1 = True
            reasoning_parts.append(f"高價值目標評分 {high_value_score:.2f}")

            if high_value_indicators["payment_flows"] > 0:
                priority_targets.append(
                    {"type": "payment", "priority": 1, "vuln_focus": ["idor", "logic_bypass", "race_condition"]}
                )
            if high_value_indicators["admin_panels"] > 0:
                priority_targets.append(
                    {"type": "admin", "priority": 2, "vuln_focus": ["auth_bypass", "privilege_escalation"]}
                )
            if high_value_indicators["file_upload_forms"] > 0:
                priority_targets.append(
                    {"type": "upload", "priority": 3, "vuln_focus": ["file_upload_rce", "path_traversal"]}
                )
            if high_value_indicators["api_endpoints"] > 3:
                priority_targets.append(
                    {"type": "api", "priority": 4, "vuln_focus": ["idor", "mass_assignment", "rate_limit_bypass"]}
                )

        attack_surface_score = urls_found * 0.01 + forms_found * 0.05 + apis_found * 0.1
        if attack_surface_score > 1.0:
            need_phase1 = True
            reasoning_parts.append(f"廣大攻擊面 (surface={attack_surface_score:.2f})")

        if waf_detected:
            reasoning_parts.append(f"WAF: {waf_vendor or 'unknown'}")
            if ai_confidence > 0.65:
                need_phase1 = True
                reasoning_parts.append("AI 建議嘗試 WAF 繞過")
            else:
                reasoning_parts.append("WAF 存在，謹慎評估")

        if tech_risk_multiplier > 1.2:
            need_phase1 = True
            reasoning_parts.append(f"高風險技術棧: {', '.join(tech_insights)}")

        if recommendations.get("needs_deep_scan") or recommendations.get("priority_high"):
            need_phase1 = True
            reasoning_parts.append("Phase0 強烈建議深度掃描")

        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "預設保守策略"

        # === 8. 構建返回結果 ===
        result: dict[str, Any] = {
            "need_phase1": need_phase1,
            "reasoning": reasoning,
            "decision_source": "neural_network" if ai_confidence > 0.6 else "rule_engine",
            "estimated_time_hours": estimated_time_hours,
            "ai_confidence": ai_confidence,
            "ai_attack_vector": ai_attack_vector,
            "ai_recommended_focus": ai_recommended_focus,
            "high_value_score": high_value_score,
            "high_value_indicators": high_value_indicators,
            "priority_targets": priority_targets,
            "tech_risk_multiplier": tech_risk_multiplier,
            "tech_insights": tech_insights,
            "technologies_detected": technologies,
            "phase1_config": {
                "scan_depth": "intensive" if high_value_score > 0.7 else "standard",
                "focus_areas": [t["type"] for t in priority_targets[:3]],
                "time_budget_minutes": int(estimated_time_hours * 60),
                "parallel_workers": 3 if not waf_detected else 1,
            },
        }

        if waf_detected:
            result["waf_bypass_plan"] = self._decide_waf_bypass(waf_vendor, ai_confidence)
            result["phase1_config"]["stealth_mode"] = True
            result["phase1_config"]["delay_between_requests"] = result["waf_bypass_plan"].get(
                "delay_multiplier", 2.0
            )
            self.logger.info(
                f"   🛡️ WAF 繞過策略已配置: delay×{result['waf_bypass_plan'].get('delay_multiplier', 1)}"
            )

        self.logger.info(
            f"   ✅ 決策: {'執行 Phase1 (優先: {})'.format(','.join([t['type'] for t in priority_targets[:2]])) if need_phase1 else '跳過 Phase1'}"
        )
        return result

    # ═══════════════════════════════════════════════════════════════
    # Phase 2 目標排序 (Step 9)
    # ═══════════════════════════════════════════════════════════════

    def decide_phase2_targets(
        self,
        phase1_result: dict[str, Any],
        max_targets: int = 10,
        program_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """步驟 9: AI 決策 Phase2 攻擊目標優先級排序

        Args:
            phase1_result: Phase1 掃描結果
            max_targets: 最大返回目標數量
            program_context: Program 上下文

        Returns:
            排序後的目標列表
        """
        self.logger.info("🎯 [Step 9: Phase2 Targets] 開始智慧目標優先級排序...")

        assets = phase1_result.get("assets", [])
        if not assets:
            self.logger.warning("   ⚠️ Phase1 未發現任何資產")
            return []

        program_context = program_context or {}
        duplicate_intel = set(program_context.get("duplicate_intel", []))

        vuln_tier_mapping = {
            "rce": 1, "remote_code_execution": 1, "command_injection": 1,
            "ssrf": 1, "server_side_request_forgery": 1,
            "account_takeover": 1, "ato": 1,
            "payment_bypass": 1, "financial_manipulation": 1,
            "sql_injection": 2, "sqli": 2,
            "idor": 2, "insecure_direct_object_reference": 2,
            "auth_bypass": 2, "authentication_bypass": 2,
            "privilege_escalation": 2, "xxe": 2, "ssti": 2,
            "path_traversal": 2, "lfi": 2, "rfi": 2,
            "xss_stored": 3, "stored_xss": 3,
            "xss_dom": 3, "xss_reflected": 3,
            "csrf": 3, "api_key_disclosure": 3,
            "cors_misconfiguration": 3, "open_redirect": 3,
        }

        targets_with_scores: list[dict[str, Any]] = []

        for asset in assets[:100]:
            vuln_type = asset.get("vuln_type", "unknown").lower().replace(" ", "_")
            tier = vuln_tier_mapping.get(vuln_type, 4)
            if tier == 4:
                continue

            waf_interference_score = self._calculate_waf_interference(asset, phase1_result)
            historical_success_score = self._query_historical_success(asset)
            duplicate_risk = 0.5 if vuln_type in duplicate_intel else 0.0

            ai_score = 0.5
            attack_vector = vuln_type
            recommended_tools = self._get_default_tools_for_vuln(vuln_type)

            if self.use_neural_decision and self.neural_engine:
                try:
                    ai_result = self.neural_engine.generate_decision(
                        task_description="target_prioritization",
                        context=f"Target: {asset.get('url', '')[:80]} | VulnType: {vuln_type} | Tier: {tier}",
                    )
                    ai_score = ai_result.get("confidence", 0.5)
                    attack_vector = ai_result.get("attack_vector", vuln_type)
                    recommended_tools = ai_result.get("recommended_tools", recommended_tools)
                except Exception as e:
                    self.logger.debug(f"   AI 評分失敗: {e}")

            final_score = (
                ai_score * 0.45
                + (1.0 - waf_interference_score) * 0.25
                + historical_success_score * 0.15
                + (1.0 - duplicate_risk) * 0.15
            )

            if tier == 1:
                final_score *= 1.3
            elif tier == 2:
                final_score *= 1.1

            targets_with_scores.append(
                {
                    "asset": asset,
                    "score": min(1.0, final_score),
                    "tier": tier,
                    "cvss_estimate": {1: 9.0, 2: 7.5, 3: 5.5}.get(tier, 5.0),
                    "attack_vector": attack_vector,
                    "recommended_tools": recommended_tools,
                    "duplicate_risk": duplicate_risk,
                    "reasoning": f"Tier{tier}|AI:{ai_score:.2f}|WAF干擾:{waf_interference_score:.2f}",
                }
            )

        targets_with_scores.sort(key=lambda x: (x["tier"], -x["score"]))
        top_targets = targets_with_scores[:max_targets]

        self.logger.info(f"   ✅ 已選出 {len(top_targets)} 個高優先級目標")
        for idx, t in enumerate(top_targets[:3], 1):
            tier_icon = {1: "🔴", 2: "🟠", 3: "🟡"}.get(t["tier"], "⚪")
            self.logger.info(
                f"   {tier_icon} #{idx} Tier{t['tier']} {t['asset'].get('url', 'N/A')[:40]}... (分數: {t['score']:.2f})"
            )

        return top_targets

    # ═══════════════════════════════════════════════════════════════
    # Phase 2 結果評估 (Step 11)
    # ═══════════════════════════════════════════════════════════════

    def evaluate_phase2_results(
        self,
        phase2_results: list[dict[str, Any]],
        time_budget_remaining: float,
        program_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """步驟 11: AI 評估 Phase2 結果並決定後續行動

        Args:
            phase2_results: Phase2 測試結果列表
            time_budget_remaining: 剩餘時間預算（秒）
            program_info: Program 資訊

        Returns:
            評估結果字典
        """
        self.logger.info("📊 [Step 11: Phase2 Evaluation] 評估攻擊結果...")

        program_info = program_info or {}
        duplicate_rate = program_info.get("duplicate_rate", 0.3)

        # === 1. 詳細統計分析 ===
        total_findings = len(phase2_results)

        severity_counts: dict[str, int] = {
            "CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0,
        }
        for r in phase2_results:
            sev = r.get("severity", "INFO").upper()
            if sev in severity_counts:
                severity_counts[sev] += 1

        critical_high_count = severity_counts["CRITICAL"] + severity_counts["HIGH"]
        poc_ready_count = sum(1 for r in phase2_results if r.get("poc_ready", False))
        reproducible_count = sum(1 for r in phase2_results if r.get("reproducible", False))

        avg_confidence = (
            sum(r.get("confidence", 0) for r in phase2_results) / max(total_findings, 1)
        )

        self.logger.info(f"   📈 發現: {total_findings} 個漏洞")
        self.logger.info(f"   🔴 Critical: {severity_counts['CRITICAL']}, High: {severity_counts['HIGH']}")
        self.logger.info(f"   🟠 Medium: {severity_counts['MEDIUM']}, Low: {severity_counts['LOW']}")
        self.logger.info(f"   📋 POC 已準備: {poc_ready_count}/{total_findings}")
        self.logger.info(f"   📈 平均信心度: {avg_confidence:.2f}")
        self.logger.info(f"   ⏱️ 剩餘時間: {time_budget_remaining / 60:.1f} 分鐘")

        # === 2. 攻擊鏈潛力分析 ===
        chain_potential = self._analyze_vulnerability_chains(phase2_results)

        # === 3. 神經網路決策輔助 ===
        ai_recommendation = None
        if self.use_neural_decision and self.neural_engine and phase2_results:
            try:
                context = (
                    f"Findings: {total_findings} | "
                    f"Critical/High: {critical_high_count} | "
                    f"Confidence: {avg_confidence:.2f} | "
                    f"POC Ready: {poc_ready_count} | "
                    f"ChainPotential: {chain_potential['score']:.2f} | "
                    f"TimeRemaining: {time_budget_remaining / 3600:.1f}h | "
                    f"DuplicateRate: {duplicate_rate:.2f}"
                )
                ai_result = self.neural_engine.generate_decision(
                    task_description="evaluate_phase2_results",
                    context=context,
                )
                ai_recommendation = ai_result.get("attack_vector", "continue")
            except Exception as e:
                self.logger.debug(f"   AI 決策失敗: {e}")

        # === 4. 決策邏輯 ===
        action = "CONTINUE_DEEP_DIVE"
        reasoning_parts: list[str] = []
        priority = "NORMAL"

        if critical_high_count >= 1 and poc_ready_count >= 1 and avg_confidence > 0.85:
            action = "SUBMIT_REPORT"
            priority = "HIGH"
            reasoning_parts.append(f"發現 {critical_high_count} 個高危漏洞，POC 已準備")
        elif chain_potential["can_chain"] and chain_potential["score"] > 0.7:
            action = "CHAIN_VULNERABILITIES"
            priority = "HIGH"
            reasoning_parts.append(f"可串聯漏洞: {chain_potential['chain_description']}")
        elif severity_counts["MEDIUM"] >= 3 and poc_ready_count >= 2:
            action = "SUBMIT_REPORT"
            priority = "MEDIUM"
            reasoning_parts.append("多個中危漏洞，建議整合報告")
        elif time_budget_remaining < 1800:
            if critical_high_count >= 1 or (total_findings > 0 and avg_confidence > 0.7):
                action = "SUBMIT_REPORT"
                priority = "URGENT"
                reasoning_parts.append("時間緊迫，提交現有發現")
            else:
                action = "ABANDON_TARGET"
                reasoning_parts.append("時間不足且收穫有限")
        elif avg_confidence < 0.4 and total_findings < 3:
            action = "SWITCH_STRATEGY"
            reasoning_parts.append("當前方法效果不佳")
            if ai_recommendation:
                reasoning_parts.append(f"AI 建議: 嘗試 {ai_recommendation}")
        elif total_findings > 0 and time_budget_remaining > 3600:
            # 根據發現數量和平均信心度決定是否繼續
            if total_findings >= 3 or avg_confidence > 0.6:
                action = "CONTINUE_DEEP_DIVE"
                reasoning_parts.append("發現較多，繼續深挖")
            else:
                action = "SWITCH_STRATEGY"
                reasoning_parts.append("發現數量較少，建議切換策略")
        elif duplicate_rate > 0.6 and severity_counts["MEDIUM"] + severity_counts["LOW"] > critical_high_count:
            action = "ABANDON_TARGET"
            reasoning_parts.append(f"高重複風險 ({duplicate_rate * 100:.0f}%)")
        else:
            action = "ABANDON_TARGET"
            reasoning_parts.append("未發現可利用漏洞")

        reasoning = " | ".join(reasoning_parts)

        # === 5. 生成下一步建議 ===
        next_steps = self._generate_next_steps(action)

        # === 6. 報告準備建議 ===
        report_guidance = None
        if action == "SUBMIT_REPORT":
            report_guidance = self._generate_report_guidance(phase2_results, program_info)

        result: dict[str, Any] = {
            "action": action,
            "priority": priority,
            "reasoning": reasoning,
            "findings_summary": {
                "total": total_findings,
                "by_severity": severity_counts,
                "critical_high": critical_high_count,
                "poc_ready": poc_ready_count,
                "reproducible": reproducible_count,
                "avg_confidence": avg_confidence,
            },
            "chain_analysis": chain_potential,
            "next_steps": next_steps,
            "report_guidance": report_guidance,
            "ai_recommendation": ai_recommendation,
            "time_metrics": {
                "remaining_minutes": time_budget_remaining / 60,
                "recommended_action_time": self._estimate_action_time(action),
            },
        }

        self.logger.info(f"   ✅ 建議行動: {action} (優先級: {priority})")
        return result

    # ---------- 輔助方法 ----------

    def _analyze_vulnerability_chains(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """分析漏洞串聯潛力"""
        vuln_types = [r.get("vuln_type", "").lower() for r in results]

        chain_patterns = [
            {"components": ["xss", "csrf"], "result": "Account Takeover", "severity_boost": "CRITICAL", "description": "XSS + CSRF → 帳戶劫持"},
            {"components": ["ssrf", "rce"], "result": "Remote Code Execution Chain", "severity_boost": "CRITICAL", "description": "SSRF → 內部服務 → RCE"},
            {"components": ["idor", "information_disclosure"], "result": "Mass Data Exposure", "severity_boost": "HIGH", "description": "IDOR + 信息洩露 → 批量數據提取"},
            {"components": ["sql_injection", "auth_bypass"], "result": "Full Database Access", "severity_boost": "CRITICAL", "description": "SQLi → 認證繞過 → 完整數據庫訪問"},
            {"components": ["open_redirect", "oauth"], "result": "OAuth Token Theft", "severity_boost": "HIGH", "description": "開放重定向 + OAuth → Token 竊取"},
        ]

        can_chain = False
        best_chain: dict[str, Any] | None = None
        chain_score = 0.0

        for pattern in chain_patterns:
            components = pattern["components"]
            matches = sum(1 for c in components if any(c in vt for vt in vuln_types))
            if matches >= len(components):
                can_chain = True
                score = matches / len(components)
                if score > chain_score:
                    chain_score = score
                    best_chain = pattern

        return {
            "can_chain": can_chain,
            "score": chain_score,
            "chain_description": best_chain["description"] if best_chain else None,
            "severity_boost": best_chain["severity_boost"] if best_chain else None,
            "matched_pattern": best_chain,
        }

    def _generate_report_guidance(
        self,
        results: list[dict[str, Any]],
        _program_info: dict[str, Any],  # reserved for future use
    ) -> dict[str, Any]:
        """生成 HackerOne 報告撰寫指南"""
        top_finding = max(results, key=lambda x: x.get("confidence", 0)) if results else {}

        return {
            "title_template": f"[{top_finding.get('severity', 'Medium')}] {top_finding.get('vuln_type', 'Vulnerability')} in {top_finding.get('url', 'application')[:50]}",
            "sections": [
                "Summary (1-2 sentences)",
                "Steps to Reproduce (numbered list)",
                "Impact (business perspective)",
                "Proof of Concept (code/screenshots)",
                "Suggested Fix",
                "References (CVE, OWASP)",
            ],
            "cvss_estimate": self._calculate_cvss(top_finding),
            "tips": [
                "使用清晰的重現步驟",
                "強調商業影響",
                "提供修復建議",
                "附上視頻 POC 可提升報告品質",
            ],
        }

    def _calculate_cvss(self, finding: dict[str, Any]) -> dict[str, Any]:
        """計算 CVSS 3.1 評分估計"""
        severity = finding.get("severity", "MEDIUM").upper()
        base_scores = {"CRITICAL": 9.0, "HIGH": 7.5, "MEDIUM": 5.5, "LOW": 3.0, "INFO": 0.0}

        return {
            "base_score": base_scores.get(severity, 5.5),
            "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N",
            "severity_rating": severity,
        }

    def _estimate_action_time(self, action: str) -> int:
        """估計各行動所需時間（分鐘）"""
        time_estimates = {
            "SUBMIT_REPORT": 30,
            "CONTINUE_DEEP_DIVE": 60,
            "CHAIN_VULNERABILITIES": 45,
            "SWITCH_STRATEGY": 15,
            "ABANDON_TARGET": 5,
        }
        return time_estimates.get(action, 30)

    def _generate_next_steps(self, action: str) -> list[str]:
        """生成下一步建議"""
        if action == "SUBMIT_REPORT":
            return [
                "1. 驗證所有高危漏洞的可重現性",
                "2. 撰寫詳細的 PoC 和影響說明",
                "3. 提交到 HackerOne 平台",
            ]
        elif action == "CONTINUE_DEEP_DIVE":
            return [
                "1. 針對高優先級目標進行手動測試",
                "2. 嘗試 WAF 繞過技術",
                "3. 探索邊緣案例和業務邏輯漏洞",
            ]
        elif action == "SWITCH_STRATEGY":
            return [
                "1. 切換到更隱蔽的掃描模式",
                "2. 嘗試不同的 payload 編碼",
                "3. 調整攻擊向量",
            ]
        else:  # ABANDON_TARGET
            return [
                "1. 記錄失敗原因和學習經驗",
                "2. 切換到下一個目標",
                "3. 更新目標篩選策略",
            ]

    def _get_default_tools_for_vuln(self, vuln_type: str) -> list[str]:
        """獲取漏洞類型對應的默認工具"""
        return {
            "sql_injection": ["sqlmap", "burp_intruder"],
            "sqli": ["sqlmap", "burp_intruder"],
            "xss_stored": ["xsstrike", "dalfox"],
            "xss_dom": ["domdig"],
            "xss_reflected": ["xsstrike"],
            "ssrf": ["ssrfmap", "burp_collaborator"],
            "rce": ["nuclei", "custom_payload"],
            "idor": ["autorize", "burp_match_replace"],
            "csrf": ["burp_csrf_poc"],
            "xxe": ["xxeinjector"],
            "ssti": ["tplmap"],
        }.get(vuln_type, ["burp_suite", "manual_analysis"])

    # ═══════════════════════════════════════════════════════════════
    # WAF/Rate Limit 自適應策略
    # ═══════════════════════════════════════════════════════════════

    def _decide_waf_bypass(
        self,
        waf_vendor: Optional[str],
        base_confidence: float,
    ) -> dict[str, Any]:
        """基於 WAF 類型和 AI 信心度決定繞過策略"""
        vendor = (waf_vendor or "unknown").lower()
        strategy = self.waf_bypass_strategies.get(vendor, self.waf_bypass_strategies["unknown"])

        delay_multiplier = strategy["delay_multiplier"]
        if base_confidence > 0.8:
            delay_multiplier *= 0.8
        elif base_confidence < 0.5:
            delay_multiplier *= 1.5

        return {**strategy, "delay_multiplier": delay_multiplier, "confidence_adjusted": True}

    def adaptive_rate_limiting(
        self,
        target_url: str,
        phase: str,
        waf_detected: bool,
    ) -> dict[str, float]:
        """自適應速率限制策略"""
        profile = self.rate_limit_profiles.get(phase, self.rate_limit_profiles["phase1"])

        base_rate = profile["requests_per_second"]

        if waf_detected:
            base_rate *= 0.2

        if self.experience_manager:
            try:
                historical_ban_rate = self.experience_manager.get_ban_rate(target_url)
                if historical_ban_rate > 0.5:
                    base_rate *= 0.5
            except Exception:
                pass

        return {
            "requests_per_second": base_rate,
            "burst_size": int(base_rate * 2),
            "retry_after_429": 60,
            "backoff_multiplier": 2.0,
        }

    # ---------- 內部工具方法 ----------

    def _calculate_waf_interference(
        self, _asset: dict[str, Any], phase1_result: dict[str, Any]  # asset reserved for future use
    ) -> float:
        """計算 WAF 干擾評分 (0-1, 越高越糟)"""
        fingerprints = phase1_result.get("fingerprints", {})
        if fingerprints.get("waf_detected"):
            return 0.8
        return 0.2

    def _query_historical_success(self, asset: dict[str, Any]) -> float:
        """查詢歷史成功率 (0-1)"""
        if self.experience_manager:
            try:
                return self.experience_manager.get_success_rate(asset.get("url", ""))
            except Exception:
                pass
        return 0.5

    def _estimate_phase1_time(self, phase0_result: dict[str, Any]) -> float:
        """估算 Phase1 所需時間（秒）"""
        urls = phase0_result.get("summary", {}).get("urls_found", 0)
        return min(1800 + urls * 10, 7200)

    def _initialize_waf_strategies(self) -> dict[str, dict[str, Any]]:
        """初始化 WAF 繞過策略庫"""
        return {
            "cloudflare": {
                "delay_multiplier": 3.0,
                "payload_encoding": ["unicode", "hex", "double_url"],
                "user_agent_rotation": True,
                "header_randomization": True,
            },
            "imperva": {
                "delay_multiplier": 2.5,
                "payload_encoding": ["case_swap", "comment_injection"],
                "chunk_encoding": True,
            },
            "aws_waf": {
                "delay_multiplier": 2.0,
                "payload_encoding": ["null_byte", "newline"],
                "ip_rotation": True,
            },
            "unknown": {
                "delay_multiplier": 2.0,
                "payload_encoding": ["basic"],
                "conservative_mode": True,
            },
        }

    def _initialize_rate_profiles(self) -> dict[str, dict[str, Any]]:
        """初始化速率限制配置文件"""
        return {
            "phase0": {"requests_per_second": 100},
            "phase1": {"requests_per_second": 50},
            "phase2": {"requests_per_second": 20},
        }
