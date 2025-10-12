from __future__ import annotations

from collections import defaultdict
from typing import Any

from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class VulnerabilityCorrelationAnalyzer:
    """
    漏洞相關性分析器

    分析不同漏洞之間的關聯性，識別攻擊鏈和複合攻擊路徑，
    幫助理解漏洞的整體影響和優先級。
    """

    def __init__(self) -> None:
        # 漏洞類型之間的相關性映射
        self._correlation_rules = {
            "xss": {"related_types": ["csrf", "session_fixation"], "multiplier": 1.3},
            "sqli": {
                "related_types": ["auth_bypass", "privilege_escalation"],
                "multiplier": 1.5,
            },
            "ssrf": {
                "related_types": ["port_scanning", "internal_disclosure"],
                "multiplier": 1.2,
            },
            "csrf": {"related_types": ["xss", "session_hijacking"], "multiplier": 1.2},
            "lfi": {"related_types": ["rce", "file_disclosure"], "multiplier": 1.4},
            "rfi": {"related_types": ["rce", "backdoor"], "multiplier": 1.6},
        }

        # 攻擊鏈模式
        self._attack_chains = [
            ["info_disclosure", "sqli", "privilege_escalation"],
            ["xss", "csrf", "account_takeover"],
            ["ssrf", "port_scanning", "lateral_movement"],
            ["lfi", "file_disclosure", "rce"],
        ]

    def analyze_correlations(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
        """
        分析漏洞相關性

        Args:
            findings: 漏洞發現列表

        Returns:
            相關性分析結果
        """
        if not findings:
            return {
                "total_findings": 0,
                "correlation_groups": [],
                "attack_chains": [],
                "risk_amplification": 0.0,
            }

        # 按類型分組
        vulns_by_type = defaultdict(list)
        vulns_by_location = defaultdict(list)

        for finding in findings:
            vuln_type = finding.get("vulnerability_type", "unknown").lower()
            location = finding.get("location", {})

            vulns_by_type[vuln_type].append(finding)

            # 按URL/路徑分組
            url_path = location.get("url", "").split("?")[0]  # 移除參數
            vulns_by_location[url_path].append(finding)

        # 尋找相關性組合
        correlation_groups = self._find_correlation_groups(vulns_by_type)

        # 識別攻擊鏈
        attack_chains = self._identify_attack_chains(vulns_by_type)

        # 分析位置聚集
        location_clusters = self._analyze_location_clusters(vulns_by_location)

        # 計算風險放大係數
        risk_amplification = self._calculate_risk_amplification(
            correlation_groups, attack_chains
        )

        result = {
            "total_findings": len(findings),
            "unique_types": len(vulns_by_type),
            "correlation_groups": correlation_groups,
            "attack_chains": attack_chains,
            "location_clusters": location_clusters,
            "risk_amplification": risk_amplification,
            "summary": self._generate_correlation_summary(
                correlation_groups, attack_chains, risk_amplification
            ),
        }

        logger.info(
            f"Correlation analysis completed: {len(correlation_groups)} groups, "
            f"{len(attack_chains)} attack chains found"
        )
        return result

    def _find_correlation_groups(
        self, vulns_by_type: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """尋找相關的漏洞組合"""
        correlation_groups = []

        for vuln_type, findings in vulns_by_type.items():
            if vuln_type in self._correlation_rules:
                rule = self._correlation_rules[vuln_type]
                related = rule.get("related_types", [])
                if isinstance(related, str):
                    related_types = [related]
                elif isinstance(related, list | tuple | set):
                    related_types = list(related)
                else:
                    related_types = []

                # 檢查是否有相關類型的漏洞
                found_related = []
                for related_type in related_types:
                    if related_type in vulns_by_type:
                        found_related.append(
                            {
                                "type": related_type,
                                "count": len(vulns_by_type[related_type]),
                                "findings": vulns_by_type[related_type],
                            }
                        )

                if found_related:
                    # 確保 multiplier 是數值類型，若無效則退回到預設 1.0
                    raw_multiplier = rule.get("multiplier", 1.0)
                    multiplier = 1.0
                    # 只在 raw_multiplier 為 int/float 或可解析為 float 的 str 時進行轉換
                    if isinstance(raw_multiplier, int | float):
                        multiplier = float(raw_multiplier)
                    elif isinstance(raw_multiplier, str):
                        try:
                            multiplier = float(raw_multiplier)
                        except ValueError:
                            multiplier = 1.0

                    correlation_groups.append(
                        {
                            "primary_type": vuln_type,
                            "primary_count": len(findings),
                            "related_vulnerabilities": found_related,
                            "multiplier": multiplier,
                            "total_impact": len(findings) * multiplier,
                        }
                    )

        return correlation_groups

    def _identify_attack_chains(
        self, vulns_by_type: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """識別可能的攻擊鏈"""
        identified_chains = []

        for chain_pattern in self._attack_chains:
            chain_matches = []
            complete_chain = True

            for step in chain_pattern:
                if step in vulns_by_type:
                    chain_matches.append(
                        {
                            "step": step,
                            "count": len(vulns_by_type[step]),
                            "findings": vulns_by_type[step][:3],  # 只保留前3個示例
                        }
                    )
                else:
                    complete_chain = False
                    break

            if complete_chain:
                # 計算攻擊鏈的風險分數
                total_findings = 0
                for match in chain_matches:
                    count = match.get("count", 0)
                    if isinstance(count, int):
                        total_findings += count
                chain_risk = len(chain_pattern) * 2  # 鏈越長風險越高

                identified_chains.append(
                    {
                        "pattern": chain_pattern,
                        "matched_steps": chain_matches,
                        "total_findings": total_findings,
                        "chain_risk_score": chain_risk,
                        "description": self._describe_attack_chain(chain_pattern),
                    }
                )

        return sorted(
            identified_chains, key=lambda x: x["chain_risk_score"], reverse=True
        )

    def _analyze_location_clusters(
        self, vulns_by_location: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        """分析漏洞位置聚集情況"""
        clusters = []

        for location, findings in vulns_by_location.items():
            if len(findings) > 1:  # 只關心有多個漏洞的位置
                vuln_types = {
                    f.get("vulnerability_type", "unknown").lower() for f in findings
                }

                clusters.append(
                    {
                        "location": location,
                        "vulnerability_count": len(findings),
                        "unique_types": len(vuln_types),
                        "vulnerability_types": list(vuln_types),
                        "risk_concentration": len(findings) / max(len(vuln_types), 1),
                    }
                )

        return sorted(clusters, key=lambda x: x["vulnerability_count"], reverse=True)

    def _calculate_risk_amplification(
        self,
        correlation_groups: list[dict[str, Any]],
        attack_chains: list[dict[str, Any]],
    ) -> float:
        """計算風險放大係數"""
        base_amplification = 1.0

        # 相關性組合的放大效果
        for group in correlation_groups:
            multiplier = group.get("multiplier", 1.0)
            related_count = len(group.get("related_vulnerabilities", []))
            base_amplification += (multiplier - 1.0) * (related_count / 10)

        # 攻擊鏈的放大效果
        for chain in attack_chains:
            chain_risk = chain.get("chain_risk_score", 0)
            base_amplification += chain_risk / 20

        return round(min(base_amplification, 3.0), 2)  # 最高3倍放大

    def _describe_attack_chain(self, chain_pattern: list[str]) -> str:
        """描述攻擊鏈"""
        chain_str = " → ".join(chain_pattern)

        # 預定義的攻擊鏈描述
        if chain_pattern == ["info_disclosure", "sqli", "privilege_escalation"]:
            return "資訊洩露 → SQL注入 → 權限提升攻擊鏈"
        elif chain_pattern == ["xss", "csrf", "account_takeover"]:
            return "XSS → CSRF → 帳號接管攻擊鏈"
        elif chain_pattern == ["ssrf", "port_scanning", "lateral_movement"]:
            return "SSRF → 埠掃描 → 橫向移動攻擊鏈"
        elif chain_pattern == ["lfi", "file_disclosure", "rce"]:
            return "LFI → 檔案洩露 → 遠端代碼執行攻擊鏈"
        else:
            return chain_str

    def _generate_correlation_summary(
        self,
        correlation_groups: list[dict[str, Any]],
        attack_chains: list[dict[str, Any]],
        risk_amplification: float,
    ) -> dict[str, Any]:
        """生成相關性分析摘要"""
        return {
            "has_correlations": len(correlation_groups) > 0,
            "has_attack_chains": len(attack_chains) > 0,
            "risk_level": (
                "critical"
                if risk_amplification > 2.0
                else (
                    "high"
                    if risk_amplification > 1.5
                    else "medium"
                    if risk_amplification > 1.2
                    else "low"
                )
            ),
            "key_findings": [
                f"發現 {len(correlation_groups)} 個相關性組合",
                f"識別出 {len(attack_chains)} 條攻擊鏈",
                f"風險放大係數: {risk_amplification}x",
            ],
        }
