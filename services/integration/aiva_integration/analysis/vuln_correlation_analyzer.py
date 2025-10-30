

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

    def analyze_code_level_root_cause(
        self, findings: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        程式碼層面的根因分析
        
        識別多個漏洞是否源於同一個有問題的共用元件、函式庫或父類別
        
        Args:
            findings: 漏洞發現列表，需包含程式碼位置資訊
            
        Returns:
            根因分析結果
        """
        if not findings:
            return {"root_causes": [], "derived_vulnerabilities": []}

        # 按程式碼路徑分組
        code_path_groups = defaultdict(list)
        for finding in findings:
            location = finding.get("location", {})
            file_path = location.get("file_path") or location.get("code_file")

            if file_path:
                code_path_groups[file_path].append(finding)

        # 尋找共用元件
        root_causes = []
        derived_vulnerabilities = []

        for file_path, file_findings in code_path_groups.items():
            if len(file_findings) < 2:
                continue

            # 檢查是否有共用的函式或類別
            common_components = self._identify_common_components(file_findings)

            if common_components:
                # 識別根本原因
                root_cause = {
                    "component_type": common_components["type"],
                    "component_name": common_components["name"],
                    "file_path": file_path,
                    "affected_vulnerabilities": len(file_findings),
                    "vulnerability_ids": [
                        f.get("finding_id") or f.get("vulnerability_id")
                        for f in file_findings
                    ],
                    "severity_distribution": self._count_severities(file_findings),
                    "recommendation": self._generate_root_cause_recommendation(
                        common_components
                    ),
                }
                root_causes.append(root_cause)

                # 標記衍生漏洞
                for finding in file_findings:
                    derived_vulnerabilities.append(
                        {
                            "vulnerability_id": finding.get("finding_id")
                            or finding.get("vulnerability_id"),
                            "root_cause_component": common_components["name"],
                            "relationship": "derived_from_shared_component",
                        }
                    )

        logger.info(
            f"Root cause analysis found {len(root_causes)} shared components "
            f"affecting {len(derived_vulnerabilities)} vulnerabilities"
        )

        return {
            "root_causes": root_causes,
            "derived_vulnerabilities": derived_vulnerabilities,
            "summary": {
                "total_root_causes": len(root_causes),
                "total_affected_vulnerabilities": len(derived_vulnerabilities),
                "fix_efficiency": (
                    f"修復 {len(root_causes)} 個根本問題可以解決 "
                    f"{len(derived_vulnerabilities)} 個漏洞"
                    if root_causes
                    else "未發現共用根本原因"
                ),
            },
        }

    def analyze_sast_dast_correlation(
        self, findings: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        SAST-DAST 資料流關聯分析
        
        將 SAST 的潛在漏洞 (Sink) 與 DAST 的可控輸入 (Source) 進行關聯，
        驗證漏洞的真實可利用性
        
        Args:
            findings: 混合的 SAST 和 DAST 發現列表
            
        Returns:
            關聯分析結果
        """
        if not findings:
            return {"confirmed_flows": [], "unconfirmed_sast": [], "orphan_dast": []}

        # 分類 SAST 和 DAST 發現
        sast_findings = []
        dast_findings = []

        for finding in findings:
            scan_type = finding.get("scan_type", "").lower()
            vuln_type = finding.get("vulnerability_type", "").lower()

            # 根據掃描類型或位置資訊判斷
            if scan_type == "sast" or "code_file" in finding.get("location", {}):
                sast_findings.append(finding)
            elif scan_type == "dast" or "url" in finding.get("location", {}):
                dast_findings.append(finding)

        # 尋找資料流匹配
        confirmed_flows = []
        unconfirmed_sast = list(sast_findings)  # 先假設都未確認
        orphan_dast = list(dast_findings)

        for sast_finding in sast_findings:
            sast_location = sast_finding.get("location", {})
            sast_vuln_type = sast_finding.get("vulnerability_type", "").lower()

            # 尋找對應的 DAST 發現
            for dast_finding in dast_findings:
                dast_location = dast_finding.get("location", {})
                dast_vuln_type = dast_finding.get("vulnerability_type", "").lower()

                # 檢查漏洞類型是否匹配
                if self._are_vulnerability_types_compatible(
                    sast_vuln_type, dast_vuln_type
                ):
                    # 檢查路徑是否匹配
                    if self._check_path_correlation(sast_location, dast_location):
                        # 找到匹配的資料流
                        confirmed_flow = {
                            "sast_finding_id": sast_finding.get("finding_id"),
                            "dast_finding_id": dast_finding.get("finding_id"),
                            "vulnerability_type": sast_vuln_type,
                            "source": {
                                "type": "external_input",
                                "location": dast_location.get("url"),
                                "parameter": dast_location.get("parameter"),
                            },
                            "sink": {
                                "type": "dangerous_function",
                                "location": sast_location.get("code_file"),
                                "line": sast_location.get("line_number"),
                                "function": sast_location.get("function_name"),
                            },
                            "confidence": "high",
                            "impact": self._calculate_flow_impact(
                                sast_finding, dast_finding
                            ),
                            "recommendation": "此漏洞已被 DAST 驗證為可利用，應立即修復",
                        }
                        confirmed_flows.append(confirmed_flow)

                        # 從未確認列表中移除
                        if sast_finding in unconfirmed_sast:
                            unconfirmed_sast.remove(sast_finding)
                        if dast_finding in orphan_dast:
                            orphan_dast.remove(dast_finding)

        logger.info(
            f"SAST-DAST correlation: {len(confirmed_flows)} confirmed flows, "
            f"{len(unconfirmed_sast)} unconfirmed SAST findings, "
            f"{len(orphan_dast)} orphan DAST findings"
        )

        return {
            "confirmed_flows": confirmed_flows,
            "unconfirmed_sast": [
                {
                    "finding_id": f.get("finding_id"),
                    "vulnerability_type": f.get("vulnerability_type"),
                    "status": "potential",
                    "recommendation": "SAST 發現但未被 DAST 驗證，可能為誤報或需要特定條件觸發",
                }
                for f in unconfirmed_sast
            ],
            "orphan_dast": [
                {
                    "finding_id": f.get("finding_id"),
                    "vulnerability_type": f.get("vulnerability_type"),
                    "status": "confirmed_by_dast",
                    "recommendation": "DAST 確認的漏洞但未找到對應的程式碼位置",
                }
                for f in orphan_dast
            ],
            "summary": {
                "total_confirmed": len(confirmed_flows),
                "total_unconfirmed_sast": len(unconfirmed_sast),
                "total_orphan_dast": len(orphan_dast),
                "confirmation_rate": (
                    round(
                        len(confirmed_flows) / len(sast_findings) * 100, 1
                    )
                    if sast_findings
                    else 0
                ),
                "key_insight": (
                    f"{len(confirmed_flows)} 個 SAST 發現已被 DAST 驗證為真實可利用漏洞"
                    if confirmed_flows
                    else "未發現 SAST-DAST 關聯"
                ),
            },
        }

    def _identify_common_components(
        self, findings: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """識別共用的程式碼元件"""
        # 收集所有可能的共用元件
        functions = []
        classes = []
        modules = []

        for finding in findings:
            location = finding.get("location", {})
            functions.append(location.get("function_name"))
            classes.append(location.get("class_name"))
            modules.append(location.get("module_name"))

        # 尋找最常見的元件
        function_counts = defaultdict(int)
        class_counts = defaultdict(int)
        module_counts = defaultdict(int)

        for func in functions:
            if func:
                function_counts[func] += 1

        for cls in classes:
            if cls:
                class_counts[cls] += 1

        for mod in modules:
            if mod:
                module_counts[mod] += 1

        # 找出出現次數最多的元件
        if function_counts and max(function_counts.values()) >= 2:
            common_func = max(function_counts.items(), key=lambda x: x[1])
            return {"type": "function", "name": common_func[0], "count": common_func[1]}

        if class_counts and max(class_counts.values()) >= 2:
            common_class = max(class_counts.items(), key=lambda x: x[1])
            return {
                "type": "class",
                "name": common_class[0],
                "count": common_class[1],
            }

        if module_counts and max(module_counts.values()) >= 2:
            common_module = max(module_counts.items(), key=lambda x: x[1])
            return {
                "type": "module",
                "name": common_module[0],
                "count": common_module[1],
            }

        return None

    def _count_severities(self, findings: list[dict[str, Any]]) -> dict[str, int]:
        """統計嚴重程度分布"""
        severity_counts = defaultdict(int)
        for finding in findings:
            severity = finding.get("severity", "unknown").upper()
            severity_counts[severity] += 1
        return dict(severity_counts)

    def _generate_root_cause_recommendation(
        self, component: dict[str, Any]
    ) -> str:
        """生成根因修復建議"""
        comp_type = component["type"]
        comp_name = component["name"]
        count = component["count"]

        return (
            f"建議重點審查和修復 {comp_type} '{comp_name}'，"
            f"該元件導致了 {count} 個相關漏洞。"
            f"修復此共用元件將同時解決所有衍生的安全問題。"
        )

    def _are_vulnerability_types_compatible(
        self, sast_type: str, dast_type: str
    ) -> bool:
        """檢查 SAST 和 DAST 的漏洞類型是否匹配"""
        # 定義類型映射關係
        type_mappings = {
            "sql_injection": ["sqli", "sql", "injection"],
            "xss": ["cross_site_scripting", "xss", "script_injection"],
            "command_injection": ["os_command", "cmd_injection", "command"],
            "path_traversal": ["lfi", "directory_traversal", "path"],
            "xxe": ["xml_external_entity", "xxe"],
        }

        # 標準化類型名稱
        sast_normalized = sast_type.replace("-", "_").replace(" ", "_")
        dast_normalized = dast_type.replace("-", "_").replace(" ", "_")

        # 檢查直接匹配
        if sast_normalized == dast_normalized:
            return True

        # 檢查映射匹配
        for category, variants in type_mappings.items():
            if sast_normalized in variants and dast_normalized in variants:
                return True

        return False

    def _check_path_correlation(
        self, sast_location: dict[str, Any], dast_location: dict[str, Any]
    ) -> bool:
        """檢查 SAST 程式碼位置與 DAST URL 路徑是否相關"""
        code_file = sast_location.get("code_file", "")
        url = dast_location.get("url", "")

        if not code_file or not url:
            return False

        # 提取 URL 路徑
        from urllib.parse import urlparse

        parsed_url = urlparse(url)
        url_path = parsed_url.path

        # 簡單的路徑匹配邏輯
        # 例如：/api/users 可能對應到 api/users.py 或 UserController
        url_segments = [seg for seg in url_path.split("/") if seg]
        code_segments = [seg for seg in code_file.replace("\\", "/").split("/") if seg]

        # 檢查是否有共同的路徑段
        common_segments = set(url_segments) & set(code_segments)
        return len(common_segments) > 0

    def _calculate_flow_impact(
        self, sast_finding: dict[str, Any], dast_finding: dict[str, Any]
    ) -> str:
        """計算資料流的影響等級"""
        sast_severity = sast_finding.get("severity", "medium").upper()
        dast_severity = dast_finding.get("severity", "medium").upper()

        # 取兩者中較高的嚴重程度
        severity_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        sast_level = (
            severity_order.index(sast_severity)
            if sast_severity in severity_order
            else 1
        )
        dast_level = (
            severity_order.index(dast_severity)
            if dast_severity in severity_order
            else 1
        )

        max_level = max(sast_level, dast_level)

        # 因為已被驗證，提升一個等級
        confirmed_level = min(max_level + 1, len(severity_order) - 1)

        return severity_order[confirmed_level]

