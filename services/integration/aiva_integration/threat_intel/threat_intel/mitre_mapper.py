"""
MITRE Mapper - MITRE ATT&CK 映射器

映射漏洞、威脅到 MITRE ATT&CK 框架，提供：
- 戰術（Tactics）映射
- 技術（Techniques）映射
- 緩解措施（Mitigations）建議
- 偵測方法（Detections）建議
"""

from datetime import datetime
from enum import Enum
import json
import os
from typing import Any

from mitreattack.stix20 import MitreAttackData
import structlog

logger = structlog.get_logger(__name__)


class AttackMatrix(str, Enum):
    """ATT&CK 矩陣類型"""

    ENTERPRISE = "enterprise-attack"
    MOBILE = "mobile-attack"
    ICS = "ics-attack"


class MitreMapper:
    """
    MITRE ATT&CK 映射器

    提供漏洞到 MITRE ATT&CK 框架的映射功能。
    """

    def __init__(
        self,
        matrix: AttackMatrix = AttackMatrix.ENTERPRISE,
        cache_dir: str = "./cache/mitre",
        auto_update: bool = False,
    ):
        """
        初始化 MITRE 映射器

        Args:
            matrix: ATT&CK 矩陣類型
            cache_dir: 快取目錄
            auto_update: 是否自動更新 ATT&CK 數據
        """
        self.matrix = matrix
        self.cache_dir = cache_dir
        self.auto_update = auto_update

        # 確保快取目錄存在
        os.makedirs(cache_dir, exist_ok=True)

        # 載入 ATT&CK 數據
        self.attack_data: MitreAttackData | None = None
        self._load_attack_data()

        logger.info(
            "mitre_mapper_initialized",
            matrix=matrix,
            cache_dir=cache_dir,
        )

    def _load_attack_data(self) -> None:
        """載入 MITRE ATT&CK 數據"""
        try:
            # 使用 mitreattack-python 載入數據
            self.attack_data = MitreAttackData(self.matrix.value)
            logger.info("mitre_attack_data_loaded", matrix=self.matrix)
        except Exception as e:
            logger.error("mitre_attack_data_load_failed", error=str(e))
            self.attack_data = None

    def get_technique_by_id(self, technique_id: str) -> dict[str, Any] | None:
        """
        根據技術 ID 獲取技術資訊

        Args:
            technique_id: 技術 ID（如 T1055）

        Returns:
            技術資訊字典
        """
        if not self.attack_data:
            return None

        try:
            techniques = self.attack_data.get_techniques(remove_revoked_deprecated=True)

            for technique in techniques:
                external_refs = technique.get("external_references", [])
                for ref in external_refs:
                    if ref.get("source_name") == "mitre-attack" and ref.get("external_id") == technique_id:
                        return {
                            "id": technique_id,
                            "name": technique.get("name"),
                            "description": technique.get("description"),
                            "tactics": self._extract_tactics(technique),
                            "platforms": technique.get("x_mitre_platforms", []),
                            "data_sources": technique.get("x_mitre_data_sources", []),
                            "detection": technique.get("x_mitre_detection"),
                            "url": ref.get("url"),
                        }

            return None
        except Exception as e:
            logger.error("get_technique_failed", technique_id=technique_id, error=str(e))
            return None

    def _extract_tactics(self, technique: dict[str, Any]) -> list[str]:
        """從技術中提取戰術"""
        kill_chain_phases = technique.get("kill_chain_phases", [])
        tactics = []

        for phase in kill_chain_phases:
            if phase.get("kill_chain_name") == "mitre-attack":
                tactic_name = phase.get("phase_name", "").replace("-", " ").title()
                tactics.append(tactic_name)

        return tactics

    def search_techniques_by_keyword(self, keyword: str, limit: int = 10) -> list[dict[str, Any]]:
        """
        根據關鍵字搜尋技術

        Args:
            keyword: 搜尋關鍵字
            limit: 返回結果數量限制

        Returns:
            技術列表
        """
        if not self.attack_data:
            return []

        try:
            techniques = self.attack_data.get_techniques(remove_revoked_deprecated=True)
            results = []
            keyword_lower = keyword.lower()

            for technique in techniques:
                name = technique.get("name", "").lower()
                description = technique.get("description", "").lower()

                if keyword_lower in name or keyword_lower in description:
                    # 獲取技術 ID
                    tech_id = None
                    external_refs = technique.get("external_references", [])
                    for ref in external_refs:
                        if ref.get("source_name") == "mitre-attack":
                            tech_id = ref.get("external_id")
                            break

                    if tech_id:
                        results.append({
                            "id": tech_id,
                            "name": technique.get("name"),
                            "description": technique.get("description", "")[:200] + "...",
                            "tactics": self._extract_tactics(technique),
                        })

                    if len(results) >= limit:
                        break

            logger.info(
                "techniques_searched",
                keyword=keyword,
                results_count=len(results),
            )
            return results

        except Exception as e:
            logger.error("search_techniques_failed", keyword=keyword, error=str(e))
            return []

    def get_tactics(self) -> list[dict[str, Any]]:
        """
        獲取所有戰術

        Returns:
            戰術列表
        """
        if not self.attack_data:
            return []

        try:
            tactics = self.attack_data.get_tactics(remove_revoked_deprecated=True)
            results = []

            for tactic in tactics:
                external_refs = tactic.get("external_references", [])
                tactic_id = None
                tactic_url = None

                for ref in external_refs:
                    if ref.get("source_name") == "mitre-attack":
                        tactic_id = ref.get("external_id")
                        tactic_url = ref.get("url")
                        break

                results.append({
                    "id": tactic_id,
                    "name": tactic.get("name"),
                    "description": tactic.get("description"),
                    "shortname": tactic.get("x_mitre_shortname"),
                    "url": tactic_url,
                })

            logger.info("tactics_retrieved", count=len(results))
            return results

        except Exception as e:
            logger.error("get_tactics_failed", error=str(e))
            return []

    def get_mitigations_for_technique(self, technique_id: str) -> list[dict[str, Any]]:
        """
        獲取技術的緩解措施

        Args:
            technique_id: 技術 ID

        Returns:
            緩解措施列表
        """
        if not self.attack_data:
            return []

        try:
            # 獲取技術物件
            techniques = self.attack_data.get_techniques(remove_revoked_deprecated=True)
            technique_obj = None

            for tech in techniques:
                external_refs = tech.get("external_references", [])
                for ref in external_refs:
                    if ref.get("source_name") == "mitre-attack" and ref.get("external_id") == technique_id:
                        technique_obj = tech
                        break
                if technique_obj:
                    break

            if not technique_obj:
                return []

            tech_id_raw = technique_obj.get("id")
            if not tech_id_raw or not isinstance(tech_id_raw, str):
                return []

            tech_id: str = tech_id_raw

            # 獲取緩解措施關係
            mitigations = self.attack_data.get_mitigations_mitigating_technique(
                tech_id
            )

            results = []
            for mitigation in mitigations:
                external_refs = mitigation.get("external_references", [])
                mitigation_id = None

                for ref in external_refs:
                    if ref.get("source_name") == "mitre-attack":
                        mitigation_id = ref.get("external_id")
                        break

                results.append({
                    "id": mitigation_id,
                    "name": mitigation.get("name"),
                    "description": mitigation.get("description"),
                })

            logger.info(
                "mitigations_retrieved",
                technique_id=technique_id,
                count=len(results),
            )
            return results

        except Exception as e:
            logger.error(
                "get_mitigations_failed",
                technique_id=technique_id,
                error=str(e),
            )
            return []

    def map_vulnerability_to_techniques(
        self, vulnerability_description: str, cwe_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        將漏洞映射到 ATT&CK 技術

        Args:
            vulnerability_description: 漏洞描述
            cwe_id: CWE ID（可選）

        Returns:
            相關技術列表
        """
        # 基於漏洞描述的關鍵字映射
        keyword_mapping = {
            "injection": ["T1190", "T1059"],  # Exploit Public-Facing, Command Injection
            "xss": ["T1189", "T1059.007"],  # Drive-by Compromise, JavaScript
            "sql": ["T1190", "T1213"],  # Exploit Public-Facing, Data from Information Repositories
            "authentication": ["T1078", "T1110"],  # Valid Accounts, Brute Force
            "authorization": ["T1548", "T1134"],  # Abuse Elevation Control, Access Token Manipulation
            "file upload": ["T1105", "T1608"],  # Ingress Tool Transfer, Stage Capabilities
            "path traversal": ["T1083", "T1005"],  # File and Directory Discovery, Data from Local System
            "ssrf": ["T1090"],  # Proxy
            "deserialization": ["T1203", "T1059"],  # Exploitation for Client Execution
            "buffer overflow": ["T1203", "T1055"],  # Exploitation, Process Injection
        }

        # CWE 映射
        cwe_mapping = {
            "CWE-79": ["T1189", "T1059.007"],  # XSS
            "CWE-89": ["T1190", "T1213"],  # SQL Injection
            "CWE-78": ["T1059"],  # OS Command Injection
            "CWE-22": ["T1083", "T1005"],  # Path Traversal
            "CWE-434": ["T1105", "T1608"],  # Unrestricted Upload
            "CWE-502": ["T1203", "T1059"],  # Deserialization
        }

        matched_technique_ids = set()
        description_lower = vulnerability_description.lower()

        # 基於描述匹配
        for keyword, technique_ids in keyword_mapping.items():
            if keyword in description_lower:
                matched_technique_ids.update(technique_ids)

        # 基於 CWE 匹配
        if cwe_id and cwe_id in cwe_mapping:
            matched_technique_ids.update(cwe_mapping[cwe_id])

        # 獲取技術詳情
        techniques = []
        for tech_id in matched_technique_ids:
            technique = self.get_technique_by_id(tech_id)
            if technique:
                techniques.append(technique)

        logger.info(
            "vulnerability_mapped",
            cwe_id=cwe_id,
            techniques_count=len(techniques),
        )

        return techniques

    def generate_attack_path(self, techniques: list[str]) -> dict[str, Any]:
        """
        生成攻擊路徑

        Args:
            techniques: 技術 ID 列表

        Returns:
            攻擊路徑資訊
        """
        path = {
            "techniques": [],
            "tactics_coverage": set(),
            "timestamp": datetime.now().isoformat(),
        }

        for tech_id in techniques:
            technique = self.get_technique_by_id(tech_id)
            if technique:
                path["techniques"].append(technique)
                path["tactics_coverage"].update(technique.get("tactics", []))

        path["tactics_coverage"] = list(path["tactics_coverage"])
        path["completeness_score"] = len(path["tactics_coverage"]) / 14  # 14 個戰術

        logger.info(
            "attack_path_generated",
            techniques_count=len(path["techniques"]),
            tactics_count=len(path["tactics_coverage"]),
        )

        return path


def main():
    """測試範例"""
    mapper = MitreMapper()

    # 1. 獲取技術資訊
    print("=== Technique Information ===")
    technique = mapper.get_technique_by_id("T1055")
    print(json.dumps(technique, indent=2, ensure_ascii=False))

    # 2. 搜尋技術
    print("\n=== Search Techniques ===")
    results = mapper.search_techniques_by_keyword("injection", limit=5)
    for result in results:
        print(f"- {result['id']}: {result['name']}")

    # 3. 獲取所有戰術
    print("\n=== Tactics ===")
    tactics = mapper.get_tactics()
    for tactic in tactics[:5]:
        print(f"- {tactic['id']}: {tactic['name']}")

    # 4. 獲取緩解措施
    print("\n=== Mitigations for T1055 ===")
    mitigations = mapper.get_mitigations_for_technique("T1055")
    for mitigation in mitigations:
        print(f"- {mitigation['id']}: {mitigation['name']}")

    # 5. 漏洞映射
    print("\n=== Vulnerability Mapping ===")
    vuln_desc = "SQL injection vulnerability allows attackers to execute arbitrary SQL commands"
    techniques = mapper.map_vulnerability_to_techniques(vuln_desc, "CWE-89")
    for tech in techniques:
        print(f"- {tech['id']}: {tech['name']}")

    # 6. 生成攻擊路徑
    print("\n=== Attack Path ===")
    path = mapper.generate_attack_path(["T1190", "T1059", "T1055", "T1078"])
    print(f"Techniques: {len(path['techniques'])}")
    print(f"Tactics Coverage: {path['tactics_coverage']}")
    print(f"Completeness: {path['completeness_score']:.2%}")


if __name__ == "__main__":
    main()
