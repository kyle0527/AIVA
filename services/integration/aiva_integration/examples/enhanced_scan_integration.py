"""
AIVA 增強功能整合示例

展示如何在實際掃描流程中使用新的資產與漏洞生命週期管理功能
"""

from __future__ import annotations

import asyncio
from typing import Any

from sqlalchemy import create_engine  # type: ignore[import-not-found]
from sqlalchemy.orm import sessionmaker  # type: ignore[import-not-found]

from services.aiva_common.schemas import FindingPayload
from services.aiva_common.utils import get_logger
from services.integration.aiva_integration.analysis.vuln_correlation_analyzer import (
    VulnerabilityCorrelationAnalyzer,
)
from services.integration.aiva_integration.reception.lifecycle_manager import (
    AssetVulnerabilityManager,
)
from services.integration.aiva_integration.reception.models_enhanced import Base

logger = get_logger(__name__)


class EnhancedScanProcessor:
    """
    增強版掃描處理器

    整合資產管理、漏洞生命週期、相關性分析等功能
    """

    def __init__(self, database_url: str):
        """
        初始化處理器

        Args:
            database_url: 資料庫連接 URL
        """
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # 創建表（如果不存在）
        Base.metadata.create_all(bind=self.engine)

    async def process_scan_results(
        self,
        scan_id: str,
        target_info: dict[str, Any],
        findings: list[FindingPayload],
    ) -> dict[str, Any]:
        """
        處理掃描結果

        Args:
            scan_id: 掃描 ID
            target_info: 目標資訊（包含業務上下文）
            findings: 掃描發現列表

        Returns:
            處理結果摘要
        """
        session = self.SessionLocal()
        try:
            # 1. 註冊或更新資產
            logger.info(f"Processing scan {scan_id}: registering asset...")
            manager = AssetVulnerabilityManager(session)

            asset = manager.register_asset(
                asset_value=target_info.get("url") or target_info.get("value"),
                asset_type=target_info.get("type", "url"),
                name=target_info.get("name"),
                business_criticality=target_info.get(
                    "business_criticality", "medium"
                ),
                environment=target_info.get("environment", "development"),
                owner=target_info.get("owner"),
                tags=target_info.get("tags", []),
                technology_stack=target_info.get("technology_stack"),
            )

            logger.info(
                f"Asset registered: {asset.asset_id} ({asset.business_criticality} / {asset.environment})"  # type: ignore[attr-defined]
            )

            # 2. 處理每個 Finding，進行漏洞去重和生命週期管理
            logger.info(f"Processing {len(findings)} findings...")
            new_vulnerabilities = []
            updated_vulnerabilities = []

            for finding in findings:
                vulnerability, is_new = manager.process_finding(
                    finding, asset.asset_id  # type: ignore[attr-defined]
                )

                if is_new:
                    new_vulnerabilities.append(vulnerability)
                    logger.info(
                        f"New vulnerability: {vulnerability.vulnerability_id} "  # type: ignore[attr-defined]
                        f"({vulnerability.severity})"  # type: ignore[attr-defined]
                    )
                else:
                    updated_vulnerabilities.append(vulnerability)

            # 3. 執行相關性分析
            logger.info("Performing correlation analysis...")
            analyzer = VulnerabilityCorrelationAnalyzer()

            # 轉換為分析器所需的格式
            finding_dicts = [self._finding_to_dict(f) for f in findings]

            # 基礎相關性分析
            correlation_result = analyzer.analyze_correlations(finding_dicts)

            # 程式碼層面根因分析
            root_cause_result = analyzer.analyze_code_level_root_cause(finding_dicts)

            # SAST-DAST 關聯分析
            sast_dast_result = analyzer.analyze_sast_dast_correlation(finding_dicts)

            # 4. 根據分析結果更新漏洞狀態和標籤
            self._apply_analysis_results(
                manager, correlation_result, root_cause_result, sast_dast_result
            )

            # 5. 生成摘要報告
            summary = self._generate_summary(
                asset=asset,
                new_vulnerabilities=new_vulnerabilities,
                updated_vulnerabilities=updated_vulnerabilities,
                correlation_result=correlation_result,
                root_cause_result=root_cause_result,
                sast_dast_result=sast_dast_result,
            )

            logger.info(f"Scan processing completed: {scan_id}")
            return summary

        except Exception as e:
            logger.error(f"Error processing scan {scan_id}: {e}")
            session.rollback()
            raise
        finally:
            session.close()

    def _finding_to_dict(self, finding: FindingPayload) -> dict[str, Any]:
        """轉換 FindingPayload 為字典"""
        return {
            "finding_id": finding.finding_id,
            "vulnerability_type": finding.vulnerability.name.value,
            "severity": finding.vulnerability.severity.value,
            "confidence": finding.vulnerability.confidence.value,
            "location": {
                "url": str(finding.target.url),
                "parameter": finding.target.parameter,
                "method": finding.target.method,
            },
        }

    def _apply_analysis_results(
        self,
        manager: AssetVulnerabilityManager,
        correlation_result: dict[str, Any],
        root_cause_result: dict[str, Any],
        sast_dast_result: dict[str, Any],
    ) -> None:
        """
        根據分析結果應用標籤和更新狀態

        Args:
            manager: 資產漏洞管理器
            correlation_result: 相關性分析結果
            root_cause_result: 根因分析結果
            sast_dast_result: SAST-DAST 關聯結果
        """
        # 標記攻擊鏈中的漏洞
        if correlation_result.get("attack_chains"):
            for chain in correlation_result["attack_chains"]:
                for step in chain.get("matched_steps", []):
                    for finding in step.get("findings", []):
                        vuln_id = finding.get("finding_id") or finding.get(
                            "vulnerability_id"
                        )
                        if vuln_id:
                            manager.add_vulnerability_tag(vuln_id, "attack_chain")
                            manager.add_vulnerability_tag(
                                vuln_id, f"chain_{chain['pattern'][0]}"
                            )

        # 標記根本原因漏洞
        if root_cause_result.get("root_causes"):
            for root_cause in root_cause_result["root_causes"]:
                for vuln_id in root_cause.get("vulnerability_ids", []):
                    manager.add_vulnerability_tag(vuln_id, "root_cause_derived")
                    manager.add_vulnerability_tag(
                        vuln_id, f"component_{root_cause['component_type']}"
                    )

        # 標記已驗證的 SAST-DAST 流
        if sast_dast_result.get("confirmed_flows"):
            for flow in sast_dast_result["confirmed_flows"]:
                sast_id = flow.get("sast_finding_id")
                dast_id = flow.get("dast_finding_id")

                if sast_id:
                    manager.add_vulnerability_tag(sast_id, "sast_dast_confirmed")
                    manager.add_vulnerability_tag(sast_id, "high_confidence")

                if dast_id:
                    manager.add_vulnerability_tag(dast_id, "sast_dast_confirmed")
                    manager.add_vulnerability_tag(dast_id, "high_confidence")

        # 標記未確認的 SAST 發現
        if sast_dast_result.get("unconfirmed_sast"):
            for unconfirmed in sast_dast_result["unconfirmed_sast"]:
                vuln_id = unconfirmed.get("finding_id")
                if vuln_id:
                    manager.add_vulnerability_tag(vuln_id, "sast_only")
                    manager.add_vulnerability_tag(vuln_id, "needs_verification")

    def _generate_summary(
        self,
        asset: Any,
        new_vulnerabilities: list[Any],
        updated_vulnerabilities: list[Any],
        correlation_result: dict[str, Any],
        root_cause_result: dict[str, Any],
        sast_dast_result: dict[str, Any],
    ) -> dict[str, Any]:
        """生成處理摘要"""
        return {
            "asset": {
                "asset_id": asset.asset_id,  # type: ignore[attr-defined]
                "name": asset.name,  # type: ignore[attr-defined]
                "type": asset.type,  # type: ignore[attr-defined]
                "business_criticality": asset.business_criticality,  # type: ignore[attr-defined]
                "environment": asset.environment,  # type: ignore[attr-defined]
            },
            "vulnerabilities": {
                "new": len(new_vulnerabilities),
                "updated": len(updated_vulnerabilities),
                "total": len(new_vulnerabilities) + len(updated_vulnerabilities),
                "by_severity": self._count_by_severity(
                    new_vulnerabilities + updated_vulnerabilities
                ),
            },
            "correlation_analysis": {
                "correlation_groups": len(correlation_result.get("correlation_groups", [])),
                "attack_chains": len(correlation_result.get("attack_chains", [])),
                "risk_amplification": correlation_result.get("risk_amplification", 1.0),
                "risk_level": correlation_result.get("summary", {}).get("risk_level", "unknown"),
            },
            "root_cause_analysis": {
                "root_causes": len(root_cause_result.get("root_causes", [])),
                "affected_vulnerabilities": len(
                    root_cause_result.get("derived_vulnerabilities", [])
                ),
                "fix_efficiency": root_cause_result.get("summary", {}).get(
                    "fix_efficiency", ""
                ),
            },
            "sast_dast_correlation": {
                "confirmed_flows": len(sast_dast_result.get("confirmed_flows", [])),
                "unconfirmed_sast": len(sast_dast_result.get("unconfirmed_sast", [])),
                "orphan_dast": len(sast_dast_result.get("orphan_dast", [])),
                "confirmation_rate": sast_dast_result.get("summary", {}).get(
                    "confirmation_rate", 0
                ),
            },
            "key_insights": self._generate_key_insights(
                correlation_result, root_cause_result, sast_dast_result
            ),
        }

    def _count_by_severity(self, vulnerabilities: list[Any]) -> dict[str, int]:
        """統計漏洞嚴重程度分布"""
        counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
        for vuln in vulnerabilities:
            severity = vuln.severity.upper()  # type: ignore[attr-defined]
            if severity in counts:
                counts[severity] += 1
        return counts

    def _generate_key_insights(
        self,
        correlation_result: dict[str, Any],
        root_cause_result: dict[str, Any],
        sast_dast_result: dict[str, Any],
    ) -> list[str]:
        """生成關鍵洞察"""
        insights = []

        # 攻擊鏈洞察
        if correlation_result.get("attack_chains"):
            chain_count = len(correlation_result["attack_chains"])
            insights.append(
                f"🔗 識別出 {chain_count} 條攻擊鏈，攻擊者可能透過這些路徑達成進階攻擊目標"
            )

        # 根因洞察
        if root_cause_result.get("root_causes"):
            root_count = len(root_cause_result["root_causes"])
            affected = len(root_cause_result.get("derived_vulnerabilities", []))
            insights.append(
                f"🎯 發現 {root_count} 個共用元件問題，影響 {affected} 個漏洞。"
                f"建議優先修復這些根本原因以提高效率"
            )

        # SAST-DAST 關聯洞察
        if sast_dast_result.get("confirmed_flows"):
            confirmed = len(sast_dast_result["confirmed_flows"])
            rate = sast_dast_result.get("summary", {}).get("confirmation_rate", 0)
            insights.append(
                f"✅ {confirmed} 個 SAST 發現已被 DAST 驗證（確認率 {rate}%），"
                f"這些是真實可利用的漏洞，應立即處理"
            )

        # 風險放大洞察
        risk_amp = correlation_result.get("risk_amplification", 1.0)
        if risk_amp > 1.5:
            insights.append(
                f"⚠️ 漏洞相關性導致風險放大 {risk_amp}x，"
                f"綜合風險遠高於單個漏洞的總和"
            )

        return insights


# ============================================================================
# 使用範例
# ============================================================================


async def example_scan_workflow():
    """完整的掃描工作流程示例"""
    # 初始化處理器
    processor = EnhancedScanProcessor(
        database_url="postgresql://user:password@localhost:5432/aiva_db"
    )

    # 目標資訊（包含業務上下文）
    target_info = {
        "url": "https://api.example.com",
        "name": "Production API Server",
        "type": "url",
        "business_criticality": "critical",  # 業務關鍵性
        "environment": "production",  # 環境
        "owner": "api-team@example.com",
        "tags": ["api", "payment", "pci-dss"],
        "technology_stack": {
            "framework": "Django",
            "language": "Python 3.11",
            "database": "PostgreSQL",
        },
    }

    # 模擬掃描發現（實際情況會從掃描引擎獲取）
    findings = []  # type: list[FindingPayload]
    # ... 從掃描引擎獲取 findings

    # 處理掃描結果
    result = await processor.process_scan_results(
        scan_id="scan_20251014_001", target_info=target_info, findings=findings
    )

    # 輸出摘要
    print("=" * 80)
    print("掃描結果摘要")
    print("=" * 80)
    print(f"\n資產: {result['asset']['name']}")
    print(
        f"業務重要性: {result['asset']['business_criticality']} | "
        f"環境: {result['asset']['environment']}"
    )
    print(
        f"\n漏洞統計: 新發現 {result['vulnerabilities']['new']} 個, "
        f"更新 {result['vulnerabilities']['updated']} 個"
    )
    print(f"嚴重程度分布: {result['vulnerabilities']['by_severity']}")

    print(f"\n相關性分析:")
    print(f"  - 相關性組合: {result['correlation_analysis']['correlation_groups']}")
    print(f"  - 攻擊鏈: {result['correlation_analysis']['attack_chains']}")
    print(f"  - 風險放大: {result['correlation_analysis']['risk_amplification']}x")

    print(f"\n根因分析:")
    print(f"  - 根本原因: {result['root_cause_analysis']['root_causes']}")
    print(f"  - {result['root_cause_analysis']['fix_efficiency']}")

    print(f"\nSAST-DAST 關聯:")
    print(f"  - 已驗證資料流: {result['sast_dast_correlation']['confirmed_flows']}")
    print(
        f"  - 確認率: {result['sast_dast_correlation']['confirmation_rate']}%"
    )

    print("\n關鍵洞察:")
    for insight in result["key_insights"]:
        print(f"  {insight}")

    print("=" * 80)

    return result


async def example_vulnerability_management():
    """漏洞生命週期管理示例"""
    from sqlalchemy import create_engine  # type: ignore[import-not-found]
    from sqlalchemy.orm import sessionmaker  # type: ignore[import-not-found]

    engine = create_engine("postgresql://user:password@localhost:5432/aiva_db")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    manager = AssetVulnerabilityManager(session)

    # 1. 查詢特定資產的所有開放漏洞
    vulnerabilities = manager.get_asset_vulnerabilities(
        asset_id="asset_abc123", include_fixed=False
    )
    print(f"找到 {len(vulnerabilities)} 個開放漏洞")

    # 2. 更新漏洞狀態
    for vuln in vulnerabilities[:3]:  # 處理前 3 個
        manager.update_vulnerability_status(
            vulnerability_id=vuln.vulnerability_id,  # type: ignore[attr-defined]
            new_status="in_progress",
            changed_by="john.doe@example.com",
            comment="已指派給開發團隊",
        )

    # 3. 指派漏洞
    manager.assign_vulnerability(
        vulnerability_id=vulnerabilities[0].vulnerability_id,  # type: ignore[attr-defined]
        assigned_to="alice@example.com",
        changed_by="manager@example.com",
    )

    # 4. 查詢逾期漏洞
    overdue_vulns = manager.get_overdue_vulnerabilities()
    print(f"⚠️ 有 {len(overdue_vulns)} 個漏洞已逾期")

    # 5. 計算 MTTR
    mttr_high = manager.calculate_mttr(severity="HIGH", days=30)
    print(f"高危漏洞平均修復時間: {mttr_high['avg_hours']:.1f} 小時")

    session.close()


if __name__ == "__main__":
    # 執行示例
    asyncio.run(example_scan_workflow())
    asyncio.run(example_vulnerability_management())
