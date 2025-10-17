"""
Enhanced Attack Path Analysis Demo
展示如何使用增強版攻擊路徑分析器和自然語言推薦系統
"""

import logging
from pathlib import Path

from services.aiva_common.enums import Confidence, Severity, VulnerabilityName
from services.aiva_common.schemas import Asset, FindingPayload, Target, Vulnerability
from services.integration.aiva_integration.attack_path_analyzer.engine import (
    AttackPathEngine,
)
from services.integration.aiva_integration.attack_path_analyzer.nlp_recommender import (
    AttackPathNLPRecommender,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def demo_basic_attack_path_analysis():
    """基礎攻擊路徑分析示範"""
    logger.info("=== 基礎攻擊路徑分析 ===")

    # 初始化引擎
    engine = AttackPathEngine(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    try:
        # 清空圖（測試用）
        engine.clear_graph()

        # 初始化
        engine.initialize_graph()

        # 模擬掃描發現的資產和漏洞
        assets = [
            Asset(asset_id="web-app-1", value="https://example.com", type="webapp"),
            Asset(
                asset_id="api-gateway-1",
                value="https://api.example.com",
                type="api",
            ),
        ]

        findings = [
            # Finding 1: SSRF on web app
            FindingPayload(
                finding_id="finding-001",
                target=Target(
                    url="https://example.com/fetch",
                    method="POST",
                    params={},
                    headers={},
                    body='{"url": "http://169.254.169.254/latest/meta-data/"}',
                ),
                vulnerability=Vulnerability(
                    name=VulnerabilityName.SSRF,
                    severity=Severity.HIGH,
                    confidence=Confidence.FIRM,
                    cwe="CWE-918",
                    description="Server-Side Request Forgery vulnerability allows access to internal resources",
                ),
            ),
            # Finding 2: SQL Injection
            FindingPayload(
                finding_id="finding-002",
                target=Target(
                    url="https://example.com/search",
                    method="GET",
                    params={"q": "test' OR '1'='1"},
                    headers={},
                    body="",
                ),
                vulnerability=Vulnerability(
                    name=VulnerabilityName.SQLI,
                    severity=Severity.CRITICAL,
                    confidence=Confidence.CERTAIN,
                    cwe="CWE-89",
                    description="SQL Injection allows direct database access",
                ),
            ),
            # Finding 3: XSS
            FindingPayload(
                finding_id="finding-003",
                target=Target(
                    url="https://example.com/comment",
                    method="POST",
                    params={},
                    headers={},
                    body='{"comment": "<script>alert(1)</script>"}',
                ),
                vulnerability=Vulnerability(
                    name=VulnerabilityName.XSS,
                    severity=Severity.HIGH,
                    confidence=Confidence.FIRM,
                    cwe="CWE-79",
                    description="Cross-Site Scripting vulnerability",
                ),
            ),
            # Finding 4: BOLA on API
            FindingPayload(
                finding_id="finding-004",
                target=Target(
                    url="https://api.example.com/users/123/profile",
                    method="GET",
                    params={},
                    headers={"Authorization": "Bearer token"},
                    body="",
                ),
                vulnerability=Vulnerability(
                    name=VulnerabilityName.BOLA,
                    severity=Severity.HIGH,
                    confidence=Confidence.CERTAIN,
                    cwe="CWE-639",
                    description="Broken Object Level Authorization",
                ),
            ),
        ]

        # 添加資產和漏洞到圖
        for asset in assets:
            engine.add_asset(asset)
            logger.info(f"Added asset: {asset.asset_id}")

        for finding in findings:
            engine.add_finding(finding)
            logger.info(
                f"Added finding: {finding.finding_id} - {finding.vulnerability.name.value}"
            )

        # 尋找攻擊路徑
        logger.info("\n=== 尋找到資料庫的攻擊路徑 ===")
        db_paths = engine.find_attack_paths(target_node_type="Database", limit=5)

        for path in db_paths:
            logger.info(f"\n路徑 {path.path_id}:")
            logger.info(f"  風險分數: {path.total_risk_score:.2f}")
            logger.info(f"  路徑長度: {path.length}")
            logger.info(f"  描述: {path.description}")

        # 尋找到內部網路的攻擊路徑
        logger.info("\n=== 尋找到內部網路的攻擊路徑 ===")
        internal_paths = engine.find_attack_paths(
            target_node_type="InternalNetwork", limit=5
        )

        for path in internal_paths:
            logger.info(f"\n路徑 {path.path_id}:")
            logger.info(f"  風險分數: {path.total_risk_score:.2f}")
            logger.info(f"  路徑長度: {path.length}")

        return db_paths + internal_paths

    finally:
        engine.close()


def demo_nlp_recommendations(paths: list):
    """自然語言推薦系統示範"""
    logger.info("\n=== 生成自然語言推薦 ===")

    # 初始化推薦器
    recommender = AttackPathNLPRecommender()

    # 分析並生成推薦
    recommendations = recommender.analyze_and_recommend(paths, top_n=5)

    # 輸出推薦
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"推薦 {i}/{len(recommendations)}")
        logger.info(f"{'=' * 80}")
        logger.info(f"路徑 ID: {rec.path_id}")
        logger.info(f"風險等級: {rec.risk_level.value.upper()}")
        logger.info(f"優先級分數: {rec.priority_score:.1f}/100")
        logger.info(f"\n{rec.executive_summary}")
        logger.info(f"\n預估工作量: {rec.estimated_effort}")
        logger.info(f"預估風險降低: {rec.estimated_risk_reduction:.0f}%")

        if rec.quick_wins:
            logger.info("\n快速修復建議:")
            for quick_win in rec.quick_wins:
                logger.info(f"  {quick_win}")

    return recommendations


def demo_generate_full_report(recommendations: list):
    """生成完整報告示範"""
    logger.info("\n=== 生成完整報告 ===")

    # 初始化推薦器
    recommender = AttackPathNLPRecommender()

    # 生成報告
    report = recommender.generate_report(recommendations)

    # 保存報告
    report_path = Path("_out/attack_path_analysis_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"報告已保存至: {report_path}")

    # 輸出報告前 1000 字元
    logger.info("\n報告預覽:")
    logger.info("=" * 80)
    logger.info(report[:1000])
    logger.info("...")
    logger.info("=" * 80)

    return report_path


def demo_critical_nodes():
    """關鍵節點分析示範"""
    logger.info("\n=== 關鍵節點分析 ===")

    engine = AttackPathEngine(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    try:
        # 尋找關鍵節點（高度中心性）
        critical_nodes = engine.find_critical_nodes(limit=10)

        logger.info(f"\n發現 {len(critical_nodes)} 個關鍵節點:")

        for i, node in enumerate(critical_nodes, 1):
            node_name = node.get("name", node.get("value", node.get("id", "Unknown")))
            node_degree = node.get("degree", 0)
            node_labels = node.get("labels", [])

            logger.info(f"\n{i}. {node_name}")
            logger.info(f"   類型: {', '.join(node_labels)}")
            logger.info(f"   連接度: {node_degree}")

            # 這些是關鍵節點，應該優先加強防護
            if node_degree >= 3:
                logger.info("   [WARN] 高連接度節點，建議加強監控和防護")

    finally:
        engine.close()


def demo_vulnerability_statistics():
    """漏洞統計示範"""
    logger.info("\n=== 漏洞統計 ===")

    engine = AttackPathEngine(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    try:
        stats = engine.get_vulnerability_statistics()

        logger.info(f"\n總漏洞數: {stats['total']}")
        logger.info("\n按嚴重程度分布:")

        for severity, count in sorted(
            stats["by_severity"].items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / stats["total"] * 100) if stats["total"] > 0 else 0
            bar = "█" * int(percentage / 5)
            logger.info(f"  {severity:12s}: {count:3d} ({percentage:5.1f}%) {bar}")

    finally:
        engine.close()


def main():
    """主示範函數"""
    logger.info("=" * 80)
    logger.info("攻擊路徑分析與自然語言推薦系統 - 完整示範")
    logger.info("=" * 80)

    try:
        # 1. 基礎攻擊路徑分析
        paths = demo_basic_attack_path_analysis()

        if not paths:
            logger.warning("未發現任何攻擊路徑，請檢查 Neo4j 連線和資料")
            return

        # 2. 生成自然語言推薦
        recommendations = demo_nlp_recommendations(paths)

        # 3. 生成完整報告
        report_path = demo_generate_full_report(recommendations)

        # 4. 關鍵節點分析
        demo_critical_nodes()

        # 5. 漏洞統計
        demo_vulnerability_statistics()

        logger.info("\n" + "=" * 80)
        logger.info("示範完成！")
        logger.info(f"完整報告已保存至: {report_path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"示範執行失敗: {e}", exc_info=True)


if __name__ == "__main__":
    main()
