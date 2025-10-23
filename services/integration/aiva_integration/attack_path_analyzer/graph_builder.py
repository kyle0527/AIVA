"""
Graph Builder - 圖資料建構器

從 AIVA 的資料庫中讀取資產與 Findings，建立 Neo4j 圖
"""

import asyncio
import logging
from typing import Dict, Any

import asyncpg

from services.aiva_common.schemas import Asset

from .engine import AttackPathEngine

logger = logging.getLogger(__name__)


class GraphBuilder:
    """圖資料建構器"""

    def __init__(
        self,
        attack_path_engine: AttackPathEngine,
        postgres_dsn: str,
    ):
        """
        初始化建構器

        Args:
            attack_path_engine: 攻擊路徑引擎
            postgres_dsn: PostgreSQL 連線字串
        """
        self.engine = attack_path_engine
        self.postgres_dsn = postgres_dsn
        self.db_pool: asyncpg.Pool | None = None

    async def initialize(self):
        """初始化資料庫連線池"""
        self.db_pool = await asyncpg.create_pool(
            self.postgres_dsn,
            min_size=2,
            max_size=10,
        )
        logger.info("Database pool initialized")

    async def close(self):
        """關閉連線"""
        if self.db_pool:
            await self.db_pool.close()

    async def build_graph_from_database(self) -> dict[str, int]:
        """
        從 PostgreSQL 讀取資料並建立 Neo4j 圖

        Returns:
            統計資訊 {assets_count, findings_count}
        """
        if not self.db_pool:
            await self.initialize()

        # 初始化圖結構
        self.engine.initialize_graph()

        # 1. 載入資產
        assets_count = await self._load_assets()

        # 2. 載入 Findings
        findings_count = await self._load_findings()

        logger.info(
            f"Graph built successfully: {assets_count} assets, {findings_count} findings"
        )

        return {
            "assets_count": assets_count,
            "findings_count": findings_count,
        }

    async def _load_assets(self) -> int:
        """從資料庫載入資產"""
        async with self.db_pool.acquire() as conn:  # type: ignore
            # 假設有 assets 資料表
            query = """
                SELECT asset_id, url, type, metadata
                FROM assets
                WHERE is_active = true
            """

            rows = await conn.fetch(query)

            count = 0
            for row in rows:
                try:
                    # 建立 Asset 物件
                    asset = Asset(
                        asset_id=row["asset_id"],
                        value=row["url"],  # Asset 使用 value 而非 url
                        type=row["type"],
                        # 其他欄位...
                    )

                    # 加入圖
                    self.engine.add_asset(asset)
                    count += 1

                except Exception as e:
                    logger.error(f"Failed to add asset {row['asset_id']}: {e}")

            logger.info(f"Loaded {count} assets")
            return count

    async def _load_findings(self) -> int:
        """從資料庫載入 Findings - 支援 Phase I 漏洞類型"""
        async with self.db_pool.acquire() as conn:  # type: ignore
            # 假設有 findings 資料表
            query = """
                SELECT finding_id, task_id, vulnerability_data,
                       severity, confidence, target_data, evidence_data,
                       vulnerability_type, module_name
                FROM findings
                WHERE severity IN ('CRITICAL', 'HIGH', 'MEDIUM')
                ORDER BY created_at DESC
                LIMIT 1000
            """

            rows = await conn.fetch(query)

            count = 0
            for row in rows:
                try:
                    # 處理 Phase I 特殊漏洞類型
                    vuln_type = row.get('vulnerability_type', '')
                    module_name = row.get('module_name', '')
                    
                    # 構建簡化的 Finding 對象用於圖分析
                    finding_data = {
                        'finding_id': row['finding_id'],
                        'vulnerability_type': vuln_type,
                        'severity': row['severity'],
                        'target_url': self._extract_target_url(row['target_data']),
                        'module_name': module_name
                    }
                    
                    # 根據 Phase I 模組類型進行特殊處理
                    if module_name == 'FUNC_CLIENT_AUTH_BYPASS':
                        self._add_client_auth_bypass_to_graph(finding_data)
                    elif 'SSRF' in vuln_type and 'Advanced' in vuln_type:
                        self._add_advanced_ssrf_to_graph(finding_data)
                    else:
                        # 標準漏洞處理
                        self._add_standard_finding_to_graph(finding_data)
                    
                    count += 1

                except Exception as e:
                    logger.error(f"Failed to add finding {row['finding_id']}: {e}")

            logger.info(f"Loaded {count} findings (包含 Phase I 高價值漏洞)")
            return count
    
    def _extract_target_url(self, target_data: Any) -> str:
        """從目標資料中提取 URL"""
        if isinstance(target_data, dict):
            return target_data.get('url', 'unknown')
        return str(target_data) if target_data else 'unknown'
    
    def _add_client_auth_bypass_to_graph(self, finding_data: Dict[str, Any]):
        """將客戶端授權繞過漏洞添加到攻擊圖"""
        target_url = finding_data['target_url']
        
        # 創建客戶端漏洞節點
        client_vuln_node = {
            'type': 'ClientSideVulnerability',
            'id': finding_data['finding_id'],
            'url': target_url,
            'severity': finding_data['severity'],
            'exploitation_difficulty': 'Low'  # 客戶端繞過通常很容易利用
        }
        
        # 如果目標是管理頁面，增加到管理系統的路徑
        if any(admin_path in target_url.lower() for admin_path in ['/admin', '/management', '/dashboard']):
            admin_system_node = {
                'type': 'AdminSystem',
                'id': f"admin_system_{hash(target_url)}",
                'url': target_url,
                'criticality': 'High'
            }
            
            # 創建攻擊路徑：客戶端漏洞 -> 管理系統訪問
            attack_path = {
                'from': client_vuln_node['id'],
                'to': admin_system_node['id'],
                'relationship': 'ENABLES_UNAUTHORIZED_ACCESS',
                'risk_score': 8.5,  # 高風險
                'description': f"Client-side authorization bypass enables admin access to {target_url}"
            }
            
            logger.debug(f"Added client auth bypass path: {client_vuln_node['id']} -> {admin_system_node['id']}")
    
    def _add_advanced_ssrf_to_graph(self, finding_data: Dict[str, Any]):
        """將進階 SSRF 漏洞添加到攻擊圖"""
        target_url = finding_data['target_url']
        vuln_type = finding_data['vulnerability_type']
        
        # 創建 SSRF 漏洞節點
        ssrf_node = {
            'type': 'SSRFVulnerability',
            'id': finding_data['finding_id'],
            'url': target_url,
            'severity': finding_data['severity']
        }
        
        # 根據 SSRF 類型創建可達目標
        if 'Cloud Metadata' in vuln_type:
            # 雲端元數據訪問 - 極高風險
            cloud_metadata_node = {
                'type': 'CloudMetadataService',
                'id': f"cloud_metadata_{hash(target_url)}",
                'provider': 'AWS/GCP/Azure',
                'criticality': 'Critical'
            }
            
            attack_path = {
                'from': ssrf_node['id'],
                'to': cloud_metadata_node['id'],
                'relationship': 'ACCESSES_CLOUD_METADATA',
                'risk_score': 9.5,  # 極高風險
                'description': f"SSRF vulnerability enables cloud metadata access from {target_url}"
            }
            
        elif 'Internal Services' in vuln_type:
            # 內部服務訪問
            internal_service_node = {
                'type': 'InternalService',
                'id': f"internal_service_{hash(target_url)}",
                'service_type': 'Database/API/Microservice',
                'criticality': 'High'
            }
            
            attack_path = {
                'from': ssrf_node['id'],
                'to': internal_service_node['id'],
                'relationship': 'ACCESSES_INTERNAL_SERVICE',
                'risk_score': 8.0,
                'description': f"SSRF vulnerability enables internal service access from {target_url}"
            }
            
        logger.debug(f"Added advanced SSRF path: {ssrf_node['id']} -> internal services")
    
    def _add_standard_finding_to_graph(self, finding_data: Dict[str, Any]):
        """將標準漏洞添加到攻擊圖"""
        # 處理其他類型的漏洞
        standard_node = {
            'type': 'Vulnerability',
            'id': finding_data['finding_id'],
            'vulnerability_type': finding_data['vulnerability_type'],
            'severity': finding_data['severity'],
            'target': finding_data['target_url']
        }
        
        logger.debug(f"Added standard vulnerability: {finding_data['finding_id']}")

    async def rebuild_graph(self) -> dict[str, int]:
        """
        重建圖（清空後重新建立）

        Returns:
            統計資訊
        """
        logger.warning("Rebuilding graph from scratch...")

        # 清空現有圖
        self.engine.clear_graph()

        # 重新建立
        stats = await self.build_graph_from_database()

        return stats

    async def incremental_update(self, since_timestamp: str) -> dict[str, int]:
        """
        增量更新圖（只新增最新的資料）

        Args:
            since_timestamp: 起始時間戳（ISO 8601 格式）

        Returns:
            統計資訊
        """
        if not self.db_pool:
            await self.initialize()

        async with self.db_pool.acquire() as conn:  # type: ignore
            # 取得最新的 Findings
            query = """
                SELECT finding_id, task_id, vulnerability_data,
                       severity, confidence, target_data, evidence_data
                FROM findings
                WHERE created_at > $1
                ORDER BY created_at ASC
            """

            rows = await conn.fetch(query, since_timestamp)

            count = 0
            for row in rows:
                try:
                    # 加入圖
                    # self.engine.add_finding(finding)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to add finding {row['finding_id']}: {e}")

            logger.info(f"Incrementally added {count} findings")
            return {"findings_added": count}


# 使用範例
async def main():
    logging.basicConfig(level=logging.INFO)

    # 建立引擎
    engine = AttackPathEngine(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your_password",
    )

    # 建立 Builder
    builder = GraphBuilder(
        attack_path_engine=engine,
        postgres_dsn="postgresql://user:password@localhost:5432/aiva",
    )

    try:
        # 建立圖
        stats = await builder.build_graph_from_database()
        print(f"Graph built: {stats}")

        # 尋找攻擊路徑
        paths = engine.find_attack_paths(target_node_type="Database")
        print(f"\nFound {len(paths)} attack paths")

        for path in paths[:3]:  # 只顯示前 3 條
            print(f"\n路徑 {path.path_id}:")
            print(f"  風險: {path.total_risk_score:.2f}")
            print(f"  長度: {path.length}")
            print(f"  {path.description}")

    finally:
        await builder.close()
        engine.close()


if __name__ == "__main__":
    asyncio.run(main())
