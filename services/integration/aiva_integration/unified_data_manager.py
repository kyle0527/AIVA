"""
統一資料管理器 (Unified Data Manager)

整合所有資料儲存和查詢功能，提供統一的資料訪問接口給各模組使用。

設計原則：
1. 遵循 aiva_common 規範
2. 統一配置管理 (使用 config.py)
3. 支援多種資料源（PostgreSQL、SQLite、NetworkX）
4. 提供跨模組資料查詢能力
5. 自動備份與清理機制

整合的資料類型：
- Experience Records (經驗記錄)
- Attack Paths (攻擊路徑圖)
- Findings (漏洞發現)
- Scan Results (掃描結果)
- Training Datasets (訓練資料集)
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker

# 遵循 aiva_common 規範
from services.aiva_common.enums import Severity, Confidence
from services.aiva_common.schemas import FindingPayload

# 本地模組
from ..config import (
    ATTACK_GRAPH_FILE,
    EXPERIENCE_DB_URL,
    POSTGRES_DSN,
    TRAINING_DATASET_DIR,
    BACKUP_DIR,
    BACKUP_RETENTION_DAYS,
)
from .reception.experience_repository import ExperienceRepository
from .reception.experience_models import ExperienceRecord
from .reception.finding_repository import FindingRepository
from .attack_path_analyzer.engine import AttackPathEngine

logger = logging.getLogger(__name__)


class UnifiedDataManager:
    """統一資料管理器
    
    提供給各模組的單一資料訪問接口
    
    使用範例：
        # Core 模組使用
        manager = UnifiedDataManager()
        manager.save_attack_experience(plan_id, attack_type, ...)
        
        # Features 模組使用
        manager.save_finding(finding_payload)
        
        # 跨模組查詢
        experiences = manager.query_high_quality_experiences(min_score=0.8)
        paths = manager.get_attack_paths_for_target(target_url)
    """
    
    def __init__(
        self,
        postgres_url: str = POSTGRES_DSN,
        experience_db_url: str = EXPERIENCE_DB_URL,
        attack_graph_file: Path | str = ATTACK_GRAPH_FILE,
    ):
        """初始化統一資料管理器
        
        Args:
            postgres_url: PostgreSQL 資料庫連接字串
            experience_db_url: 經驗資料庫連接字串（SQLite）
            attack_graph_file: 攻擊路徑圖檔案路徑
        """
        logger.info("🚀 Initializing UnifiedDataManager...")
        
        # 1. 經驗資料庫（SQLite）
        self.experience_repo = ExperienceRepository(experience_db_url)
        logger.info(f"  ✅ Experience DB: {experience_db_url}")
        
        # 2. 攻擊路徑引擎（NetworkX）
        self.attack_path_engine = AttackPathEngine(graph_file=attack_graph_file)
        logger.info(f"  ✅ Attack Path Engine: {attack_graph_file}")
        
        # 3. PostgreSQL（漏洞發現、掃描結果）
        try:
            self.postgres_engine = create_engine(postgres_url, echo=False)
            self.PostgresSession = sessionmaker(bind=self.postgres_engine)
            
            # 4. 漏洞發現資料庫（PostgreSQL）
            self.finding_repo = FindingRepository(postgres_url)
            logger.info(f"  ✅ PostgreSQL + FindingRepository: {postgres_url}")
        except Exception as e:
            logger.warning(f"  ⚠️  PostgreSQL connection failed: {e}")
            self.postgres_engine = None
            self.PostgresSession = None
            self.finding_repo = None
        
        logger.info("✅ UnifiedDataManager initialized successfully")
    
    # ========================================================================
    # 經驗記錄相關（供 Core 模組使用）
    # ========================================================================
    
    def save_attack_experience(
        self,
        plan_id: str,
        attack_type: str,
        ast_graph: dict[str, Any],
        execution_trace: dict[str, Any],
        metrics: dict[str, Any],
        feedback: dict[str, Any],
        target_info: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExperienceRecord:
        """保存攻擊執行經驗
        
        供 AICommander 和 AttackExecutor 使用
        
        Args:
            plan_id: 攻擊計畫 ID
            attack_type: 攻擊類型
            ast_graph: AST 圖結構
            execution_trace: 執行軌跡
            metrics: 評估指標
            feedback: 回饋數據
            target_info: 目標資訊
            metadata: 額外元數據
            
        Returns:
            儲存的經驗記錄
        """
        return self.experience_repo.save_experience(
            plan_id=plan_id,
            attack_type=attack_type,
            ast_graph=ast_graph,
            execution_trace=execution_trace,
            metrics=metrics,
            feedback=feedback,
            target_info=target_info,
            metadata=metadata,
        )
    
    def query_high_quality_experiences(
        self,
        attack_type: str | None = None,
        min_score: float = 0.7,
        limit: int = 100,
    ) -> list[ExperienceRecord]:
        """查詢高質量經驗記錄
        
        供 RAG Engine 和 ExperienceManager 使用
        
        Args:
            attack_type: 攻擊類型過濾
            min_score: 最低分數閾值
            limit: 返回數量限制
            
        Returns:
            經驗記錄列表
        """
        return self.experience_repo.query_experiences(
            attack_type=attack_type,
            min_score=min_score,
            limit=limit,
        )
    
    def get_experience_statistics(self) -> dict[str, Any]:
        """獲取經驗庫統計資訊
        
        Returns:
            統計資訊字典
        """
        session = self.experience_repo.get_session()
        try:
            total_count = session.query(func.count(ExperienceRecord.id)).scalar()
            avg_score = session.query(func.avg(ExperienceRecord.overall_score)).scalar()
            
            # 按攻擊類型分組統計
            attack_type_stats = (
                session.query(
                    ExperienceRecord.attack_type,
                    func.count(ExperienceRecord.id),
                    func.avg(ExperienceRecord.overall_score),
                )
                .group_by(ExperienceRecord.attack_type)
                .all()
            )
            
            return {
                "total_experiences": total_count or 0,
                "average_score": float(avg_score) if avg_score else 0.0,
                "attack_types": [
                    {
                        "type": attack_type,
                        "count": count,
                        "avg_score": float(avg_score) if avg_score else 0.0,
                    }
                    for attack_type, count, avg_score in attack_type_stats
                ],
            }
        finally:
            session.close()
    
    # ========================================================================
    # 攻擊路徑相關（供 Integration 模組使用）
    # ========================================================================
    
    def add_asset_to_attack_graph(
        self,
        asset_id: str,
        asset_type: str,
        url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加資產到攻擊路徑圖
        
        供 Scan 模組使用
        
        Args:
            asset_id: 資產 ID
            asset_type: 資產類型
            url: 資產 URL
            metadata: 額外元數據
        """
        self.attack_path_engine.add_asset(
            asset_id=asset_id,
            asset_type=asset_type,
            url=url,
            metadata=metadata or {},
        )
    
    def add_vulnerability_to_attack_graph(
        self,
        vuln_id: str,
        vuln_type: str,
        severity: str,
        affected_asset: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """添加漏洞到攻擊路徑圖
        
        供 Features 模組使用
        
        Args:
            vuln_id: 漏洞 ID
            vuln_type: 漏洞類型
            severity: 嚴重程度
            affected_asset: 受影響的資產 ID
            metadata: 額外元數據
        """
        self.attack_path_engine.add_vulnerability(
            vuln_id=vuln_id,
            vuln_type=vuln_type,
            severity=severity,
            affected_asset=affected_asset,
            metadata=metadata or {},
        )
    
    def get_attack_paths_for_target(
        self,
        target: str,
        max_paths: int = 10,
    ) -> list[list[str]]:
        """獲取目標的攻擊路徑
        
        供 AI 決策使用
        
        Args:
            target: 目標資產 ID
            max_paths: 最大路徑數量
            
        Returns:
            攻擊路徑列表
        """
        try:
            return self.attack_path_engine.find_attack_paths(
                target_asset=target,
                max_paths=max_paths,
            )
        except Exception as e:
            logger.error(f"Failed to get attack paths: {e}")
            return []
    
    def save_attack_graph(self) -> None:
        """保存攻擊路徑圖到磁碟"""
        self.attack_path_engine.save_graph()
    
    def get_attack_graph_statistics(self) -> dict[str, Any]:
        """獲取攻擊路徑圖統計資訊"""
        return self.attack_path_engine.get_statistics()
    
    # ========================================================================
    # 漏洞發現相關（供 Features 模組使用）
    # ========================================================================
    
    def save_finding(
        self,
        finding: FindingPayload,
        scan_id: str | None = None,
        task_id: str | None = None,
    ) -> None:
        """保存漏洞發現
        
        供所有 Features Worker 使用
        
        Args:
            finding: 漏洞發現資料
            scan_id: 掃描 ID（可選）
            task_id: 任務 ID（可選）
        """
        if not self.finding_repo:
            logger.warning("FindingRepository not available, skipping finding save")
            return
        
        try:
            # 保存到 PostgreSQL
            record = self.finding_repo.save_finding(
                finding=finding,
                scan_id=scan_id or f"scan_{finding.finding_id[:8]}",
                task_id=task_id,
            )
            
            logger.info(
                f"✅ Saved finding to PostgreSQL: {finding.finding_id} "
                f"({finding.severity.value}, {finding.confidence.value})"
            )
            
            # 同時更新攻擊路徑圖
            self.add_vulnerability_to_attack_graph(
                vuln_id=finding.finding_id,
                vuln_type=finding.vulnerability_type or "unknown",
                severity=finding.severity.value,
                affected_asset=finding.affected_url or "",
                metadata={
                    "confidence": finding.confidence.value,
                    "cwe": finding.cwe,
                    "description": finding.description,
                },
            )
            
            logger.info(
                f"✅ Updated attack path graph with finding: {finding.finding_id}"
            )
            
        except Exception as e:
            logger.error(f"Failed to save finding: {e}")
    
    def query_findings(
        self,
        scan_id: str | None = None,
        severity: Severity | None = None,
        confidence: Confidence | None = None,
        vulnerability_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """查詢漏洞發現
        
        Args:
            scan_id: 掃描 ID 過濾
            severity: 嚴重程度過濾
            confidence: 信心度過濾
            vulnerability_type: 漏洞類型過濾
            status: 狀態過濾
            limit: 返回數量限制
            offset: 偏移量（分頁）
            
        Returns:
            漏洞發現列表
        """
        if not self.finding_repo:
            logger.warning("FindingRepository not available")
            return []
        
        try:
            findings = self.finding_repo.query_findings(
                scan_id=scan_id,
                severity=severity,
                confidence=confidence,
                vulnerability_type=vulnerability_type,
                status=status,
                limit=limit,
                offset=offset,
            )
            
            return [finding.to_dict() for finding in findings]
            
        except Exception as e:
            logger.error(f"Failed to query findings: {e}")
            return []
    
    def get_finding_statistics(
        self,
        scan_id: str | None = None,
    ) -> dict[str, Any]:
        """獲取漏洞統計資訊
        
        Args:
            scan_id: 掃描 ID 過濾（可選）
        
        Returns:
            統計資訊字典
        """
        if not self.finding_repo:
            logger.warning("FindingRepository not available")
            return {"total_findings": 0}
        
        try:
            return self.finding_repo.get_statistics(scan_id=scan_id)
        except Exception as e:
            logger.error(f"Failed to get finding statistics: {e}")
            return {"total_findings": 0}
    
    def update_finding_status(
        self,
        finding_id: str,
        new_status: str,
    ) -> bool:
        """更新漏洞狀態
        
        Args:
            finding_id: 漏洞 ID
            new_status: 新狀態（active, fixed, false_positive, accepted_risk）
        
        Returns:
            是否成功更新
        """
        if not self.finding_repo:
            logger.warning("FindingRepository not available")
            return False
        
        try:
            return self.finding_repo.update_finding_status(finding_id, new_status)
        except Exception as e:
            logger.error(f"Failed to update finding status: {e}")
            return False
    
    # ========================================================================
    # 訓練資料集相關
    # ========================================================================
    
    def create_training_dataset(
        self,
        name: str,
        attack_types: list[str] | None = None,
        min_score: float = 0.7,
        max_samples: int = 1000,
    ) -> str:
        """創建訓練資料集
        
        供訓練流程使用
        
        Args:
            name: 資料集名稱
            attack_types: 包含的攻擊類型
            min_score: 最低分數閾值
            max_samples: 最大樣本數
            
        Returns:
            資料集 ID
        """
        return self.experience_repo.create_dataset(
            name=name,
            attack_types=attack_types,
            min_score=min_score,
            max_samples=max_samples,
        )
    
    def export_training_dataset(
        self,
        dataset_id: str,
        output_format: str = "jsonl",
    ) -> Path:
        """匯出訓練資料集
        
        Args:
            dataset_id: 資料集 ID
            output_format: 匯出格式（jsonl, csv）
            
        Returns:
            匯出檔案路徑
        """
        output_file = TRAINING_DATASET_DIR / f"{dataset_id}.{output_format}"
        self.experience_repo.export_dataset(
            dataset_id=dataset_id,
            output_path=output_file,
            format=output_format,
        )
        return output_file
    
    # ========================================================================
    # 資料維護相關
    # ========================================================================
    
    def cleanup_old_data(self, days: int = 30) -> dict[str, int]:
        """清理舊資料
        
        Args:
            days: 保留天數
            
        Returns:
            清理統計資訊
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        stats = {
            "experiences_deleted": 0,
            "backup_files_deleted": 0,
        }
        
        # 清理舊經驗記錄
        session = self.experience_repo.get_session()
        try:
            deleted_count = (
                session.query(ExperienceRecord)
                .filter(ExperienceRecord.created_at < cutoff_date)
                .delete()
            )
            session.commit()
            stats["experiences_deleted"] = deleted_count
            logger.info(f"Deleted {deleted_count} old experience records")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to cleanup experiences: {e}")
        finally:
            session.close()
        
        # 清理舊備份檔案
        if BACKUP_DIR.exists():
            for backup_file in BACKUP_DIR.rglob("*"):
                if backup_file.is_file():
                    file_age = datetime.now() - datetime.fromtimestamp(
                        backup_file.stat().st_mtime
                    )
                    if file_age.days > days:
                        backup_file.unlink()
                        stats["backup_files_deleted"] += 1
        
        logger.info(f"Cleanup completed: {stats}")
        return stats
    
    def create_backup(self) -> dict[str, Path]:
        """創建資料備份
        
        Returns:
            備份檔案路徑字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_paths = {}
        
        # 備份攻擊路徑圖
        attack_graph_backup = BACKUP_DIR / "attack_paths" / f"attack_graph_{timestamp}.pkl"
        attack_graph_backup.parent.mkdir(parents=True, exist_ok=True)
        self.attack_path_engine.save_graph(str(attack_graph_backup))
        backup_paths["attack_graph"] = attack_graph_backup
        
        # 備份經驗資料庫（可以實現為複製 SQLite 檔案）
        # 這裡省略具體實現
        
        logger.info(f"Backup created: {backup_paths}")
        return backup_paths
    
    def get_unified_statistics(self) -> dict[str, Any]:
        """獲取統一的資料統計資訊
        
        供監控和報告使用
        
        Returns:
            統計資訊字典
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "experiences": self.get_experience_statistics(),
            "attack_graph": self.get_attack_graph_statistics(),
            "backup_retention": BACKUP_RETENTION_DAYS,
        }


# 單例實例（可選）
_unified_manager_instance: UnifiedDataManager | None = None


def get_unified_data_manager() -> UnifiedDataManager:
    """獲取統一資料管理器單例實例
    
    使用範例：
        manager = get_unified_data_manager()
        manager.save_attack_experience(...)
    """
    global _unified_manager_instance
    if _unified_manager_instance is None:
        _unified_manager_instance = UnifiedDataManager()
    return _unified_manager_instance
