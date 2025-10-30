"""
統一存儲適配器

將 AIVA 標準 StorageManager 適配到 DataReceptionLayer 接口
遵循 aiva_common 規範，實現數據庫架構統一
"""



from datetime import datetime
from typing import Any

from services.aiva_common.schemas import FindingPayload
from services.aiva_common.utils import get_logger
from services.core.aiva_core.storage import StorageManager

from .test_result_database import TestResultDatabase

logger = get_logger(__name__)


class UnifiedStorageAdapter(TestResultDatabase):
    """
    統一存儲適配器
    
    將 StorageManager 的 PostgreSQL 後端適配到現有的 TestResultDatabase 接口
    確保與現有代碼的兼容性，同時使用統一的存儲架構
    """

    def __init__(
        self,
        data_root: str = "./data",
        db_config: dict[str, Any] | None = None,
    ):
        """
        初始化統一存儲適配器

        Args:
            data_root: 數據根目錄
            db_config: PostgreSQL 配置
        """
        self.data_root = data_root
        self.db_config = db_config or {
            "host": "localhost",
            "port": 5432,
            "database": "aiva_db",
            "user": "postgres",
            "password": "aiva123",
        }

        # 使用 aiva_common 標準的 StorageManager with PostgreSQL 後端
        self.storage_manager = StorageManager(
            data_root=data_root,
            db_type="postgres",
            db_config=self.db_config,
            auto_create_dirs=True,
        )

        logger.info(
            f"UnifiedStorageAdapter initialized with PostgreSQL backend: "
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )

    async def save_finding(self, finding: FindingPayload) -> None:
        """
        保存漏洞發現到統一存儲
        
        將 FindingPayload 轉換為 ExperienceSample 並保存到 PostgreSQL
        """
        try:
            # 使用 StorageManager 的後端直接保存
            # 由於 StorageManager 使用 PostgreSQL 後端，這會保存到 PostgreSQL
            from services.aiva_common.schemas import ExperienceSample
            import uuid
            
            # 將 FindingPayload 轉換為 ExperienceSample
            experience = ExperienceSample(
                sample_id=str(uuid.uuid4()),
                session_id=finding.scan_id,
                plan_id=finding.task_id,
                state_before={
                    "target_url": str(finding.target.url),
                    "target_method": finding.target.method,
                    "target_parameter": finding.target.parameter or "",
                    "scan_status": "started",
                },
                action_taken={
                    "action_type": "vulnerability_scan",
                    "vulnerability_type": finding.vulnerability.name,
                    "scan_parameters": {
                        "url": str(finding.target.url),
                        "method": finding.target.method,
                        "parameter": finding.target.parameter or "",
                    },
                },
                state_after={
                    "vulnerability_found": finding.status == "confirmed",
                    "vulnerability_type": finding.vulnerability.name,
                    "severity": finding.vulnerability.severity,
                    "confidence": finding.vulnerability.confidence,
                    "scan_status": "completed",
                },
                reward=self._calculate_reward(finding),
                reward_breakdown={
                    "completion": 1.0,  # 掃描完成
                    "success": 1.0 if finding.status == "confirmed" else 0.0,
                    "severity": self._get_severity_score(finding.vulnerability.severity),
                    "confidence": self._get_confidence_score(finding.vulnerability.confidence),
                },
                context={
                    "scan_id": finding.scan_id,
                    "task_id": finding.task_id,
                    "finding_id": finding.finding_id,
                },
                target_info={
                    "url": str(finding.target.url),
                    "method": finding.target.method,
                    "parameter": finding.target.parameter or "",
                    "vulnerability": finding.vulnerability.model_dump(),
                    "evidence": finding.evidence.model_dump() if finding.evidence else {},
                },
                timestamp=datetime.utcnow(),
                quality_score=self._calculate_quality_score(finding),
                is_positive=finding.status == "confirmed",
                confidence=self._get_confidence_score(finding.vulnerability.confidence),
                learning_tags=[
                    finding.vulnerability.severity,
                    finding.vulnerability.confidence,
                    finding.status,
                    "vulnerability_scan",
                    finding.vulnerability.name.lower(),
                ],
                difficulty_level=self._get_difficulty_level(finding),
            )

            # 保存到統一存儲
            success = await self.storage_manager.save_unified_experience_sample(experience)
            
            if success:
                logger.info(f"Successfully saved finding {finding.finding_id} to unified storage")
            else:
                logger.error(f"Failed to save finding {finding.finding_id} to unified storage")
                
        except Exception as e:
            logger.error(f"Error saving finding {finding.finding_id}: {str(e)}")
            raise

    async def get_finding(self, finding_id: str) -> FindingPayload | None:
        """
        根據 ID 獲取漏洞發現
        
        從統一存儲中搜索並還原 FindingPayload
        """
        try:
            # 從經驗樣本中搜索
            samples = await self.storage_manager.get_experience_samples(
                limit=1000,  # 搜索較多樣本以找到匹配項
                min_quality=0.0,
            )
            
            for sample in samples:
                # 檢查是否匹配 finding_id
                if (
                    sample.result
                    and sample.result.data
                    and sample.result.data.get("finding_id") == finding_id
                ):
                    # 從結果數據重建 FindingPayload
                    raw_data = sample.result.data.get("raw_data")
                    if raw_data:
                        return FindingPayload.model_validate(raw_data)
            
            logger.warning(f"Finding {finding_id} not found in unified storage")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving finding {finding_id}: {str(e)}")
            return None

    async def list_findings(
        self,
        scan_id: str | None = None,
        severity: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FindingPayload]:
        """
        列出漏洞發現
        
        從統一存儲中篩選和返回漏洞發現
        """
        try:
            # 從經驗樣本中搜索
            # TODO: 實現更精確的篩選邏輯，現在先返回所有樣本並過濾
            samples = await self.storage_manager.get_experience_samples(
                limit=limit + offset,  # 獲取更多樣本以便過濾
                min_quality=0.0,
            )
            
            findings = []
            processed = 0
            
            for sample in samples:
                # 跳過 offset 數量的記錄
                if processed < offset:
                    processed += 1
                    continue
                    
                # 檢查是否是漏洞發現記錄
                if (
                    sample.result
                    and sample.result.data
                    and sample.result.data.get("finding_id")
                    and sample.metadata
                    and sample.metadata.get("original_type") == "finding"
                ):
                    # 應用過濾條件
                    raw_data = sample.result.data.get("raw_data")
                    if raw_data:
                        finding = FindingPayload.model_validate(raw_data)
                        
                        # 篩選條件
                        if scan_id and finding.scan_id != scan_id:
                            continue
                        if severity and finding.vulnerability.severity != severity:
                            continue
                            
                        findings.append(finding)
                        
                        # 達到限制數量
                        if len(findings) >= limit:
                            break
            
            logger.info(f"Retrieved {len(findings)} findings from unified storage")
            return findings
            
        except Exception as e:
            logger.error(f"Error listing findings: {str(e)}")
            return []

    async def count_findings(
        self,
        scan_id: str | None = None,
        severity: str | None = None,
    ) -> int:
        """
        統計漏洞發現數量
        """
        try:
            # 獲取所有相關樣本並計數
            findings = await self.list_findings(
                scan_id=scan_id,
                severity=severity,
                limit=10000,  # 設置較大限制以獲取準確計數
                offset=0,
            )
            return len(findings)
            
        except Exception as e:
            logger.error(f"Error counting findings: {str(e)}")
            return 0

    async def get_scan_summary(self, scan_id: str) -> dict[str, Any]:
        """
        獲取掃描摘要統計
        """
        try:
            # 獲取該掃描的所有發現
            findings = await self.list_findings(scan_id=scan_id, limit=10000)
            
            # 統計數據
            total = len(findings)
            by_severity: dict[str, int] = {}
            by_vulnerability_type: dict[str, int] = {}
            
            for finding in findings:
                # 按嚴重程度統計
                severity = finding.vulnerability.severity
                by_severity[severity] = by_severity.get(severity, 0) + 1
                
                # 按漏洞類型統計
                vuln_type = finding.vulnerability.name
                by_vulnerability_type[vuln_type] = by_vulnerability_type.get(vuln_type, 0) + 1
            
            summary = {
                "scan_id": scan_id,
                "total_findings": total,
                "by_severity": by_severity,
                "by_vulnerability_type": by_vulnerability_type,
            }
            
            logger.info(f"Generated scan summary for {scan_id}: {total} findings")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating scan summary for {scan_id}: {str(e)}")
            return {
                "scan_id": scan_id,
                "total_findings": 0,
                "by_severity": {},
                "by_vulnerability_type": {},
            }

    def _calculate_quality_score(self, finding: FindingPayload) -> float:
        """
        計算漏洞發現的質量分數
        
        基於嚴重程度、置信度和狀態計算分數
        """
        score = 0.5  # 基礎分數
        
        # 嚴重程度加分
        severity_scores = {
            "critical": 0.4,
            "high": 0.3,
            "medium": 0.2,
            "low": 0.1,
            "info": 0.05,
        }
        score += severity_scores.get(finding.vulnerability.severity.lower(), 0.1)
        
        # 置信度加分
        confidence_scores = {
            "certain": 0.1,
            "firm": 0.08,
            "tentative": 0.05,
        }
        score += confidence_scores.get(finding.vulnerability.confidence.lower(), 0.03)
        
        # 狀態調整
        if finding.status == "confirmed":
            score += 0.1
        elif finding.status == "false_positive":
            score -= 0.2
        
        # 確保分數在 0-1 範圍內
        return max(0.0, min(1.0, score))

    def _calculate_reward(self, finding: FindingPayload) -> float:
        """
        計算獎勵值
        
        基於漏洞發現的成功程度和重要性計算獎勵
        """
        base_reward = 0.5  # 基礎獎勵
        
        # 成功發現漏洞的獎勵
        if finding.status == "confirmed":
            success_reward = 1.0
        elif finding.status == "potential":
            success_reward = 0.6
        else:
            success_reward = 0.1  # 即使是誤報也有基礎學習價值
        
        # 嚴重程度獎勵
        severity_reward = self._get_severity_score(finding.vulnerability.severity)
        
        # 置信度獎勵
        confidence_reward = self._get_confidence_score(finding.vulnerability.confidence)
        
        # 計算總獎勵 (加權平均)
        total_reward = (
            success_reward * 0.4 +
            severity_reward * 0.3 +
            confidence_reward * 0.2 +
            base_reward * 0.1
        )
        
        return max(0.0, min(1.0, total_reward))

    def _get_severity_score(self, severity: str) -> float:
        """獲取嚴重程度對應的分數"""
        severity_scores = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4,
            "info": 0.2,
        }
        return severity_scores.get(severity.lower(), 0.3)

    def _get_confidence_score(self, confidence: str) -> float:
        """獲取置信度對應的分數"""
        confidence_scores = {
            "certain": 1.0,
            "firm": 0.8,
            "tentative": 0.6,
        }
        return confidence_scores.get(confidence.lower(), 0.5)

    def _get_difficulty_level(self, finding: FindingPayload) -> int:
        """
        獲取難度級別 (1-5)
        
        基於漏洞類型和嚴重程度評估難度
        """
        severity = finding.vulnerability.severity.lower()
        vulnerability_type = finding.vulnerability.name.lower()
        
        # 基礎難度
        base_difficulty = {
            "critical": 5,
            "high": 4,
            "medium": 3,
            "low": 2,
            "info": 1,
        }.get(severity, 3)
        
        # 某些漏洞類型調整難度
        if any(keyword in vulnerability_type for keyword in ["sql injection", "code injection", "command injection"]):
            base_difficulty = min(5, base_difficulty + 1)
        elif any(keyword in vulnerability_type for keyword in ["xss", "csrf"]):
            base_difficulty = max(1, base_difficulty - 1)
        
        return base_difficulty