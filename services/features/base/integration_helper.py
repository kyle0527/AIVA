"""
整合模組助手 - 提供統一的整合功能，無需修改現有檢測邏輯

這個助手類別提供了一個簡單的方式來將 Features 模組的結果自動儲存到整合模組，
而不需要修改各個功能模組的核心檢測邏輯。

設計原則：
1. **非侵入性**: 不修改現有的檢測邏輯
2. **可選使用**: Worker 可以選擇是否啟用整合
3. **簡單易用**: 只需要在發布結果時調用一個方法
4. **向後兼容**: 現有代碼繼續正常工作

使用範例：
    # 在 Worker 初始化時創建助手
    integration = IntegrationHelper()
    
    # 在發布 Finding 時自動儲存到資料庫
    for finding in findings:
        await publisher.publish_finding(finding, trace_id=trace_id)
        await integration.save_finding(finding)  # 新增這一行即可
"""

from typing import Any, Dict, List, Optional
from services.aiva_common.schemas.findings import FindingPayload
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class IntegrationHelper:
    """
    整合模組助手
    
    提供簡單的方法來自動儲存 Finding、記錄經驗、更新攻擊路徑
    """
    
    def __init__(self, enable_integration: bool = True):
        """
        初始化整合助手
        
        Args:
            enable_integration: 是否啟用整合功能（預設為 True）
        """
        self.enable_integration = enable_integration
        self._data_manager = None
        
        if enable_integration:
            self._initialize_data_manager()
    
    def _initialize_data_manager(self):
        """延遲初始化 UnifiedDataManager"""
        try:
            from services.integration.aiva_integration.unified_data_manager import UnifiedDataManager
            self._data_manager = UnifiedDataManager()
            logger.info("IntegrationHelper initialized with UnifiedDataManager")
        except ImportError as e:
            logger.warning(
                "Could not import UnifiedDataManager, integration disabled",
                extra={"error": str(e)}
            )
            self.enable_integration = False
    
    async def save_finding(self, finding: FindingPayload) -> bool:
        """
        儲存 Finding 到資料庫
        
        這個方法會自動將 Finding 儲存到整合模組的資料庫中。
        如果整合功能被禁用或發生錯誤，會記錄日誌但不影響原始流程。
        
        Args:
            finding: 要儲存的 Finding
            
        Returns:
            bool: 是否成功儲存
            
        範例：
            >>> integration = IntegrationHelper()
            >>> await integration.save_finding(finding)
            True
        """
        if not self.enable_integration or not self._data_manager:
            return False
        
        try:
            await self._data_manager.save_finding(finding)
            logger.debug(
                "Finding saved to database",
                extra={"finding_id": finding.finding_id}
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to save finding to database",
                extra={
                    "finding_id": finding.finding_id,
                    "error": str(e)
                },
                exc_info=True
            )
            return False
    
    async def save_findings_batch(self, findings: List[FindingPayload]) -> int:
        """
        批量儲存多個 Finding
        
        Args:
            findings: Finding 列表
            
        Returns:
            int: 成功儲存的數量
            
        範例：
            >>> integration = IntegrationHelper()
            >>> saved_count = await integration.save_findings_batch(findings)
            >>> print(f"Saved {saved_count}/{len(findings)} findings")
        """
        if not self.enable_integration or not self._data_manager:
            return 0
        
        saved_count = 0
        for finding in findings:
            if await self.save_finding(finding):
                saved_count += 1
        
        logger.info(
            "Batch saved findings",
            extra={
                "total": len(findings),
                "saved": saved_count,
                "failed": len(findings) - saved_count
            }
        )
        return saved_count
    
    async def record_experience(
        self,
        task_id: str,
        action: str,
        outcome: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        記錄檢測經驗
        
        這個方法會記錄檢測過程的經驗，用於 AI 學習和優化。
        
        Args:
            task_id: 任務 ID
            action: 執行的動作（如 "sql_injection_detection"）
            outcome: 執行結果（如 {"findings_count": 3, "success": True}）
            metadata: 額外的元數據（如使用的引擎、參數等）
            
        Returns:
            bool: 是否成功記錄
            
        範例：
            >>> integration = IntegrationHelper()
            >>> await integration.record_experience(
            ...     task_id="task_123",
            ...     action="sqli_boolean_detection",
            ...     outcome={"findings_count": 2, "payloads_tested": 50},
            ...     metadata={"engine": "boolean", "timeout": 20}
            ... )
            True
        """
        if not self.enable_integration or not self._data_manager:
            return False
        
        try:
            await self._data_manager.save_experience(
                task_id=task_id,
                action=action,
                outcome=outcome,
                metadata=metadata or {}
            )
            logger.debug(
                "Experience recorded",
                extra={
                    "task_id": task_id,
                    "action": action
                }
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to record experience",
                extra={
                    "task_id": task_id,
                    "action": action,
                    "error": str(e)
                },
                exc_info=True
            )
            return False
    
    async def update_attack_path(
        self,
        finding: FindingPayload,
        target: str
    ) -> bool:
        """
        更新攻擊路徑圖
        
        當發現高危漏洞時，將其加入攻擊路徑分析圖中。
        
        Args:
            finding: 發現的漏洞
            target: 目標 URL 或資源
            
        Returns:
            bool: 是否成功更新
            
        範例：
            >>> integration = IntegrationHelper()
            >>> if finding.severity in ["critical", "high"]:
            ...     await integration.update_attack_path(finding, target_url)
        """
        if not self.enable_integration or not self._data_manager:
            return False
        
        # 只為高危漏洞更新攻擊路徑
        if finding.severity not in ["critical", "high"]:
            return False
        
        try:
            await self._data_manager.add_vulnerability_to_graph(
                vuln_id=finding.finding_id,
                vuln_type=finding.vulnerability.vulnerability_type,
                target=target
            )
            logger.debug(
                "Attack path updated",
                extra={
                    "finding_id": finding.finding_id,
                    "severity": finding.severity
                }
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to update attack path",
                extra={
                    "finding_id": finding.finding_id,
                    "error": str(e)
                },
                exc_info=True
            )
            return False


# 提供一個全局單例（可選）
_global_integration_helper: Optional[IntegrationHelper] = None


def get_integration_helper(
    enable_integration: bool = True,
    use_global: bool = True
) -> IntegrationHelper:
    """
    獲取整合助手實例
    
    Args:
        enable_integration: 是否啟用整合功能
        use_global: 是否使用全局單例
        
    Returns:
        IntegrationHelper 實例
        
    範例：
        >>> # 使用全局單例
        >>> integration = get_integration_helper()
        >>> 
        >>> # 或創建新實例
        >>> integration = get_integration_helper(use_global=False)
    """
    global _global_integration_helper
    
    if use_global:
        if _global_integration_helper is None:
            _global_integration_helper = IntegrationHelper(enable_integration)
        return _global_integration_helper
    else:
        return IntegrationHelper(enable_integration)


# 便利函數：直接調用整合功能
async def save_finding(finding: FindingPayload) -> bool:
    """
    便利函數：直接儲存 Finding
    
    範例：
        >>> from services.features.base.integration_helper import save_finding
        >>> await save_finding(finding)
    """
    helper = get_integration_helper()
    return await helper.save_finding(finding)


async def record_experience(
    task_id: str,
    action: str,
    outcome: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    便利函數：直接記錄經驗
    
    範例：
        >>> from services.features.base.integration_helper import record_experience
        >>> await record_experience(
        ...     task_id="task_123",
        ...     action="xss_detection",
        ...     outcome={"findings_count": 1}
        ... )
    """
    helper = get_integration_helper()
    return await helper.record_experience(task_id, action, outcome, metadata)
