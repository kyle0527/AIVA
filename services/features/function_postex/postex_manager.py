# 此文件已廢棄
# 請直接使用 services.features.function_postex.detector.postex_detector.PostExDetector
# 不需要額外的 Manager 包裝層

"""
已廢棄：請使用 PostExDetector

根據 SIMPLE_ARCHITECTURE.md 核心理念，每個模組按自己的最佳實踐組織。
function_postex 使用 PostExDetector，不需要統一的 Manager 模式。

正確用法：
    from services.features.function_postex.detector.postex_detector import PostExDetector
    
    detector = PostExDetector()
    findings = detector.analyze(
        test_type="privilege_escalation",
        target="localhost",
        task_id=task_id,
        scan_id=scan_id,
        safe_mode=True
    )
"""

from typing import Any, Dict, Optional

from services.aiva_common.utils import get_logger
from services.features.function_postex.detector.postex_detector import PostExDetector

logger = get_logger(__name__)


class PostExManager:
    """後滲透測試管理器 - 同步接口"""
    
    def __init__(self):
        """初始化後滲透測試管理器"""
        self._detector = PostExDetector()
        logger.info("PostExManager 初始化完成")
    
    def scan(
        self, 
        test_type: str, 
        target: Optional[str] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> dict:
        """
        執行後滲透測試（同步接口）
        
        Args:
            test_type: 測試類型
                - "privilege_escalation": 權限提升測試
                - "lateral_movement": 橫向移動測試
                - "persistence": 持久化檢測
            target: 目標系統（可選，預設為 localhost）
            options: 測試選項
                - safe_mode: bool - 安全模式（預設: True，僅檢測不執行）
                - auth_token: str - 授權令牌（必須提供以執行真實測試）
                - deep_scan: bool - 深度掃描（預設: False）
                - task_id: str - 任務 ID（用於追蹤）
                - scan_id: str - 掃描 ID（用於追蹤）
        
        Returns:
            dict: 測試結果
                {
                    "success": bool,
                    "test_type": str,
                    "target": str,
                    "safe_mode": bool,
                    "findings": list[FindingPayload],
                    "summary": dict
                }
        
        Example:
            >>> manager = PostExManager()
            >>> # 安全檢測模式（不執行實際操作）
            >>> result = manager.scan("privilege_escalation", safe_mode=True)
            >>> print(f"發現 {len(result['findings'])} 個潛在權限提升向量")
        """
        options = options or {}
        target = target or "localhost"
        safe_mode = options.get("safe_mode", False)  # 預設執行真實檢測
        auth_token = options.get("auth_token")
        task_id = options.get("task_id", "postex_task")
        scan_id = options.get("scan_id", "postex_scan")
        
        # 安全檢查
        if not safe_mode and not auth_token:
            logger.warning("嘗試在非安全模式下執行，但未提供授權令牌，強制啟用安全模式")
            safe_mode = True
        
        try:
            logger.info(
                f"開始後滲透測試",
                extra={
                    "test_type": test_type,
                    "target": target,
                    "safe_mode": safe_mode
                }
            )
            
            # 執行檢測
            findings = self._detector.analyze(
                test_type=test_type,
                target=target,
                task_id=task_id,
                scan_id=scan_id,
                safe_mode=safe_mode,
                auth_token=auth_token
            )
            
            # 生成摘要
            summary = self._generate_summary(findings, test_type)
            
            return {
                "success": True,
                "test_type": test_type,
                "target": target,
                "safe_mode": safe_mode,
                "findings": [f.model_dump() for f in findings],
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"後滲透測試失敗: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "test_type": test_type,
                "target": target
            }
    
    def scan_privilege_escalation(
        self, 
        target: Optional[str] = None, 
        safe_mode: bool = False
    ) -> dict:
        """
        權限提升測試（快捷方法）
        
        Args:
            target: 目標系統
            safe_mode: 安全模式（預設: False）
        
        Returns:
            dict: 測試結果
        """
        return self.scan("privilege_escalation", target, {"safe_mode": safe_mode})
    
    def scan_lateral_movement(
        self, 
        target: Optional[str] = None, 
        safe_mode: bool = False
    ) -> dict:
        """
        橫向移動測試（快捷方法）
        
        Args:
            target: 目標網段
            safe_mode: 安全模式（預設: False）
        
        Returns:
            dict: 測試結果
        """
        return self.scan("lateral_movement", target, {"safe_mode": safe_mode})
    
    def scan_persistence(
        self, 
        target: Optional[str] = None, 
        safe_mode: bool = False,
        deep_scan: bool = False
    ) -> dict:
        """
        持久化測試（快捷方法）
        
        Args:
            target: 目標系統
            safe_mode: 安全模式（預設: False）
            deep_scan: 深度掃描（預設: False）
        
        Returns:
            dict: 測試結果
        """
        return self.scan(
            "persistence", 
            target, 
            {"safe_mode": safe_mode, "deep_scan": deep_scan}
        )
    
    def _generate_summary(self, findings: list, test_type: str) -> dict:
        """生成測試摘要"""
        if not findings:
            return {
                "total_findings": 0,
                "test_type": test_type,
                "message": "未發現明顯的安全問題"
            }
        
        # 按嚴重性分類
        severity_counts = {}
        for finding in findings:
            severity = finding.vulnerability.severity.name
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_findings": len(findings),
            "test_type": test_type,
            "severity_breakdown": severity_counts,
            "high_priority": severity_counts.get("CRITICAL", 0) + severity_counts.get("HIGH", 0)
        }


# 為了與其他功能模組保持一致性，提供快捷函數
def scan_target(
    test_type: str, 
    target: Optional[str] = None, 
    options: Optional[Dict[str, Any]] = None
) -> dict:
    """
    快捷函數：執行後滲透測試
    
    這是一個便捷函數，用於快速執行後滲透測試，無需手動創建 Manager 實例。
    
    Args:
        test_type: 測試類型
        target: 目標系統
        options: 測試選項
    
    Returns:
        dict: 測試結果
    
    Example:
        >>> from services.features.function_postex.postex_manager import scan_target
        >>> result = scan_target("privilege_escalation", safe_mode=True)
    """
    manager = PostExManager()
    return manager.scan(test_type, target, options)


if __name__ == "__main__":
    # 測試代碼
    import json
    
    print("=== Post-Exploitation Manager 測試 ===\n")
    print("⚠️  所有測試均在安全模式下執行（僅檢測，不執行實際操作）\n")
    
    manager = PostExManager()
    
    # 測試 1: 權限提升檢測
    print("測試 1: 權限提升向量檢測")
    result = manager.scan_privilege_escalation(safe_mode=True)
    print(f"狀態: {'成功' if result['success'] else '失敗'}")
    print(f"發現: {result.get('summary', {}).get('total_findings', 0)} 個潛在向量")
    print(f"安全模式: {result.get('safe_mode')}")
    
    print("\n" + "="*50 + "\n")
    
    # 測試 2: 橫向移動檢測
    print("測試 2: 橫向移動向量檢測")
    result = manager.scan_lateral_movement(target="192.168.1.0/24", safe_mode=True)
    print(f"狀態: {'成功' if result['success'] else '失敗'}")
    print(f"發現: {result.get('summary', {}).get('total_findings', 0)} 個潛在向量")
    
    print("\n" + "="*50 + "\n")
    
    # 測試 3: 持久化檢測
    print("測試 3: 持久化機制檢測")
    result = manager.scan_persistence(safe_mode=True, deep_scan=False)
    print(f"狀態: {'成功' if result['success'] else '失敗'}")
    print(f"發現: {result.get('summary', {}).get('total_findings', 0)} 個持久化機制")
    
    print("\n" + "="*50)
    print("\n✅ 所有測試完成（安全模式）")
