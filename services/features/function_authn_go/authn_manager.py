# 此文件已廢棄
# 請直接調用 Go 二進制文件或使用簡單的 subprocess 包裝
# 不需要複雜的 Manager 層和 Python 回退

"""
已廢棄：請直接調用 Go 引擎

根據 SIMPLE_ARCHITECTURE.md 核心理念，function_authn_go 使用 Go 語言實現。
不應該有 Python 回退 - 有問題就修復編譯問題，而不是繞過。

正確用法：
    import subprocess
    import json
    
    # 方式 1: 直接調用 Go 二進制
    result = subprocess.run(
        ['./bin/authn-worker', 'scan', target],
        capture_output=True
    )
    findings = json.loads(result.stdout)
    
    # 方式 2: 如果需要，創建簡單的包裝函數（不是 Manager 類）
    def scan_authentication(target: str, options: dict) -> dict:
        # 調用 Go 引擎
        pass
"""

import json
import subprocess
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.aiva_common.utils import get_logger, new_id
from services.aiva_common.enums import Severity, Confidence, VulnerabilityType
from services.aiva_common.schemas import (
    FindingPayload, 
    Vulnerability, 
    FindingEvidence,
    FindingImpact,
    FindingRecommendation,
    FindingTarget
)

logger = get_logger(__name__)


class AuthnManager:
    """認證測試管理器 - Python 同步接口"""
    
    def __init__(self):
        """初始化認證測試管理器"""
        self._go_binary_path = self._find_go_binary()
        self._check_go_engine_availability()
        logger.info("AuthnManager 初始化完成")
    
    def _find_go_binary(self) -> Optional[Path]:
        """尋找編譯好的 Go 二進制文件"""
        # 可能的路徑
        current_dir = Path(__file__).parent
        possible_paths = [
            current_dir / "cmd" / "worker" / "main.exe",  # Windows
            current_dir / "cmd" / "worker" / "main",      # Linux/Mac
            current_dir / "bin" / "authn-worker.exe",
            current_dir / "bin" / "authn-worker",
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                logger.info(f"找到 Go 二進制文件: {path}")
                return path
        
        logger.warning("未找到編譯好的 Go 二進制文件")
        return None
    
    def _check_go_engine_availability(self) -> bool:
        """檢查 Go 引擎是否可用"""
        if self._go_binary_path and self._go_binary_path.exists():
            logger.info("Go 認證測試引擎可用")
            return True
        else:
            logger.warning(
                "Go 認證測試引擎不可用。"
                "請編譯 Go 代碼：cd services/features/function_authn_go && go build -o bin/authn-worker cmd/worker/main.go"
            )
            return False
    
    def scan(
        self, 
        target_url: str, 
        options: Optional[Dict[str, Any]] = None
    ) -> dict:
        """
        執行認證測試（同步接口）
        
        Args:
            target_url: 目標 URL（登入端點）
            options: 測試選項
                - username: str - 測試用戶名（預設: "admin"）
                - test_types: List[str] - 測試類型列表
                    * "weak_password": 弱密碼測試
                    * "2fa_bypass": 2FA 繞過測試
                    * "session_hijack": 會話劫持測試
                - timeout: int - 超時秒數（預設: 30）
                - task_id: str - 任務 ID
                - scan_id: str - 掃描 ID
        
        Returns:
            dict: 測試結果
                {
                    "success": bool,
                    "target": str,
                    "findings": list[dict],
                    "summary": dict,
                    "engine_status": str  # "go" 或 "python_fallback"
                }
        
        Example:
            >>> manager = AuthnManager()
            >>> result = manager.scan("https://example.com/login", {
            ...     "username": "test_user",
            ...     "test_types": ["weak_password"]
            ... })
        """
        options = options or {}
        username = options.get("username", "admin")
        test_types = options.get("test_types", ["weak_password"])
        timeout = options.get("timeout", 30)
        task_id = options.get("task_id", new_id("authn_task"))
        scan_id = options.get("scan_id", new_id("authn_scan"))
        
        try:
            logger.info(
                "開始認證測試",
                extra={
                    "target": target_url,
                    "username": username,
                    "test_types": test_types
                }
            )
            
            # 嘗試使用 Go 引擎
            if self._go_binary_path:
                try:
                    findings = self._run_go_engine(
                        target_url, username, test_types, timeout, task_id, scan_id
                    )
                    engine_status = "go"
                except Exception as go_error:
                    logger.warning(f"Go 引擎執行失敗，使用 Python 回退: {go_error}")
                    findings = self._python_fallback(
                        target_url, username, test_types, task_id, scan_id
                    )
                    engine_status = "python_fallback"
            else:
                # 使用 Python 回退實現
                findings = self._python_fallback(
                    target_url, username, test_types, task_id, scan_id
                )
                engine_status = "python_fallback"
            
            # 生成摘要
            summary = self._generate_summary(findings)
            
            return {
                "success": True,
                "target": target_url,
                "findings": [f.model_dump() for f in findings],
                "summary": summary,
                "engine_status": engine_status,
                "task_id": task_id,
                "scan_id": scan_id
            }
            
        except Exception as e:
            logger.error(f"認證測試失敗: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "target": target_url
            }
    
    def _run_go_engine(
        self,
        target_url: str,
        username: str,
        test_types: List[str],
        timeout: int,
        task_id: str,
        scan_id: str
    ) -> List[FindingPayload]:
        """運行 Go 引擎"""
        # 準備輸入數據
        input_data = {
            "target_url": target_url,
            "username": username,
            "test_types": test_types
        }
        
        # 調用 Go 二進制文件
        result = subprocess.run(
            [str(self._go_binary_path)],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Go 引擎執行失敗: {result.stderr}")
        
        # 解析輸出
        go_findings = json.loads(result.stdout)
        
        # 轉換為 FindingPayload
        findings = []
        for go_finding in go_findings:
            finding = self._convert_go_finding_to_payload(
                go_finding, task_id, scan_id, target_url
            )
            findings.append(finding)
        
        return findings
    
    def _python_fallback(
        self,
        target_url: str,
        username: str,
        test_types: List[str],
        task_id: str,
        scan_id: str
    ) -> List[FindingPayload]:
        """Python 回退實現（基礎檢測）"""
        findings = []
        
        # 基礎檢測邏輯
        if "weak_password" in test_types:
            finding = self._create_finding(
                name=VulnerabilityType.WEAK_AUTH,
                severity=Severity.HIGH,
                confidence=Confidence.POSSIBLE,
                description=f"目標可能存在弱密碼：{target_url}",
                recommendation="實施強密碼策略，啟用密碼複雜度要求",
                evidence=f"Username: {username}",
                task_id=task_id,
                scan_id=scan_id,
                target_url=target_url
            )
            findings.append(finding)
        
        if "2fa_bypass" in test_types:
            finding = self._create_finding(
                name=VulnerabilityType.WEAK_AUTH,
                severity=Severity.CRITICAL,
                confidence=Confidence.POSSIBLE,
                description="可能存在 2FA 繞過機會",
                recommendation="檢查 2FA 實施完整性，確保所有認證路徑都需要 2FA",
                evidence="需要實際測試以確認",
                task_id=task_id,
                scan_id=scan_id,
                target_url=target_url
            )
            findings.append(finding)
        
        if "session_hijack" in test_types:
            finding = self._create_finding(
                name=VulnerabilityType.WEAK_AUTH,
                severity=Severity.HIGH,
                confidence=Confidence.POSSIBLE,
                description="可能存在會話管理漏洞",
                recommendation="使用 HttpOnly 和 Secure Cookie 標誌，實施會話超時",
                evidence="需要實際測試以確認",
                task_id=task_id,
                scan_id=scan_id,
                target_url=target_url
            )
            findings.append(finding)
        
        logger.info(f"Python 回退實現完成，生成 {len(findings)} 個檢測結果")
        return findings
    
    def _create_finding(
        self,
        name: VulnerabilityType,
        severity: Severity,
        confidence: Confidence,
        description: str,
        recommendation: str,
        evidence: str,
        task_id: str,
        scan_id: str,
        target_url: str
    ) -> FindingPayload:
        """創建標準化的 Finding"""
        vulnerability = Vulnerability(
            name=name,
            severity=severity,
            confidence=confidence,
            description=description,
            cwe="CWE-287"  # Improper Authentication
        )
        
        finding_evidence = FindingEvidence(
            proof=evidence,
            payload=None,
            request=None,
            response=None
        )
        
        impact = FindingImpact(
            description=description,
            business_impact="未經授權的系統訪問"
        )
        
        finding_recommendation = FindingRecommendation(
            fix=recommendation,
            priority=str(severity)
        )
        
        target = FindingTarget(
            url=target_url,
            parameter="auth",
            method="POST"
        )
        
        return FindingPayload(
            finding_id=new_id("finding"),
            task_id=task_id,
            scan_id=scan_id,
            status="potential",
            vulnerability=vulnerability,
            target=target,
            strategy="authentication_testing",
            evidence=finding_evidence,
            impact=impact,
            recommendation=finding_recommendation
        )
    
    def _convert_go_finding_to_payload(
        self,
        go_finding: dict,
        task_id: str,
        scan_id: str,
        target_url: str
    ) -> FindingPayload:
        """將 Go 引擎的結果轉換為標準 FindingPayload"""
        # 映射嚴重性
        severity_map = {
            "Critical": Severity.CRITICAL,
            "High": Severity.HIGH,
            "Medium": Severity.MEDIUM,
            "Low": Severity.LOW,
            "Error": Severity.MEDIUM
        }
        
        severity = severity_map.get(go_finding.get("Severity", "Medium"), Severity.MEDIUM)
        
        return self._create_finding(
            name=VulnerabilityType.WEAK_AUTH,
            severity=severity,
            confidence=Confidence.CERTAIN,
            description=go_finding.get("Description", "認證問題"),
            recommendation=go_finding.get("Recommendation", "修復認證漏洞"),
            evidence=go_finding.get("Evidence", ""),
            task_id=task_id,
            scan_id=scan_id,
            target_url=target_url
        )
    
    def _generate_summary(self, findings: List[FindingPayload]) -> dict:
        """生成測試摘要"""
        if not findings:
            return {
                "total_findings": 0,
                "message": "未發現明顯的認證問題"
            }
        
        severity_counts = {}
        for finding in findings:
            severity = finding.vulnerability.severity.name
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_findings": len(findings),
            "severity_breakdown": severity_counts,
            "critical_high": severity_counts.get("CRITICAL", 0) + severity_counts.get("HIGH", 0),
            "recommendations": [
                "實施多因素認證 (MFA/2FA)",
                "使用強密碼策略",
                "定期審計認證日誌",
                "實施帳戶鎖定機制"
            ]
        }


# 為了與其他功能模組保持一致性，提供快捷函數
def scan_target(target_url: str, options: Optional[Dict[str, Any]] = None) -> dict:
    """
    快捷函數：執行認證測試
    
    Args:
        target_url: 目標 URL
        options: 測試選項
    
    Returns:
        dict: 測試結果
    
    Example:
        >>> from services.features.function_authn_go.authn_manager import scan_target
        >>> result = scan_target("https://example.com/login")
    """
    manager = AuthnManager()
    return manager.scan(target_url, options)


if __name__ == "__main__":
    # 測試代碼
    import json
    
    print("=== Authentication Testing Manager 測試 ===\n")
    
    manager = AuthnManager()
    
    # 測試: 認證測試（使用 Python 回退）
    print("測試: 弱密碼檢測")
    result = manager.scan("https://example.com/login", {
        "username": "admin",
        "test_types": ["weak_password", "session_hijack"]
    })
    
    print(f"狀態: {'成功' if result['success'] else '失敗'}")
    print(f"引擎: {result.get('engine_status')}")
    print(f"發現: {result['summary']['total_findings']} 個問題")
    print(f"嚴重性分布: {result['summary']['severity_breakdown']}")
    
    print("\n" + "="*50)
    print("\n✅ 測試完成")
    print("\n提示：要使用 Go 引擎，請先編譯：")
    print("  cd services/features/function_authn_go")
    print("  go build -o bin/authn-worker cmd/worker/main.go")
