"""
BizLogic Attack Executor - 業務邏輯漏洞攻擊執行器

執行業務邏輯類型的攻擊，包括：
- 價格操縱 (Price Manipulation)
- 越權訪問 (IDOR - Insecure Direct Object Reference)
- 工作流繞過 (Workflow Bypass)
- 競態條件 (Race Condition)
- 優惠券濫用 (Coupon Abuse)

架構設計：
- 本模組屬於 core_capabilities (攻擊能力層)
- 接收來自 AICommander 的決策指令
- 調用 function_bizlogic 服務執行具體測試
- 將結果回報給 TaskPlanning 模組
"""

import asyncio
import logging
from typing import Any
from datetime import datetime

from services.aiva_common.schemas import Finding, SeverityLevel, VulnerabilityType
from services.aiva_common.error_handling import AIVAError, ErrorType, ErrorSeverity, create_error_context

logger = logging.getLogger(__name__)


class BizLogicAttackType:
    """業務邏輯攻擊類型"""
    
    PRICE_MANIPULATION = "price_manipulation"  # 價格操縱
    IDOR = "idor"  # 越權訪問
    WORKFLOW_BYPASS = "workflow_bypass"  # 工作流繞過
    RACE_CONDITION = "race_condition"  # 競態條件
    COUPON_ABUSE = "coupon_abuse"  # 優惠券濫用
    QUANTITY_LIMIT_BYPASS = "quantity_limit_bypass"  # 數量限制繞過
    DISCOUNT_STACKING = "discount_stacking"  # 折扣堆疊
    CAPTCHA_BYPASS = "captcha_bypass"  # 驗證碼繞過


class BizLogicAttackExecutor:
    """業務邏輯攻擊執行器
    
    核心能力模組的業務邏輯攻擊實現，負責：
    1. 接收 AI 決策指令
    2. 執行業務邏輯測試
    3. 驗證漏洞存在性
    4. 生成標準化結果
    
    注意：
    - 本類不做決策，只執行 AI Commander 下達的命令
    - 具體測試邏輯可委派給 function_bizlogic 服務
    - 返回標準化的 Finding 對象
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """初始化業務邏輯攻擊執行器
        
        Args:
            config: 配置字典，包含超時、重試等參數
        """
        self.config = config or {}
        self.timeout = self.config.get("timeout", 30)
        self.max_retries = self.config.get("max_retries", 2)
        
        logger.info("BizLogicAttackExecutor initialized")
    
    async def execute_attack(
        self,
        attack_type: str,
        target_url: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """執行業務邏輯攻擊
        
        這是主要的執行入口，接收來自 AICommander 的指令。
        
        Args:
            attack_type: 攻擊類型（如 "price_manipulation"）
            target_url: 目標 URL
            parameters: 攻擊參數（由 AI 決策生成）
        
        Returns:
            執行結果字典，包含發現的漏洞和執行狀態
        
        Example:
            >>> executor = BizLogicAttackExecutor()
            >>> result = await executor.execute_attack(
            ...     attack_type="price_manipulation",
            ...     target_url="http://localhost:3000",
            ...     parameters={"product_id": 1, "payload": -1000}
            ... )
        """
        logger.info(f"Executing BizLogic attack: {attack_type} on {target_url}")
        
        start_time = datetime.now()
        
        try:
            # 根據攻擊類型分發到對應的處理器
            if attack_type == BizLogicAttackType.PRICE_MANIPULATION:
                findings = await self._test_price_manipulation(target_url, parameters)
            elif attack_type == BizLogicAttackType.IDOR:
                findings = await self._test_idor(target_url, parameters)
            elif attack_type == BizLogicAttackType.WORKFLOW_BYPASS:
                findings = await self._test_workflow_bypass(target_url, parameters)
            elif attack_type == BizLogicAttackType.RACE_CONDITION:
                findings = await self._test_race_condition(target_url, parameters)
            elif attack_type == BizLogicAttackType.COUPON_ABUSE:
                findings = await self._test_coupon_abuse(target_url, parameters)
            else:
                raise AIVAError(
                    f"Unknown bizlogic attack type: {attack_type}",
                    error_type=ErrorType.VALIDATION,
                    severity=ErrorSeverity.MEDIUM,
                    context=create_error_context(
                        module="bizlogic_attack_executor",
                        function="execute_attack"
                    )
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "attack_type": attack_type,
                "target_url": target_url,
                "findings": [f.model_dump() for f in findings],
                "findings_count": len(findings),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"BizLogic attack execution failed: {e}", exc_info=True)
            return {
                "success": False,
                "attack_type": attack_type,
                "target_url": target_url,
                "error": str(e),
                "findings": [],
                "execution_time": (datetime.now() - start_time).total_seconds(),
            }
    
    async def _test_price_manipulation(
        self,
        target_url: str,
        parameters: dict[str, Any],
    ) -> list[Finding]:
        """測試價格操縱漏洞
        
        嘗試修改商品價格為負數、零或極小值，檢測是否能成功購買。
        """
        findings = []
        
        product_id = parameters.get("product_id", 1)
        payloads = parameters.get("payloads", [-1000, 0, 0.01, -0.01])
        
        logger.info(f"Testing price manipulation on product_id={product_id}")
        
        for payload in payloads:
            try:
                # 模擬價格操縱測試
                # 實際環境中會調用 function_bizlogic 服務或直接發送 HTTP 請求
                is_vulnerable = await self._send_price_manipulation_request(
                    target_url, product_id, payload
                )
                
                if is_vulnerable:
                    finding = Finding(
                        finding_id=f"bizlogic_price_{product_id}_{abs(payload)}",
                        vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_FLAW,
                        severity=SeverityLevel.HIGH,
                        title=f"價格操縱漏洞 - 產品 {product_id}",
                        description=f"可將產品價格修改為 {payload}，導致經濟損失",
                        affected_url=f"{target_url}/api/BasketItems",
                        evidence={
                            "product_id": product_id,
                            "manipulated_price": payload,
                            "original_price": "未知",
                            "attack_vector": "POST /api/BasketItems with negative price",
                        },
                        remediation="實施服務端價格驗證，拒絕負數和零值價格",
                        confidence="high",
                        cve_id=None,
                        cvss_score=7.5,
                    )
                    findings.append(finding)
                    logger.warning(f"Found price manipulation vulnerability: payload={payload}")
                    
            except Exception as e:
                logger.error(f"Price manipulation test failed for payload {payload}: {e}")
        
        return findings
    
    async def _test_idor(
        self,
        target_url: str,
        parameters: dict[str, Any],
    ) -> list[Finding]:
        """測試越權訪問 (IDOR) 漏洞
        
        嘗試訪問其他用戶的資源（如訂單、個人資料等）。
        """
        findings = []
        
        resource_type = parameters.get("resource_type", "user")
        target_ids = parameters.get("target_ids", [1, 2, 3, 4, 5])
        
        logger.info(f"Testing IDOR on resource_type={resource_type}")
        
        for user_id in target_ids:
            try:
                is_vulnerable = await self._send_idor_request(
                    target_url, resource_type, user_id
                )
                
                if is_vulnerable:
                    finding = Finding(
                        finding_id=f"bizlogic_idor_{resource_type}_{user_id}",
                        vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_FLAW,
                        severity=SeverityLevel.HIGH,
                        title=f"越權訪問 (IDOR) - {resource_type.upper()} {user_id}",
                        description=f"可未授權訪問用戶 {user_id} 的 {resource_type} 資源",
                        affected_url=f"{target_url}/api/{resource_type}s/{user_id}",
                        evidence={
                            "resource_type": resource_type,
                            "accessed_user_id": user_id,
                            "attack_vector": f"GET /api/{resource_type}s/{user_id} without auth",
                        },
                        remediation="實施嚴格的授權檢查，驗證當前用戶是否有權訪問該資源",
                        confidence="high",
                        cvss_score=8.0,
                    )
                    findings.append(finding)
                    logger.warning(f"Found IDOR vulnerability: user_id={user_id}")
                    
            except Exception as e:
                logger.error(f"IDOR test failed for user_id {user_id}: {e}")
        
        return findings
    
    async def _test_workflow_bypass(
        self,
        target_url: str,
        parameters: dict[str, Any],
    ) -> list[Finding]:
        """測試工作流繞過漏洞
        
        嘗試跳過必要步驟（如跳過付款直接發貨）。
        """
        findings = []
        
        workflow_type = parameters.get("workflow_type", "checkout")
        
        logger.info(f"Testing workflow bypass: {workflow_type}")
        
        # 示例：嘗試直接訪問完成頁面
        is_vulnerable = await self._send_workflow_bypass_request(
            target_url, workflow_type
        )
        
        if is_vulnerable:
            finding = Finding(
                finding_id=f"bizlogic_workflow_{workflow_type}",
                vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_FLAW,
                severity=SeverityLevel.CRITICAL,
                title=f"工作流繞過 - {workflow_type}",
                description="可跳過關鍵業務步驟（如付款），直接完成交易",
                affected_url=f"{target_url}/api/{workflow_type}/complete",
                evidence={
                    "workflow_type": workflow_type,
                    "bypassed_step": "payment",
                    "attack_vector": "Direct access to completion endpoint",
                },
                remediation="實施狀態機驗證，確保每個步驟按順序完成",
                confidence="high",
                cvss_score=9.0,
            )
            findings.append(finding)
        
        return findings
    
    async def _test_race_condition(
        self,
        target_url: str,
        parameters: dict[str, Any],
    ) -> list[Finding]:
        """測試競態條件漏洞
        
        並發發送請求，檢測資源競爭問題（如重複優惠券使用）。
        """
        findings = []
        
        concurrent_requests = parameters.get("concurrent_requests", 10)
        target_endpoint = parameters.get("endpoint", "/api/coupon/redeem")
        
        logger.info(f"Testing race condition with {concurrent_requests} concurrent requests")
        
        # 並發發送請求
        tasks = [
            self._send_race_condition_request(target_url, target_endpoint)
            for _ in range(concurrent_requests)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_count = sum(1 for r in results if r is True)
        
        if successful_count > 1:
            finding = Finding(
                finding_id="bizlogic_race_condition",
                vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_FLAW,
                severity=SeverityLevel.HIGH,
                title="競態條件漏洞",
                description=f"並發請求導致資源重複使用 ({successful_count} 次成功)",
                affected_url=f"{target_url}{target_endpoint}",
                evidence={
                    "concurrent_requests": concurrent_requests,
                    "successful_requests": successful_count,
                    "attack_vector": "Concurrent POST requests",
                },
                remediation="實施分佈式鎖或資料庫事務隔離",
                confidence="high",
                cvss_score=7.0,
            )
            findings.append(finding)
        
        return findings
    
    async def _test_coupon_abuse(
        self,
        target_url: str,
        parameters: dict[str, Any],
    ) -> list[Finding]:
        """測試優惠券濫用漏洞
        
        嘗試重複使用優惠券、堆疊折扣等。
        """
        findings = []
        
        coupon_code = parameters.get("coupon_code", "TEST100")
        
        logger.info(f"Testing coupon abuse: {coupon_code}")
        
        # 嘗試多次使用同一優惠券
        reuse_count = 0
        for attempt in range(3):
            is_successful = await self._send_coupon_request(
                target_url, coupon_code
            )
            if is_successful:
                reuse_count += 1
        
        if reuse_count > 1:
            finding = Finding(
                finding_id=f"bizlogic_coupon_{coupon_code}",
                vulnerability_type=VulnerabilityType.BUSINESS_LOGIC_FLAW,
                severity=SeverityLevel.MEDIUM,
                title=f"優惠券重複使用 - {coupon_code}",
                description=f"優惠券可重複使用 {reuse_count} 次",
                affected_url=f"{target_url}/api/coupon/apply",
                evidence={
                    "coupon_code": coupon_code,
                    "reuse_count": reuse_count,
                    "attack_vector": "Multiple POST /api/coupon/apply",
                },
                remediation="添加優惠券使用狀態追蹤，限制單次使用",
                confidence="high",
                cvss_score=5.0,
            )
            findings.append(finding)
        
        return findings
    
    # === 模擬 HTTP 請求方法（實際環境中應替換為真實 HTTP 客戶端） ===
    
    async def _send_price_manipulation_request(
        self,
        target_url: str,
        product_id: int,
        price: float,
    ) -> bool:
        """發送價格操縱請求（模擬）"""
        # TODO: 實際環境中使用 httpx 或 aiohttp
        await asyncio.sleep(0.1)  # 模擬網絡延遲
        
        # 模擬：負數價格通常能成功繞過驗證
        return price < 0 or price == 0
    
    async def _send_idor_request(
        self,
        target_url: str,
        resource_type: str,
        user_id: int,
    ) -> bool:
        """發送 IDOR 請求（模擬）"""
        await asyncio.sleep(0.1)
        
        # 模擬：某些用戶 ID 可未授權訪問
        return user_id in [1, 2, 3]
    
    async def _send_workflow_bypass_request(
        self,
        target_url: str,
        workflow_type: str,
    ) -> bool:
        """發送工作流繞過請求（模擬）"""
        await asyncio.sleep(0.1)
        
        # 模擬：某些工作流可繞過
        return workflow_type == "checkout"
    
    async def _send_race_condition_request(
        self,
        target_url: str,
        endpoint: str,
    ) -> bool:
        """發送競態條件請求（模擬）"""
        await asyncio.sleep(0.05)
        
        # 模擬：有一定概率成功
        import random
        return random.random() > 0.7
    
    async def _send_coupon_request(
        self,
        target_url: str,
        coupon_code: str,
    ) -> bool:
        """發送優惠券請求（模擬）"""
        await asyncio.sleep(0.1)
        
        # 模擬：優惠券可重複使用
        return True


# === 整合示例 ===

async def integration_example():
    """展示如何與 AICommander 整合使用"""
    
    # 1. AI Commander 做出決策
    ai_decision = {
        "attack_type": "price_manipulation",
        "target_url": "http://localhost:3000",
        "parameters": {
            "product_id": 1,
            "payloads": [-1000, 0, -0.01],
        },
        "reasoning": "目標網站存在價格驗證漏洞，嘗試負數價格攻擊",
    }
    
    # 2. 創建執行器
    executor = BizLogicAttackExecutor(config={"timeout": 30})
    
    # 3. 執行攻擊
    result = await executor.execute_attack(
        attack_type=ai_decision["attack_type"],
        target_url=ai_decision["target_url"],
        parameters=ai_decision["parameters"],
    )
    
    # 4. 返回結果給 Task Planning 模組
    logger.info(f"Attack execution completed: {result['findings_count']} findings")
    
    return result


if __name__ == "__main__":
    # 快速測試
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        result = await integration_example()
        print(f"\n✅ 執行完成: 發現 {result['findings_count']} 個漏洞")
        print(f"⏱️  執行時間: {result['execution_time']:.2f} 秒")
    
    asyncio.run(test())
