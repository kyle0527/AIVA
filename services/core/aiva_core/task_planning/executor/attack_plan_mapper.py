# services/core/aiva_core/execution/attack_plan_mapper.py
"""AI 攻擊計畫映射器

負責將 AI 生成的抽象攻擊計畫 (來自 5M Decision Engine) 轉換為
一系列具體的可執行的 FunctionTaskPayload。

**重構說明** (2025-12-26):
- 修復FunctionTaskPayload參數錯誤
- 所有task必須包含scan_id
- 移除不存在的參數: module_name, function_name, config
- 使用正確的參數: task_id, scan_id, target, context, strategy, test_config
- 降低認知複雜度

**5M 架構說明**:
- 使用神經網路決策輸出（attack_vector, confidence, tools）
- 基於預定義模板生成執行計畫
"""

import logging
from typing import Any
from urllib.parse import urlparse
import uuid

from aiva_common.schemas import AivaMessage
from aiva_common.schemas.tasks import (
    FunctionTaskContext,
    FunctionTaskPayload,
    FunctionTaskTarget,
    FunctionTaskTestConfig,
)

logger = logging.getLogger(__name__)


class AttackPlanMapper:
    """將 AI 決策映射到具體執行任務"""

    def __init__(self, default_scan_id: str | None = None):
        """初始化攻擊計畫映射器
        
        Args:
            default_scan_id: 預設scan ID，如果scan_context中沒有提供
        """
        logger.info("AttackPlanMapper initialized.")
        self.default_scan_id = default_scan_id or self._generate_scan_id()

    async def map_decision_to_tasks(
        self, ai_decision: AivaMessage, scan_context: dict[str, Any]
    ) -> list[FunctionTaskPayload]:
        """將單個 AI 決策步驟轉換為一個或多個 FunctionTaskPayload
        
        Args:
            ai_decision: AI 生成的決策對象
            scan_context: 掃描上下文，必須包含：
                - scan_id: 掃描ID (可選，使用default_scan_id)
                - target_url: 目標URL
                - session_id: 會話ID (可選)
                
        Returns:
            FunctionTaskPayload列表
        """
        tasks: list[FunctionTaskPayload] = []
        
        # 獲取或生成scan_id
        scan_id = scan_context.get("scan_id", self.default_scan_id)
        if not scan_id.startswith("scan_"):
            scan_id = f"scan_{scan_id}"
        
        logger.info(f"Mapping AI decision: {ai_decision.header.message_id}")

        try:
            if not hasattr(ai_decision, "payload") or not ai_decision.payload:
                logger.warning(f"AI decision {ai_decision.header.message_id} has no payload")
                return tasks

            decision_data = ai_decision.payload
            action_type = decision_data.get("action_type", "")

            # 根據決策類型路由
            if action_type == "vulnerability_scan":
                tasks.extend(self._create_vulnerability_scan_task(
                    ai_decision, scan_context, scan_id
                ))
            elif action_type == "information_gathering":
                tasks.extend(self._create_info_gathering_tasks(
                    scan_context, scan_id
                ))
            elif action_type == "exploitation":
                tasks.extend(self._create_exploitation_tasks(
                    scan_context, scan_id
                ))
            else:
                logger.warning(f"Unknown action_type: {action_type}")

        except Exception as e:
            logger.error(
                f"Error mapping decision {ai_decision.header.message_id}: {e}",
                exc_info=True,
            )

        if not tasks:
            logger.warning(
                f"No tasks generated for AI decision: {ai_decision.header.message_id}"
            )

        return tasks

    def _create_vulnerability_scan_task(
        self,
        ai_decision: AivaMessage,
        scan_context: dict[str, Any],
        scan_id: str
    ) -> list[FunctionTaskPayload]:
        """創建漏洞掃描任務"""
        tasks = []
        decision_data = ai_decision.payload
        target_info = decision_data.get("target", {})
        
        if not target_info.get("url"):
            logger.warning("No URL in target_info, skipping task creation")
            return tasks

        # 創建目標
        target = FunctionTaskTarget(
            url=target_info.get("url"),
            method=target_info.get("method", "GET"),
            headers=scan_context.get("base_headers", {}),
        )

        # 創建上下文
        context = FunctionTaskContext(
            # 使用預設配置
        )

        # 創建測試配置
        vuln_type = decision_data.get("vulnerability_type", "general")
        test_config = self._create_test_config_for_vuln(vuln_type)

        # 創建payload
        task_id = f"task_{ai_decision.header.message_id}_{vuln_type}"
        payload = FunctionTaskPayload(
            task_id=task_id,
            scan_id=scan_id,
            priority=self._get_priority_for_vuln(vuln_type),
            target=target,
            context=context,
            strategy=decision_data.get("strategy", "full"),
            test_config=test_config,
        )
        
        tasks.append(payload)
        logger.info(f"Generated {vuln_type} task: {task_id}")
        
        return tasks

    def _create_info_gathering_tasks(
        self, scan_context: dict[str, Any], scan_id: str
    ) -> list[FunctionTaskPayload]:
        """創建信息收集任務
        
        基於 OWASP WSTG 4.1 Information Gathering
        """
        tasks = []
        target_url = scan_context.get("target_url", "")
        
        if not target_url:
            logger.warning("No target_url in scan_context")
            return tasks

        logger.info(f"Creating information gathering tasks for {target_url}")
        
        # 創建通用的目標和上下文
        base_target = FunctionTaskTarget(url=target_url, method="GET")
        base_context = FunctionTaskContext(
            # 信息收集階段使用預設配置
        )

        # 1. Web Server Fingerprinting
        tasks.append(FunctionTaskPayload(
            task_id=self._generate_task_id("info_fingerprint"),
            scan_id=scan_id,
            priority=7,
            target=base_target,
            context=base_context,
            strategy="full",
            test_config=FunctionTaskTestConfig(),
        ))

        # 2. Directory Enumeration
        tasks.append(FunctionTaskPayload(
            task_id=self._generate_task_id("info_directory"),
            scan_id=scan_id,
            priority=6,
            target=base_target,
            context=base_context,
            strategy="full",
            test_config=FunctionTaskTestConfig(),
        ))

        # 3. Technology Detection
        tasks.append(FunctionTaskPayload(
            task_id=self._generate_task_id("info_techstack"),
            scan_id=scan_id,
            priority=8,
            target=base_target,
            context=base_context,
            strategy="full",
            test_config=FunctionTaskTestConfig(),
        ))

        # 4. Entry Points Discovery
        tasks.append(FunctionTaskPayload(
            task_id=self._generate_task_id("info_entrypoints"),
            scan_id=scan_id,
            priority=7,
            target=base_target,
            context=base_context,
            strategy="full",
            test_config=FunctionTaskTestConfig(),
        ))

        # 5. Subdomain Enumeration
        domain = self._extract_domain(target_url)
        if domain:
            tasks.append(FunctionTaskPayload(
                task_id=self._generate_task_id("info_subdomains"),
                scan_id=scan_id,
                priority=6,
                target=FunctionTaskTarget(url=domain, method="GET"),
                context=base_context,
                strategy="full",
                test_config=FunctionTaskTestConfig(),
            ))

        logger.info(f"Created {len(tasks)} information gathering tasks")
        return tasks

    def _create_exploitation_tasks(
        self, scan_context: dict[str, Any], scan_id: str
    ) -> list[FunctionTaskPayload]:
        """創建漏洞利用任務
        
        基於已發現的漏洞信息創建對應的利用任務
        """
        tasks = []
        target_url = scan_context.get("target_url", "")
        findings = scan_context.get("findings", [])

        if not target_url:
            logger.warning("No target_url in scan_context")
            return tasks

        logger.info(f"Creating exploitation tasks for {len(findings)} findings")

        # 根據嚴重程度過濾
        high_severity_findings = [
            f for f in findings 
            if f.get("severity", "").lower() in ["high", "critical", "medium"]
        ]

        # 為每個高危發現創建利用任務
        for finding in high_severity_findings:
            task = self._create_exploitation_task_for_finding(
                finding, target_url, scan_id, scan_context
            )
            if task:
                tasks.append(task)

        # 如果沒有發現，創建通用掃描任務
        if not findings and target_url:
            tasks.append(self._create_comprehensive_scan_task(
                target_url, scan_id
            ))

        logger.info(f"Created {len(tasks)} exploitation tasks")
        return tasks

    def _create_exploitation_task_for_finding(
        self,
        finding: dict[str, Any],
        default_url: str,
        scan_id: str,
        scan_context: dict[str, Any]
    ) -> FunctionTaskPayload | None:
        """為單個發現創建利用任務"""
        vuln_type = finding.get("vulnerability_type", "").lower()
        
        # 創建目標
        target = FunctionTaskTarget(
            url=finding.get("url", default_url),
            method=finding.get("method", "GET"),
        )
        
        # 創建上下文
        context = FunctionTaskContext(
            # 利用階段使用預設配置
        )
        
        # 根據漏洞類型創建測試配置
        test_config = self._create_test_config_for_exploitation(finding)
        
        # 確定策略
        strategy = self._determine_exploitation_strategy(vuln_type)
        
        # 確定優先級
        severity = finding.get("severity", "medium").lower()
        priority = {"critical": 10, "high": 9, "medium": 6}.get(severity, 5)
        
        return FunctionTaskPayload(
            task_id=self._generate_task_id(f"exploit_{vuln_type}"),
            scan_id=scan_id,
            priority=priority,
            target=target,
            context=context,
            strategy=strategy,
            test_config=test_config,
        )

    def _create_comprehensive_scan_task(
        self, target_url: str, scan_id: str
    ) -> FunctionTaskPayload:
        """創建通用掃描任務"""
        return FunctionTaskPayload(
            task_id=self._generate_task_id("exploit_comprehensive"),
            scan_id=scan_id,
            priority=5,
            target=FunctionTaskTarget(url=target_url, method="GET"),
            context=FunctionTaskContext(
                # 利用階段使用預設配置
            ),
            strategy="full",
            test_config=FunctionTaskTestConfig(),
        )

    async def map_entire_plan(
        self, attack_plan: list[AivaMessage], initial_context: dict[str, Any]
    ) -> list[FunctionTaskPayload]:
        """將整個 AI 攻擊計畫映射為任務列表
        
        Args:
            attack_plan: AI 生成的決策步驟列表
            initial_context: 初始掃描上下文
            
        Returns:
            所有任務的列表
        """
        all_tasks: list[FunctionTaskPayload] = []
        current_context = initial_context.copy()

        # 確保有scan_id
        if "scan_id" not in current_context:
            current_context["scan_id"] = self.default_scan_id

        logger.info(f"Mapping entire attack plan with {len(attack_plan)} steps.")

        for i, decision in enumerate(attack_plan):
            logger.debug(f"Mapping step {i+1}/{len(attack_plan)}")
            step_tasks = await self.map_decision_to_tasks(decision, current_context)
            all_tasks.extend(step_tasks)

            # 更新上下文
            if step_tasks:
                current_context["previous_task_ids"] = [
                    task.task_id for task in step_tasks
                ]
                current_context["step_number"] = i + 1

        logger.info(f"Generated {len(all_tasks)} tasks from the attack plan.")
        return all_tasks

    # ==================== 輔助方法 ====================

    def _generate_task_id(self, prefix: str) -> str:
        """生成唯一的任務 ID (符合task_前綴要求)"""
        return f"task_{prefix}_{uuid.uuid4().hex[:8]}"

    def _generate_scan_id(self) -> str:
        """生成掃描ID (符合scan_前綴要求)"""
        return f"scan_{uuid.uuid4().hex[:12]}"

    def _extract_domain(self, url: str) -> str | None:
        """從 URL 中提取域名"""
        try:
            parsed = urlparse(url)
            return parsed.netloc or parsed.path
        except Exception as e:
            logger.warning(f"Failed to extract domain from {url}: {e}")
            return None

    def _create_test_config_for_vuln(self, vuln_type: str) -> FunctionTaskTestConfig:
        """根據漏洞類型創建測試配置"""
        # 基本配置
        config = FunctionTaskTestConfig()
        
        # 根據漏洞類型定制
        if "xss" in vuln_type:
            config.payloads = ["basic", "encoded", "bypass"]
            config.blind_xss = True
            config.dom_testing = True
        elif "sql" in vuln_type:
            config.payloads = ["boolean", "time", "error"]
        
        return config

    def _create_test_config_for_exploitation(
        self, finding: dict[str, Any]
    ) -> FunctionTaskTestConfig:
        """根據漏洞類型和發現信息創建利用測試配置"""
        config = FunctionTaskTestConfig()
        
        # 從finding中提取自定義payload
        if finding.get("suggested_payloads"):
            config.custom_payloads = finding["suggested_payloads"]
        
        return config

    def _get_priority_for_vuln(self, vuln_type: str) -> int:
        """根據漏洞類型確定優先級 (1-10)"""
        priority_map = {
            "sqli": 9,
            "xss": 7,
            "rce": 10,
            "ssrf": 8,
            "idor": 8,
            "lfi": 8,
            "auth": 9,
            "jwt": 8,
            "general": 5,
        }
        return priority_map.get(vuln_type, 5)

    def _determine_exploitation_strategy(self, vuln_type: str) -> str:
        """根據漏洞類型確定利用策略"""
        strategy_map = {
            "idor": "full",
            "sqli": "full",
            "xss": "full",
            "auth": "full",
            "jwt": "full",
        }
        return strategy_map.get(vuln_type, "full")

