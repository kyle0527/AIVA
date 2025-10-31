"""Tool Selector - 工具選擇器

根據任務類型和參數決定使用哪個功能服務/工具來執行
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any

from .task_converter import ExecutableTask

logger = logging.getLogger(__name__)


class ServiceType(str, Enum):
    """服務類型"""

    SCAN_SERVICE = "scan_service"  # 掃描服務
    FUNCTION_SQLI = "function_sqli"  # SQL 注入功能
    FUNCTION_XSS = "function_xss"  # XSS 功能
    FUNCTION_SSRF = "function_ssrf"  # SSRF 功能
    FUNCTION_IDOR = "function_idor"  # IDOR 功能
    INTEGRATION_SERVICE = "integration_service"  # 整合服務
    CORE_ANALYZER = "core_analyzer"  # 核心分析器


@dataclass
class ToolDecision:
    """工具選擇決策

    描述執行任務應該使用哪個工具/服務
    """

    task_id: str
    service_type: ServiceType
    service_endpoint: str | None = None  # RPC/HTTP 端點
    service_function: str | None = None  # 具體調用的函數
    parameters: dict[str, Any] | None = None  # 傳遞給服務的參數
    routing_key: str | None = None  # RabbitMQ routing key
    confidence: float = 1.0  # 選擇信心度

    def __repr__(self) -> str:
        return f"ToolDecision({self.service_type.value}:{self.service_function})"


class ToolSelector:
    """工具選擇器

    根據任務特性決定使用哪個工具/功能服務
    """

    def __init__(self) -> None:
        """初始化工具選擇器"""
        # 任務類型到服務的映射規則
        self.task_service_map: dict[str, ServiceType] = {
            "scan": ServiceType.SCAN_SERVICE,
            "analyze": ServiceType.CORE_ANALYZER,
            "exploit": ServiceType.INTEGRATION_SERVICE,  # 由整合層決定
            "validate": ServiceType.INTEGRATION_SERVICE,
        }

        # 攻擊類型到功能服務的映射
        self.attack_function_map: dict[str, ServiceType] = {
            "sqli": ServiceType.FUNCTION_SQLI,
            "sql_injection": ServiceType.FUNCTION_SQLI,
            "xss": ServiceType.FUNCTION_XSS,
            "cross_site_scripting": ServiceType.FUNCTION_XSS,
            "ssrf": ServiceType.FUNCTION_SSRF,
            "server_side_request_forgery": ServiceType.FUNCTION_SSRF,
            "idor": ServiceType.FUNCTION_IDOR,
            "insecure_direct_object_reference": ServiceType.FUNCTION_IDOR,
        }

        logger.info("ToolSelector initialized with mapping rules")

    def select_tool(self, task: ExecutableTask) -> ToolDecision:
        """選擇執行任務的工具

        Args:
            task: 可執行任務

        Returns:
            工具選擇決策
        """
        logger.debug(f"Selecting tool for task {task.task_id} ({task.task_type})")

        # 根據任務類型選擇服務
        service_type = self._select_service_type(task)

        # 確定服務端點和函數
        endpoint, function = self._determine_endpoint_and_function(service_type, task)

        # 準備參數
        parameters = self._prepare_parameters(task)

        # 確定 routing key (用於 RabbitMQ)
        routing_key = self._determine_routing_key(service_type, task)

        decision = ToolDecision(
            task_id=task.task_id,
            service_type=service_type,
            service_endpoint=endpoint,
            service_function=function,
            parameters=parameters,
            routing_key=routing_key,
            confidence=1.0,
        )

        logger.info(f"Selected {service_type.value}.{function} for task {task.task_id}")
        return decision

    def _select_service_type(self, task: ExecutableTask) -> ServiceType:
        """選擇服務類型

        Args:
            task: 任務

        Returns:
            服務類型
        """
        # 優先檢查任務參數中是否指定了攻擊類型
        attack_type = (
            task.parameters.get("attack_type")
            or task.parameters.get("vulnerability_type")
            or task.metadata.get("attack_type")
        )

        if attack_type and attack_type in self.attack_function_map:
            return self.attack_function_map[attack_type]

        # 根據任務類型選擇
        return self.task_service_map.get(
            task.task_type, ServiceType.INTEGRATION_SERVICE
        )

    def _determine_endpoint_and_function(
        self, service_type: ServiceType, task: ExecutableTask
    ) -> tuple[str | None, str | None]:
        """確定服務端點和函數

        Args:
            service_type: 服務類型
            task: 任務

        Returns:
            (端點, 函數名)
        """
        endpoint_map = {
            ServiceType.SCAN_SERVICE: (
                "http://localhost:8001",
                "scan_target",
            ),
            ServiceType.FUNCTION_SQLI: (
                "http://localhost:8101",
                "test_sql_injection",
            ),
            ServiceType.FUNCTION_XSS: (
                "http://localhost:8102",
                "test_xss",
            ),
            ServiceType.FUNCTION_SSRF: (
                "http://localhost:8103",
                "test_ssrf",
            ),
            ServiceType.FUNCTION_IDOR: (
                "http://localhost:8104",
                "test_idor",
            ),
            ServiceType.INTEGRATION_SERVICE: (
                "http://localhost:8002",
                "execute_test",
            ),
            ServiceType.CORE_ANALYZER: (None, "analyze_vulnerability"),
        }

        return endpoint_map.get(service_type, (None, None))

    def _prepare_parameters(self, task: ExecutableTask) -> dict[str, Any]:
        """準備傳遞給服務的參數

        Args:
            task: 任務

        Returns:
            參數字典
        """
        # 合併任務參數和元數據
        params = task.parameters.copy()
        params["task_id"] = task.task_id
        params["task_type"] = task.task_type
        params["action"] = task.action

        return params

    def _determine_routing_key(
        self, service_type: ServiceType, task: ExecutableTask
    ) -> str | None:
        """確定 RabbitMQ routing key

        Args:
            service_type: 服務類型
            task: 任務

        Returns:
            routing key
        """
        routing_key_map = {
            ServiceType.SCAN_SERVICE: "task.scan",
            ServiceType.FUNCTION_SQLI: "task.sqli",
            ServiceType.FUNCTION_XSS: "task.xss",
            ServiceType.FUNCTION_SSRF: "task.ssrf",
            ServiceType.FUNCTION_IDOR: "task.idor",
            ServiceType.INTEGRATION_SERVICE: "task.integration",
            ServiceType.CORE_ANALYZER: "task.analyze",
        }

        return routing_key_map.get(service_type)
