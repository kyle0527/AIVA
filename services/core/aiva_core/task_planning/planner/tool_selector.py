"""Tool Selector - 工具選擇器

根據任務類型和參數決定使用哪個功能服務/工具來執行

架構升級（v2.0 - 2026-02-09）:
- 支持 CLI 參數包驅動架構
- 從 internal_classification.json 動態加載 flows
- AI 產出 CLICommand，執行器轉換為實際 CLI 調用
"""

from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any

# 導入 CLI 命令模型
from aiva_common.schemas.commands import CLICommand

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

    # ========== CLI 架構擴展（v2.0）==========

    def select_cli_command(
        self,
        intent: str,
        target: str,
        context: dict[str, Any] | None = None
    ) -> CLICommand | None:
        """根據意圖選擇 CLI 命令（新架構）

        Args:
            intent: AI 意圖（scan, exploit, analyze 等）
            target: 目標 URL/IP
            context: 上下文信息（包含 intensity, mode 等）

        Returns:
            CLICommand 或 None（未找到匹配的 flow）

        Examples:
            >>> selector = ToolSelector()
            >>> cmd = selector.select_cli_command(
            ...     intent="sqli_detection",
            ...     target="https://example.com",
            ...     context={"intensity": 0.8, "mode": "stealth"}
            ... )
            >>> print(cmd.to_shell_command())
            'python -m services.core.aiva_core.core_capabilities.cli.aiva_cli flow8 --target https://example.com --intensity 0.8 --mode stealth'
        """
        context = context or {}

        # 加載 flow 定義
        flows = self._load_internal_flows()
        if not flows:
            logger.warning("未找到可用的 flows")
            return None

        # 根據意圖匹配 flow
        matched_flow = self._match_flow_by_intent(intent, flows)
        if not matched_flow:
            logger.warning(f"未找到匹配意圖的 flow: {intent}")
            return None

        # 構建 CLI 命令
        flow_id = matched_flow.get("id")
        flags = {
            "intensity": context.get("intensity", 0.5),
            "mode": context.get("mode", "normal")
        }

        # 添加額外參數
        for key in ["data", "query", "param", "concurrency"]:
            if key in context:
                flags[key] = context[key]

        cmd = CLICommand(
            flow_id=f"flow_{flow_id}",
            target=target,
            flags=flags,
            metadata={
                "intent": intent,
                "flow_name": matched_flow.get("name"),
                "primary_module": matched_flow.get("primary_module"),
                "component_type": matched_flow.get("component_type")
            }
        )

        logger.info(f"✅ 選擇 CLI 命令: {cmd.flow_id} for {intent}")
        return cmd

    def _load_internal_flows(self) -> list[dict[str, Any]]:
        """加載內部 flows 定義

        Returns:
            Flow 定義列表
        """
        # 嘗試多個可能路徑
        possible_paths = [
            Path("C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/internal_classification.json"),
            Path("services/integration/data/internal_exploration/internal_classification.json"),
            Path("../../../integration/data/internal_exploration/internal_classification.json"),
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, encoding='utf-8') as f:
                        data = json.load(f)
                        flows = data.get('flows', [])
                        if flows:
                            logger.info(f"📋 已載入 {len(flows)} 個 internal flows")
                            return flows
                except Exception as e:
                    logger.warning(f"讀取 {path} 失敗: {e}")
                    continue

        logger.error("未找到 internal_classification.json")
        return []

    def _match_flow_by_intent(self, intent: str, flows: list[dict[str, Any]]) -> dict[str, Any] | None:
        """根據意圖匹配最佳 flow

        Args:
            intent: 意圖字符串
            flows: Flow 列表

        Returns:
            匹配的 flow 或 None
        """
        # 意圖關鍵字映射
        intent_keywords = {
            "sqli": ["sql", "injection", "sqli"],
            "xss": ["xss", "cross", "script"],
            "scan": ["scan", "nmap", "port"],
            "exploit": ["exploit", "attack"],
            "analyze": ["analyze", "analysis"],
            "learning": ["learn", "train", "model"],
            "rag": ["rag", "retrieval", "knowledge"]
        }

        # 提取意圖類型
        intent_lower = intent.lower()
        intent_type = None
        for key, keywords in intent_keywords.items():
            if any(kw in intent_lower for kw in keywords):
                intent_type = key
                break

        if not intent_type:
            # 降級：返回第一個可操作的 flow
            for flow in flows:
                if flow.get("component_type") in ["AI對外能力", "混合組件"]:
                    return flow
            return flows[0] if flows else None

        # 匹配對應模組的 flow
        module_mapping = {
            "sqli": "function_sqli",
            "xss": "function_xss",
            "scan": "scan",
            "exploit": "task_planning",
            "analyze": "cognitive_core",
            "learning": "learning_system",
            "rag": "cognitive_core"
        }

        target_module = module_mapping.get(intent_type)
        if target_module:
            for flow in flows:
                primary = flow.get("primary_module", "")
                if target_module in primary:
                    return flow

        # 降級：返回匹配組件類型的 flow
        for flow in flows:
            if flow.get("component_type") == "AI對外能力":
                return flow

        return flows[0] if flows else None
