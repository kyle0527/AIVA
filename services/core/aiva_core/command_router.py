"""AIVA Command Router - 智能命令路由系統
從 aiva_core_v2 遷移到核心模組

智能命令路由器 - 支持 AI vs 非AI 自動判斷和複雜性分析
"""

from dataclasses import dataclass
from enum import Enum
import logging
import time
from typing import Any

from ...aiva_common.cross_language import AIVAError


class CommandType(Enum):
    """命令類型枚舉"""

    SIMPLE = "simple"  # 簡單命令，無需 AI
    ANALYSIS = "analysis"  # 分析類命令，可能需要 AI
    COMPLEX = "complex"  # 複雜命令，需要 AI 推理
    INTERACTIVE = "interactive"  # 交互式命令，需要對話
    SCAN = "scan"  # 掃描命令，需要安全掃描
    REPORT = "report"  # 報告生成命令


class ExecutionMode(Enum):
    """執行模式"""

    SYNCHRONOUS = "sync"  # 同步執行
    ASYNCHRONOUS = "async"  # 異步執行
    BACKGROUND = "background"  # 後台執行
    STREAMING = "streaming"  # 流式執行


@dataclass
class CommandContext:
    """命令上下文"""

    command: str
    args: dict[str, Any]
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    timestamp: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionResult:
    """執行結果"""

    success: bool
    result: Any = None
    error: AIVAError | None = None
    execution_time: float = 0.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CommandRouter:
    """智能命令路由器 - 支持 AI vs 非AI 自動判斷和複雜性分析
    """

    def __init__(self):
        self.logger = logging.getLogger("command_router")
        self._route_map = self._initialize_intelligent_routes()
        self._ai_keywords = self._initialize_ai_keywords()
        self._complexity_patterns = self._initialize_complexity_patterns()
        self._command_stats = {}  # 命令統計和學習

    def _initialize_intelligent_routes(self) -> dict[str, dict[str, Any]]:
        """初始化智能路由映射"""
        return {
            # 系統管理命令 - 低複雜性，無需 AI
            "help": {
                "type": CommandType.SIMPLE,
                "mode": ExecutionMode.SYNCHRONOUS,
                "requires_ai": False,
                "complexity": "low",
                "priority": 5,
                "description": "Display help information",
            },
            "version": {
                "type": CommandType.SIMPLE,
                "mode": ExecutionMode.SYNCHRONOUS,
                "requires_ai": False,
                "complexity": "low",
                "priority": 5,
                "description": "Show version information",
            },
            "status": {
                "type": CommandType.SIMPLE,
                "mode": ExecutionMode.SYNCHRONOUS,
                "requires_ai": False,
                "complexity": "low",
                "priority": 5,
                "description": "Check system status",
            },
            "config": {
                "type": CommandType.SIMPLE,
                "mode": ExecutionMode.SYNCHRONOUS,
                "requires_ai": False,
                "complexity": "medium",
                "priority": 4,
                "description": "Configuration management",
            },
            # 掃描和檢測命令 - 中等複雜性
            "scan": {
                "type": CommandType.SCAN,
                "mode": ExecutionMode.BACKGROUND,
                "requires_ai": False,
                "complexity": "medium",
                "priority": 4,
                "description": "Security scanning operations",
            },
            "test": {
                "type": CommandType.SCAN,
                "mode": ExecutionMode.ASYNCHRONOUS,
                "requires_ai": False,
                "complexity": "medium",
                "priority": 3,
                "description": "Testing operations",
            },
            "probe": {
                "type": CommandType.SCAN,
                "mode": ExecutionMode.BACKGROUND,
                "requires_ai": False,
                "complexity": "medium",
                "priority": 3,
                "description": "Network probing",
            },
            "audit": {
                "type": CommandType.ANALYSIS,
                "mode": ExecutionMode.ASYNCHRONOUS,
                "requires_ai": True,
                "complexity": "high",
                "priority": 4,
                "description": "Security audit with AI analysis",
            },
            # 分析命令 - 高複雜性，可能需要 AI
            "analyze": {
                "type": CommandType.ANALYSIS,
                "mode": ExecutionMode.ASYNCHRONOUS,
                "requires_ai": True,
                "complexity": "high",
                "priority": 3,
                "description": "Deep analysis with AI insights",
            },
            "inspect": {
                "type": CommandType.ANALYSIS,
                "mode": ExecutionMode.ASYNCHRONOUS,
                "requires_ai": False,
                "complexity": "medium",
                "priority": 3,
                "description": "Detailed inspection",
            },
            "validate": {
                "type": CommandType.ANALYSIS,
                "mode": ExecutionMode.ASYNCHRONOUS,
                "requires_ai": False,
                "complexity": "medium",
                "priority": 3,
                "description": "Validation checks",
            },
            "check": {
                "type": CommandType.ANALYSIS,
                "mode": ExecutionMode.SYNCHRONOUS,
                "requires_ai": False,
                "complexity": "low",
                "priority": 3,
                "description": "Quick checks",
            },
            # AI 互動命令 - 高複雜性，需要 AI
            "ask": {
                "type": CommandType.INTERACTIVE,
                "mode": ExecutionMode.SYNCHRONOUS,
                "requires_ai": True,
                "complexity": "medium",
                "priority": 2,
                "description": "AI question answering",
            },
            "chat": {
                "type": CommandType.INTERACTIVE,
                "mode": ExecutionMode.SYNCHRONOUS,
                "requires_ai": True,
                "complexity": "medium",
                "priority": 2,
                "description": "AI conversation",
            },
            "explain": {
                "type": CommandType.COMPLEX,
                "mode": ExecutionMode.SYNCHRONOUS,
                "requires_ai": True,
                "complexity": "high",
                "priority": 2,
                "description": "AI explanation and reasoning",
            },
            # 報告命令 - 中高複雜性
            "report": {
                "type": CommandType.REPORT,
                "mode": ExecutionMode.BACKGROUND,
                "requires_ai": True,
                "complexity": "high",
                "priority": 3,
                "description": "Generate comprehensive reports",
            },
            "summarize": {
                "type": CommandType.REPORT,
                "mode": ExecutionMode.ASYNCHRONOUS,
                "requires_ai": True,
                "complexity": "medium",
                "priority": 3,
                "description": "Generate summaries",
            },
        }

    def _initialize_ai_keywords(self) -> set[str]:
        """初始化 AI 關鍵詞"""
        return {
            # AI 推理關鍵詞
            "analyze",
            "analysis",
            "reason",
            "reasoning",
            "infer",
            "inference",
            "predict",
            "prediction",
            "learn",
            "learning",
            "train",
            "training",
            "classify",
            "classification",
            "detect",
            "detection",
            "recognize",
            "pattern",
            "patterns",
            "trend",
            "trends",
            "insight",
            "insights",
            # 自然語言處理關鍵詞
            "understand",
            "interpret",
            "generate",
            "create",
            "compose",
            "write",
            "translate",
            "summarize",
            "explain",
            "describe",
            "clarify",
            # 決策和建議關鍵詞
            "recommend",
            "recommendation",
            "suggest",
            "suggestion",
            "advice",
            "decide",
            "decision",
            "choose",
            "select",
            "optimize",
            "improve",
            # 交互和對話關鍵詞
            "chat",
            "talk",
            "discuss",
            "conversation",
            "dialogue",
            "ask",
            "question",
            "answer",
            "respond",
            "reply",
            "interact",
            # 複雜分析關鍵詞
            "correlate",
            "correlation",
            "compare",
            "comparison",
            "evaluate",
            "assessment",
            "judge",
            "judgment",
            "estimate",
            "calculation",
            "research",
            "investigate",
            "exploration",
            # 複雜性指標詞彙
            "complex",
            "complexity",
            "advanced",
            "sophisticated",
            "intelligent",
            "smart",
            "adaptive",
            "personalized",
            "contextual",
            "dynamic",
            # 問句關鍵詞
            "how",
            "why",
            "what",
            "when",
            "where",
            "who",
            "which",
            "whether",
        }

    def _initialize_complexity_patterns(self) -> dict[str, int]:
        """初始化複雜性模式權重"""
        return {
            # 高複雜性模式 (權重 3)
            "multi-step": 3,
            "cross-reference": 3,
            "contextual": 3,
            "strategic": 3,
            "optimization": 3,
            "integration": 3,
            "correlation": 3,
            "prediction": 3,
            "machine learning": 3,
            # 中等複雜性模式 (權重 2)
            "analysis": 2,
            "comparison": 2,
            "evaluation": 2,
            "synthesis": 2,
            "assessment": 2,
            "validation": 2,
            "classification": 2,
            "categorization": 2,
            # 低複雜性模式 (權重 1)
            "lookup": 1,
            "simple": 1,
            "basic": 1,
            "direct": 1,
            "immediate": 1,
            "quick": 1,
            "fast": 1,
            "instant": 1,
        }

    def _analyze_command_complexity(
        self, command: str, args: list[str] | dict[str, Any]
    ) -> str:
        """分析命令複雜性"""
        # 處理不同類型的 args
        if isinstance(args, dict):
            args_str = " ".join([f"{k}={v}" for k, v in args.items()])
            arg_count = len(args)
        else:
            args_str = " ".join(args) if args else ""
            arg_count = len(args) if args else 0

        full_text = f"{command} {args_str}".lower()
        complexity_score = 0

        # 檢查複雜性模式
        for pattern, weight in self._complexity_patterns.items():
            if pattern in full_text:
                complexity_score += weight

        # 參數數量影響複雜性
        if arg_count > 5:
            complexity_score += 2
        elif arg_count > 3:
            complexity_score += 1

        # 文本長度影響複雜性
        if len(full_text) > 100:
            complexity_score += 2
        elif len(full_text) > 50:
            complexity_score += 1

        # 返回複雜性等級
        if complexity_score >= 6:
            return "high"
        elif complexity_score >= 3:
            return "medium"
        else:
            return "low"

    def _requires_ai_analysis(
        self, command: str, args: list[str] | dict[str, Any]
    ) -> bool:
        """智能判斷是否需要 AI 處理"""
        # 處理不同類型的 args
        if isinstance(args, dict):
            args_str = " ".join([f"{k}={v}" for k, v in args.items()])
        else:
            args_str = " ".join(args) if args else ""

        full_text = f"{command} {args_str}".lower()

        # 檢查 AI 關鍵詞
        ai_keyword_count = sum(
            1 for keyword in self._ai_keywords if keyword in full_text
        )

        # 複雜性分析
        complexity = self._analyze_command_complexity(command, args)

        # 決策邏輯
        if ai_keyword_count >= 2 or ai_keyword_count >= 1 and complexity == "high" or complexity == "high" and len(full_text) > 50:  # 包含多個 AI 關鍵詞
            return True
        else:
            return False

    def route_command(self, context: CommandContext) -> dict[str, Any]:
        """智能路由命令"""
        command = context.command.lower()

        # 記錄命令統計
        if command not in self._command_stats:
            self._command_stats[command] = {"count": 0, "ai_usage": 0}
        self._command_stats[command]["count"] += 1

        # 檢查預定義路由
        if command in self._route_map:
            route_info = self._route_map[command].copy()
        else:
            # 動態分析未知命令
            requires_ai = self._requires_ai_analysis(command, context.args)
            complexity = self._analyze_command_complexity(command, context.args)

            # 根據分析結果確定命令類型
            if requires_ai:
                if any(
                    keyword in f"{command} {context.args}".lower()
                    for keyword in ["chat", "ask", "talk", "discuss"]
                ):
                    cmd_type = CommandType.INTERACTIVE
                elif complexity == "high":
                    cmd_type = CommandType.COMPLEX
                else:
                    cmd_type = CommandType.ANALYSIS
            else:
                if command in ["scan", "test", "probe"]:
                    cmd_type = CommandType.SCAN
                elif command in ["report", "summarize"]:
                    cmd_type = CommandType.REPORT
                else:
                    cmd_type = CommandType.SIMPLE

            # 確定執行模式
            if (
                cmd_type in [CommandType.SCAN, CommandType.REPORT]
                and complexity == "high"
            ):
                exec_mode = ExecutionMode.BACKGROUND
            elif cmd_type in [CommandType.ANALYSIS, CommandType.COMPLEX]:
                exec_mode = ExecutionMode.ASYNCHRONOUS
            else:
                exec_mode = ExecutionMode.SYNCHRONOUS

            route_info = {
                "type": cmd_type,
                "mode": exec_mode,
                "requires_ai": requires_ai,
                "complexity": complexity,
                "priority": 3,  # 默認優先級
                "description": f"Dynamic route for {command}",
            }

        # 記錄 AI 使用統計
        if route_info["requires_ai"]:
            self._command_stats[command]["ai_usage"] += 1

        # 添加路由元數據
        route_info["route_timestamp"] = time.time()
        route_info["context_id"] = context.request_id

        self.logger.debug(
            f"Routed '{command}' -> {route_info['type'].value} "
            f"(AI: {route_info['requires_ai']}, "
            f"Complexity: {route_info['complexity']})"
        )

        return route_info

    def get_command_stats(self) -> dict[str, Any]:
        """獲取命令統計信息"""
        return {
            "total_commands": sum(
                stats["count"] for stats in self._command_stats.values()
            ),
            "ai_commands": sum(
                stats["ai_usage"] for stats in self._command_stats.values()
            ),
            "command_breakdown": self._command_stats.copy(),
            "ai_usage_rate": (
                sum(stats["ai_usage"] for stats in self._command_stats.values())
                / max(sum(stats["count"] for stats in self._command_stats.values()), 1)
            )
            * 100,
        }

    def update_route(self, command: str, route_info: dict[str, Any]):
        """更新路由規則"""
        self._route_map[command.lower()] = route_info
        self.logger.info(f"Updated route for command: {command}")

    def get_available_commands(self) -> list[str]:
        """獲取可用命令列表"""
        return list(self._route_map.keys())


# 全局命令路由器實例
_command_router_instance = None


def get_command_router() -> CommandRouter:
    """獲取命令路由器實例"""
    global _command_router_instance
    if _command_router_instance is None:
        _command_router_instance = CommandRouter()
    return _command_router_instance
