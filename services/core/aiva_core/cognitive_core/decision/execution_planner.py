"""执行计划生成器

职责:
1. 根据目标信息生成执行计划
2. 决定使用的能力组合 (CLI指令组合)
3. 设定能力参数和间隔时间
4. 首次扫描策略: 使用 Rust引擎深入扫描
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from uuid import uuid4
from enum import Enum

from services.aiva_common.schemas import (
    CommandType,
    Phase0StartPayload,
    Phase1StartPayload,
    ScanScope,
    Authentication,
    RateLimit,
)
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class ScanStrategy(str, Enum):
    """扫描策略"""
    INITIAL_DEEP = "initial_deep"           # 首次深入扫描
    INFORMED = "informed"                   # 基于历史的知情扫描
    MULTI_ENGINE = "multi_engine"           # 多引擎协同
    TARGETED = "targeted"                   # 针对性扫描


class ExecutionStep:
    """执行步骤"""
    def __init__(
        self,
        step_id: int,
        capability: str,
        module: str,
        command_type: CommandType,
        parameters: Dict[str, Any],
        estimated_duration: int = 300,
        depends_on: Optional[List[int]] = None
    ):
        self.step_id = step_id
        self.capability = capability
        self.module = module
        self.command_type = command_type
        self.parameters = parameters
        self.estimated_duration = estimated_duration
        self.depends_on = depends_on or []


class ExecutionPlan:
    """执行计划"""
    def __init__(
        self,
        plan_id: str,
        objective: str,
        strategy: ScanStrategy,
        steps: List[ExecutionStep],
        total_estimated_duration: int = 0
    ):
        self.plan_id = plan_id
        self.objective = objective
        self.strategy = strategy
        self.steps = steps
        self.total_estimated_duration = total_estimated_duration or sum(
            step.estimated_duration for step in steps
        )
        self.created_at = datetime.now(timezone.utc)


class NextPhaseDecision:
    """下一阶段决策"""
    def __init__(
        self,
        action: str,
        reason: str,
        next_step: str,
        engines: Optional[List[str]] = None
    ):
        self.action = action
        self.reason = reason
        self.next_step = next_step
        self.engines = engines or []


class ExecutionPlanner:
    """执行计划生成器
    
    这是步骤2的核心实现：生成规划
    """
    
    def __init__(self):
        self.logger = logger
    
    async def generate_plan(
        self,
        targets: List[str],
        constraints: Dict[str, Any],
        is_new_target: bool = True,
        historical_data: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """生成执行计划
        
        Args:
            targets: 目标 URL 列表
            constraints: 能力限制和排除网页
                {
                    "allowed_capabilities": [...],
                    "disallowed_capabilities": [...],
                    "excluded_paths": [...]
                }
            is_new_target: 是否为新目标
            historical_data: 历史扫描数据 (如有)
            
        Returns:
            ExecutionPlan: 包含步骤、能力、参数的完整计划
        """
        self.logger.info(f"🎯 生成执行计划: {len(targets)} 个目标, 新目标={is_new_target}")
        
        if is_new_target:
            # 首次扫描: 使用 Rust 引擎深入扫描
            return await self._generate_initial_scan_plan(targets, constraints)
        else:
            # 后续扫描: 基于历史资料决策
            return await self._generate_informed_scan_plan(targets, constraints, historical_data)
    
    async def _generate_initial_scan_plan(
        self,
        targets: List[str],
        constraints: Dict[str, Any]
    ) -> ExecutionPlan:
        """生成首次扫描计划
        
        策略: 使用 Rust引擎进行深入扫描
        目的: 获取技术栈、端点、漏洞线索等参考依据
        
        CLI命令: python scripts/ui/aiva_cli.py --attack "rust深入扫描 <目标>"
        """
        self.logger.info("📋 生成首次深入扫描计划 (Rust引擎)")
        
        plan_id = f"plan_{uuid4().hex[:12]}"
        
        # 构建 Rust 扫描参数
        scan_params = {
            "scan_id": plan_id,
            "targets": targets,
            "mode": "deep_analysis",  # 深入分析模式
            "timeout": 600,           # 10分钟超时
            "scan_scope": self._build_scan_scope(targets, constraints),
        }
        
        step = ExecutionStep(
            step_id=1,
            capability="rust_deep_scan",
            module="scan",
            command_type=CommandType.SCAN_COMPREHENSIVE,
            parameters=scan_params,
            estimated_duration=600,  # 10分钟
            depends_on=[]
        )
        
        plan = ExecutionPlan(
            plan_id=plan_id,
            objective="首次深入扫描 - 获取目标信息",
            strategy=ScanStrategy.INITIAL_DEEP,
            steps=[step],
            total_estimated_duration=600
        )
        
        self.logger.info(f"✅ 计划生成完成: {plan.plan_id}, 预计耗时 {plan.total_estimated_duration}s")
        return plan
    
    async def _generate_informed_scan_plan(
        self,
        targets: List[str],
        constraints: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]]
    ) -> ExecutionPlan:
        """生成基于历史的知情扫描计划
        
        基于之前的扫描结果，有针对性地进行扫描
        """
        self.logger.info("📋 生成知情扫描计划 (基于历史数据)")
        
        plan_id = f"plan_{uuid4().hex[:12]}"
        steps = []
        
        # 分析历史数据，决定扫描策略
        if historical_data:
            tech_stack = historical_data.get("tech_stack", {})
            
            # 根据历史发现的技术栈，选择合适的扫描器
            if tech_stack.get("is_spa"):
                # SPA应用 - 使用 TypeScript 引擎
                steps.append(self._create_typescript_scan_step(targets, constraints))
            
            if "api" in tech_stack.get("detected_technologies", []):
                # API端点 - 使用 Go 引擎 (SSRF/API扫描)
                steps.append(self._create_go_scan_step(targets, constraints))
            
            # 如果需要深度爬虫
            if historical_data.get("needs_deep_crawl"):
                steps.append(self._create_python_crawl_step(targets, constraints))
        
        # 如果没有特定策略，使用综合扫描
        if not steps:
            steps.append(ExecutionStep(
                step_id=1,
                capability="comprehensive_scan",
                module="scan",
                command_type=CommandType.SCAN_COMPREHENSIVE,
                parameters={
                    "scan_id": plan_id,
                    "targets": targets,
                    "scan_scope": self._build_scan_scope(targets, constraints),
                },
                estimated_duration=1800,  # 30分钟
                depends_on=[]
            ))
        
        return ExecutionPlan(
            plan_id=plan_id,
            objective="知情扫描 - 基于历史数据",
            strategy=ScanStrategy.INFORMED,
            steps=steps
        )
    
    def _build_scan_scope(
        self,
        targets: List[str],
        constraints: Dict[str, Any]
    ) -> ScanScope:
        """构建扫描范围"""
        # 提取主机名
        from urllib.parse import urlparse
        allowed_hosts = []
        for target in targets:
            parsed = urlparse(target)
            if parsed.hostname:
                allowed_hosts.append(parsed.hostname)
        
        return ScanScope(
            allowed_hosts=allowed_hosts,
            include_subdomains=True,
            exclusions=constraints.get("exclusions", [])
        )
    
    def _create_typescript_scan_step(
        self,
        targets: List[str],
        constraints: Dict[str, Any]
    ) -> ExecutionStep:
        """创建 TypeScript 扫描步骤 (SPA)"""
        return ExecutionStep(
            step_id=2,
            capability="typescript_spa_scan",
            module="scan",
            command_type=CommandType.SCAN_PHASE1,
            parameters={
                "targets": targets,
                "engines": ["typescript"],
                "scan_scope": self._build_scan_scope(targets, constraints),
            },
            estimated_duration=900,  # 15分钟
            depends_on=[1]
        )
    
    def _create_go_scan_step(
        self,
        targets: List[str],
        constraints: Dict[str, Any]
    ) -> ExecutionStep:
        """创建 Go 扫描步骤 (SSRF/API)"""
        return ExecutionStep(
            step_id=3,
            capability="go_ssrf_scan",
            module="scan",
            command_type=CommandType.SCAN_PHASE1,
            parameters={
                "targets": targets,
                "engines": ["go"],
                "scan_scope": self._build_scan_scope(targets, constraints),
            },
            estimated_duration=600,  # 10分钟
            depends_on=[1]
        )
    
    def _create_python_crawl_step(
        self,
        targets: List[str],
        constraints: Dict[str, Any]
    ) -> ExecutionStep:
        """创建 Python 爬虫步骤"""
        return ExecutionStep(
            step_id=4,
            capability="python_deep_crawl",
            module="scan",
            command_type=CommandType.SCAN_PHASE1,
            parameters={
                "targets": targets,
                "engines": ["python"],
                "scan_scope": self._build_scan_scope(targets, constraints),
            },
            estimated_duration=1800,  # 30分钟
            depends_on=[1]
        )
    
    async def decide_next_phase(
        self,
        rust_scan_result: Dict[str, Any]
    ) -> NextPhaseDecision:
        """根据 Rust扫描结果决定下一阶段
        
        这是步骤4的实现：AI决定引擎组合
        
        决策逻辑:
        1. 结果充分 → 直接进入 Phase 2 (功能模块测试)
        2. 需要补充 → 使用其他引擎组合
           - 发现 SPA → 添加 TypeScript引擎
           - 发现 API → 添加 Go引擎 (SSRF/CSPM)
           - 需要深度爬虫 → 添加 Python引擎
        """
        self.logger.info("🤔 分析 Rust扫描结果，决定下一阶段...")
        
        analysis = self._analyze_rust_result(rust_scan_result)
        
        if analysis["is_sufficient"]:
            # 结果充分，直接进入攻击测试
            self.logger.info("✅ Rust扫描结果充分，跳过多引擎扫描")
            return NextPhaseDecision(
                action="skip_to_phase2",
                reason="Rust扫描已获得足够信息",
                next_step="功能模块测试"
            )
        
        # 需要补充扫描
        engines = ["rust"]  # 保留 Rust 结果
        reasons = []
        
        if analysis.get("detected_spa"):
            engines.append("typescript")
            reasons.append("检测到SPA应用")
        
        if analysis.get("detected_api_endpoints"):
            engines.append("go")
            reasons.append("检测到API端点")
        
        if analysis.get("needs_deep_crawl"):
            engines.append("python")
            reasons.append("需要深度爬虫")
        
        self.logger.info(f"📊 需要补充扫描: {', '.join(reasons)}")
        self.logger.info(f"🔧 选择引擎: {', '.join(engines)}")
        
        return NextPhaseDecision(
            action="multi_engine_scan",
            engines=engines,
            reason=f"需要补充: {', '.join(reasons)}",
            next_step="多引擎协同扫描"
        )
    
    def _analyze_rust_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """分析 Rust 扫描结果"""
        summary = result.get("summary", {})
        
        # 检查是否发现足够信息
        urls_found = summary.get("urls_found", 0)
        endpoints_found = summary.get("endpoints_found", 0)
        tech_stack = summary.get("tech_stack", {})
        
        is_sufficient = (
            urls_found >= 10 and
            endpoints_found >= 5 and
            len(tech_stack) > 0
        )
        
        # 检测特征
        is_spa = tech_stack.get("is_spa", False)
        has_api = len(summary.get("apis_found", [])) > 0
        needs_crawl = urls_found < 20  # URL数量少，需要深度爬虫
        
        return {
            "is_sufficient": is_sufficient,
            "detected_spa": is_spa,
            "detected_api_endpoints": has_api,
            "needs_deep_crawl": needs_crawl,
            "missing_info": self._identify_missing_info(summary)
        }
    
    def _identify_missing_info(self, summary: Dict[str, Any]) -> str:
        """识别缺失的信息"""
        missing = []
        
        if summary.get("urls_found", 0) < 10:
            missing.append("URL数量不足")
        
        if not summary.get("tech_stack"):
            missing.append("技术栈未识别")
        
        if not summary.get("forms_found"):
            missing.append("未发现表单")
        
        return ", ".join(missing) if missing else "信息完整"
