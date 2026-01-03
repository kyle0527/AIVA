"""执行编排器

职责:
1. 接收执行计划
2. 下令给各模块执行 (通过CLI命令)
3. 监控执行状态
4. 收集执行结果
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from services.aiva_common.command_center import get_command_center
from services.aiva_common.schemas import (
    AICommand,
    AICommandResult,
    CommandStatus,
    CommandContext,
    CommandPriority,
)
from services.aiva_common.utils import get_logger

from .execution_planner import ExecutionPlan, ExecutionStep

logger = get_logger(__name__)


class ExecutionResult:
    """执行结果"""
    def __init__(
        self,
        plan_id: str,
        success: bool,
        steps_executed: int,
        results: List[AICommandResult],
        execution_time: float = 0.0,
        error: Optional[str] = None
    ):
        self.plan_id = plan_id
        self.success = success
        self.steps_executed = steps_executed
        self.results = results
        self.execution_time = execution_time
        self.error = error
        self.completed_at = datetime.now(timezone.utc)


class ExecutionOrchestrator:
    """执行编排器
    
    这是步骤2的下半部分：下令执行
    通过 CLI 命令调用各模块，避免跨语言问题
    """
    
    def __init__(self):
        self.command_center = get_command_center()
        self.logger = logger
        self._active_executions: Dict[str, Dict[str, Any]] = {}
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        context: Optional[CommandContext] = None
    ) -> ExecutionResult:
        """执行计划
        
        Args:
            plan: 执行计划
            context: 命令上下文
            
        Returns:
            ExecutionResult: 执行结果
        """
        self.logger.info(f"🚀 开始执行计划: {plan.plan_id}")
        self.logger.info(f"   目标: {plan.objective}")
        self.logger.info(f"   策略: {plan.strategy}")
        self.logger.info(f"   步骤数: {len(plan.steps)}")
        
        start_time = datetime.now(timezone.utc)
        results = []
        success = True
        error_msg = None
        
        # 标记为活跃执行
        self._active_executions[plan.plan_id] = {
            "plan": plan,
            "start_time": start_time,
            "status": "running"
        }
        
        try:
            # 按顺序执行步骤
            for step in plan.steps:
                self.logger.info(f"📌 执行步骤 {step.step_id}: {step.capability}")
                
                # 检查依赖
                if not self._check_dependencies(step, results):
                    self.logger.warning(f"⚠️  步骤 {step.step_id} 依赖未满足，跳过")
                    continue
                
                # 构建 AI 命令
                command = self._build_command(step, plan.plan_id, context)
                
                # 执行命令
                try:
                    result = await self.command_center.execute(command, context)
                    results.append(result)
                    
                    if result.status == CommandStatus.FAILED:
                        self.logger.error(f"❌ 步骤 {step.step_id} 执行失败: {result.error}")
                        success = False
                        error_msg = result.error
                        break  # 失败则停止执行
                    
                    self.logger.info(f"✅ 步骤 {step.step_id} 执行成功")
                    
                except Exception as e:
                    self.logger.error(f"❌ 步骤 {step.step_id} 执行异常: {e}")
                    success = False
                    error_msg = str(e)
                    break
            
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()
            
            # 更新执行状态
            self._active_executions[plan.plan_id]["status"] = "completed" if success else "failed"
            self._active_executions[plan.plan_id]["end_time"] = end_time
            
            result = ExecutionResult(
                plan_id=plan.plan_id,
                success=success,
                steps_executed=len(results),
                results=results,
                execution_time=execution_time,
                error=error_msg
            )
            
            self.logger.info(f"🏁 计划执行完成: {plan.plan_id}")
            self.logger.info(f"   成功: {success}")
            self.logger.info(f"   步骤: {len(results)}/{len(plan.steps)}")
            self.logger.info(f"   耗时: {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ 计划执行异常: {e}")
            self._active_executions[plan.plan_id]["status"] = "error"
            
            return ExecutionResult(
                plan_id=plan.plan_id,
                success=False,
                steps_executed=len(results),
                results=results,
                error=str(e)
            )
    
    def _build_command(
        self,
        step: ExecutionStep,
        plan_id: str,
        context: Optional[CommandContext]
    ) -> AICommand:
        """构建 AI 命令
        
        这里生成的命令将通过 CLI 执行：
        python scripts/ui/aiva_cli.py --attack "..."
        """
        # 從 context 提取追蹤信息（如果有）
        trace_id = context.trace_id if context else None
        session_id = context.session_id if context else None
        
        command = AICommand(
            command_id=f"{plan_id}_step_{step.step_id}",
            command_type=step.command_type,
            target_module=step.module,
            payload=step.parameters,
            priority=CommandPriority.LOW,
            timeout=step.estimated_duration,
            trace_id=trace_id,
            session_id=session_id,
            parent_command_id=plan_id,
            callback_url=None
        )
        
        self.logger.debug(f"🔨 构建命令: {command.command_id}")
        self.logger.debug(f"   类型: {command.command_type}")
        self.logger.debug(f"   模块: {command.target_module}")
        
        return command
    
    def _check_dependencies(
        self,
        step: ExecutionStep,
        completed_results: List[AICommandResult]
    ) -> bool:
        """检查步骤依赖是否满足"""
        if not step.depends_on:
            return True
        
        completed_step_ids = [
            int(result.command_id.split("_step_")[1])
            for result in completed_results
            if result.status == CommandStatus.COMPLETED
        ]
        
        for dep_id in step.depends_on:
            if dep_id not in completed_step_ids:
                self.logger.warning(f"⚠️  依赖步骤 {dep_id} 未完成")
                return False
        
        return True
    
    def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """获取执行状态"""
        return self._active_executions.get(plan_id)
    
    def list_active_executions(self) -> List[str]:
        """列出活跃的执行"""
        return [
            plan_id
            for plan_id, info in self._active_executions.items()
            if info["status"] == "running"
        ]
