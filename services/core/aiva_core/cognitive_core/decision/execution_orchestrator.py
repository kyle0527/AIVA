"""执行编排器

⚠️ DEPRECATED: 此模組已廢棄，請使用 PlanExecutor 代替
   路徑: aiva_core.task_planning.executor.plan_executor.PlanExecutor
   
原因:
- ExecutionOrchestrator 使用本地 CLI 架構（subprocess）
- PlanExecutor 使用分布式微服務架構（MessageBroker + RabbitMQ）
- 兩套系統並存導致狀態不對等和架構衝突

建議:
- 新代碼請使用 PlanExecutor
- 現有代碼請逐步遷移到 PlanExecutor
- 此模組將在未來版本中移除

职责（舊）:
1. 接收执行计划
2. 下令给各模块执行 (通过CLI命令)
3. 监控执行状态
4. 收集执行结果
"""

import warnings

# 發出廢棄警告
warnings.warn(
    "ExecutionOrchestrator is deprecated. Use PlanExecutor instead: "
    "from aiva_core.task_planning.executor.plan_executor import PlanExecutor",
    DeprecationWarning,
    stacklevel=2
)

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import subprocess
import json

from services.aiva_common.schemas import (
    CommandContext,
)
from services.aiva_common.utils import get_logger

from .execution_planner import ExecutionPlan, ExecutionStep

logger = get_logger(__name__)


class ExecutionResult:
    """执行结果（CLI 架構）"""
    def __init__(
        self,
        plan_id: str,
        success: bool,
        steps_executed: int,
        command_outputs: List[Dict[str, Any]],
        execution_time: float = 0.0,
        error: Optional[str] = None
    ):
        self.plan_id = plan_id
        self.success = success
        self.steps_executed = steps_executed
        self.command_outputs = command_outputs  # {step_id, stdout, stderr, exit_code, cli_cmd}
        self.execution_time = execution_time
        self.error = error
        self.completed_at = datetime.now(timezone.utc)


class ExecutionOrchestrator:
    """执行编排器
    
    这是步骤2的下半部分：下令执行
    通过 CLI 命令调用各模块，避免跨语言问题
    """
    
    def __init__(self):
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
                
                # 构建 CLI 命令
                cli_cmd = self._build_cli_command(step, plan.plan_id, context)
                
                # 执行命令 (使用 subprocess)
                try:
                    result = subprocess.run(
                        cli_cmd,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=step.estimated_duration
                    )
                    
                    output = {
                        "step_id": step.step_id,
                        "cli_cmd": cli_cmd,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "exit_code": result.returncode
                    }
                    results.append(output)
                    
                    if result.returncode != 0:
                        self.logger.error(f"❌ 步骤 {step.step_id} 执行失败: {result.stderr}")
                        success = False
                        error_msg = result.stderr
                        break  # 失败则停止执行
                    
                    self.logger.info(f"✅ 步骤 {step.step_id} 执行成功")
                    
                except subprocess.TimeoutExpired:
                    self.logger.error(f"⏱️ 步骤 {step.step_id} 超時")
                    success = False
                    error_msg = "Command timeout"
                    break
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
                command_outputs=results,
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
                command_outputs=results,
                error=str(e)
            )
    
    def _build_cli_command(
        self,
        step: ExecutionStep,
        plan_id: str,
        context: Optional[CommandContext]
    ) -> str:
        """构建 CLI 命令（基於 Manifest 架構）
        
        生成符合 aiva_common 規範的 CLI 命令：
        aiva-cli <capability_id> --target <target> [--param value ...]
        """
        # 基本命令結構
        params_json = json.dumps(step.parameters)
        
        # 構建 CLI 命令（使用能力 ID）
        cli_cmd = f"aiva-cli {step.capability} --params '{params_json}'"
        
        # 添加追蹤信息（如果有）
        if context:
            if context.trace_id:
                cli_cmd += f" --trace-id {context.trace_id}"
            if context.session_id:
                cli_cmd += f" --session-id {context.session_id}"
        
        self.logger.debug(f"🔨 构建 CLI 命令: {cli_cmd}")
        self.logger.debug(f"   能力: {step.capability}")
        self.logger.debug(f"   模块: {step.module}")
        
        return cli_cmd
    
    def _check_dependencies(
        self,
        step: ExecutionStep,
        completed_results: List[Dict[str, Any]]
    ) -> bool:
        """检查步骤依赖是否满足（CLI 架構）"""
        if not step.depends_on:
            return True
        
        # 從 CLI 輸出中提取已完成的步驟 ID
        completed_step_ids = [
            result["step_id"]
            for result in completed_results
            if result.get("exit_code") == 0
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
