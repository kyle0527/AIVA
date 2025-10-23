"""
Attack Executor - 攻擊執行器

負責執行實際的安全測試攻擊操作
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Union
from enum import Enum

try:
    from services.aiva_common.schemas import (
        AttackPlan,
        AttackStep,
        AttackTarget,
        PlanExecutionResult,
        PlanExecutionMetrics,
        TraceRecord,
    )
    from services.aiva_common.enums import AttackStatus
    _SCHEMAS_AVAILABLE = True
except ImportError:
    # 容錯導入 - 使用 TYPE_CHECKING 避免運行時錯誤
    _SCHEMAS_AVAILABLE = False
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from services.aiva_common.schemas import (
            AttackPlan,
            AttackStep,
            AttackTarget,
            PlanExecutionResult,
            PlanExecutionMetrics,
            TraceRecord,
        )
        from services.aiva_common.enums import AttackStatus

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """執行模式"""
    SAFE = "safe"          # 安全模式 - 僅模擬
    TESTING = "testing"    # 測試模式 - 受控環境
    AGGRESSIVE = "aggressive"  # 激進模式 - 完整測試


class AttackExecutor:
    """
    攻擊執行器
    
    負責執行 AI 生成的攻擊計劃，包括:
    - 攻擊步驟編排
    - 並發執行控制
    - 結果收集
    - 安全檢查
    """
    
    def __init__(
        self,
        mode: ExecutionMode = ExecutionMode.TESTING,
        max_concurrent: int = 5,
        timeout: int = 300,
        safety_enabled: bool = True,
    ):
        """
        初始化攻擊執行器
        
        Args:
            mode: 執行模式
            max_concurrent: 最大並發攻擊數
            timeout: 執行超時時間(秒)
            safety_enabled: 是否啟用安全檢查
        """
        self.mode = mode
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.safety_enabled = safety_enabled
        
        self.active_attacks: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"AttackExecutor initialized: mode={mode}, "
            f"max_concurrent={max_concurrent}, timeout={timeout}s"
        )
    
    async def execute_plan(
        self,
        plan: "Union[AttackPlan, Dict[str, Any]]",
        target: "Union[AttackTarget, Dict[str, Any]]",
    ) -> "Union[PlanExecutionResult, Dict[str, Any]]":
        """
        執行攻擊計劃
        
        Args:
            plan: 攻擊計劃
            target: 攻擊目標
            
        Returns:
            執行結果
        """
        plan_id = plan.plan_id if hasattr(plan, 'plan_id') else plan.get('plan_id', 'unknown')
        
        logger.info(f"開始執行攻擊計劃: {plan_id}")
        
        # 安全檢查
        if self.safety_enabled:
            if not await self._safety_check(target):
                logger.warning(f"安全檢查失敗，中止攻擊: {plan_id}")
                return self._create_aborted_result(plan_id, "Safety check failed")
        
        # 記錄開始時間
        start_time = datetime.now(UTC)
        
        # 執行攻擊步驟
        trace_records = []
        findings = []
        
        try:
            steps = plan.steps if hasattr(plan, 'steps') else plan.get('steps', [])
            
            for step in steps:
                step_result = await self._execute_step(step, target)
                trace_records.append(step_result['trace'])
                
                if step_result.get('findings'):
                    findings.extend(step_result['findings'])
                
                # 檢查是否應該繼續
                if not step_result.get('success') and step_result.get('critical', False):
                    logger.warning(f"關鍵步驟失敗，中止計劃: {plan_id}")
                    break
            
            status = "completed"
            
        except asyncio.TimeoutError:
            logger.error(f"攻擊執行超時: {plan_id}")
            status = "timeout"
        except Exception as e:
            logger.error(f"攻擊執行錯誤: {plan_id}, error={e}")
            status = "failed"
        
        # 計算執行時間
        end_time = datetime.now(UTC)
        execution_time = (end_time - start_time).total_seconds()
        
        # 創建執行結果
        result = {
            "result_id": f"result_{plan_id}_{int(start_time.timestamp())}",
            "plan_id": plan_id,
            "session_id": f"session_{int(start_time.timestamp())}",
            "plan": plan,
            "trace": trace_records,
            "metrics": {
                "total_steps": len(steps) if 'steps' in locals() else 0,
                "successful_steps": sum(1 for t in trace_records if t.get('success')),
                "failed_steps": sum(1 for t in trace_records if not t.get('success')),
                "total_execution_time": execution_time,
                "timestamp": datetime.now(UTC),
            },
            "findings": findings,
            "anomalies": [],
            "recommendations": self._generate_recommendations(trace_records, findings),
            "status": status,
            "completed_at": end_time,
            "metadata": {
                "executor_mode": self.mode.value,
                "safety_enabled": self.safety_enabled,
            },
        }
        
        # 記錄到歷史
        self.execution_history.append({
            "plan_id": plan_id,
            "timestamp": end_time,
            "status": status,
            "execution_time": execution_time,
            "findings_count": len(findings),
        })
        
        logger.info(
            f"攻擊計劃執行完成: {plan_id}, status={status}, "
            f"findings={len(findings)}, time={execution_time:.2f}s"
        )
        
        return result
    
    async def _execute_step(
        self,
        step: "Union[AttackStep, Dict[str, Any]]",
        target: "Union[AttackTarget, Dict[str, Any]]",
    ) -> Dict[str, Any]:
        """執行單個攻擊步驟"""
        
        step_id = step.step_id if hasattr(step, 'step_id') else step.get('step_id', 'unknown')
        action = step.action if hasattr(step, 'action') else step.get('action', 'unknown')
        
        logger.debug(f"執行攻擊步驟: {step_id}, action={action}")
        
        start_time = datetime.now(UTC)
        
        try:
            # 根據執行模式決定實際執行還是模擬
            if self.mode == ExecutionMode.SAFE:
                # 安全模式 - 僅模擬
                result = await self._simulate_step(step, target)
            else:
                # 測試/激進模式 - 實際執行
                result = await self._real_execute_step(step, target)
            
            success = result.get('success', False)
            findings = result.get('findings', [])
            
        except Exception as e:
            logger.error(f"步驟執行失敗: {step_id}, error={e}")
            success = False
            findings = []
            result = {"error": str(e)}
        
        end_time = datetime.now(UTC)
        execution_time = (end_time - start_time).total_seconds()
        
        # 創建追蹤記錄
        trace = {
            "trace_id": f"trace_{step_id}_{int(start_time.timestamp())}",
            "plan_id": "unknown",  # 由上層填充
            "step_id": step_id,
            "action": action,
            "request": step,
            "response": result,
            "success": success,
            "execution_time": execution_time,
            "timestamp": start_time,
            "metadata": {
                "mode": self.mode.value,
            },
        }
        
        return {
            "trace": trace,
            "success": success,
            "findings": findings,
            "critical": step.get('critical', False) if isinstance(step, dict) else False,
        }
    
    async def _simulate_step(
        self,
        step: "Union[AttackStep, Dict[str, Any]]",
        target: "Union[AttackTarget, Dict[str, Any]]",
    ) -> Dict[str, Any]:
        """模擬執行攻擊步驟 (安全模式)"""
        
        # 模擬延遲
        await asyncio.sleep(0.1)
        
        return {
            "success": True,
            "simulated": True,
            "findings": [],
            "message": "Step simulated in safe mode",
        }
    
    async def _real_execute_step(
        self,
        step: "Union[AttackStep, Dict[str, Any]]",
        target: "Union[AttackTarget, Dict[str, Any]]",
    ) -> Dict[str, Any]:
        """實際執行攻擊步驟"""
        
        # TODO: 實現實際的攻擊執行邏輯
        # 這裡應該調用具體的漏洞利用工具
        
        await asyncio.sleep(0.5)  # 模擬執行時間
        
        return {
            "success": True,
            "findings": [],
            "message": "Step executed",
        }
    
    async def _safety_check(
        self,
        target: "Union[AttackTarget, Dict[str, Any]]",
    ) -> bool:
        """執行安全檢查"""
        
        # 檢查目標是否在允許的測試範圍內
        target_url = target.target_url if hasattr(target, 'target_url') else target.get('target_url', '')
        
        # 禁止攻擊的域名/IP
        forbidden_patterns = [
            'google.com',
            'facebook.com',
            '8.8.8.8',
            '1.1.1.1',
        ]
        
        for pattern in forbidden_patterns:
            if pattern in target_url:
                logger.warning(f"安全檢查: 禁止攻擊目標 {target_url}")
                return False
        
        # 檢查是否為測試環境
        test_indicators = ['localhost', '127.0.0.1', '.test', '.local', 'test.']
        is_test_env = any(indicator in target_url for indicator in test_indicators)
        
        if not is_test_env and self.mode == ExecutionMode.AGGRESSIVE:
            logger.warning(f"安全檢查: 非測試環境使用激進模式 {target_url}")
            return False
        
        return True
    
    def _create_aborted_result(self, plan_id: str, reason: str) -> Dict[str, Any]:
        """創建中止結果"""
        return {
            "result_id": f"result_{plan_id}_aborted",
            "plan_id": plan_id,
            "session_id": f"session_aborted_{int(datetime.now(UTC).timestamp())}",
            "status": "aborted",
            "reason": reason,
            "completed_at": datetime.now(UTC),
        }
    
    def _generate_recommendations(
        self,
        trace_records: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
    ) -> List[str]:
        """生成建議"""
        
        recommendations = []
        
        # 基於執行結果生成建議
        failed_steps = [t for t in trace_records if not t.get('success')]
        if failed_steps:
            recommendations.append(
                f"有 {len(failed_steps)} 個步驟執行失敗，建議檢查目標可達性"
            )
        
        if findings:
            recommendations.append(
                f"發現 {len(findings)} 個潛在漏洞，建議進行修復"
            )
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取執行統計"""
        
        total_executions = len(self.execution_history)
        if total_executions == 0:
            return {"total_executions": 0}
        
        successful = sum(
            1 for h in self.execution_history
            if h['status'] == 'completed'
        )
        
        total_findings = sum(
            h['findings_count'] for h in self.execution_history
        )
        
        avg_execution_time = sum(
            h['execution_time'] for h in self.execution_history
        ) / total_executions
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": total_executions - successful,
            "success_rate": successful / total_executions if total_executions > 0 else 0,
            "total_findings": total_findings,
            "avg_execution_time": avg_execution_time,
            "mode": self.mode.value,
        }
