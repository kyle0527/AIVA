"""攻擊執行協調器

負責攻擊執行、漏洞檢測和多引擎掃描協調
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class AttackCoordinator:
    """攻擊執行協調器
    
    協調攻擊執行、漏洞檢測和多引擎掃描
    """

    def __init__(
        self,
        unified_executor: Any,
        multilang_coordinator: Any,
        internal_loop: Any,
    ):
        """初始化攻擊協調器
        
        Args:
            unified_executor: 統一執行器
            multilang_coordinator: 多語言協調器
            internal_loop: 內部循環連接器
        """
        self.unified_executor = unified_executor
        self.multilang_coordinator = multilang_coordinator
        self.internal_loop = internal_loop

    async def detect_vulnerabilities(self, context: dict[str, Any]) -> dict[str, Any]:
        """檢測漏洞（調用功能模組）

        Args:
            context: 檢測上下文 {
                "target": str,
                "vulnerability_types": list[str],
                "deep_scan": bool,
            }

        Returns:
            檢測結果
        """
        logger.info("🔍 AI 控制: 開始漏洞檢測...")

        target = context.get("target")
        if not target:
            return {"success": False, "error": "No target specified"}

        vuln_types = context.get("vulnerability_types", ["sqli", "xss", "ssrf", "idor"])
        deep_scan = context.get("deep_scan", False)

        results = {
            "success": True,
            "target": target,
            "vulnerabilities_found": [],
            "modules_executed": [],
            "total_findings": 0,
        }

        try:
            module_map = {
                "sqli": "services.features.function_sqli.worker",
                "xss": "services.features.function_xss.worker",
                "ssrf": "services.features.function_ssrf.worker",
                "idor": "services.features.function_idor.worker",
            }

            for vuln_type in vuln_types:
                if vuln_type not in module_map:
                    logger.warning(f"⚠️ 未知漏洞類型: {vuln_type}")
                    continue

                try:
                    module_path = module_map[vuln_type]
                    module = __import__(module_path, fromlist=["*"])
                    
                    worker_class_name = f"{vuln_type.capitalize()}WorkerService"
                    if vuln_type == "sqli":
                        worker_class_name = "SqliWorkerService"
                    elif vuln_type == "xss":
                        worker_class_name = "XssWorkerService"
                    elif vuln_type == "ssrf":
                        worker_class_name = "SsrfWorkerService"
                    elif vuln_type == "idor":
                        worker_class_name = "IdorWorkerService"
                    
                    worker_class = getattr(module, worker_class_name)
                    worker = worker_class()

                    from services.aiva_common.schemas.tasks import (
                        FunctionTaskPayload,
                        FunctionTaskTarget,
                    )

                    task = FunctionTaskPayload(
                        task_id=f"task_ai_{vuln_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        scan_id=f"scan_ai_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        target=FunctionTaskTarget(url=target, method="GET"),
                        priority=8 if deep_scan else 5,
                    )

                    logger.info(f"   🎯 執行 {vuln_type.upper()} 檢測...")
                    detection_result = await worker.process_task(task)

                    if detection_result:
                        findings_count = len(detection_result.get("findings", []))
                        results["vulnerabilities_found"].extend(
                            detection_result.get("findings", [])
                        )
                        results["modules_executed"].append(vuln_type)
                        results["total_findings"] += findings_count
                        logger.info(f"   ✅ {vuln_type.upper()}: 發現 {findings_count} 個漏洞")

                except Exception as e:
                    logger.error(f"   ❌ {vuln_type.upper()} 模組執行失敗: {e}")
                    results[vuln_type + "_error"] = str(e)

            logger.info(
                f"✅ AI 控制: 漏洞檢測完成 - 共發現 {results['total_findings']} 個漏洞"
            )

        except Exception as e:
            logger.error(f"❌ AI 控制: 漏洞檢測失敗 - {e}", exc_info=True)
            results["success"] = False
            results["error"] = str(e)

        return results

    async def coordinate_multilang(self, context: dict[str, Any]) -> dict[str, Any]:
        """協調掃描引擎（Python/TypeScript/Rust/Go）

        Args:
            context: 協調上下文

        Returns:
            掃描結果
        """
        logger.info("🌐 AI 控制: 協調多引擎掃描...")

        targets = context.get("targets", [])
        if not targets:
            return {"success": False, "error": "No targets specified"}

        strategy = context.get("scan_strategy", "balanced")
        scan_id = context.get("scan_id", f"scan_ai_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        max_depth = context.get("max_depth", 3)

        try:
            try:
                from services.scan.coordinators.multi_engine_coordinator import (  # type: ignore[import-not-found]
                    MultiEngineCoordinator,
                )
            except ImportError:
                logger.error("❌ MultiEngineCoordinator 模組尚未實現")
                return {
                    "success": False,
                    "error": "MultiEngineCoordinator module not available",
                    "scan_id": scan_id,
                }

            coordinator = MultiEngineCoordinator()
            await coordinator.initialize()

            logger.info(f"   🎯 使用策略: {strategy}")
            logger.info(f"   🎯 目標數量: {len(targets)}")

            strategy_methods = {
                "fast": coordinator.execute_strategy_fast,
                "balanced": coordinator.execute_strategy_balanced,
                "comprehensive": coordinator.execute_strategy_comprehensive,
                "aggressive": coordinator.execute_strategy_aggressive,
                "smart": coordinator.execute_strategy_smart,
            }

            if strategy not in strategy_methods:
                logger.warning(f"⚠️ 未知策略 '{strategy}'，使用 'balanced'")
                strategy = "balanced"

            scan_method = strategy_methods[strategy]
            result = await scan_method(
                scan_id=scan_id,
                targets=targets,
                max_depth=max_depth,
            )

            urls_found = result.summary.urls_found if result.summary else 0
            assets_found = len(result.assets) if result.assets else 0
            execution_time = result.execution_time if hasattr(result, "execution_time") else 0
            engines_used = [
                engine for engine, data in result.engine_results.items()
                if data.get("status") == "completed"
            ] if result.engine_results else []

            logger.info(
                f"✅ AI 控制: 掃描完成 - 發現 {urls_found} 個 URL, {assets_found} 個資產, 耗時 {execution_time:.2f}s"
            )

            return {
                "success": True,
                "scan_id": scan_id,
                "strategy_used": strategy,
                "targets_scanned": len(targets),
                "urls_found": urls_found,
                "assets_found": assets_found,
                "execution_time": execution_time,
                "engines_used": engines_used,
                "full_result": result,
            }

        except Exception as e:
            logger.error(f"❌ AI 控制: 多引擎掃描失敗 - {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "scan_id": scan_id,
            }

    async def execute_attack(self, context: dict[str, Any]) -> dict[str, Any]:
        """執行攻擊計畫

        Args:
            context: 執行上下文

        Returns:
            執行結果
        """
        logger.info("⚔️ AI 控制: 執行攻擊計畫...")

        plan = context.get("plan")
        target = context.get("target")

        if not plan or not target:
            return {"success": False, "error": "Missing plan or target"}

        try:
            try:
                from services.features.function_exploit.executor.attack_executor import (
                    AttackExecutor,
                    ExecutionMode,
                )
            except ImportError:
                logger.warning("⚠️ AttackExecutor 模組不可用，使用 unified_executor")
                result = await self.unified_executor.execute(
                    target=target if isinstance(target, str) else target.get("target_url", ""),
                    objective=plan.get("objective", "Execute attack plan") if isinstance(plan, dict) else str(plan),
                )
                return {
                    "success": result.success if hasattr(result, 'success') else False,
                    "result": result,
                }

            mode_str = context.get("mode", "testing").lower()
            mode_map = {
                "safe": ExecutionMode.SAFE,
                "testing": ExecutionMode.TESTING,
                "aggressive": ExecutionMode.AGGRESSIVE,
            }
            mode = mode_map.get(mode_str, ExecutionMode.TESTING)

            executor = AttackExecutor(
                mode=mode,
                max_concurrent=context.get("max_concurrent", 5),
                timeout=context.get("timeout", 300),
                safety_enabled=context.get("safety_enabled", True),
            )

            ai_analysis = context.get("ai_analysis")

            logger.info(f"   🎯 執行模式: {mode.value}")
            logger.info(f"   🎯 安全檢查: {'啟用' if executor.safety_enabled else '禁用'}")

            execution_result = await executor.execute_plan_with_ai_analysis(
                plan=plan,
                target=target,
                ai_analysis_results=ai_analysis,
            )

            # 提取執行結果 - 支持多種返回類型
            if isinstance(execution_result, dict):
                success = execution_result.get("success", False)
                steps_completed = execution_result.get("steps_completed", 0)
                steps_failed = execution_result.get("steps_failed", 0)
            elif hasattr(execution_result, "status"):
                # PlanExecutionResult 類型
                success = execution_result.status == "completed"  # type: ignore[union-attr]
                steps_completed = len(execution_result.trace) if hasattr(execution_result, "trace") else 0  # type: ignore[union-attr]
                steps_failed = 1 if execution_result.status == "failed" else 0  # type: ignore[union-attr]
            else:
                # 其他對象類型（可能有 success 屬性）
                success = getattr(execution_result, "success", False)
                steps_completed = getattr(execution_result, "steps_completed", 0)
                steps_failed = getattr(execution_result, "steps_failed", 0)

            logger.info(
                f"✅ AI 控制: 攻擊執行完成 - 成功: {success}, "
                f"完成步驟: {steps_completed}, 失敗步驟: {steps_failed}"
            )

            return {
                "success": success,
                "mode": mode.value,
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "execution_result": execution_result
                if isinstance(execution_result, dict)
                else execution_result.__dict__,
            }

        except Exception as e:
            logger.error(f"❌ AI 控制: 攻擊執行失敗 - {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def execute_two_phase_scan(self, context: dict[str, Any]) -> dict[str, Any]:
        """執行兩階段掃描

        Args:
            context: 掃描上下文

        Returns:
            掃描結果
        """
        logger.info("🔍 AI 控制: 執行兩階段掃描...")

        targets = context.get("targets", [])
        if not targets:
            return {"success": False, "error": "No targets specified"}

        broker = context.get("broker")
        if not broker:
            return {
                "success": False,
                "error": "RabbitMQ broker is required for two-phase scan",
            }

        try:
            from services.core.aiva_core.core_capabilities.orchestration.two_phase_scan_orchestrator import (
                TwoPhaseScanOrchestrator,
            )

            orchestrator = TwoPhaseScanOrchestrator(broker=broker)

            trace_id = context.get("trace_id", f"ai_scan_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            max_depth = context.get("max_depth", 3)
            max_urls = context.get("max_urls", 1000)

            logger.info(f"   🎯 目標數量: {len(targets)}")
            logger.info(f"   🎯 最大深度: {max_depth}")
            logger.info(f"   🎯 最大 URL: {max_urls}")

            result = await orchestrator.execute_two_phase_scan(
                targets=targets,
                trace_id=trace_id,
                max_depth=max_depth,
                max_urls=max_urls,
            )

            urls_found = result.summary.urls_found if hasattr(result, "summary") else 0
            assets_found = len(result.assets) if hasattr(result, "assets") else 0
            execution_time = result.execution_time if hasattr(result, "execution_time") else 0

            logger.info(
                f"✅ AI 控制: 兩階段掃描完成 - 發現 {urls_found} 個 URL, "
                f"{assets_found} 個資產, 耗時 {execution_time:.2f}s"
            )

            return {
                "success": True,
                "trace_id": trace_id,
                "targets_scanned": len(targets),
                "urls_found": urls_found,
                "assets_found": assets_found,
                "execution_time": execution_time,
                "phase0_summary": result.phase0_summary if hasattr(result, "phase0_summary") else {},
                "full_result": result.__dict__ if hasattr(result, "__dict__") else result,
            }

        except Exception as e:
            logger.error(f"❌ AI 控制: 兩階段掃描失敗 - {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def query_capabilities(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5
    ) -> dict[str, Any]:
        """查詢自身能力

        Args:
            query: 查詢文本
            filters: 過濾條件
            top_k: 返回結果數量

        Returns:
            包含匹配能力的結果字典
        """
        logger.info(f"🧠 AI Commander querying self capabilities: '{query}'")
        
        try:
            rag_result = self.internal_loop.query_capabilities(
                query=query,
                filters=filters,
                top_k=top_k
            )
            
            capabilities = []
            for result in rag_result.results:
                cap_data = result.get('metadata', {})
                capabilities.append({
                    "name": cap_data.get('name', 'unknown'),
                    "description": cap_data.get('description', ''),
                    "module": cap_data.get('module', 'unknown'),
                    "category": cap_data.get('category', 'unknown'),
                    "file_path": cap_data.get('file_path', ''),
                    "relevance_score": result.get('score', 0.0)
                })
            
            logger.info(f"✅ Found {len(capabilities)} matching capabilities")
            
            return {
                "success": True,
                "query": query,
                "total_found": rag_result.total_found,
                "capabilities": capabilities,
                "timestamp": rag_result.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to query capabilities: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "total_found": 0,
                "capabilities": []
            }

    async def unified_attack(
        self,
        target: str,
        objective: str,
        user_input: str | None = None
    ) -> dict[str, Any]:
        """統一攻擊執行接口

        Args:
            target: 目標 URL/IP
            objective: 攻擊目標描述
            user_input: 原始用戶輸入

        Returns:
            執行結果
        """
        logger.info(f"🚀 Unified Attack: {target} - {objective}")
        
        try:
            result = await self.unified_executor.execute(
                target=target,
                objective=objective,
                scenario=None,
                constraints=None
            )
            
            return {
                "success": result.success,
                "vulnerabilities": result.vulnerabilities,
                "learning_info": result.learning_info,
                "execution_details": result.execution_details
            }
        except Exception as e:
            logger.error(f"Unified attack failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "vulnerabilities": []
            }

    async def process_scan_command(self, user_input: str) -> dict[str, Any]:
        """處理用戶掃描命令

        Args:
            user_input: 用戶自然語言輸入

        Returns:
            掃描結果
        """
        try:
            from ...core_capabilities.task_context import parse_user_input_to_context
            scan_context = parse_user_input_to_context(user_input)
            
            logger.info(f"📋 已解析任務參數: target={scan_context.target}, "
                       f"intent={scan_context.intent}")
            
            from ...cognitive_core.decision.enhanced_decision_agent import EnhancedDecisionAgent
            decision_agent = EnhancedDecisionAgent()
            ai_decision = decision_agent.decide_scan_strategy(scan_context)
            
            scan_context.ai_decision.selected_tool = ai_decision["selected_tool"]
            scan_context.ai_decision.confidence_score = ai_decision["confidence"]
            scan_context.ai_decision.reasoning = ai_decision["reasoning"]
            
            logger.info(f"🎯 AI 決策: {ai_decision['selected_tool']} "
                       f"(信心度 {ai_decision['confidence']:.2f})")
            
            result = await self.unified_executor.execute(
                target=scan_context.target,
                objective=scan_context.intent,
            )
            
            total_found = 0
            if hasattr(result, 'vulnerabilities'):
                total_found = len(result.vulnerabilities)
            elif isinstance(result, dict):
                total_found = result.get("findings_count", 0)
            
            logger.info(f"✅ 掃描完成: {total_found} 個發現")
            
            return {
                "status": "success",
                "task_id": scan_context.task_id,
                "ai_decision": ai_decision,
                "scan_result": result if isinstance(result, dict) else str(result)
            }
        
        except Exception as e:
            logger.error(f"❌ 掃描命令處理失敗: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
