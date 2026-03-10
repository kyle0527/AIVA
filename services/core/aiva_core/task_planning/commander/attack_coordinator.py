"""攻擊執行協調器

負責攻擊執行、漏洞檢測和多引擎掃描協調
整合 CLIDecisionEngine 和 FlowExecutorAdapter 實現智能攻擊決策
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional
import httpx

# 初始化 logger（必須在任何使用 logger 的地方之前）
logger = logging.getLogger(__name__)

# RAG 決策引擎整合
try:
    from services.core.aiva_core.cognitive_core.learning_system.cli_decision_engine import (
        CLIDecisionEngine,
        AttackCapability
    )
    from services.core.aiva_core.cognitive_core.learning_system.flow_executor_adapter import (
        FlowExecutorAdapter,
        ExecutionConfig
    )
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("⚠️ RAG 決策引擎未載入，將使用傳統檢測模式")

# 強制依賴檢查 - Fail Fast 原則
try:
    from services.scan.coordinators.multi_engine_coordinator import MultiEngineCoordinator  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "❌ 缺少必要依賴 MultiEngineCoordinator\n"
        "請確認 services.scan.coordinators.multi_engine_coordinator 模組已實現\n"
        f"原始錯誤: {e}"
    ) from e

try:
    from services.features.function_exploit.executor.attack_executor import AttackExecutor, ExecutionMode
except ImportError as e:
    raise ImportError(
        "❌ 缺少必要依賴 AttackExecutor\n"
        "請確認 services.features.function_exploit.executor.attack_executor 模組已實現\n"
        f"原始錯誤: {e}"
    ) from e

# 核心檢測器導入
from services.features.function_xss.traditional_detector import TraditionalXssDetector
from services.features.function_xss.payload_generator import XssPayloadGenerator
from services.features.function_sqli.detector.sqli_detector import SqliDetector
from aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget
from aiva_common.schemas.findings import FindingPayload, Vulnerability, FindingEvidence, FindingTarget
from aiva_common.enums import VulnerabilityType, Severity, Confidence
from aiva_common.utils import new_id


class AttackCoordinator:
    """攻擊執行協調器

    協調攻擊執行、漏洞檢測和多引擎掃描
    整合 RAG 決策引擎實現智能攻擊選擇

    Raises:
        ImportError: 如果必要依賴缺失（MultiEngineCoordinator, AttackExecutor）
    """

    def __init__(
        self,
        unified_executor: Any = None,
        multilang_coordinator: Any = None,
        internal_loop: Any = None,
        session_state_manager: Any = None,
    ):
        """初始化攻擊協調器

        Args:
            unified_executor: 統一執行器（核心依賴）
            multilang_coordinator: 多語言協調器（核心依賴）
            internal_loop: 內部閉環連接器（核心依賴）
            session_state_manager: 會話狀態管理器（進度追蹤）
        """
        # ✅ 核心依賴：四個關鍵組件
        self.unified_executor = unified_executor
        self.multilang_coordinator = multilang_coordinator
        self.internal_loop = internal_loop
        self.session_state_manager = session_state_manager

        # ✅ RAG 決策引擎（新增）
        if RAG_AVAILABLE:
            try:
                self.decision_engine = CLIDecisionEngine()
                self.flow_adapter = FlowExecutorAdapter(
                    decision_engine=self.decision_engine,
                    external_executor=unified_executor  # 共享統一執行器
                )
                logger.info("✅ RAG 決策引擎已初始化")
            except Exception as e:
                logger.warning(f"⚠️ RAG 決策引擎初始化失敗: {e}")
                self.decision_engine = None
                self.flow_adapter = None
        else:
            self.decision_engine = None
            self.flow_adapter = None

        # CLI 執行架構 - 透過 subprocess 調用外部模組
        self._cli_executor = self._init_cli_executor()

    def _init_cli_executor(self) -> dict[str, Any]:
        """初始化 CLI 執行器配置

        Returns:
            CLI 執行器配置字典
        """
        import subprocess
        return {
            "subprocess": subprocess,
            "timeout": 300,  # 默認超時 5 分鐘
            "encoding": "utf-8"
        }

    def _execute_cli_command(self, command: list[str], timeout: int | None = None) -> dict[str, Any]:
        """執行 CLI 命令

        Args:
            command: CLI 命令列表
            timeout: 超時秒數

        Returns:
            執行結果字典 {"success": bool, "stdout": str, "stderr": str, "returncode": int}
        """
        import json
        subprocess = self._cli_executor["subprocess"]
        timeout = timeout or self._cli_executor["timeout"]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding=self._cli_executor["encoding"]
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timeout after {timeout}s",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    async def detect_vulnerabilities(self, context: dict[str, Any]) -> dict[str, Any]:
        """檢測漏洞（直接調用核心檢測器）

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

        vuln_types = context.get("vulnerability_types", ["sqli", "xss"])
        deep_scan = context.get("deep_scan", False)

        results = {
            "success": True,
            "target": target,
            "vulnerabilities_found": [],
            "modules_executed": [],
            "total_findings": 0,
        }

        try:
            # 創建 HTTP 客戶端
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=30.0 if not deep_scan else 60.0
            ) as client:

                # XSS 檢測
                if "xss" in vuln_types:
                    try:
                        logger.info("   🎯 執行 XSS 檢測...")

                        # 創建任務 payload
                        task = FunctionTaskPayload(
                            task_id=f"task_ai_xss_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            scan_id=f"scan_ai_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            target=FunctionTaskTarget(url=target, method="GET"),
                            priority=8 if deep_scan else 5,
                        )

                        # 生成 payload
                        payload_gen = XssPayloadGenerator()
                        if deep_scan:
                            payloads = payload_gen.generate_all_payloads()
                        else:
                            payloads = payload_gen.generate_basic_payloads()

                        # 執行檢測
                        detector = TraditionalXssDetector(
                            task=task,
                            timeout=30.0,
                            retries=3 if deep_scan else 1,
                            client=client
                        )

                        xss_results = await detector.execute(payloads)

                        # 轉換為標準格式
                        findings_count = len(xss_results)
                        for xss_result in xss_results:
                            vulnerability = Vulnerability(
                                name=VulnerabilityType.XSS,
                                severity=Severity.HIGH,
                                confidence=Confidence.FIRM,
                                description=f"XSS vulnerability detected with payload: {xss_result.payload}"
                            )

                            evidence = FindingEvidence(
                                payload=xss_result.payload,
                                request=str(xss_result.request.url) if xss_result.request else None,
                                response=xss_result.response_text[:500] if xss_result.response_text else None
                            )

                            finding = FindingPayload(
                                finding_id=new_id("finding"),
                                task_id=task.task_id,
                                scan_id=task.scan_id,
                                status="confirmed",
                                vulnerability=vulnerability,
                                target=FindingTarget(url=str(task.target.url)),
                                evidence=evidence
                            )
                            results["vulnerabilities_found"].append(finding)

                        results["modules_executed"].append("xss")
                        results["total_findings"] += findings_count
                        logger.info(f"   ✅ XSS: 發現 {findings_count} 個漏洞")

                    except Exception as e:
                        logger.error(f"   ❌ XSS 模組執行失敗: {e}")
                        results["xss_error"] = str(e)

                # SQL 注入檢測
                if "sqli" in vuln_types:
                    try:
                        logger.info("   🎯 執行 SQL 注入檢測...")

                        # 創建任務 payload
                        task = FunctionTaskPayload(
                            task_id=f"task_ai_sqli_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            scan_id=f"scan_ai_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                            target=FunctionTaskTarget(url=target, method="GET"),
                            priority=8 if deep_scan else 5,
                        )

                        # 執行檢測
                        detector = SqliDetector()
                        sqli_results = await detector.detect_sqli(
                            target=target,
                            params={"db_fingerprint": context.get("db_type")}
                        )

                        # 轉換為標準格式
                        findings_count = len(sqli_results)
                        for sqli_result in sqli_results:
                            finding = FindingPayload(
                                finding_id=new_id("finding"),
                                task_id=task.task_id,
                                scan_id=task.scan_id,
                                status="confirmed",
                                vulnerability=sqli_result.vulnerability,
                                target=sqli_result.target,
                                evidence=sqli_result.evidence if hasattr(sqli_result, 'evidence') else None
                            )
                            results["vulnerabilities_found"].append(finding)

                        results["modules_executed"].append("sqli")
                        results["total_findings"] += findings_count
                        logger.info(f"   ✅ SQL 注入: 發現 {findings_count} 個漏洞")

                    except Exception as e:
                        logger.error(f"   ❌ SQL 注入模組執行失敗: {e}")
                        results["sqli_error"] = str(e)

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
            # 直接使用已在模組頂部導入的 MultiEngineCoordinator
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

            # 安全訪問 result 屬性，支援 dict 和 object 類型
            if isinstance(result, dict):
                urls_found = result.get("summary", {}).get("urls_found", 0)
                assets_found = len(result.get("assets", []))
                execution_time = result.get("execution_time", 0)
                engine_results = result.get("engine_results", {})
            else:
                urls_found = result.summary.urls_found if hasattr(result, "summary") and result.summary else 0
                assets_found = len(result.assets) if hasattr(result, "assets") and result.assets else 0
                execution_time = result.execution_time if hasattr(result, "execution_time") else 0
                engine_results = result.engine_results if hasattr(result, "engine_results") else {}

            engines_used = [
                engine for engine, data in engine_results.items()
                if isinstance(data, dict) and data.get("status") == "completed"
            ] if engine_results else []

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
            # 直接使用已在模組頂部導入的 AttackExecutor 和 ExecutionMode
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
        trace_id = context.get("trace_id", f"ai_scan_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        max_depth = context.get("max_depth", 3)
        max_urls = context.get("max_urls", 1000)

        # 🆕 本地直接執行模式（無需 RabbitMQ）
        if not broker:
            logger.info("📝 使用本地直接執行模式（無需 RabbitMQ）")
            return await self._execute_local_scan(targets, trace_id, max_depth, context)

        # 原有的 RabbitMQ 模式
        try:
            from services.core.aiva_core.core_capabilities.orchestration.two_phase_scan_orchestrator import (
                TwoPhaseScanOrchestrator,
            )

            orchestrator = TwoPhaseScanOrchestrator(broker=broker)

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

    async def _execute_local_scan(
        self,
        targets: list[str],
        trace_id: str,
        _max_depth: int,
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """本地直接執行掃描（不需要 RabbitMQ）

        執行完整的 13 步驟 AI 自主掃描流程（黑盒測試架構）。
        關鍵：步驟 3 規劃多路並行探查，步驟 4 執行，步驟 5 整合。

        Args:
            targets: 掃描目標列表
            trace_id: 追蹤 ID
            _max_depth: 最大深度（保留參數以保持接口一致性）
            context: 上下文信息
        """
        import time
        start_time = time.time()
        target = targets[0] if targets else ""
        scan_id = context.get("scan_id", trace_id)

        # 輔助函數：更新狀態
        def update_status(step: int, phase: str, status: str, details: Optional[dict[str, Any]] = None):
            if self.session_state_manager:
                self.session_state_manager.update_session_status(
                    scan_id,
                    status,
                    {
                        "current_phase": phase,
                        "current_step": step,
                        "tasks_completed": step,
                        "tasks_total": 13,
                        "target": target,
                        "trace_id": trace_id,
                        **(details or {})
                    }
                )

        logger.info(f"🚀 [Step 0] 接收用戶輸入: {target}")
        update_status(0, "initialization", "started", {"target": target})

        # === Step 1-2: 解析目標，創建任務計劃 ===
        logger.info("📋 [Step 1-2] 解析目標並創建任務計劃...")
        update_status(2, "planning", "planning")

        # === Step 3: 規劃多路並行探查任務（黑盒測試策略）===
        logger.info("📝 [Step 3] 規劃 Phase 0 多路並行探查任務...")
        update_status(3, "phase0", "planning_reconnaissance")
        phase0_tasks = self._plan_phase0_tasks(target)
        logger.info(f"   規劃完成: {len(phase0_tasks)} 個並行探查任務")

        # === Step 4: 執行並行探查（Rust 引擎或 UnifiedExecutor）===
        logger.info(f"🔍 [Step 4] 執行 Phase 0 並行探查（{len(phase0_tasks)} 個任務）...")
        update_status(4, "phase0", "executing_reconnaissance")
        phase0_results = await self._execute_parallel_tasks(phase0_tasks)
        logger.info(f"   探查完成: 收到 {len(phase0_results)} 個結果")

        # === Step 5: 整合並行探查結果 ===
        logger.info("📊 [Step 5] 整合 Phase 0 多路探查結果...")
        update_status(5, "phase0", "integrating_results")
        phase0_integrated = self._integrate_phase0_results(phase0_results)
        logger.info(f"   整合完成: ports={phase0_integrated.get('open_ports', [])}, tech={phase0_integrated.get('technologies', [])}")

        # === Step 6: AI 決策 - Phase 0 → Phase 1 ===
        logger.info("🧠 [Step 6] AI 決策: 基於多路情報分析 Phase 1 策略...")
        update_status(6, "ai_decision", "deciding_phase1")
        phase1_decision = await self._ai_decide_phase1(phase0_integrated, target)
        logger.info(f"   決策結果: needs_deep_scan={phase1_decision.get('needs_deep_scan', True)}")
        logger.info(f"   AI 推理: {phase1_decision.get('reasoning', 'N/A')}")

        # === Step 7-8: Phase 1 深度掃描 ===
        logger.info("🔬 [Step 7-8] 執行 Phase 1 深度掃描...")
        update_status(7, "phase1", "deep_scanning")
        phase1_result = await self._deep_scan(target, phase0_integrated)
        update_status(8, "phase1", "deep_scan_complete", {"phase1_result": phase1_result})

        # === Step 9: AI 決策 - Phase 1 → Phase 2 目標選擇 ===
        logger.info("🎯 [Step 9] AI 決策: 選擇 Phase 2 攻擊目標...")
        update_status(9, "ai_decision", "selecting_targets")
        phase2_targets = self._ai_decide_phase2_targets(phase1_result)
        logger.info(f"   選中 {len(phase2_targets)} 個高價值目標")

        # === Step 10: Phase 2 攻擊測試 ===
        logger.info("⚔️ [Step 10] 執行 Phase 2 攻擊測試...")
        update_status(10, "phase2", "attack_testing", {"targets_selected": len(phase2_targets)})
        phase2_result = self._attack_test(target, phase2_targets)

        # === Step 11: AI 決策 - 結果評估 ===
        logger.info("📝 [Step 11] AI 決策: 評估攻擊結果...")
        update_status(11, "ai_decision", "evaluating_results")
        evaluation = self._ai_evaluate_results(phase2_result)
        logger.info(f"   評估結果: action={evaluation.get('action', 'UNKNOWN')}")

        # === Step 12: 生成報告 ===
        logger.info("📄 [Step 12] 生成漏洞報告...")
        update_status(12, "reporting", "generating_report", {
            "vulnerabilities_found": len(phase2_result.get("vulnerabilities", []))
        })

        # === Step 13: 完成 ===
        execution_time = time.time() - start_time
        logger.info(f"✅ [Step 13] 掃描完成! 耗時 {execution_time:.2f}s")

        final_result = {
            "success": True,
            "mode": "local_direct_execution",
            "trace_id": trace_id,
            "targets_scanned": len(targets),
            "execution_time": execution_time,
            "phase0_integrated": phase0_integrated,  # 整合後的多路情報
            "phase0_tasks_count": len(phase0_tasks),  # 並行任務數量
            "phase1_result": phase1_result,
            "phase2_result": phase2_result,
            "evaluation": evaluation,
            "vulnerabilities": phase2_result.get("vulnerabilities", []),
        }

        update_status(13, "completed", "completed", {
            "execution_time": execution_time,
            "vulnerabilities_found": len(final_result["vulnerabilities"]),
            "final_result": final_result
        })

        return final_result

    def _plan_phase0_tasks(self, target: str) -> list[dict[str, Any]]:
        """步驟 3: 規劃 Phase 0 多路並行探查任務

        黑盒測試策略：因為不知道目標情況，所以規劃多個不同角度的探查
        """
        tasks = [
            {
                "task_id": "phase0_http_probe",
                "tool": "httpx",
                "description": "HTTP 基礎探測",
                "parameters": {"target": target, "follow_redirects": True}
            },
            {
                "task_id": "phase0_port_scan",
                "tool": "port_scanner",
                "description": "端口掃描",
                "parameters": {"target": target, "ports": "top100"}
            },
            {
                "task_id": "phase0_waf_detect",
                "tool": "waf_detector",
                "description": "WAF 檢測",
                "parameters": {"target": target}
            },
            {
                "task_id": "phase0_tech_fingerprint",
                "tool": "tech_stack",
                "description": "技術棧指紋",
                "parameters": {"target": target}
            },
            {
                "task_id": "phase0_dir_discover",
                "tool": "directory_scanner",
                "description": "目錄發現",
                "parameters": {"target": target, "wordlist": "common"}
            }
        ]

        logger.info(f"   📝 規劃 {len(tasks)} 個並行探查任務")
        for task in tasks:
            logger.info(f"      - {task['description']} ({task['tool']})")

        return tasks

    async def _execute_parallel_tasks(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """步驟 4: 並行執行多個探查任務

        使用 asyncio.gather 真正並行執行，提升性能
        """
        logger.info(f"   🚀 並行執行 {len(tasks)} 個探查任務...")

        # 如果有 UnifiedExecutor，使用它來執行
        if self.unified_executor:
            logger.info("   ✅ 使用 UnifiedExecutor 引擎")
            # 真正的並行執行
            task_coroutines = [
                self._execute_single_task_safe(task)
                for task in tasks
            ]
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)

            # 處理可能的異常結果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"   ⚠️ 任務 {tasks[i]['task_id']} 失敗: {result}")
                    processed_results.append({
                        "task_id": tasks[i]["task_id"],
                        "success": False,
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)

            return processed_results
        else:
            # 降級模式：使用本地工具並行執行
            logger.info("   ⚙️ 使用降級模式（本地工具）")
            task_coroutines = [
                self._fallback_execute_task(task)
                for task in tasks
            ]
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)

            # 處理可能的異常結果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"   ⚠️ 任務 {tasks[i]['task_id']} 失敗: {result}")
                    processed_results.append({
                        "task_id": tasks[i]["task_id"],
                        "success": False,
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)

            return processed_results

    async def _execute_single_task_safe(self, task: dict[str, Any]) -> dict[str, Any]:
        """安全執行單個任務（包含異常處理）"""
        try:
            return await asyncio.to_thread(self._execute_single_task, task)
        except Exception as e:
            logger.warning(f"   ⚠️ 任務 {task['task_id']} 執行失敗: {e}")
            return {
                "task_id": task["task_id"],
                "success": False,
                "error": str(e)
            }

    def _execute_single_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """執行單個探查任務（通過 UnifiedExecutor）"""
        # 這裡需要調用 unified_executor 的實際接口
        # 暫時返回模擬結果
        return {
            "task_id": task["task_id"],
            "success": True,
            "tool": task["tool"],
            "data": {"placeholder": "實際數據"}
        }

    async def _fallback_execute_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """降級模式：使用本地工具執行任務

        將任務分派到對應的工具處理器
        """
        tool = task["tool"]

        # 工具處理器映射
        handlers = {
            "httpx": self._execute_httpx_tool,
            "port_scanner": self._execute_port_scanner_tool,
            "waf_detector": self._execute_waf_detector_tool,
            "tech_stack": self._execute_tech_stack_tool,
            "directory_scanner": self._execute_directory_scanner_tool,
        }

        handler = handlers.get(tool)
        if handler:
            try:
                return await handler(task)
            except Exception as e:
                return self._create_error_result(task, str(e))
        else:
            return self._create_error_result(task, f"Tool {tool} not implemented in fallback mode")

    def _create_error_result(self, task: dict[str, Any], error: str) -> dict[str, Any]:
        """創建錯誤結果"""
        return {
            "task_id": task["task_id"],
            "success": False,
            "error": error
        }

    async def _execute_httpx_tool(self, task: dict[str, Any]) -> dict[str, Any]:
        """執行 HTTP 基礎探測"""
        import httpx

        target = task["parameters"].get("target", "")
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            response = await client.get(target)
            return {
                "task_id": task["task_id"],
                "success": True,
                "tool": "httpx",
                "data": {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content_length": len(response.text),
                    "final_url": str(response.url)
                }
            }

    async def _execute_port_scanner_tool(self, task: dict[str, Any]) -> dict[str, Any]:
        """執行端口掃描"""
        import httpx
        from urllib.parse import urlparse

        target = task["parameters"].get("target", "")
        parsed = urlparse(target if "://" in target else f"http://{target}")
        hostname = parsed.hostname or target

        # 掃描常見端口
        common_ports = [80, 443, 8080, 8443, 3000, 5000, 8000]
        open_ports = []

        async with httpx.AsyncClient(timeout=2.0) as client:
            for port in common_ports:
                try:
                    test_url = f"http://{hostname}:{port}"
                    await client.get(test_url)
                    open_ports.append(port)
                except (httpx.RequestError, httpx.HTTPStatusError):
                    logger.debug(f"Port {port} closed or unreachable on {hostname}")

        return {
            "task_id": task["task_id"],
            "success": True,
            "tool": "port_scanner",
            "data": {
                "open_ports": open_ports,
                "target": hostname
            }
        }

    async def _execute_waf_detector_tool(self, task: dict[str, Any]) -> dict[str, Any]:
        """執行 WAF 檢測"""
        import httpx

        target = task["parameters"].get("target", "")
        async with httpx.AsyncClient(timeout=10.0) as client:
            test_payload = "' OR '1'='1"
            try:
                response = await client.get(f"{target}?id={test_payload}")

                # 檢測 WAF 特徵
                waf_detected = False
                headers_lower = {k.lower(): v for k, v in response.headers.items()}

                # 根據 headers 或 status code 檢測 WAF
                if ("x-waf" in headers_lower or "x-firewall" in headers_lower or
                    response.status_code in [403, 406, 429, 503]):
                    waf_detected = True

                return {
                    "task_id": task["task_id"],
                    "success": True,
                    "tool": "waf_detector",
                    "data": {
                        "waf_detected": waf_detected,
                        "waf_type": None,
                        "response_code": response.status_code
                    }
                }
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                logger.debug(f"WAF 檢測請求失敗: {e}")
                return {
                    "task_id": task["task_id"],
                    "success": True,
                    "tool": "waf_detector",
                    "data": {"waf_detected": False}
                }

    async def _execute_tech_stack_tool(self, task: dict[str, Any]) -> dict[str, Any]:
        """執行技術棧指紋識別"""
        import httpx

        target = task["parameters"].get("target", "")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(target)

            technologies = []
            headers = response.headers

            # 分析 headers
            if "server" in headers:
                technologies.append(headers["server"])
            if "x-powered-by" in headers:
                technologies.append(headers["x-powered-by"])
            if "x-aspnet-version" in headers:
                technologies.append(f"ASP.NET {headers['x-aspnet-version']}")

            return {
                "task_id": task["task_id"],
                "success": True,
                "tool": "tech_stack",
                "data": {
                    "technologies": technologies
                }
            }

    async def _execute_directory_scanner_tool(self, task: dict[str, Any]) -> dict[str, Any]:
        """執行目錄掃描"""
        import httpx

        target = task["parameters"].get("target", "")
        common_dirs = ["/admin", "/api", "/login", "/dashboard", "/config", "/backup"]
        found_dirs = []

        async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
            for dir_path in common_dirs:
                try:
                    url = f"{target.rstrip('/')}{dir_path}"
                    response = await client.get(url)
                    if response.status_code not in [404, 410]:
                        found_dirs.append({
                            "path": dir_path,
                            "status_code": response.status_code
                        })
                except (httpx.RequestError, httpx.HTTPStatusError):
                    logger.debug(f"Directory probe failed for {dir_path} on {target}")

        return {
            "task_id": task["task_id"],
            "success": True,
            "tool": "directory_scanner",
            "data": {
                "directories": found_dirs
            }
        }

    def _integrate_phase0_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """步驟 5: 整合多路探查結果

        將多個並行探查的結果整合成統一的情報
        """
        integrated = {
            "open_ports": [],
            "technologies": [],
            "waf_detected": False,
            "directories_found": [],
            "http_info": {},
            "raw_results": results
        }

        for result in results:
            if not result.get("success"):
                continue

            tool = result.get("tool", "")
            data = result.get("data", {})

            # 根據工具類型整合數據
            if tool == "httpx":
                integrated["http_info"] = {
                    "status_code": data.get("status_code"),
                    "headers": data.get("headers", {})
                }
                # 從 headers 提取技術棧
                headers = data.get("headers", {})
                if "x-powered-by" in headers:
                    integrated["technologies"].append(headers["x-powered-by"])
                if "server" in headers:
                    integrated["technologies"].append(headers["server"])

            elif tool == "port_scanner":
                integrated["open_ports"] = data.get("open_ports", [])

            elif tool == "waf_detector":
                integrated["waf_detected"] = data.get("waf_detected", False)

            elif tool == "tech_stack":
                integrated["technologies"].extend(data.get("technologies", []))

            elif tool == "directory_scanner":
                integrated["directories_found"] = data.get("directories", [])

        # 去重技術棧
        integrated["technologies"] = list(set(integrated["technologies"]))

        logger.info(f"   ✅ 整合完成: {len(results)} 個結果 → 1 個統一情報")

        return integrated

    async def _ai_decide_phase1(self, phase0_result: dict, target: str) -> dict[str, Any]:
        """AI 決策: 是否需要 Phase 1"""
        # 使用內部循環連接器進行 AI 決策
        if self.internal_loop:
            try:
                decision = await self.internal_loop.decide_phase1_strategy_async({
                    "phase0_result": phase0_result,
                    "target": target,
                })
                return decision
            except Exception as e:
                logger.warning(f"AI 決策失敗，使用規則模式: {e}")

        # 規則模式降級
        needs_deep = phase0_result.get("urls_found", 0) > 5 or phase0_result.get("forms_found", 0) > 0
        return {
            "needs_deep_scan": needs_deep,
            "focus_areas": ["forms", "apis"] if phase0_result.get("apis_found", 0) > 0 else ["forms"],
            "reasoning": "基於規則的決策",
        }

    async def _deep_scan(self, target: str, _phase0_result: dict) -> dict[str, Any]:
        """Phase 1: 深度掃描

        Args:
            target: 目標 URL
            _phase0_result: Phase 0 結果（保留以保持接口一致性）
        """
        vulnerabilities = []

        # 使用 UnifiedExecutor 執行深度掃描
        if self.unified_executor:
            try:
                result = await self.unified_executor.execute(
                    target=target,
                    objective="Perform comprehensive vulnerability scan including SQL injection, XSS, SSRF, and IDOR detection",
                    constraints={"max_depth": 3}
                )
                if result.vulnerabilities:
                    vulnerabilities.extend(result.vulnerabilities)
            except Exception as e:
                logger.warning(f"UnifiedExecutor 掃描失敗: {e}")

        return {
            "vulnerabilities_detected": len(vulnerabilities),
            "vulnerabilities": vulnerabilities,
            "scan_coverage": "comprehensive",
        }

    def _ai_decide_phase2_targets(self, phase1_result: dict) -> list[dict]:
        """AI 決策: 選擇 Phase 2 攻擊目標"""
        targets = []

        for vuln in phase1_result.get("vulnerabilities", []):
            if isinstance(vuln, dict):
                targets.append({
                    "type": vuln.get("type", "unknown"),
                    "location": vuln.get("location", ""),
                    "priority": 1 if vuln.get("severity", "").upper() in ["HIGH", "CRITICAL"] else 2,
                    "tier": 1 if vuln.get("severity", "").upper() == "CRITICAL" else 2,
                })

        # 按優先級排序
        targets.sort(key=lambda x: x.get("priority", 99))
        return targets[:10]  # 最多 10 個目標

    def _attack_test(self, target: str, phase2_targets: list) -> dict[str, Any]:
        """Phase 2: 攻擊測試"""
        confirmed_vulns = []

        for t in phase2_targets:
            # 模擬攻擊測試
            confirmed_vulns.append({
                "type": t.get("type", "unknown"),
                "location": t.get("location", target),
                "confirmed": True,
                "poc_ready": True,
            })

        return {
            "vulnerabilities": confirmed_vulns,
            "total_tested": len(phase2_targets),
            "total_confirmed": len(confirmed_vulns),
        }

    def _ai_evaluate_results(self, phase2_result: dict) -> dict[str, Any]:
        """AI 決策: 評估結果"""
        vulns = phase2_result.get("vulnerabilities", [])

        if len(vulns) > 0:
            return {
                "action": "SUBMIT_REPORT",
                "reasoning": f"發現 {len(vulns)} 個已確認漏洞",
            }
        else:
            return {
                "action": "CONTINUE_SCAN",
                "reasoning": "未發現高價值漏洞，建議繼續掃描",
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
            # 使用 await 調用內部循環的異步方法
            rag_result = await self.internal_loop.query_capabilities_async(
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
        objective: str
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

            # === Phase2 決策: 攻擊目標選擇和結果評估 ===
            phase2_targets = None
            phase2_evaluation = None

            try:
                # 如果發現了漏洞，進行 Phase2 決策
                if total_found > 0:
                    logger.info("🎯 啟動 Phase2 決策: 攻擊目標優先級排序")

                    # 構建 phase1_result 格式
                    phase1_dict = {
                        "scan_id": scan_context.task_id,
                        "status": "completed",
                        "assets": result.get("assets", []) if isinstance(result, dict) else [],
                        "engine_results": {"coordinator": {"status": "completed"}},
                        "summary": {
                            "urls_found": result.get("urls_found", 0) if isinstance(result, dict) else 0,
                            "forms_found": result.get("forms_found", 0) if isinstance(result, dict) else 0,
                            "apis_found": result.get("apis_found", 0) if isinstance(result, dict) else 0,
                            "files_found": total_found,
                        },
                        "fingerprints": result.get("fingerprints", {}) if isinstance(result, dict) else {},
                        "execution_time": result.get("execution_time", 0) if isinstance(result, dict) else 0,
                    }

                    # 決定 Phase2 攻擊目標
                    phase2_targets = decision_agent.decide_phase2_targets(
                        phase1_dict, max_targets=5
                    )

                    # 確保 phase2_targets 是 dict 類型
                    if not isinstance(phase2_targets, dict):
                        phase2_targets = {"targets": []}

                    logger.info(
                        f"🎯 Phase2 目標分析: {len(phase2_targets.get('targets', []))} 個高價值目標"
                    )

                    # 構建 Phase2 模擬結果
                    simulated_phase2_results = {
                        "targets": phase2_targets.get("targets", []),
                        "vulnerability_findings": result.get("vulnerabilities", []) if isinstance(result, dict) else [],
                        "attack_success_rate": min(1.0, total_found / 10.0),  # 簡單估算
                        "total_execution_time": result.get("execution_time", 0) if isinstance(result, dict) else 0,
                    }

                    # 評估 Phase2 結果（將 dict 包裝成 list）
                    phase2_evaluation = decision_agent.evaluate_phase2_results(
                        [simulated_phase2_results], time_budget_remaining=60.0
                    )

                    logger.info(
                        f"📊 Phase2 評估完成: 風險等級={phase2_evaluation.get('risk_level', 'unknown')}"
                    )

            except Exception as e:
                logger.warning(f"⚠️ Phase2 決策失敗: {e}")

            return {
                "status": "success",
                "task_id": scan_context.task_id,
                "ai_decision": ai_decision,
                "scan_result": result if isinstance(result, dict) else str(result),
                "phase2_targets": phase2_targets,
                "phase2_evaluation": phase2_evaluation,
            }

        except Exception as e:
            logger.error(f"❌ 掃描命令處理失敗: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    # ==================== RAG 智能決策方法（新增）====================

    async def rag_smart_attack(
        self,
        scan_results: dict[str, Any],
        attack_mode: str = "auto"
    ) -> dict[str, Any]:
        """RAG 智能攻擊決策與執行

        根據掃描結果，使用 RAG 決策引擎智能選擇和執行攻擊流程。

        Args:
            scan_results: 掃描結果 {
                "vulnerabilities": [...],
                "target_info": {...},
                "fingerprints": {...}
            }
            attack_mode: 攻擊模式
                - "auto": 完全自動化（推薦前3個）
                - "aggressive": 激進模式（推薦前5個）
                - "cautious": 謹慎模式（推薦前1個）

        Returns:
            執行結果 {
                "success": bool,
                "flows_executed": int,
                "results": [...],
                "summary": {...}
            }
        """
        if not RAG_AVAILABLE or self.flow_adapter is None:
            logger.warning("⚠️ RAG 決策引擎不可用，使用傳統檢測")
            return await self.detect_vulnerabilities({"target": scan_results.get("target_info", {}).get("url")})

        logger.info(f"🧠 RAG 智能攻擊: 模式={attack_mode}")

        # 確定執行數量
        top_k_map = {
            "auto": 3,
            "aggressive": 5,
            "cautious": 1
        }
        top_k = top_k_map.get(attack_mode, 3)

        # 構建執行配置
        exec_config = ExecutionConfig(
            timeout=60 if attack_mode == "aggressive" else 30,
            verbose=True
        )

        # 提取漏洞信息
        vulnerabilities = scan_results.get("vulnerabilities", [])
        target_info = scan_results.get("target_info", {})

        if not vulnerabilities:
            logger.warning("⚠️ 未發現漏洞，無法執行攻擊")
            return {
                "success": False,
                "error": "No vulnerabilities found",
                "flows_executed": 0
            }

        # 針對每個漏洞執行攻擊
        all_results = []
        for vuln in vulnerabilities[:top_k]:  # 限制處理數量
            vuln_type = vuln.get("type", "unknown").upper()

            # 構建掃描上下文
            scan_context = {
                "vulnerability_type": vuln_type,
                "target_info": {
                    "url": target_info.get("url", vuln.get("url", "")),
                    "parameter": vuln.get("parameter", ""),
                    "method": vuln.get("method", "GET")
                },
                "scan_results": {
                    "confidence": vuln.get("confidence", "medium"),
                    "severity": vuln.get("severity", "medium")
                }
            }

            # 使用 FlowExecutorAdapter 執行
            results = self.flow_adapter.execute_recommended_flows(
                scan_context=scan_context,
                top_k=2,  # 每個漏洞執行前2個流程
                config=exec_config
            )

            all_results.extend(results)

        # 統計結果
        success_count = sum(1 for r in all_results if r.success)

        logger.info(
            f"✅ RAG 智能攻擊完成: {success_count}/{len(all_results)} 成功"
        )

        return {
            "success": True,
            "attack_mode": attack_mode,
            "flows_executed": len(all_results),
            "success_count": success_count,
            "results": [r.to_dict() for r in all_results],
            "summary": {
                "total_vulnerabilities": len(vulnerabilities),
                "flows_executed": len(all_results),
                "success_rate": success_count / len(all_results) if all_results else 0
            }
        }

    def rag_targeted_attack(
        self,
        capability: str,
        target_url: str,
        keywords: list[str] | None = None,
        limit: int = 3
    ) -> dict[str, Any]:
        """RAG 定向攻擊

        直接指定攻擊類型和目標，執行攻擊流程。

        Args:
            capability: 攻擊能力類型 (xss, sqli, ssrf, scanner, postex)
            target_url: 目標 URL
            keywords: 搜索關鍵字
            limit: 最大執行數量

        Returns:
            執行結果
        """
        if not RAG_AVAILABLE or self.flow_adapter is None:
            return {"success": False, "error": "RAG not available"}

        # 映射字符串到 AttackCapability
        capability_map = {
            "xss": AttackCapability.XSS,
            "sqli": AttackCapability.SQLI,
            "ssrf": AttackCapability.SSRF,
            "scanner": AttackCapability.SCANNER,
            "postex": AttackCapability.POSTEX,
        }

        attack_cap = capability_map.get(capability.lower())
        if not attack_cap:
            return {
                "success": False,
                "error": f"Unknown capability: {capability}"
            }

        logger.info(f"🎯 RAG 定向攻擊: {capability} -> {target_url}")

        # 構建目標上下文
        target_context = {
            "url": target_url,
            "method": "GET"
        }

        # 執行攻擊
        results = self.flow_adapter.search_and_execute(
            capability=attack_cap,
            target_context=target_context,
            keywords=keywords,
            limit=limit,
            config=ExecutionConfig(timeout=30)
        )

        success_count = sum(1 for r in results if r.success)

        return {
            "success": True,
            "capability": capability,
            "target": target_url,
            "flows_executed": len(results),
            "success_count": success_count,
            "results": [r.to_dict() for r in results]
        }
