"""攻擊執行協調器

負責攻擊執行、漏洞檢測和多引擎掃描協調
"""

import logging
from datetime import datetime
from typing import Any
import httpx

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
from services.aiva_common.schemas.tasks import FunctionTaskPayload, FunctionTaskTarget
from services.aiva_common.schemas.findings import FindingPayload, Vulnerability, FindingEvidence, FindingTarget
from services.aiva_common.enums import VulnerabilityType, Severity, Confidence
from services.aiva_common.utils import new_id

logger = logging.getLogger(__name__)


class AttackCoordinator:
    """攻擊執行協調器
    
    協調攻擊執行、漏洞檢測和多引擎掃描
    
    Raises:
        ImportError: 如果必要依賴缺失（MultiEngineCoordinator, AttackExecutor）
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
