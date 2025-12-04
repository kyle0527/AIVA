"""兩階段掃描編排器 - Phase0/Phase1 流程控制

此模組負責編排 Phase0 快速偵察和 Phase1 深度掃描的完整流程,
包括命令發送、結果接收、AI 分析決策、引擎選擇等功能。

流程:
1. 發送 Phase0 命令 (tasks.scan.phase0)
2. 接收 Phase0 結果 (scan.phase0.completed)
3. AI 分析決策是否需要 Phase1
4. 引擎選擇決策樹 (根據 Phase0 結果)
5. 發送 Phase1 命令 (tasks.scan.phase1)
6. 接收 Phase1 結果 (scan.completed)
7. 進入七階段處理流程
"""

import asyncio
import json
from typing import Any
from uuid import uuid4

from services.aiva_common.enums import Topic
from services.aiva_common.mq import AbstractBroker
from services.aiva_common.schemas import (
    Phase0StartPayload,
    Phase0CompletedPayload,
    Phase1StartPayload,
    Phase1CompletedPayload,
)
from services.aiva_common.utils import get_logger

logger = get_logger(__name__)


class TwoPhaseOrchestratorError(Exception):
    """兩階段編排器異常基類"""

    pass


class Phase0TimeoutError(TwoPhaseOrchestratorError):
    """Phase0 超時異常"""

    pass


class Phase1TimeoutError(TwoPhaseOrchestratorError):
    """Phase1 超時異常"""

    pass


class AIDecisionError(TwoPhaseOrchestratorError):
    """AI 決策異常"""

    pass


class TwoPhaseScanOrchestrator:
    """兩階段掃描編排器

    負責控制 Phase0 快速偵察和 Phase1 深度掃描的完整流程,
    包括 AI 決策和引擎選擇邏輯。
    """

    def __init__(self, broker: AbstractBroker):
        """初始化編排器

        Args:
            broker: RabbitMQ 消息代理
        """
        self.broker = broker
        self.phase0_timeout = 600  # 10 分鐘
        self.phase1_timeout = 1800  # 30 分鐘

    async def execute_two_phase_scan(
        self,
        targets: list[str],
        trace_id: str,
        max_depth: int = 3,
        max_urls: int = 1000,
    ) -> Phase1CompletedPayload:
        """執行完整的兩階段掃描流程

        Args:
            targets: 目標列表 (URLs)
            trace_id: 追蹤 ID
            max_depth: Phase1 最大爬取深度
            max_urls: Phase1 最大 URL 數量

        Returns:
            Phase1 完成的結果

        Raises:
            Phase0TimeoutError: Phase0 超時
            Phase1TimeoutError: Phase1 超時
            AIDecisionError: AI 決策失敗
        """
        scan_id = str(uuid4())
        logger.info(
            f"[🚀] Starting two-phase scan for {len(targets)} targets (scan_id={scan_id})"
        )

        # === Phase 0: 快速偵察 (5-10 分鐘) ===
        logger.info(f"[Phase0] Launching quick reconnaissance (scan_id={scan_id})")
        phase0_result = await self._execute_phase0(scan_id, targets, trace_id)

        logger.info(
            f"[Phase0] Completed - URLs: {phase0_result.summary.urls_found}, "
            f"Forms: {phase0_result.summary.forms_found}, "
            f"APIs: {phase0_result.summary.apis_found}, "
            f"Assets: {len(phase0_result.assets)}"
        )

        # === AI 決策: 是否需要 Phase1 ===
        need_phase1, reason = self._analyze_phase0_and_decide(
            scan_id, phase0_result
        )

        if not need_phase1:
            logger.info(
                f"[AI Decision] Phase1 not needed for {scan_id}. Reason: {reason}"
            )
            # 不需要 Phase1，轉換 Phase0 結果為 Phase1 格式返回
            return Phase1CompletedPayload(
                scan_id=scan_id,
                status="completed",
                execution_time=phase0_result.execution_time,
                summary=phase0_result.summary,
                fingerprints=phase0_result.fingerprints,
                assets=phase0_result.assets,
                engine_results={"rust": {"status": "completed"}},
                phase0_summary={
                    "urls": phase0_result.summary.urls_found,
                    "execution_time": phase0_result.execution_time,
                },
                error_info=None,
            )

        logger.info(f"[AI Decision] Phase1 required for {scan_id}. Reason: {reason}")

        # === 引擎選擇決策樹 ===
        selected_engines = self._select_engines_for_phase1(phase0_result)
        logger.info(
            f"[Engine Selection] Selected engines for Phase1: {selected_engines}"
        )

        # === Phase 1: 深度掃描 (10-30 分鐘) ===
        logger.info(
            f"[Phase1] Launching deep scan with {len(selected_engines)} engines (scan_id={scan_id})"
        )
        phase1_result = await self._execute_phase1(
            scan_id,
            targets,
            trace_id,
            phase0_result,
            selected_engines,
            max_depth,
            max_urls,
        )

        logger.info(
            f"[Phase1] Completed - Total assets: {len(phase1_result.assets)}, "
            f"Engines: {list(phase1_result.engine_results.keys())}, "
            f"Status: {phase1_result.status}"
        )

        return phase1_result

    async def _execute_phase0(
        self, scan_id: str, targets: list[str], trace_id: str
    ) -> Phase0CompletedPayload:
        """執行 Phase0 快速偵察

        Args:
            scan_id: 掃描 ID
            targets: 目標列表
            trace_id: 追蹤 ID

        Returns:
            Phase0 完成的結果

        Raises:
            Phase0TimeoutError: 超時異常
        """
        # 構建 Phase0 命令
        from pydantic import HttpUrl
        
        # 轉換 targets 為 HttpUrl 類型
        http_targets = [HttpUrl(t) if not isinstance(t, HttpUrl) else t for t in targets]
        
        phase0_command = Phase0StartPayload(
            scan_id=scan_id,
            targets=http_targets,  # type: ignore
            timeout=self.phase0_timeout,
        )

        # 發送到 tasks.scan.phase0
        message = {
            "trace_id": trace_id,
            "correlation_id": scan_id,
            "payload": phase0_command.model_dump(),
        }

        await self.broker.publish(
            Topic.TASK_SCAN_PHASE0, json.dumps(message).encode("utf-8")
        )
        logger.info(
            f"[Phase0] Command sent to {Topic.TASK_SCAN_PHASE0.value} (scan_id={scan_id})"
        )

        # 接收 Phase0 結果
        logger.info(f"[Phase0] Waiting for results (timeout={self.phase0_timeout}s)...")
        
        timeout_task = None
        try:
            # 訂閱 Phase0 完成消息
            subscription = await self.broker.subscribe(Topic.RESULTS_SCAN_PHASE0_COMPLETED)
            
            # 設置超時
            timeout_task = asyncio.create_task(asyncio.sleep(self.phase0_timeout))
            
            async for msg in subscription:
                # 解析消息
                if isinstance(msg, dict):
                    msg_data = msg
                else:
                    # MQMessage 使用 body 屬性 - 安全類型檢查
                    if hasattr(msg, 'body') and hasattr(msg.body, 'decode'):
                        msg_body = msg.body.decode("utf-8")
                    elif hasattr(msg, 'body'):
                        msg_body = str(msg.body)
                    else:
                        msg_body = str(msg)
                    msg_data = json.loads(msg_body)
                
                # 檢查是否是我們的掃描結果
                if isinstance(msg_data, dict) and msg_data.get("payload", {}).get("scan_id") == scan_id:
                    logger.info(f"[Phase0] Received result for scan_id={scan_id}")
                    
                    # 解析 Phase0CompletedPayload - 使用 Pydantic v2 model_validate
                    payload_data = msg_data.get("payload", {})
                    if isinstance(payload_data, dict):
                        phase0_result = Phase0CompletedPayload.model_validate(payload_data)
                    else:
                        phase0_result = payload_data  # 假設已經是正確的實例
                    
                    logger.info(
                        f"[Phase0] Completed: technologies={len(phase0_result.fingerprints.web_server or {}) if phase0_result.fingerprints else 0}, "
                        f"assets={len(phase0_result.assets)}"
                    )
                    
                    return phase0_result
                    
        except Exception as e:
            logger.error(f"[Phase0] Error or timeout: {e}")
        finally:
            if timeout_task:
                timeout_task.cancel()
        
        # 超時或錯誤返回
        from services.aiva_common.schemas import Summary
        return Phase0CompletedPayload(
            scan_id=scan_id,
            status="timeout",
            execution_time=float(self.phase0_timeout),
            summary=Summary(),
            fingerprints=None,
            assets=[],
            recommendations={},
            error_info="Phase0 scan timeout or error",
        )

    async def _execute_phase1(
        self,
        scan_id: str,
        targets: list[str],
        trace_id: str,
        phase0_result: Phase0CompletedPayload,
        selected_engines: list[str],
        max_depth: int,
        max_urls: int,
    ) -> Phase1CompletedPayload:
        """執行 Phase1 深度掃描

        Args:
            scan_id: 掃描 ID
            targets: 目標列表
            trace_id: 追蹤 ID
            phase0_result: Phase0 結果
            selected_engines: 選中的引擎列表
            max_depth: 最大爬取深度
            max_urls: 最大 URL 數量

        Returns:
            Phase1 完成的結果

        Raises:
            Phase1TimeoutError: 超時異常
        """
        # 構建 Phase1 命令
        # 轉換 targets 為 HttpUrl 類型
        from pydantic import HttpUrl
        http_targets = [HttpUrl(t) if not isinstance(t, HttpUrl) else t for t in targets]
        
        phase1_command = Phase1StartPayload(
            scan_id=scan_id,
            targets=http_targets,  # type: ignore
            phase0_result=phase0_result,
            selected_engines=selected_engines,
            max_depth=max_depth,
            max_pages=max_urls,
            timeout=self.phase1_timeout,
        )

        # 發送到 tasks.scan.phase1
        message = {
            "trace_id": trace_id,
            "correlation_id": scan_id,
            "payload": phase1_command.model_dump(),
        }

        await self.broker.publish(
            Topic.TASK_SCAN_PHASE1, json.dumps(message).encode("utf-8")
        )
        logger.info(
            f"[Phase1] Command sent to {Topic.TASK_SCAN_PHASE1.value} with engines {selected_engines} (scan_id={scan_id})"
        )

        # 接收 Phase1 結果
        logger.info(f"[Phase1] Waiting for results (timeout={self.phase1_timeout}s)...")
        
        timeout_task = None
        try:
            # 訂閱 Phase1 完成消息
            subscription = await self.broker.subscribe(Topic.RESULTS_SCAN_COMPLETED)
            
            # 設置超時
            timeout_task = asyncio.create_task(asyncio.sleep(self.phase1_timeout))
            
            async for msg in subscription:
                # 解析消息
                if isinstance(msg, dict):
                    msg_data = msg
                else:
                    # MQMessage 使用 body 屬性 - 安全類型檢查
                    if hasattr(msg, 'body') and hasattr(msg.body, 'decode'):
                        msg_body = msg.body.decode("utf-8")
                    elif hasattr(msg, 'body'):
                        msg_body = str(msg.body)
                    else:
                        msg_body = str(msg)
                    msg_data = json.loads(msg_body)
                
                # 檢查是否是我們的掃描結果
                if isinstance(msg_data, dict) and msg_data.get("payload", {}).get("scan_id") == scan_id:
                    logger.info(f"[Phase1] Received result for scan_id={scan_id}")
                    
                    # 解析 Phase1CompletedPayload - 使用 Pydantic v2 model_validate
                    payload_data = msg_data.get("payload", {})
                    if isinstance(payload_data, dict):
                        phase1_result = Phase1CompletedPayload.model_validate(payload_data)
                    else:
                        phase1_result = payload_data  # 假設已經是正確的實例
                    
                    logger.info(
                        f"[Phase1] Completed: status={phase1_result.status}, "
                        f"assets={len(phase1_result.assets)}, "
                        f"engines={list(phase1_result.engine_results.keys())}"
                    )
                    
                    return phase1_result
                    
        except Exception as e:
            logger.error(f"[Phase1] Error or timeout: {e}")
        finally:
            if timeout_task:
                timeout_task.cancel()
        
        # 超時或錯誤返回
        from services.aiva_common.schemas import Summary
        return Phase1CompletedPayload(
            scan_id=scan_id,
            status="timeout",
            execution_time=float(self.phase1_timeout),
            summary=Summary(),
            fingerprints=None,
            assets=[],
            engine_results={
                engine: {"status": "timeout", "findings": 0}
                for engine in selected_engines
            },
            phase0_summary={
                "urls": phase0_result.summary.urls_found,
                "execution_time": phase0_result.execution_time,
            },
            error_info="Phase1 scan timeout or error",
        )

    def _analyze_phase0_and_decide(
        self, scan_id: str, phase0_result: Phase0CompletedPayload
    ) -> tuple[bool, str]:
        """AI 分析 Phase0 結果並決策是否需要 Phase1

        決策邏輯:
        1. 發現敏感資料 → 需要 Phase1 (高風險)
        2. 發現多種技術棧 → 需要 Phase1 (複雜目標)
        3. 端點數量超過閾值 → 需要 Phase1 (大型應用)
        4. 攻擊面風險等級 >= medium → 需要 Phase1
        5. 其他情況 → 可選 Phase1 (根據策略)

        Args:
            scan_id: 掃描 ID
            phase0_result: Phase0 結果

        Returns:
            (需要Phase1, 原因)
        """
        logger.info(f"[AI Decision] Analyzing Phase0 results for {scan_id}")

        # 獲取統計資訊
        summary = phase0_result.summary
        fingerprints = phase0_result.fingerprints
        recommendations = phase0_result.recommendations

        # 規則1: 檢查 recommendations (Rust 引擎的建議)
        if recommendations.get("needs_deep_scan", False):
            return True, "Rust engine recommends deep scan"
        
        if recommendations.get("sensitive_data_detected", False):
            return True, "Sensitive data detected by Phase0"

        # 規則2: 多種技術棧 (檢查 fingerprints)
        tech_count = 0
        if fingerprints:
            if fingerprints.framework:
                tech_count += len(fingerprints.framework)
            if fingerprints.language:
                tech_count += len(fingerprints.language)
            if tech_count >= 3:
                return True, f"Complex tech stack: {tech_count} technologies"

        # 規則3: 端點數量閾值 (>20)
        if summary.urls_found > 20:
            return True, f"Large application: {summary.urls_found} URLs found"

        # 規則4: 發現表單或 API (需要深度測試)
        if summary.forms_found > 0 or summary.apis_found > 0:
            return True, f"Forms ({summary.forms_found}) or APIs ({summary.apis_found}) detected"

        # 規則5: WAF 檢測 (需要繞過測試)
        if fingerprints and fingerprints.waf_detected:
            return True, f"WAF detected: {fingerprints.waf_vendor or 'unknown'}"

        # 規則6: 默認策略 - 建議執行 Phase1 (保守策略)
        # 在生產環境可改為更激進的策略 (如 return False)
        return True, "Default strategy: comprehensive scan recommended"

    def _select_engines_for_phase1(
        self, phase0_result: Phase0CompletedPayload
    ) -> list[str]:
        """引擎選擇決策樹 - 根據 Phase0 結果選擇最適合的引擎

        決策邏輯:
        1. 檢測到 JavaScript/TypeScript → 添加 "typescript" 引擎
        2. 檢測到表單或 API → 添加 "python" 引擎
        3. URL 數量大 (>50) → 添加 "go" 引擎 (並發優勢)
        4. 需要快速掃描 → 添加 "rust" 引擎 (性能優勢)
        5. 默認使用 "python" 引擎

        Args:
            scan_id: 掃描 ID
            phase0_result: Phase0 結果

        Returns:
            選中的引擎列表
        """
        logger.info("[Engine Selection] Analyzing Phase0 results")

        selected_engines: list[str] = []
        summary = phase0_result.summary
        fingerprints = phase0_result.fingerprints
        recommendations = phase0_result.recommendations

        # 優先使用 Phase0 的 recommendations
        if recommendations.get("needs_js_engine", False):
            selected_engines.append("typescript")
            logger.info("[Engine Selection] Added 'typescript' - Recommended by Phase0")

        if recommendations.get("needs_form_testing", False):
            if "python" not in selected_engines:
                selected_engines.append("python")
            logger.info("[Engine Selection] Added 'python' - Form testing recommended")

        if recommendations.get("needs_api_testing", False):
            if "python" not in selected_engines:
                selected_engines.append("python")
            logger.info("[Engine Selection] Added 'python' - API testing recommended")

        # 決策樹邏輯 (基於 fingerprints 和 summary)
        # 檢測 JavaScript/TypeScript
        if fingerprints and fingerprints.language:
            if any("javascript" in lang.lower() or "typescript" in lang.lower() 
                   for lang in fingerprints.language.values()):
                if "typescript" not in selected_engines:
                    selected_engines.append("typescript")
                    logger.info("[Engine Selection] Added 'typescript' - JavaScript detected")

        # 檢測表單或 API
        if summary.forms_found > 0 or summary.apis_found > 0:
            if "python" not in selected_engines:
                selected_engines.append("python")
                logger.info(
                    f"[Engine Selection] Added 'python' - Forms ({summary.forms_found}) or APIs ({summary.apis_found}) detected"
                )

        # URL 數量大 (>50) → 使用 Go 並發優勢
        if summary.urls_found > 50:
            selected_engines.append("go")
            logger.info(
                f"[Engine Selection] Added 'go' - Large URL count ({summary.urls_found})"
            )

        # 需要快速掃描 (WAF 或高風險)
        needs_fast_scan = False
        if fingerprints and fingerprints.waf_detected:
            needs_fast_scan = True
        if recommendations.get("high_risk", False):
            needs_fast_scan = True
        
        if needs_fast_scan and "rust" not in selected_engines:
            selected_engines.append("rust")
            logger.info("[Engine Selection] Added 'rust' - Fast scan required")

        # 默認引擎: 如果沒有選擇任何引擎,使用 Python
        if not selected_engines:
            selected_engines.append("python")
            logger.info("[Engine Selection] Added 'python' - Default engine")

        return selected_engines

    # _build_engine_criteria 方法已被移除
    # 引擎選擇邏輯已整合到 _select_engines_for_phase1 中
    # 直接使用 Phase0CompletedPayload 的欄位進行決策
