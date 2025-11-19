"""
Rust 引擎 Worker - Phase0 快速偵察 + Phase1 高性能掃描
專注於性能和大規模掃描
"""

import json

from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    Phase0CompletedPayload,
    Phase0StartPayload,
    Phase1CompletedPayload,
    Phase1StartPayload,
    Asset,
)
from services.aiva_common.utils import get_logger, new_id
from pydantic import HttpUrl

logger = get_logger(__name__)


async def run() -> None:
    """
    Rust Worker 主函數
    訂閱 Phase0 和 Phase1 任務，執行高性能掃描
    """
    broker = await get_broker()

    logger.info("[Rust Worker] Started, subscribing to Phase0 and Phase1 tasks...")

    # 訂閱 Phase0 任務
    async for mqmsg in await broker.subscribe(Topic.TASK_SCAN_PHASE0):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            await _handle_phase0(msg, broker)
        except Exception as exc:
            logger.exception("[Rust] Phase0 task failed: %s", exc)
    
    # 訂閱 Phase1 任務
    async for mqmsg in await broker.subscribe(Topic.TASK_SCAN_PHASE1):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            await _handle_phase1(msg, broker)
        except Exception as exc:
            logger.exception("[Rust] Phase1 task failed: %s", exc)


async def _handle_phase0(msg: AivaMessage, broker) -> None:
    """處理 Phase0 快速偵察任務"""
    req = Phase0StartPayload(**msg.payload)
    logger.info(f"[Rust] Processing Phase0 scan: {req.scan_id}")

    payload = _execute_rust_phase0(req)

    out = AivaMessage(
        header=MessageHeader(
            message_id=new_id("msg"),
            trace_id=msg.header.trace_id,
            correlation_id=req.scan_id,
            source_module=ModuleName.SCAN,
        ),
        topic=Topic.RESULTS_SCAN_PHASE0_COMPLETED,
        payload=payload.model_dump(),
    )
    await broker.publish(
        Topic.RESULTS_SCAN_PHASE0_COMPLETED,
        json.dumps(out.model_dump()).encode("utf-8"),
    )

    logger.info(
        f"[Rust] Phase0 completed: {req.scan_id}, "
        f"URLs: {payload.summary.urls_found}, "
        f"Assets: {len(payload.assets)}"
    )


async def _handle_phase1(msg: AivaMessage, broker) -> None:
    """處理 Phase1 深度掃描任務"""
    req = Phase1StartPayload(**msg.payload)

    # 檢查是否選擇了 Rust 引擎
    if "rust" not in req.selected_engines:
        logger.debug(
            f"[Rust] Skipping scan {req.scan_id} - engine not selected"
        )
        return

    logger.info(f"[Rust] Processing Phase1 scan: {req.scan_id}")

    payload = _execute_rust_phase1(req)

    out = AivaMessage(
        header=MessageHeader(
            message_id=new_id("msg"),
            trace_id=msg.header.trace_id,
            correlation_id=req.scan_id,
            source_module=ModuleName.SCAN,
        ),
        topic=Topic.RESULTS_SCAN_COMPLETED,
        payload=payload.model_dump(),
    )
    await broker.publish(
        Topic.RESULTS_SCAN_COMPLETED,
        json.dumps(out.model_dump()).encode("utf-8"),
    )

    logger.info(
        f"[Rust] Phase1 completed: {req.scan_id}, "
        f"assets: {len(payload.assets)}"
    )


def _execute_rust_phase0(
    req: Phase0StartPayload,
) -> Phase0CompletedPayload:
    """
    執行 Rust Phase0 快速偵察
    
    專注於:
    1. 敏感資訊快速掃描 (API Key, Token, 密碼)
    2. 技術棧指紋識別
    3. 基礎端點發現 (深度1)
    4. 初步攻擊面評估
    
    Args:
        req: Phase0 請求
        
    Returns:
        Phase0 完成結果
    """
    logger.info(f"[Rust] Starting Phase0 scan for {req.scan_id}")
    
    # 執行掃描並收集結果
    scan_results = _perform_phase0_scan(req)
    
    # 生成結果摘要
    return _build_phase0_result(req.scan_id, scan_results)


def _perform_phase0_scan(req: Phase0StartPayload) -> dict:
    """執行 Phase0 掃描並收集所有結果"""
    from .python_bridge import rust_info_gatherer
    
    discovered_technologies: list[str] = []
    sensitive_data_found: list[str] = []
    basic_endpoints: list[str] = []
    
    if not rust_info_gatherer.is_available():
        logger.warning(f"[Rust] Scanner not available for {req.scan_id}, using fallback")
        return {
            "technologies": discovered_technologies,
            "sensitive_data": sensitive_data_found,
            "endpoints": basic_endpoints
        }
    
    # 對每個目標進行快速掃描
    for target in req.targets:
        target_results = _scan_single_target_phase0(target, req)
        
        discovered_technologies.extend(target_results["technologies"])
        sensitive_data_found.extend(target_results["sensitive_data"])
        basic_endpoints.extend(target_results["endpoints"])
    
    return {
        "technologies": discovered_technologies,
        "sensitive_data": sensitive_data_found,
        "endpoints": basic_endpoints
    }


def _scan_single_target_phase0(target: HttpUrl, req: Phase0StartPayload) -> dict:
    """掃描單個目標 (Phase0)"""
    from .python_bridge import rust_info_gatherer
    
    target_url = str(target)
    logger.info(f"[Rust] Scanning {target_url}")
    
    technologies = []
    sensitive_data = []
    endpoints = []
    
    try:
        scan_config = {
            "mode": "fast_discovery",
            "timeout": req.timeout,
            "max_depth": req.max_depth,
        }
        
        result = rust_info_gatherer.scan_target(target_url, scan_config)
        
        if result.get("success"):
            scan_data = result.get("results", {})
            
            # 提取各種數據
            technologies = scan_data.get("technologies", [])
            
            for info in scan_data.get("sensitive_info", []):
                sensitive_data.append(f"{info.get('type')}: {info.get('location')}")
            
            endpoints = scan_data.get("endpoints", [])
            
            logger.info(f"[Rust] Completed scan for {target_url}")
        else:
            logger.error(f"[Rust] Scan failed for {target_url}: {result.get('error')}")
    
    except Exception as e:
        logger.error(f"[Rust] Error scanning {target_url}: {e}")
    
    return {
        "technologies": technologies,
        "sensitive_data": sensitive_data,
        "endpoints": endpoints
    }


def _build_phase0_result(scan_id: str, scan_results: dict) -> Phase0CompletedPayload:
    """構建 Phase0 結果"""
    from services.aiva_common.schemas import Summary, Fingerprints
    
    discovered_technologies = scan_results["technologies"]
    sensitive_data_found = scan_results["sensitive_data"]
    basic_endpoints = scan_results["endpoints"]
    
    return Phase0CompletedPayload(
        scan_id=scan_id,
        status="completed",
        execution_time=0.0,
        summary=Summary(
            urls_found=len(basic_endpoints),
            forms_found=0,
            apis_found=0,
            scan_duration_seconds=0,
        ),
        fingerprints=Fingerprints(
            web_server=dict.fromkeys(discovered_technologies[:1], "unknown"),
            framework={},
            language={},
        ) if discovered_technologies else None,
        assets=[],
        recommendations={
            "needs_js_engine": False,
            "needs_form_testing": False,
            "needs_api_testing": False,
            "sensitive_data_detected": len(sensitive_data_found) > 0,
            "high_risk": False,
        },
        error_info=None,
    )


def _execute_rust_phase1(
    req: Phase1StartPayload,
) -> Phase1CompletedPayload:
    """
    執行 Rust Phase1 深度掃描
    
    專注於:
    1. 高性能並發掃描
    2. 大規模目標處理
    3. 快速網路掃描
    4. 資源高效利用
    
    Args:
        req: Phase1 請求
        
    Returns:
        Phase1 完成結果
    """
    logger.info(f"[Rust] Starting Phase1 scan for {req.scan_id}")
    
    # 執行深度掃描並收集資產
    all_assets = _perform_phase1_scan(req)
    
    # 生成結果摘要
    return _build_phase1_result(req.scan_id, all_assets)


def _perform_phase1_scan(req: Phase1StartPayload) -> list[Asset]:
    """執行 Phase1 深度掃描並收集所有資產"""
    from .python_bridge import rust_info_gatherer
    
    all_assets: list[Asset] = []
    
    if not rust_info_gatherer.is_available():
        logger.warning(f"[Rust] Scanner not available for {req.scan_id}, using fallback")
        return all_assets
    
    # 基於Phase0結果進行深度掃描
    for target in req.targets:
        target_assets = _scan_single_target_phase1(target, req)
        all_assets.extend(target_assets)
    
    return all_assets


def _scan_single_target_phase1(target: HttpUrl, req: Phase1StartPayload) -> list[Asset]:
    """掃描單個目標 (Phase1)"""
    from .python_bridge import rust_info_gatherer
    
    target_url = str(target)
    logger.info(f"[Rust] Deep scanning {target_url}")
    
    target_assets: list[Asset] = []
    
    try:
        scan_config = {
            "mode": "deep_analysis",
            "timeout": 1800,
        }
        
        result = rust_info_gatherer.scan_target(target_url, scan_config)
        
        if result.get("success"):
            scan_data = result.get("results", {})
            
            # 將結果轉換為 Asset 格式
            if "assets" in scan_data:
                for idx, asset_data in enumerate(scan_data["assets"]):
                    asset = Asset(
                        asset_id=f"{req.scan_id}_rust_{len(target_assets)}_{idx}",
                        type=asset_data.get("type", "endpoint"),
                        value=asset_data.get("url", target_url),
                        parameters=asset_data.get("parameters", []),
                        has_form=asset_data.get("has_form", False),
                    )
                    target_assets.append(asset)
            
            logger.info(f"[Rust] Completed deep scan for {target_url}, found {len(target_assets)} assets")
        else:
            logger.error(f"[Rust] Deep scan failed for {target_url}: {result.get('error')}")
    
    except Exception as e:
        logger.error(f"[Rust] Error in deep scan for {target_url}: {e}")
    
    return target_assets


def _build_phase1_result(scan_id: str, all_assets: list[Asset]) -> Phase1CompletedPayload:
    """構建 Phase1 結果"""
    from services.aiva_common.schemas import Summary
    
    return Phase1CompletedPayload(
        scan_id=scan_id,
        status="completed",
        execution_time=0.0,
        summary=Summary(
            urls_found=len(all_assets),
            forms_found=sum(1 for a in all_assets if a.has_form),
            apis_found=sum(1 for a in all_assets if a.type == "api"),
            scan_duration_seconds=0,
        ),
        fingerprints=None,
        assets=all_assets,
        engine_results={"rust": {"status": "completed", "assets_found": len(all_assets)}},
        phase0_summary=None,
        error_info=None,
    )


def _call_rust_sensitive_scanner(_targets: list[str]) -> list[str]:
    """
    調用 Rust 敏感資訊掃描器
    
    Args:
        _targets: 目標 URL 列表
        
    Returns:
        發現的敏感資訊列表
    """
    from .python_bridge import rust_info_gatherer
    
    sensitive_findings: list[str] = []
    
    if not rust_info_gatherer.is_available():
        logger.warning("[Rust] Sensitive scanner not available")
        return sensitive_findings
    
    for target in _targets:
        try:
            scan_config = {"mode": "deep_analysis", "focus": "sensitive_data"}
            result = rust_info_gatherer.scan_target(target, scan_config)
            
            if result.get("success"):
                scan_data = result.get("results", {})
                if "sensitive_info" in scan_data:
                    for info in scan_data["sensitive_info"]:
                        sensitive_findings.append(
                            f"{info.get('type')}: {info.get('value')} at {info.get('location')}"
                        )
        except Exception as e:
            logger.error(f"[Rust] Error scanning {target}: {e}")
    
    return sensitive_findings


def _call_rust_fingerprint(_target: str) -> list[str]:
    """
    調用 Rust 指紋識別
    
    Args:
        _target: 目標 URL
        
    Returns:
        識別的技術棧列表
    """
    from .python_bridge import rust_info_gatherer
    
    if not rust_info_gatherer.is_available():
        logger.warning("[Rust] Fingerprint scanner not available")
        return []
    
    try:
        scan_config = {"mode": "fast_discovery", "focus": "fingerprint"}
        result = rust_info_gatherer.scan_target(_target, scan_config)
        
        if result.get("success"):
            scan_data = result.get("results", {})
            return scan_data.get("technologies", [])
    except Exception as e:
        logger.error(f"[Rust] Error fingerprinting {_target}: {e}")
    
    return []
