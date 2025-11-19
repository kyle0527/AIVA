"""
Go 引擎 Worker - Phase1 深度掃描
專注於高並發掃描和服務發現
"""

import json
import asyncio
import tempfile
from pathlib import Path
from typing import Sequence

from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    Phase1CompletedPayload,
    Phase1StartPayload,
    Asset,
)
from services.aiva_common.utils import get_logger, new_id

logger = get_logger(__name__)

# 常數定義 (根據 aiva_common README 規範)
WORKER_EXECUTABLE = "worker.exe"
TASK_FILE_SUFFIX = ".json"
HEALTH_CHECK_TIMEOUT = 10
SSRF_SCAN_TIMEOUT = 60
CSPM_SCAN_TIMEOUT = 120
SCA_SCAN_TIMEOUT = 180
MAX_CONCURRENT_SSRF_TARGETS = 10
MAX_CONCURRENT_CSPM_TARGETS = 5
MAX_CONCURRENT_SCA_TARGETS = 3


async def run() -> None:
    """
    Go Worker 主函數
    訂閱 Phase1 任務，執行高並發掃描
    """
    broker = await get_broker()

    logger.info("[Go Worker] Started, subscribing to Phase1 tasks...")

    async for mqmsg in await broker.subscribe(Topic.TASK_SCAN_PHASE1):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            req = Phase1StartPayload(**msg.payload)

            # 檢查是否選擇了 Go 引擎
            if "go" not in req.selected_engines:
                logger.debug(
                    f"[Go] Skipping scan {req.scan_id} - engine not selected"
                )
                continue

            logger.info(
                f"[Go] Processing Phase1 scan: {req.scan_id} "
                f"with {len(req.targets)} targets"
            )

            # 執行 Go 掃描
            payload = await _execute_go_scan(req)

            # 發送結果
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
                f"[Go] Phase1 completed: {req.scan_id}, "
                f"assets: {len(payload.assets)}"
            )

        except Exception as exc:
            logger.exception("[Go] Scan task failed: %s", exc)


async def _execute_go_scan(
    req: Phase1StartPayload,
) -> Phase1CompletedPayload:
    """
    執行 Go 引擎掃描
    
    專注於:
    1. 高並發 URL 掃描 (適合大量 URL)
    2. 端口掃描和服務發現
    3. SSRF (Server-Side Request Forgery) 檢測
    4. CSPM (Cloud Security Posture Management)
    5. SCA (Software Composition Analysis)
    
    Args:
        req: Phase1 請求
        
    Returns:
        Phase1 完成結果
    """
    import time
    
    start_time = time.time()
    logger.info(f"[Go] Starting scan for {req.scan_id} with {len(req.targets)} targets")
    
    assets: list[Asset] = []
    engine_results = {"go": {"status": "success", "scanners_used": []}}
    
    try:
        # 檢查和執行 Go 掃描器
        scan_result = await _execute_go_scanners(req)
        assets = scan_result["assets"]
        engine_results["go"] = scan_result["engine_results"]
        
    except asyncio.TimeoutError:
        logger.error(f"[Go] Scan timeout for {req.scan_id}")
        engine_results["go"]["status"] = "timeout"
    except Exception as exc:
        logger.exception(f"[Go] Scan failed for {req.scan_id}: {exc}")
        engine_results["go"]["status"] = "error"
        engine_results["go"]["error"] = str(exc)
    
    execution_time = time.time() - start_time
    
    return _build_phase1_result(req.scan_id, assets, execution_time, engine_results)


async def _execute_go_scanners(req: Phase1StartPayload) -> dict:
    """執行所有可用的 Go 掃描器"""
    go_engine_path = Path(__file__).parent
    available_scanners = await _check_go_scanners_availability(go_engine_path)
    
    if not available_scanners:
        logger.warning(f"[Go] No Go scanners available for {req.scan_id}")
        return {"assets": [], "engine_results": {"status": "no_scanners_available"}}
    
    logger.info(f"[Go] Available scanners: {list(available_scanners.keys())}")
    
    # 準備掃描任務
    tasks = _prepare_scanner_tasks(req, available_scanners)
    
    # 並行執行掃描器，設置超時
    results = await asyncio.wait_for(
        asyncio.gather(*tasks, return_exceptions=True),
        timeout=300  # 5分鐘超時
    )
    
    # 合併結果
    assets = []
    scanners_used = []
    
    for i, result in enumerate(results):
        scanner_name = ["ssrf", "cspm", "sca"][i] if i < 3 else f"scanner_{i}"
        if isinstance(result, Exception):
            logger.error(f"[Go] {scanner_name} scanner error: {result}")
            continue
        if isinstance(result, list):
            assets.extend(result)
            scanners_used.append(scanner_name)
    
    return {
        "assets": assets,
        "engine_results": {"status": "success", "scanners_used": scanners_used}
    }


def _prepare_scanner_tasks(req: Phase1StartPayload, available_scanners: dict[str, bool]) -> list:
    """準備掃描器任務列表"""
    tasks = []
    targets_str = [str(target) for target in req.targets]  # 轉換 HttpUrl 為 str
    
    if available_scanners.get("ssrf") and _should_run_ssrf_scan(req):
        tasks.append(_call_ssrf_scanner(targets_str, req.scan_id))
    
    if available_scanners.get("cspm") and _should_run_cspm_scan(req):
        tasks.append(_call_cspm_scanner(targets_str, req.scan_id))
    
    if available_scanners.get("sca") and _should_run_sca_scan(req):
        tasks.append(_call_sca_scanner(targets_str, req.scan_id))
    
    return tasks


def _build_phase1_result(
    scan_id: str, 
    assets: list[Asset], 
    execution_time: float, 
    engine_results: dict
) -> Phase1CompletedPayload:
    """建立 Phase1 完成結果"""
    from services.aiva_common.schemas import Summary
    
    return Phase1CompletedPayload(
        scan_id=scan_id,
        status="completed",
        execution_time=execution_time,
        summary=Summary(
            urls_found=sum(1 for a in assets if a.type == "url"),
            forms_found=sum(1 for a in assets if a.type == "form"),
            apis_found=sum(1 for a in assets if a.type == "api"),
            scan_duration_seconds=int(execution_time),  # 轉換為 int
        ),
        fingerprints=None,
        assets=assets,
        engine_results=engine_results,
        phase0_summary=None,
        error_info=None,
    )


async def _check_go_scanners_availability(go_engine_path: Path) -> dict[str, bool]:
    """
    檢查 Go 掃描器的可用性
    
    Args:
        go_engine_path: Go 引擎目錄路徑
        
    Returns:
        掃描器可用性字典 {scanner_name: available}
    """
    scanners = {
        "ssrf": go_engine_path / "ssrf_scanner" / WORKER_EXECUTABLE,
        "cspm": go_engine_path / "cspm_scanner" / WORKER_EXECUTABLE, 
        "sca": go_engine_path / "sca_scanner" / WORKER_EXECUTABLE
    }
    
    availability = {}
    
    for name, executable in scanners.items():
        try:
            # 檢查執行檔是否存在
            if executable.exists():
                # 嘗試執行健康檢查
                proc = await asyncio.create_subprocess_exec(
                    str(executable), "--health-check",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                _, stderr = await asyncio.wait_for(
                    proc.communicate(), 
                    timeout=HEALTH_CHECK_TIMEOUT
                )
                availability[name] = proc.returncode == 0
                
                if proc.returncode != 0:
                    logger.warning(
                        f"[Go] Scanner {name} health check failed: "
                        f"exit={proc.returncode}, stderr={stderr.decode()}"
                    )
            else:
                availability[name] = False
                logger.debug(f"[Go] Scanner {name} executable not found: {executable}")
        except asyncio.TimeoutError:
            availability[name] = False
            logger.warning(f"[Go] Scanner {name} health check timeout")
        except Exception as exc:
            availability[name] = False
            logger.debug(f"[Go] Scanner {name} check failed: {exc}")
    
    return availability


def _should_run_ssrf_scan(req: Phase1StartPayload) -> bool:
    """判斷是否應該執行 SSRF 掃描"""
    # 如果目標包含參數化 URL，適合 SSRF 掃描
    return any("?" in str(target) for target in req.targets)


def _should_run_cspm_scan(req: Phase1StartPayload) -> bool:
    """判斷是否應該執行 CSPM 掃描"""
    # 如果目標是雲服務相關 URL，適合 CSPM 掃描
    cloud_indicators = ["aws", "azure", "gcp", "cloud", "s3", "blob"]
    return any(
        any(indicator in str(target).lower() for indicator in cloud_indicators)
        for target in req.targets
    )


def _should_run_sca_scan(req: Phase1StartPayload) -> bool:
    """判斷是否應該執行 SCA 掃描"""
    # 如果目標是程式碼倉庫 URL，適合 SCA 掃描  
    repo_indicators = ["github.com", "gitlab.com", "bitbucket.org", ".git"]
    return any(
        any(indicator in str(target).lower() for indicator in repo_indicators)
        for target in req.targets
    )


async def _call_ssrf_scanner(targets: list[str], scan_id: str) -> list[Asset]:
    """調用 SSRF 掃描器"""
    import asyncio
    import json
    import tempfile
    from pathlib import Path
    
    try:
        go_engine_path = Path(__file__).parent
        ssrf_scanner = go_engine_path / "ssrf_scanner" / "worker.exe"
        
        if not ssrf_scanner.exists():
            logger.warning(f"[Go] SSRF scanner not found: {ssrf_scanner}")
            return []
        
        # 準備掃描任務
        tasks = []
        for i, target in enumerate(targets[:10]):  # 限制同時掃描的目標數量
            task = {
                "task_id": f"{scan_id}_ssrf_{i}",
                "scan_id": scan_id,
                "target": {"url": target},
                "config": {"timeout": 30}
            }
            tasks.append(task)
        
        # 執行掃描
        assets = []
        for task in tasks:
            # 創建臨時文件傳遞任務
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(task, f)
                task_file = f.name
            
            try:
                # 執行 SSRF 掃描器
                proc = await asyncio.create_subprocess_exec(
                    str(ssrf_scanner), "--task-file", task_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                
                if proc.returncode == 0 and stdout:
                    # 解析掃描結果
                    result = json.loads(stdout.decode())
                    assets.extend(_convert_ssrf_results_to_assets(result, task["target"]["url"]))
                else:
                    logger.warning(
                        f"[Go] SSRF scanner failed for {task['target']['url']}: "
                        f"exit={proc.returncode}, stderr={stderr.decode()[:200]}"
                    )
            
            finally:
                # 清理臨時文件
                try:
                    Path(task_file).unlink()
                except Exception:
                    pass
        
        logger.info(f"[Go] SSRF scan completed: {len(assets)} assets found")
        return assets
    
    except Exception as exc:
        logger.exception(f"[Go] SSRF scanner error: {exc}")
        return []


async def _call_cspm_scanner(targets: list[str], scan_id: str) -> list[Asset]:
    """調用 CSPM 掃描器"""
    try:
        go_engine_path = Path(__file__).parent
        cspm_scanner = go_engine_path / "cspm_scanner" / WORKER_EXECUTABLE
        
        if not cspm_scanner.exists():
            logger.warning(f"[Go] CSPM scanner not found: {cspm_scanner}")
            return []
        
        # 準備掃描任務
        assets = []
        for i, target in enumerate(targets[:MAX_CONCURRENT_CSPM_TARGETS]):
            task = {
                "task_id": f"{scan_id}_cspm_{i}", 
                "scan_id": scan_id,
                "target": {"url": target},
                "config": {
                    "check_public_buckets": True,
                    "check_security_groups": True,
                    "check_iam_policies": True
                }
            }
            
            # 使用 aiofiles 來處理異步檔案操作
            task_file = await _create_async_temp_file(task)
            
            try:
                # 執行 CSPM 掃描器
                proc = await asyncio.create_subprocess_exec(
                    str(cspm_scanner), "--task-file", task_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), 
                    timeout=CSPM_SCAN_TIMEOUT
                )
                
                if proc.returncode == 0 and stdout:
                    # 解析掃描結果
                    result = json.loads(stdout.decode())
                    assets.extend(_convert_cspm_results_to_assets(result, target))
                else:
                    logger.warning(
                        f"[Go] CSPM scanner failed for {target}: "
                        f"exit={proc.returncode}, stderr={stderr.decode()[:200]}"
                    )
            
            finally:
                # 清理臨時文件
                await _cleanup_temp_file(task_file)
        
        logger.info(f"[Go] CSPM scan completed: {len(assets)} assets found")
        return assets
    
    except Exception as exc:
        logger.exception(f"[Go] CSPM scanner error: {exc}")
        return []


async def _call_sca_scanner(targets: list[str], scan_id: str) -> list[Asset]:
    """調用 SCA 掃描器"""
    try:
        go_engine_path = Path(__file__).parent
        sca_scanner = go_engine_path / "sca_scanner" / WORKER_EXECUTABLE
        
        if not sca_scanner.exists():
            logger.warning(f"[Go] SCA scanner not found: {sca_scanner}")
            return []
        
        # 準備掃描任務
        assets = []
        for i, target in enumerate(targets[:MAX_CONCURRENT_SCA_TARGETS]):
            task = {
                "task_id": f"{scan_id}_sca_{i}",
                "scan_id": scan_id,
                "target": {"url": target},
                "config": {
                    "scan_dependencies": True,
                    "check_vulnerabilities": True,
                    "include_dev_deps": False
                }
            }
            
            # 使用 aiofiles 來處理異步檔案操作
            task_file = await _create_async_temp_file(task)
            
            try:
                # 執行 SCA 掃描器
                proc = await asyncio.create_subprocess_exec(
                    str(sca_scanner), "--task-file", task_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), 
                    timeout=SCA_SCAN_TIMEOUT
                )
                
                if proc.returncode == 0 and stdout:
                    # 解析掃描結果
                    result = json.loads(stdout.decode())
                    assets.extend(_convert_sca_results_to_assets(result, target))
                else:
                    logger.warning(
                        f"[Go] SCA scanner failed for {target}: "
                        f"exit={proc.returncode}, stderr={stderr.decode()[:200]}"
                    )
            
            finally:
                # 清理臨時文件
                await _cleanup_temp_file(task_file)
        
        logger.info(f"[Go] SCA scan completed: {len(assets)} assets found")
        return assets
    
    except Exception as exc:
        logger.exception(f"[Go] SCA scanner error: {exc}")
        return []


# ===== 異步檔案輔助函數 =====

async def _create_async_temp_file(task: dict) -> str:
    """創建異步臨時文件"""
    try:
        import aiofiles
        import os
        
        # 創建臨時文件路徑
        temp_dir = tempfile.gettempdir()
        task_file = os.path.join(temp_dir, f"aiva_go_task_{task['task_id']}{TASK_FILE_SUFFIX}")
        
        # 異步寫入文件
        async with aiofiles.open(task_file, mode='w', encoding='utf-8') as f:
            await f.write(json.dumps(task, ensure_ascii=False, indent=2))
        
        return task_file
    except ImportError:
        # fallback to sync if aiofiles not available
        with tempfile.NamedTemporaryFile(mode='w', suffix=TASK_FILE_SUFFIX, delete=False) as f:
            json.dump(task, f)
            return f.name


async def _cleanup_temp_file(task_file: str) -> None:
    """清理臨時文件"""
    try:
        import aiofiles.os
        await aiofiles.os.remove(task_file)
    except ImportError:
        # fallback to sync if aiofiles not available
        try:
            Path(task_file).unlink()
        except FileNotFoundError:
            pass
    except FileNotFoundError:
        pass  # 文件已不存在，無需處理
    except Exception as exc:
        logger.debug(f"[Go] Failed to cleanup temp file {task_file}: {exc}")


# ===== 結果轉換函數 =====

def _convert_ssrf_results_to_assets(result: dict, target_url: str) -> list[Asset]:
    """將 SSRF 掃描結果轉換為 Asset 列表"""
    assets = []
    
    # 模擬 SSRF 掃描結果處理
    findings = result.get("findings", [])
    for i, finding in enumerate(findings):
        if finding.get("severity") in ["HIGH", "CRITICAL"]:
            asset = Asset(
                asset_id=finding.get("id", f"ssrf_{i}"),
                type="vulnerability",
                value=target_url,
                parameters=[finding.get("parameter", "unknown")],
                has_form=False
            )
            assets.append(asset)
    
    return assets


def _convert_cspm_results_to_assets(result: dict, target_url: str) -> list[Asset]:
    """將 CSPM 掃描結果轉換為 Asset 列表"""
    assets = []
    
    # 模擬 CSPM 掃描結果處理
    findings = result.get("findings", [])
    for i, finding in enumerate(findings):
        if finding.get("severity") in ["HIGH", "CRITICAL", "MEDIUM"]:
            asset = Asset(
                asset_id=finding.get("id", f"cspm_{i}"),
                type="misconfiguration",
                value=target_url,
                parameters=[finding.get("rule_id", "unknown")],
                has_form=False
            )
            assets.append(asset)
    
    return assets


def _convert_sca_results_to_assets(result: dict, target_url: str) -> list[Asset]:
    """將 SCA 掃描結果轉換為 Asset 列表"""
    assets = []
    
    # 模擬 SCA 掃描結果處理
    findings = result.get("findings", [])
    for i, finding in enumerate(findings):
        # SCA 主要關注中高風險漏洞
        if finding.get("severity") in ["HIGH", "CRITICAL"]:
            asset = Asset(
                asset_id=finding.get("id", f"sca_{i}"),
                type="dependency_vulnerability", 
                value=target_url,
                parameters=[
                    finding.get("package_name", "unknown"),
                    finding.get("package_version", "unknown")
                ],
                has_form=False
            )
            assets.append(asset)
    
    return assets
