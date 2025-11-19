"""
TypeScript 引擎 Worker - Phase1 深度掃描
專注於 JavaScript/TypeScript 動態內容掃描
"""

import json
import asyncio
import subprocess
from pathlib import Path
from typing import Any

from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import (
    AivaMessage,
    MessageHeader,
    Phase1CompletedPayload,
    Phase1StartPayload,
    Asset,
    Summary,
)
from services.aiva_common.utils import get_logger, new_id

logger = get_logger(__name__)

# TypeScript 引擎目錄
TYPESCRIPT_ENGINE_DIR = Path(__file__).parent / "src"
NODE_EXECUTABLE = "node"  # 或使用完整路徑


async def run() -> None:
    """
    TypeScript Worker 主函數
    訂閱 Phase1 任務，執行 JavaScript/TypeScript 動態掃描
    """
    broker = await get_broker()

    logger.info("[TypeScript Worker] Started, subscribing to Phase1 tasks...")

    async for mqmsg in await broker.subscribe(Topic.TASK_SCAN_PHASE1):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            req = Phase1StartPayload(**msg.payload)

            # 檢查是否選擇了 TypeScript 引擎
            if "typescript" not in req.selected_engines:
                logger.debug(
                    f"[TypeScript] Skipping scan {req.scan_id} - engine not selected"
                )
                continue

            logger.info(
                f"[TypeScript] Processing Phase1 scan: {req.scan_id} "
                f"with {len(req.targets)} targets"
            )

            # 執行 TypeScript 掃描
            payload = await _execute_typescript_scan(req)

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
                f"[TypeScript] Phase1 completed: {req.scan_id}, "
                f"assets: {len(payload.assets)}"
            )

        except Exception as exc:
            logger.exception("[TypeScript] Scan task failed: %s", exc)


async def _execute_typescript_scan(
    req: Phase1StartPayload,
) -> Phase1CompletedPayload:
    """
    執行 TypeScript 引擎掃描
    
    專注於:
    1. JavaScript 渲染頁面
    2. SPA (Single Page Application) 路由發現
    3. 動態加載內容
    4. AJAX/Fetch API 端點
    5. WebSocket 連接
    
    Args:
        req: Phase1 請求
        
    Returns:
        Phase1 完成結果
    """
    import time
    start_time = time.time()
    
    logger.info(f"[TypeScript] Starting scan for {req.scan_id}")

    try:
        # 確認 TypeScript 掃描器可用性
        if not await _check_typescript_scanner_availability():
            logger.error("[TypeScript] Scanner not available - missing Node.js or build")
            return Phase1CompletedPayload(
                scan_id=req.scan_id,
                status="error",
                execution_time=time.time() - start_time,
                summary=Summary(
                    urls_found=0,
                    forms_found=0,
                    apis_found=0,
                    scan_duration_seconds=0,
                ),
                fingerprints=None,
                assets=[],
                engine_results={
                    "typescript": {
                        "status": "error",
                        "error": "Scanner not available"
                    }
                },
                phase0_summary=None,
                error_info="TypeScript scanner not available",
            )

        # 執行 TypeScript 掃描
        assets = await _launch_typescript_scanner(
            [str(t) for t in req.targets],  # 轉換 HttpUrl 為 str
            req.max_depth or 3,
            req.timeout or 300,
            req.scan_id,
        )

        execution_time = time.time() - start_time

        # 統計資產類型
        urls_found = len([a for a in assets if a.type == "url"])
        forms_found = len([a for a in assets if a.type == "form"])
        apis_found = len([a for a in assets if a.type in ["api", "ajax"]])

        logger.info(
            f"[TypeScript] Scan completed: {req.scan_id}, "
            f"assets: {len(assets)}, time: {execution_time:.2f}s"
        )

        return Phase1CompletedPayload(
            scan_id=req.scan_id,
            status="completed",
            execution_time=execution_time,
            summary=Summary(
                urls_found=urls_found,
                forms_found=forms_found,
                apis_found=apis_found,
                scan_duration_seconds=int(execution_time),
            ),
            fingerprints=None,
            assets=assets,
            engine_results={
                "typescript": {
                    "status": "success",
                    "assets_found": len(assets),
                    "execution_time": execution_time,
                }
            },
            phase0_summary=None,
            error_info=None,
        )

    except Exception as exc:
        logger.exception(f"[TypeScript] Scan failed for {req.scan_id}: %s", exc)
        return Phase1CompletedPayload(
            scan_id=req.scan_id,
            status="error",
            execution_time=time.time() - start_time,
            summary=Summary(
                urls_found=0,
                forms_found=0,
                apis_found=0,
                scan_duration_seconds=0,
            ),
            fingerprints=None,
            assets=[],
            engine_results={
                "typescript": {
                    "status": "error",
                    "error": str(exc)
                }
            },
            phase0_summary=None,
            error_info=str(exc),
        )


async def _check_typescript_scanner_availability() -> bool:
    """
    檢查 TypeScript 掃描器是否可用
    
    Returns:
        是否可用
    """
    try:
        # 檢查 Node.js
        result = await asyncio.create_subprocess_exec(
            NODE_EXECUTABLE,
            "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await result.communicate()
        
        if result.returncode != 0:
            logger.error("[TypeScript] Node.js not found")
            return False
            
        node_version = stdout.decode().strip()
        logger.info(f"[TypeScript] Node.js version: {node_version}")

        # 檢查 dist 目錄
        dist_dir = TYPESCRIPT_ENGINE_DIR.parent / "dist"
        index_js = dist_dir / "index.js"
        
        if not index_js.exists():
            logger.error(f"[TypeScript] Built scanner not found at {index_js}")
            logger.info("[TypeScript] Please run: npm run build")
            return False

        logger.info("[TypeScript] Scanner available")
        return True

    except Exception as exc:
        logger.error(f"[TypeScript] Availability check failed: {exc}")
        return False


async def _launch_typescript_scanner(
    targets: list[str],
    max_depth: int,
    _timeout: int,
    scan_id: str,
) -> list[Asset]:
    """
    啟動 TypeScript 掃描器
    
    Args:
        targets: 目標 URL 列表
        max_depth: 最大爬取深度
        _timeout: 超時時間 (保留供未來使用)
        scan_id: 掃描 ID
        
    Returns:
        發現的資產列表
    """
    import tempfile
    import os
    
    assets: list[Asset] = []

    # 為每個目標啟動獨立掃描
    for target_url in targets:
        try:
            # 準備掃描任務 JSON
            task = {
                "scan_id": scan_id,
                "target_url": target_url,
                "max_depth": max_depth,
                "max_pages": 100,  # 最多掃描 100 頁
                "enable_javascript": True,
            }

            # 寫入臨時文件
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(task, f)
                task_file = f.name

            # 構建 Node.js 命令
            dist_dir = TYPESCRIPT_ENGINE_DIR.parent / "dist"
            
            # 使用環境變數方式
            env = {
                **os.environ,
                "AIVA_SCAN_TASK_FILE": task_file,
            }

            logger.info(f"[TypeScript] Launching scanner for {target_url}")

            # 執行掃描 (使用構建好的 dist/index.js)
            proc = await asyncio.create_subprocess_exec(
                NODE_EXECUTABLE,
                str(dist_dir / "index.js"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                # 使用 asyncio.timeout (Python 3.11+)
                async with asyncio.timeout(300):  # 5分鐘超時
                    stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    logger.error(
                        f"[TypeScript] Scanner failed for {target_url}: "
                        f"{stderr.decode()}"
                    )
                    continue

                # 解析結果
                result = json.loads(stdout.decode())
                target_assets = _parse_typescript_result(result)
                assets.extend(target_assets)

                logger.info(
                    f"[TypeScript] Scanned {target_url}: "
                    f"{len(target_assets)} assets"
                )

            except asyncio.TimeoutError:
                logger.error(f"[TypeScript] Timeout scanning {target_url}")
                proc.kill()
                await proc.wait()

        except Exception as exc:
            logger.error(f"[TypeScript] Failed to scan {target_url}: {exc}")

        finally:
            # 清理臨時文件
            try:
                Path(task_file).unlink()
            except Exception:
                pass

    return assets


def _parse_typescript_result(result: dict[str, Any]) -> list[Asset]:
    """
    解析 TypeScript 掃描器結果
    
    Args:
        result: 掃描器輸出
        
    Returns:
        Asset 列表
    """
    assets: list[Asset] = []

    # 解析掃描結果中的 assets
    raw_assets = result.get("assets", [])

    for raw_asset in raw_assets:
        try:
            asset = Asset(
                asset_id=new_id("asset"),
                type=raw_asset.get("type", "unknown"),
                value=raw_asset.get("value", ""),
                confidence=1.0,
                **raw_asset.get("metadata", {}),
            )
            assets.append(asset)
        except Exception as exc:
            logger.warning(f"[TypeScript] Failed to parse asset: {exc}")

    return assets
