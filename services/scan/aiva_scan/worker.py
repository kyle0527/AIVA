from __future__ import annotations

import json
import time  # 新增：匯入 time 模組

from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import (
    AivaMessage,
    Asset,
    MessageHeader,
    ScanCompletedPayload,
    ScanStartPayload,
    Summary,
)
from services.aiva_common.utils import get_logger, new_id

from .authentication_manager import AuthenticationManager
from .core_crawling_engine.http_client_hi import HiHttpClient
from .core_crawling_engine.static_content_parser import StaticContentParser
from .core_crawling_engine.url_queue_manager import UrlQueueManager
from .fingerprint_manager import FingerprintCollector
from .header_configuration import HeaderConfiguration

logger = get_logger(__name__)


async def run() -> None:
    broker = await get_broker()
    async for mqmsg in broker.subscribe(Topic.TASK_SCAN_START):
        try:
            msg = AivaMessage.model_validate_json(mqmsg.body)
            req = ScanStartPayload(**msg.payload)
            payload = await _perform_scan(req)
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
        except Exception as exc:  # pragma: no cover
            logger.exception("scan failed: %s", exc)


async def _perform_scan(req: ScanStartPayload) -> ScanCompletedPayload:
    start_time = time.time()  # 新增：記錄開始時間

    # Initialize subcomponents
    # cfg = ConfigControlCenter()
    # scope = ScopeManager(req.scope)
    # strat = StrategyController(req.strategy)
    auth = AuthenticationManager(req.authentication)
    headers = HeaderConfiguration(req.custom_headers)

    urlq = UrlQueueManager([str(t) for t in req.targets])
    http = HiHttpClient(auth, headers)
    static = StaticContentParser()
    fingerprint_collector = FingerprintCollector()  # 使用新的指紋收集器

    assets: list[Asset] = []
    urls_found = 0
    forms_found = 0

    while urlq.has_next():
        url = urlq.next()
        r = await http.get(url)
        if r is None:
            continue
        urls_found += 1
        assets.append(
            Asset(asset_id=new_id("asset"), type="URL", value=url, has_form=False)
        )
        # Passive fingerprinting - 使用模組化的指紋收集器
        await fingerprint_collector.process_response(r)

        # Static parse
        parsed_assets, n_forms = static.extract(url, r)
        assets.extend(parsed_assets)
        forms_found += n_forms

    scan_duration = int(time.time() - start_time)  # 新增：計算掃描時長

    summary = Summary(
        urls_found=urls_found,
        forms_found=forms_found,
        apis_found=0,
        scan_duration_seconds=scan_duration,  # 修正：使用計算出的時長
    )

    return ScanCompletedPayload(
        scan_id=req.scan_id,
        status="completed",
        summary=summary,
        assets=assets,
        # 使用指紋收集器的結果
        fingerprints=fingerprint_collector.get_final_fingerprints(),
    )
