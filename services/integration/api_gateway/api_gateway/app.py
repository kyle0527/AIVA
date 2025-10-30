

from datetime import UTC, datetime
import json
from typing import Any

from fastapi import FastAPI

from services.aiva_common.enums import ModuleName, Topic
from services.aiva_common.mq import get_broker
from services.aiva_common.schemas import AivaMessage, MessageHeader, ScanStartPayload
from services.aiva_common.utils import get_logger, new_id

app = FastAPI(title="AIVA UI API Gateway")
logger = get_logger(__name__)


class StartScanRequest(ScanStartPayload):
    pass


@app.get("/api/v1/system/health")
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/api/v1/scans", status_code=202)
async def start_scan(req: StartScanRequest) -> dict[str, Any]:
    broker = await get_broker()
    msg = AivaMessage(
        header=MessageHeader(
            message_id=new_id("msg"),
            trace_id=new_id("trace"),
            correlation_id=req.scan_id,
            source_module=ModuleName.API_GATEWAY,
        ),
        topic=Topic.TASK_SCAN_START,
        payload=req.model_dump(),
    )
    await broker.publish(
        Topic.TASK_SCAN_START, json.dumps(msg.model_dump()).encode("utf-8")
    )
    return {"dispatched": True, "scan_id": req.scan_id}


@app.get("/api/v1/scans")
async def list_scans() -> dict[str, Any]:
    # Placeholder: should query Integration DB
    return {"items": []}


@app.get("/api/v1/scans/{scan_id}")
async def get_scan(scan_id: str) -> dict[str, Any]:
    return {"scan_id": scan_id, "status": "unknown"}


@app.post("/api/v1/scans/{scan_id}/cancel", status_code=202)
async def cancel_scan(scan_id: str) -> dict[str, Any]:
    broker = await get_broker()
    msg = AivaMessage(
        header=MessageHeader(
            message_id=new_id("msg"),
            trace_id=new_id("trace"),
            correlation_id=scan_id,
            source_module=ModuleName.API_GATEWAY,
        ),
        topic=Topic.COMMAND_TASK_CANCEL,
        payload={"scan_id": scan_id},
    )
    await broker.publish(
        Topic.COMMAND_TASK_CANCEL, json.dumps(msg.model_dump()).encode("utf-8")
    )
    return {"cancel_requested": True, "scan_id": scan_id}


@app.post("/api/v1/scans/{scan_id}/pause", status_code=202)
async def pause_scan(scan_id: str) -> dict[str, Any]:
    # Placeholder: would publish pause semantics; here we reuse config update as stub
    broker = await get_broker()
    msg = AivaMessage(
        header=MessageHeader(
            message_id=new_id("msg"),
            trace_id=new_id("trace"),
            correlation_id=scan_id,
            source_module=ModuleName.API_GATEWAY,
        ),
        topic=Topic.CONFIG_GLOBAL_UPDATE,
        payload={
            "update_id": new_id("upd"),
            "config_items": {"pause_scan": scan_id},
        },
    )
    await broker.publish(
        Topic.CONFIG_GLOBAL_UPDATE,
        json.dumps(msg.model_dump()).encode("utf-8"),
    )
    return {"pause_requested": True, "scan_id": scan_id}


@app.get("/api/v1/scans/{scan_id}/findings")
async def scan_findings(scan_id: str) -> dict[str, Any]:
    return {"scan_id": scan_id, "findings": []}


@app.get("/api/v1/findings/{finding_id}")
async def get_finding(finding_id: str) -> dict[str, Any]:
    return {"finding_id": finding_id}


@app.get("/api/v1/assets")
async def list_assets() -> dict[str, Any]:
    return {"items": []}


@app.post("/api/v1/reports", status_code=202)
async def generate_report() -> dict[str, Any]:
    return {"report_id": new_id("rpt")}


@app.get("/api/v1/reports/{report_id}")
async def get_report(report_id: str) -> dict[str, Any]:
    return {"report_id": report_id, "status": "ready"}
