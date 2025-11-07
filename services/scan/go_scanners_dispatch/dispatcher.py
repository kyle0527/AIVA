"""
Lightweight dispatcher for sending tasks to Go scanners via AMQP.
This module is optional; it does not override existing unified_scan_engine.
"""
from __future__ import annotations
import os, json, asyncio
from typing import Iterable, Dict, Any

try:
    # Prefer the project's broker if available
    from services.core.aiva_core.messaging.message_broker import MessageBroker  # type: ignore
except Exception:
    MessageBroker = None  # Fallback: use local aio-pika style if needed

DEFAULT_AMQP_URL = os.getenv("AIVA_AMQP_URL", "amqp://guest:guest@localhost:5672/")

SCAN_QUEUES = {
    "ssrf_go": os.getenv("SCAN_TASKS_SSRF_GO", "SCAN_TASKS_SSRF_GO"),
    "cspm_go": os.getenv("SCAN_TASKS_CSPM_GO", "SCAN_TASKS_CSPM_GO"),
    "sca_go":  os.getenv("SCAN_TASKS_SCA_GO",  "SCAN_TASKS_SCA_GO"),
}

RESULT_QUEUE = os.getenv("SCAN_RESULTS_QUEUE", "SCAN_RESULTS")

async def dispatch_go_scanners(targets: Iterable[str], config: Dict[str, Any] | None = None) -> None:
    """
    Dispatch a set of targets to all three Go scanners.
    This function only publishes tasks and returns; result collection stays with your existing consumers.
    """
    config = config or {}
    tasks = []
    for url in targets:
        payload = {
            "task_id": config.get("task_id", ""),
            "scan_id": config.get("scan_id", ""),
            "session_id": config.get("session_id", ""),
            "target": {"url": url},
            "config": config.get("scanner_config", {}),
        }
        tasks.append(("ssrf_go", payload))
        tasks.append(("cspm_go", payload))
        tasks.append(("sca_go",  payload))

    if MessageBroker:
        broker = MessageBroker("SCAN")  # module enum string is OK
        for qname, pl in tasks:
            await broker.publish(SCAN_QUEUES[qname], json.dumps(pl).encode("utf-8"))
    else:
        # Minimal fallback publisher using aio-pika
        import aio_pika  # type: ignore
        conn = await aio_pika.connect_robust(DEFAULT_AMQP_URL)
        ch = await conn.channel()
        for qname, pl in tasks:
            q = await ch.declare_queue(SCAN_QUEUES[qname], durable=True)
            await ch.default_exchange.publish(aio_pika.Message(body=json.dumps(pl).encode("utf-8")), routing_key=q.name)
        await conn.close()
