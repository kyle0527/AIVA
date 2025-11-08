from __future__ import annotations
import os, time, json, asyncio, re
from typing import Any, Dict
from .schemas import NodeResult, Plan, NodeSpec
from .events import EventWriter
from .guard import Guard
from .state import StateStore
from .registry import CapabilityRegistry

_ARTIFACT_RE = re.compile(r"^\$\{([a-zA-Z0-9_]+)\.artifacts\.([a-zA-Z0-9_]+)\}$")

def _resolve_args(args: Dict[str, Any], node_results: Dict[str, NodeResult]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, str):
            m = _ARTIFACT_RE.match(v.strip())
            if m:
                nd, key = m.group(1), m.group(2)
                nr = node_results.get(nd)
                if nr and nr.artifacts and key in nr.artifacts:
                    out[k] = nr.artifacts[key]
                    continue
        out[k] = v
    return out

class Executor:
    def __init__(self, *, base_dir: str = ".", log_dir: str = "logs/aiva_core") -> None:
        self.base_dir = base_dir
        os.makedirs(log_dir, exist_ok=True)
        self.events = EventWriter(os.path.join(log_dir, "events.log"))

    async def run_plan(self, plan: Plan, registry: CapabilityRegistry, state: StateStore, guard: Guard) -> Dict[str, Any]:
        run_dir = state.init_run_dir(plan.run_id)
        self.events.write({"type":"run_started","run_id":plan.run_id,"nodes": [n.id for n in plan.nodes]})
        summary: Dict[str, Any] = {"run_id": plan.run_id, "nodes": {}, "ok": True}
        node_results: Dict[str, NodeResult] = {}

        for nd in plan.nodes:
            state.save_plan(plan)
            if guard.block(nd, plan):
                nr = NodeResult(ok=False, name=nd.id, started_at=time.time(), ended_at=time.time(), blocked=True,
                                error="blocked_by_guard")
                summary["ok"] = False
                summary["nodes"][nd.id] = {"blocked": True, "ok": False}
                self.events.write({"type":"node_blocked","run_id":plan.run_id,"node":nd.id})
                state.save_result(plan.run_id, nd.id, nr)
                node_results[nd.id] = nr
                continue

            # resolve args with previous artifacts like ${index.artifacts.index}
            call_args = _resolve_args(nd.args, node_results)

            att = 0
            while True:
                att += 1
                started = time.time()
                self.events.write({"type":"node_started","run_id":plan.run_id,"node":nd.id,"attempt":att,"args":call_args})
                res = await registry.run(nd.cap, **call_args)
                ended = time.time()
                nr = NodeResult(ok=bool(res.get("ok")), name=nd.id, started_at=started, ended_at=ended,
                                metrics=res.get("metrics", {}),
                                findings=res.get("findings", []),
                                artifacts=res.get("artifacts", {}),
                                error=res.get("error"))
                state.save_result(plan.run_id, nd.id, nr)
                node_results[nd.id] = nr
                self.events.write({"type":"node_finished","run_id":plan.run_id,"node":nd.id,"ok":nr.ok,"error":nr.error})
                summary["nodes"][nd.id] = {"ok": nr.ok, "error": nr.error}
                if nr.ok or att > (nd.retries or plan.policy.retry):
                    if not nr.ok:
                        summary["ok"] = False
                    break
                await asyncio.sleep(0.1)  # backoff
        self.events.write({"type":"run_finished","run_id":plan.run_id,"ok":summary["ok"]})
        state.save_summary(plan.run_id, summary)
        return summary
