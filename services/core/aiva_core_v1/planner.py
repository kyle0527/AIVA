from __future__ import annotations
import json, uuid
from typing import Any, Dict, List
from .schemas import NodeSpec, Plan, PlanPolicy

def _load_flow(path: str) -> Dict[str, Any]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    # try yaml
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Cannot load flow file {path}: {e}")

def _toposort(nodes: List[NodeSpec]) -> List[NodeSpec]:
    m = {n.id: set(n.needs or []) for n in nodes}
    result: List[NodeSpec] = []
    while m:
        acyclic = [k for k, deps in m.items() if not deps]
        if not acyclic:
            raise RuntimeError("Flow has cycles or unmet dependencies")
        for k in acyclic:
            result.append(next(n for n in nodes if n.id == k))
            del m[k]
        for deps in m.values():
            deps.difference_update(set(acyclic))
    return result

def build_plan(flow_path: str, *, vars: Dict[str, Any] | None = None) -> Plan:
    d = _load_flow(flow_path)
    vs = vars or {}
    nodes: List[NodeSpec] = []
    for nd in d.get("nodes", []):
        # simple ${var} interpolation in args
        args = {}
        for k, v in (nd.get("args") or {}).items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                key = v[2:-1]
                args[k] = vs.get(key, v)
            else:
                args[k] = v
        nodes.append(NodeSpec(
            id=nd["id"],
            cap=nd["cap"],
            needs=nd.get("needs", []),
            args=args,
            timeout_sec=nd.get("timeout"),
            retries=nd.get("retries", 0),
            risk=nd.get("risk", "L0"),
            tags=nd.get("tags", []),
        ))
    policy = d.get("policy", {})
    plan = Plan(run_id=str(uuid.uuid4()), nodes=_toposort(nodes),
                policy=PlanPolicy(retry=policy.get("retry", 0), risk_cap=policy.get("risk_cap", "L0,L1")))
    return plan
