from __future__ import annotations
import dataclasses as dc
from typing import Any, Dict, List, Optional

@dc.dataclass
class NodeSpec:
    id: str
    cap: str
    needs: List[str] = dc.field(default_factory=list)
    args: Dict[str, Any] = dc.field(default_factory=dict)
    timeout_sec: Optional[int] = None
    retries: int = 0
    risk: str = "L0"
    tags: List[str] = dc.field(default_factory=list)

@dc.dataclass
class PlanPolicy:
    retry: int = 0
    risk_cap: str = "L0,L1"  # L0=安全、L1=低風險、L2=中風險、L3=高風險

@dc.dataclass
class Plan:
    run_id: str
    nodes: List[NodeSpec]
    policy: PlanPolicy

@dc.dataclass
class NodeResult:
    ok: bool
    name: str
    started_at: float
    ended_at: float
    blocked: bool = False
    metrics: Dict[str, Any] = dc.field(default_factory=dict)
    findings: List[Dict[str, Any]] = dc.field(default_factory=list)
    artifacts: Dict[str, Any] = dc.field(default_factory=dict)
    error: Optional[str] = None
