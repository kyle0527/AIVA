from __future__ import annotations
import os, json, dataclasses as dc
from typing import Any, Dict
from .schemas import Plan, NodeResult

class StateStore:
    def __init__(self, *, base: str = "data/run") -> None:
        self.base = base
        os.makedirs(self.base, exist_ok=True)

    def init_run_dir(self, run_id: str) -> str:
        d = os.path.join(self.base, run_id)
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, "nodes"), exist_ok=True)
        return d

    def _plan_path(self, run_id: str) -> str:
        return os.path.join(self.base, run_id, "plan.json")

    def _node_path(self, run_id: str, node: str) -> str:
        return os.path.join(self.base, run_id, "nodes", f"{node}.json")

    def _summary_path(self, run_id: str) -> str:
        return os.path.join(self.base, run_id, "summary.json")

    def save_plan(self, plan: Plan) -> None:
        with open(self._plan_path(plan.run_id), "w", encoding="utf-8") as f:
            json.dump(dc.asdict(plan), f, ensure_ascii=False, indent=2)

    def save_result(self, run_id: str, node: str, res: NodeResult) -> None:
        with open(self._node_path(run_id, node), "w", encoding="utf-8") as f:
            json.dump(dc.asdict(res), f, ensure_ascii=False, indent=2)

    def save_summary(self, run_id: str, summary: Dict[str, Any]) -> None:
        with open(self._summary_path(run_id), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def load_summary(self, run_id: str) -> Dict[str, Any]:
        with open(self._summary_path(run_id), "r", encoding="utf-8") as f:
            return json.load(f)
