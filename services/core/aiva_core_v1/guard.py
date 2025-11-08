from __future__ import annotations
import os
from .schemas import NodeSpec, Plan

class Guard:
    # very simple gate: block risky ops unless explicitly allowed
    def __init__(self) -> None:
        # comma separated allowed risk levels, e.g., "L0,L1,L2"
        self.allowed = set((os.getenv("AIVA_ALLOWED_RISK") or "L0,L1").split(","))

    def block(self, node: NodeSpec, plan: Plan) -> bool:
        # block if node risk not allowed
        risk = node.risk or "L0"
        if risk not in self.allowed:
            return True
        # optionally: block tags like "attack" unless AIVA_ALLOW_ATTACK=1
        if "attack" in (node.tags or []):
            if os.getenv("AIVA_ALLOW_ATTACK") != "1":
                return True
        return False
