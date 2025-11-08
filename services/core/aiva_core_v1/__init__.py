from .schemas import NodeSpec, Plan, PlanPolicy, NodeResult
from .registry import CapabilityRegistry, try_autoload_features
from .planner import build_plan
from .executor import Executor
from .state import StateStore
from .guard import Guard

class AivaCore:
    def __init__(self) -> None:
        self.registry = CapabilityRegistry()
        self.state = StateStore()
        self.guard = Guard()
        self.executor = Executor()
        # register builtins
        from .capabilities.builtin import register_builtins
        register_builtins(self.registry)
        # try autoload external features
        try_autoload_features(self.registry)

    def list_caps(self):
        return self.registry.list()

    def plan(self, flow_path: str, **vars):
        return build_plan(flow_path, vars=vars)

    async def exec(self, plan: Plan):
        return await self.executor.run_plan(plan, self.registry, self.state, self.guard)
