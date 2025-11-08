from __future__ import annotations
import asyncio, inspect, traceback
from typing import Any, Callable, Dict, Optional

class CapabilityRegistry:
    def __init__(self) -> None:
        self._caps: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, fn: Callable[..., Any], *, desc: str = "", version: str = "v1") -> None:
        self._caps[name] = {"fn": fn, "desc": desc, "version": version}

    def list(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._caps)

    async def run(self, name: str, **kwargs) -> Dict[str, Any]:
        if name not in self._caps:
            return {"ok": False, "error": f"unknown capability: {name}"}
        fn = self._caps[name]["fn"]
        try:
            if inspect.iscoroutinefunction(fn):
                out = await fn(**kwargs)
            else:
                loop = asyncio.get_running_loop()
                out = await loop.run_in_executor(None, lambda: fn(**kwargs))
            if isinstance(out, dict) and "ok" in out:
                return out
            return {"ok": True, "result": out}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}

# optional autoload of external feature modules
def try_autoload_features(registry: CapabilityRegistry) -> None:
    try:
        import pkgutil, importlib
        pkg = importlib.import_module("services.features")
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                mod = importlib.import_module(m.name)
                if hasattr(mod, "register_capabilities"):
                    mod.register_capabilities(registry)
            except Exception:
                # best effort; ignore failures here
                continue
    except Exception:
        pass
