from __future__ import annotations
import json, time, os, threading

class EventWriter:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.Lock()

    def write(self, ev: dict) -> None:
        ev = {"ts": time.time(), **ev}
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
