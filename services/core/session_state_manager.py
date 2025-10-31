# core/session_state_manager.py
from datetime import datetime
import threading
from typing import Any


class SessionStateManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: dict[str, dict[str, Any]] = {}

    def create(self, session_id: str, initial: dict[str, Any] | None = None):
        with self._lock:
            self._sessions[session_id] = {
                "created_at": datetime.utcnow(),
                "data": initial or {},
            }
            return self._sessions[session_id]

    def get(self, session_id: str):
        with self._lock:
            return self._sessions.get(session_id)

    def update(self, session_id: str, update: dict[str, Any]):
        with self._lock:
            if session_id not in self._sessions:
                return None
            self._sessions[session_id]["data"].update(update)
            return self._sessions[session_id]

    def delete(self, session_id: str):
        with self._lock:
            return self._sessions.pop(session_id, None)


# module-level singleton
_global_session_manager = SessionStateManager()


def get_session_manager() -> SessionStateManager:
    return _global_session_manager
