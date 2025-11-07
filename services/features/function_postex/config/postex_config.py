from pydantic import BaseModel

class PostExConfig(BaseModel):
    DEFAULT_SAFE_MODE: bool = True
    ENABLED_CHECKS: dict[str,bool] = {
        "privilege_escalation": True, "lateral_movement": True, "persistence": True
    }
    WHITELISTED_ACCOUNTS: list[str] = []
