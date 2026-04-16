from pydantic import BaseModel


class PostExConfig(BaseModel):
    DEFAULT_SAFE_MODE: bool = False  # 預設執行真實檢測
    ENABLED_CHECKS: dict[str,bool] = {
        "privilege_escalation": True, "lateral_movement": True, "persistence": True
    }
    WHITELISTED_ACCOUNTS: list[str] = []
