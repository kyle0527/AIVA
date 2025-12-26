from pydantic import BaseModel

class SqliConfig(BaseModel):
    enable_error: bool = True
    enable_boolean: bool = True
    enable_time: bool = True
    enable_union: bool = True
    enable_oob: bool = False

    timeout_seconds: float = 30.0
    max_retries: int = 3
    time_delay_threshold: float = 3.0
    boolean_diff_threshold: float = 0.1
