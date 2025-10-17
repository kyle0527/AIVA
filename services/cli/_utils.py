# 路徑：services/cli/_utils.py
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_SYSTEM = 2
EXIT_BUSINESS_BASE = 10  # 10+ 給業務錯


def load_config_file(path: str | None) -> dict[str, Any]:
    """載入設定檔
    
    Args:
        path: 設定檔路徑（支援 JSON）
        
    Returns:
        設定字典
        
    Raises:
        FileNotFoundError: 檔案不存在
        ValueError: 不支援的檔案格式
    """
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    if p.suffix.lower() in {".json"}:
        return json.loads(p.read_text(encoding="utf-8"))
    # 簡化：若需要 YAML/TOML，可再加
    raise ValueError(f"Unsupported config type: {p.suffix}")


def merge_params(
    flags: dict[str, Any], env_prefix: str, config: dict[str, Any]
) -> dict[str, Any]:
    """合併參數（優先級：flags > ENV > config）
    
    Args:
        flags: 命令列旗標參數
        env_prefix: 環境變數前綴（如 AIVA_）
        config: 設定檔內容
        
    Returns:
        合併後的參數字典
    """
    # 合併順序：flags > ENV > config
    out = dict(config)
    # ENV：以 AIVA_ 作為前綴（例如 AIVA_TARGETS, AIVA_TIMEOUT）
    for k, v in os.environ.items():
        if not k.startswith(env_prefix):
            continue
        out[k[len(env_prefix) :].lower()] = v
    for k, v in flags.items():
        if v is not None:
            out[k] = v
    return out


def echo(data: Any, fmt: str = "human") -> None:
    """輸出資料
    
    Args:
        data: 要輸出的資料
        fmt: 輸出格式（human|json）
    """
    if fmt == "json":
        sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        # human：簡單列印；可換成 rich/table
        if isinstance(data, (dict, list)):
            sys.stdout.write(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            sys.stdout.write(str(data))
    sys.stdout.write("\n")


def get_exit_code(status: str) -> int:
    """根據狀態字串返回對應的退出碼
    
    Args:
        status: 狀態字串（ok, error, usage, system, business）
        
    Returns:
        退出碼
    """
    status_map = {
        "ok": EXIT_OK,
        "success": EXIT_OK,
        "error": EXIT_SYSTEM,
        "usage": EXIT_USAGE,
        "system": EXIT_SYSTEM,
        "business": EXIT_BUSINESS_BASE,
    }
    return status_map.get(status.lower(), EXIT_SYSTEM)
