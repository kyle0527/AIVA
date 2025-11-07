import importlib
from typing import List, Tuple

try:
    crypto_engine = importlib.import_module("crypto_engine")
except Exception as e:
    crypto_engine = None
    import logging
    logging.getLogger(__name__).error("crypto_engine import failed: %s", e)

def scan_code(code: str) -> List[Tuple[str,str]]:
    if crypto_engine is None:
        raise RuntimeError("Rust crypto_engine not available; build with maturin first.")
    return crypto_engine.scan_crypto_weaknesses(code)
