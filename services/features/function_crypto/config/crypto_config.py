from pydantic import BaseModel
from typing import List

class CryptoConfig(BaseModel):
    WEAK_HASH_ALGOS: List[str] = ["MD5", "SHA1"]
    WEAK_CIPHERS: List[str] = ["DES", "RC4", "ECB"]
    MIN_TLS_VERSION: str = "TLS1.2"
    KEY_PATTERNS: List[str] = [r"SECRET_KEY", r"API_KEY", r"PRIVATE_KEY", r"BEGIN RSA PRIVATE KEY"]
