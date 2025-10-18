# -*- coding: utf-8 -*-
import os
import re
import time
from typing import Dict, Any, Optional, Tuple
import requests
from urllib.parse import urlparse

# 從環境變數讀取允許的目標域名，避免誤掃
ALLOWLIST = {d.strip().lower() for d in os.getenv("ALLOWLIST_DOMAINS", "").split(",") if d.strip()}

class SafeHttp:
    """
    具 Allowlist、速率限制、逾時、基本重試的輕量 HTTP 客戶端。
    
    安全特性：
    - ALLOWLIST_DOMAINS 環境變數控制允許的目標域名
    - 速率限制防止對目標服務造成壓力
    - 自動重試機制提高檢測穩定性
    - 合理的超時設定避免長時間等待
    
    用法：
    1. 設定環境變數：ALLOWLIST_DOMAINS=example.com,api.example.com
    2. 創建實例：http = SafeHttp()
    3. 發送請求：response = http.request("GET", "https://api.example.com/test")
    """
    
    def __init__(self, timeout: int = 12, rate_limit_qps: float = 5.0, retries: int = 1):
        """
        初始化安全 HTTP 客戶端
        
        Args:
            timeout: 請求超時時間（秒）
            rate_limit_qps: 每秒請求數限制
            retries: 失敗重試次數
        """
        self.timeout = timeout
        self.rate_limit_interval = 1.0 / max(rate_limit_qps, 0.01)
        self.retries = max(retries, 0)
        self._last_ts = 0.0
        self.s = requests.Session()
        # 設定合理的 User-Agent
        self.s.headers.update({
            "User-Agent": "AIVA Security Scanner/2.0 (Authorized Testing)"
        })

    def _check_allow(self, url: str) -> Tuple[bool, str]:
        """
        檢查 URL 是否在允許列表中
        
        Args:
            url: 要檢查的 URL
            
        Returns:
            (是否允許, 域名)
        """
        try:
            host = urlparse(url).hostname or ""
            host = host.lower()
        except Exception:
            return False, "invalid_url"
            
        if not ALLOWLIST:
            return False, "ALLOWLIST empty; set env ALLOWLIST_DOMAINS=example.com,api.example.com"
        
        # 支援精確匹配和子域名匹配
        allowed = any(host == a or host.endswith(f".{a}") for a in ALLOWLIST)
        return allowed, host

    def _pace(self):
        """實施速率限制"""
        now = time.time()
        delta = now - self._last_ts
        if delta < self.rate_limit_interval:
            time.sleep(self.rate_limit_interval - delta)
        self._last_ts = time.time()

    def request(self, method: str, url: str, **kw) -> requests.Response:
        """
        發送 HTTP 請求
        
        Args:
            method: HTTP 方法（GET, POST 等）
            url: 目標 URL
            **kw: 其他 requests 參數
            
        Returns:
            requests.Response 物件
            
        Raises:
            PermissionError: 當目標不在允許列表時
            Exception: 當所有重試都失敗時
        """
        ok, host = self._check_allow(url)
        if not ok:
            raise PermissionError(f"Target `{host}` not in ALLOWLIST_DOMAINS. Current allowlist: {list(ALLOWLIST)}")
        
        last_err = None
        for attempt in range(self.retries + 1):
            self._pace()
            try:
                response = self.s.request(
                    method=method.upper(), 
                    url=url, 
                    timeout=self.timeout, 
                    **kw
                )
                return response
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    # 短暫等待後重試
                    time.sleep(0.5 * (attempt + 1))
                    
        raise last_err

    def get_session_info(self) -> Dict[str, Any]:
        """
        取得當前會話資訊，用於除錯
        
        Returns:
            包含設定資訊的字典
        """
        return {
            "timeout": self.timeout,
            "rate_limit_qps": 1.0 / self.rate_limit_interval,
            "retries": self.retries,
            "allowlist": list(ALLOWLIST),
            "last_request_time": self._last_ts
        }