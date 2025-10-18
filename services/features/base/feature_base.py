# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .result_schema import FeatureResult, Finding

class FeatureBase(ABC):
    """
    所有功能模組（IDOR/XSS/SQLi/OAuth/JWT/GraphQL 等）的共同介面。
    每個 worker 實作 run()，輸入統一、輸出統一。
    
    設計原則：
    - 高價值漏洞優先：專注於能在 Bug Bounty 平台拿到高額獎金的漏洞類型
    - 證據完整：提供可直接貼進 HackerOne/Bugcrowd 報告的結構化證據
    - 低誤報：嚴格的檢測邏輯，避免浪費 triage 時間
    - 安全防護：ALLOWLIST 機制防止誤掃，避免法律風險
    """

    name: str = "base"
    version: str = "1.0.0"
    tags: List[str] = []

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def run(self, params: Dict[str, Any]) -> FeatureResult:
        """
        執行功能檢測
        
        params 通用結構：
          - target: "https://app.example.com" (目標基礎 URL)
          - headers: {...} (認證頭等)
          - options: {...} (模組特定選項)
          
        其他參數依模組類型而定：
          IDOR: path_template, candidate_ids, method, compare_headers
          XSS: path, param_name, method
          SQLi: path, param_name, method, baseline_value
          JWT: validate_endpoint, victim_token, jwks_url
          OAuth: auth_endpoint, client_id, redirect_base
          GraphQL: endpoint, test_queries, headers_admin
          SSRF: probe_endpoint, url_param, oob_http, oob_dns
        """
        ...

    def build_command_record(self, command: str, description: str,
                             parameters: Optional[Dict[str, Any]] = None,
                             tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        產出面板可用的 CLI JSON（標準化命令記錄）
        這讓 AI 的每個步驟都有清晰的操作記錄，便於 UI 渲染和審計
        """
        return {
            "command": command,
            "description": description,
            "parameters": parameters or {},
            "tags": tags or self.tags,
            "feature": self.name,
            "feature_version": self.version,
            "timestamp": None  # 將由執行器填入
        }