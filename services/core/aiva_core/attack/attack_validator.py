"""
Attack Validator - 攻擊結果驗證器

驗證攻擊執行結果的真實性和有效性
"""



import logging
import re
from typing import Any, Dict, List
from enum import Enum


logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """驗證級別"""
    BASIC = "basic"          # 基礎驗證
    STANDARD = "standard"    # 標準驗證
    STRICT = "strict"        # 嚴格驗證


class AttackValidator:
    """
    攻擊結果驗證器
    
    驗證攻擊執行結果，防止誤報，包括:
    - 響應內容驗證
    - 狀態碼驗證
    - 特徵匹配
    - 誤報過濾
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        初始化驗證器
        
        Args:
            validation_level: 驗證級別
        """
        self.validation_level = validation_level
        self.validation_count = 0
        self.false_positive_patterns = self._load_false_positive_patterns()
        
        logger.info(f"AttackValidator initialized: level={validation_level}")
    
    def _load_false_positive_patterns(self) -> Dict[str, List[str]]:
        """加載已知的誤報模式"""
        return {
            "sql_injection": [
                r"SQL syntax.*MySQL",  # 實際的 SQL 錯誤
                r"You have an error in your SQL syntax",
            ],
            "xss": [
                r"<script>alert\(['\"]XSS['\"]\)</script>",  # 回顯的 payload
                r"onerror=alert",
            ],
            "command_injection": [
                r"uid=\d+.*gid=\d+",  # whoami 或 id 命令的輸出
                r"root:x:0:0:",  # /etc/passwd 內容
            ],
        }
    
    def validate_result(
        self,
        attack_type: str,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        驗證攻擊結果
        
        Args:
            attack_type: 攻擊類型
            result: 攻擊結果
            
        Returns:
            驗證結果
        """
        self.validation_count += 1
        
        logger.debug(f"驗證攻擊結果: type={attack_type}")
        
        # 基礎驗證
        if not self._basic_validation(result):
            return {
                "valid": False,
                "reason": "Basic validation failed",
                "confidence": 0.0,
            }
        
        # 根據攻擊類型進行特定驗證
        validator_method = getattr(
            self,
            f"_validate_{attack_type.replace(' ', '_').lower()}",
            None,
        )
        
        if validator_method:
            specific_result = validator_method(result)
        else:
            specific_result = self._default_validation(result)
        
        # 檢查誤報
        is_false_positive = self._check_false_positive(attack_type, result)
        
        if is_false_positive:
            specific_result['valid'] = False
            specific_result['reason'] = "Detected as false positive"
            specific_result['confidence'] *= 0.1
        
        logger.info(
            f"驗證完成: valid={specific_result['valid']}, "
            f"confidence={specific_result['confidence']:.2f}"
        )
        
        return specific_result
    
    def _basic_validation(self, result: Dict[str, Any]) -> bool:
        """基礎驗證"""
        # 檢查結果結構
        if not isinstance(result, dict):
            return False
        
        # 檢查必要字段
        required_fields = ['success', 'response']
        if not all(field in result for field in required_fields):
            return False
        
        return True
    
    def _default_validation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """默認驗證邏輯"""
        success = result.get('success', False)
        response = result.get('response', {})
        
        # 基於 HTTP 狀態碼
        status_code = response.get('status_code', 0)
        
        if status_code >= 500:
            # 服務器錯誤可能表示成功的攻擊
            confidence = 0.7
        elif status_code == 200:
            # 成功響應，需要進一步檢查
            confidence = 0.5
        else:
            confidence = 0.3
        
        return {
            "valid": success,
            "confidence": confidence,
            "details": {
                "status_code": status_code,
            },
        }
    
    def _validate_sql_injection(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """驗證 SQL 注入結果"""
        response = result.get('response', {})
        content = response.get('content', '')
        
        # 檢查 SQL 錯誤消息
        sql_error_patterns = [
            r"SQL syntax",
            r"mysql_fetch",
            r"ORA-\d+",
            r"PostgreSQL.*ERROR",
            r"Microsoft SQL Server",
        ]
        
        has_sql_error = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in sql_error_patterns
        )
        
        if has_sql_error:
            confidence = 0.9
            valid = True
        else:
            confidence = 0.3
            valid = False
        
        return {
            "valid": valid,
            "confidence": confidence,
            "details": {
                "has_sql_error": has_sql_error,
            },
        }
    
    def _validate_xss(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """驗證 XSS 結果"""
        response = result.get('response', {})
        content = response.get('content', '')
        payload = result.get('payload', '')
        
        # 檢查 payload 是否被完整反射
        payload_reflected = payload in content
        
        # 檢查是否有過濾或編碼
        escaped_payload = (
            payload.replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
        )
        
        is_escaped = escaped_payload in content
        
        if payload_reflected and not is_escaped:
            confidence = 0.9
            valid = True
        elif payload_reflected and is_escaped:
            confidence = 0.3
            valid = False
        else:
            confidence = 0.1
            valid = False
        
        return {
            "valid": valid,
            "confidence": confidence,
            "details": {
                "payload_reflected": payload_reflected,
                "is_escaped": is_escaped,
            },
        }
    
    def _validate_command_injection(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """驗證命令注入結果"""
        response = result.get('response', {})
        content = response.get('content', '')
        
        # 檢查命令執行的證據
        command_evidence_patterns = [
            r"uid=\d+",
            r"gid=\d+",
            r"root:x:0:0",
            r"Windows.*Version",
            r"\[System\.Process\]",
        ]
        
        has_evidence = any(
            re.search(pattern, content, re.IGNORECASE)
            for pattern in command_evidence_patterns
        )
        
        if has_evidence:
            confidence = 0.95
            valid = True
        else:
            confidence = 0.2
            valid = False
        
        return {
            "valid": valid,
            "confidence": confidence,
            "details": {
                "has_command_evidence": has_evidence,
            },
        }
    
    def _check_false_positive(
        self,
        attack_type: str,
        result: Dict[str, Any],
    ) -> bool:
        """檢查是否為誤報"""
        
        patterns = self.false_positive_patterns.get(attack_type, [])
        response = result.get('response', {})
        content = response.get('content', '')
        
        # 檢查是否匹配已知的誤報模式
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def batch_validate(
        self,
        results: List[Dict[str, Any]],
        attack_type: str,
    ) -> List[Dict[str, Any]]:
        """
        批量驗證結果
        
        Args:
            results: 結果列表
            attack_type: 攻擊類型
            
        Returns:
            驗證結果列表
        """
        validated_results = []
        
        for result in results:
            validation = self.validate_result(attack_type, result)
            validated_results.append({
                "original_result": result,
                "validation": validation,
            })
        
        return validated_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取統計信息"""
        return {
            "total_validations": self.validation_count,
            "validation_level": self.validation_level.value,
            "false_positive_patterns_count": sum(
                len(patterns) for patterns in self.false_positive_patterns.values()
            ),
        }
