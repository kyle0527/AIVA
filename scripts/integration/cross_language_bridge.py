#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIVA 跨語言通訊橋接器
提供多種跨語言實現方式作為預備方案
"""

import json
import subprocess
import asyncio
import websockets
import zmq
import logging
import struct
import ctypes
from typing import Dict, Any, List, Optional, Union, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import socket
import time
import uuid

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CommunicationMethod(Enum):
    """跨語言通訊方式枚舉"""
    FFI_CTYPES = "ffi_ctypes"          # C 函數庫調用
    SUBPROCESS = "subprocess"          # 子進程調用
    WEBSOCKET = "websocket"           # WebSocket 通訊
    ZMQ = "zmq"                       # ZeroMQ 訊息佇列
    NAMED_PIPE = "named_pipe"         # 命名管道
    TCP_SOCKET = "tcp_socket"         # TCP Socket
    SHARED_MEMORY = "shared_memory"   # 共享記憶體
    FILE_BASED = "file_based"         # 檔案基礎通訊
    REST_API = "rest_api"             # REST API 調用
    GRPC = "grpc"                     # gRPC 通訊

@dataclass
class CrossLanguageMessage:
    """跨語言訊息格式"""
    id: str
    method: str
    params: Dict[str, Any]
    language: str
    timestamp: float
    timeout: float = 30.0

@dataclass
class CrossLanguageResponse:
    """跨語言回應格式"""
    id: str
    result: Any
    error: Optional[str]
    timestamp: float
    duration: float

class CrossLanguageBridge(ABC):
    """跨語言橋接器抽象基類"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_connected = False
        self.logger = logging.getLogger(f"Bridge.{name}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """建立連接"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """斷開連接"""
        pass
    
    @abstractmethod
    async def call(self, message: CrossLanguageMessage) -> CrossLanguageResponse:
        """發送調用請求"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """檢查是否可用"""
        pass

class FFIBridge(CrossLanguageBridge):
    """FFI (Foreign Function Interface) 橋接器"""
    
    def __init__(self, library_path: str):
        super().__init__("FFI")
        self.library_path = library_path
        self.library = None
    
    async def connect(self) -> bool:
        try:
            self.library = ctypes.CDLL(self.library_path)
            self.is_connected = True
            self.logger.info(f"FFI library loaded: {self.library_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load FFI library: {e}")
            return False
    
    async def disconnect(self) -> bool:
        self.library = None
        self.is_connected = False
        return True
    
    async def call(self, message: CrossLanguageMessage) -> CrossLanguageResponse:
        start_time = time.time()
        try:
            if not self.library:
                raise Exception("Library not loaded")
            
            # 獲取函數
            func = getattr(self.library, message.method)
            
            # 調用函數 (需要根據實際情況調整參數類型)
            result = func(*message.params.values())
            
            return CrossLanguageResponse(
                id=message.id,
                result=result,
                error=None,
                timestamp=time.time(),
                duration=time.time() - start_time
            )
        except Exception as e:
            return CrossLanguageResponse(
                id=message.id,
                result=None,
                error=str(e),
                timestamp=time.time(),
                duration=time.time() - start_time
            )
    
    def is_available(self) -> bool:
        return Path(self.library_path).exists()

class SubprocessBridge(CrossLanguageBridge):
    """子進程調用橋接器"""
    
    def __init__(self, executable: str, working_dir: Optional[str] = None):
        super().__init__("Subprocess")
        self.executable = executable
        self.working_dir = working_dir
    
    async def connect(self) -> bool:
        # 子進程不需要持久連接
        self.is_connected = True
        return True
    
    async def disconnect(self) -> bool:
        self.is_connected = False
        return True
    
    async def call(self, message: CrossLanguageMessage) -> CrossLanguageResponse:
        start_time = time.time()
        try:
            # 建構命令
            cmd = [self.executable, message.method]
            
            # 準備輸入資料
            input_data = json.dumps(message.params)
            
            # 執行子進程
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir
            )
            
            # 發送資料並等待結果
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_data.encode()),
                timeout=message.timeout
            )
            
            if process.returncode == 0:
                result = json.loads(stdout.decode())
                error = None
            else:
                result = None
                error = stderr.decode()
            
            return CrossLanguageResponse(
                id=message.id,
                result=result,
                error=error,
                timestamp=time.time(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return CrossLanguageResponse(
                id=message.id,
                result=None,
                error=str(e),
                timestamp=time.time(),
                duration=time.time() - start_time
            )
    
    def is_available(self) -> bool:
        try:
            subprocess.run([self.executable, "--version"], 
                         capture_output=True, timeout=5)
            return True
        except:
            return False

class WebSocketBridge(CrossLanguageBridge):
    """WebSocket 通訊橋接器"""
    
    def __init__(self, uri: str):
        super().__init__("WebSocket")
        self.uri = uri
        self.websocket = None
    
    async def connect(self) -> bool:
        try:
            self.websocket = await websockets.connect(self.uri)
            self.is_connected = True
            self.logger.info(f"WebSocket connected: {self.uri}")
            return True
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        self.is_connected = False
        return True
    
    async def call(self, message: CrossLanguageMessage) -> CrossLanguageResponse:
        start_time = time.time()
        try:
            if not self.websocket:
                raise Exception("WebSocket not connected")
            
            # 發送訊息
            message_json = json.dumps({
                "id": message.id,
                "method": message.method,
                "params": message.params,
                "language": message.language
            })
            
            await self.websocket.send(message_json)
            
            # 等待回應
            response_json = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=message.timeout
            )
            
            response_data = json.loads(response_json)
            
            return CrossLanguageResponse(
                id=response_data.get("id", message.id),
                result=response_data.get("result"),
                error=response_data.get("error"),
                timestamp=time.time(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return CrossLanguageResponse(
                id=message.id,
                result=None,
                error=str(e),
                timestamp=time.time(),
                duration=time.time() - start_time
            )
    
    def is_available(self) -> bool:
        # 簡單的可用性檢查
        return True

class ZMQBridge(CrossLanguageBridge):
    """ZeroMQ 訊息佇列橋接器"""
    
    def __init__(self, endpoint: str, pattern: str = "REQ"):
        super().__init__("ZMQ")
        self.endpoint = endpoint
        self.pattern = pattern
        self.context = None
        self.socket = None
    
    async def connect(self) -> bool:
        try:
            self.context = zmq.Context()
            
            if self.pattern == "REQ":
                self.socket = self.context.socket(zmq.REQ)
            elif self.pattern == "PUSH":
                self.socket = self.context.socket(zmq.PUSH)
            else:
                raise ValueError(f"Unsupported pattern: {self.pattern}")
            
            self.socket.connect(self.endpoint)
            self.is_connected = True
            self.logger.info(f"ZMQ connected: {self.endpoint}")
            return True
        except Exception as e:
            self.logger.error(f"ZMQ connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.is_connected = False
        return True
    
    async def call(self, message: CrossLanguageMessage) -> CrossLanguageResponse:
        start_time = time.time()
        try:
            if not self.socket:
                raise Exception("ZMQ socket not connected")
            
            # 準備訊息
            message_data = {
                "id": message.id,
                "method": message.method,
                "params": message.params,
                "language": message.language
            }
            
            # 發送訊息
            self.socket.send_json(message_data)
            
            if self.pattern == "REQ":
                # 等待回應
                response_data = self.socket.recv_json()
                
                return CrossLanguageResponse(
                    id=response_data.get("id", message.id),
                    result=response_data.get("result"),
                    error=response_data.get("error"),
                    timestamp=time.time(),
                    duration=time.time() - start_time
                )
            else:
                # PUSH 模式不等待回應
                return CrossLanguageResponse(
                    id=message.id,
                    result="sent",
                    error=None,
                    timestamp=time.time(),
                    duration=time.time() - start_time
                )
                
        except Exception as e:
            return CrossLanguageResponse(
                id=message.id,
                result=None,
                error=str(e),
                timestamp=time.time(),
                duration=time.time() - start_time
            )
    
    def is_available(self) -> bool:
        try:
            # 嘗試建立測試連接
            test_context = zmq.Context()
            test_socket = test_context.socket(zmq.REQ)
            test_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒超時
            test_socket.connect(self.endpoint)
            test_socket.close()
            test_context.term()
            return True
        except:
            return False

class TCPSocketBridge(CrossLanguageBridge):
    """TCP Socket 橋接器"""
    
    def __init__(self, host: str, port: int):
        super().__init__("TCPSocket")
        self.host = host
        self.port = port
        self.socket = None
    
    async def connect(self) -> bool:
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            self.logger.info(f"TCP connected: {self.host}:{self.port}")
            return True
        except Exception as e:
            self.logger.error(f"TCP connection failed: {e}")
            return False
    
    async def disconnect(self) -> bool:
        if self.socket:
            self.socket.close()
            self.socket = None
        self.is_connected = False
        return True
    
    async def call(self, message: CrossLanguageMessage) -> CrossLanguageResponse:
        start_time = time.time()
        try:
            if not self.socket:
                raise Exception("TCP socket not connected")
            
            # 準備訊息
            message_json = json.dumps({
                "id": message.id,
                "method": message.method,
                "params": message.params,
                "language": message.language
            })
            
            # 發送訊息 (先發送長度，再發送內容)
            message_bytes = message_json.encode()
            message_length = struct.pack("!I", len(message_bytes))
            
            self.socket.sendall(message_length + message_bytes)
            
            # 接收回應
            response_length_bytes = self.socket.recv(4)
            response_length = struct.unpack("!I", response_length_bytes)[0]
            
            response_bytes = self.socket.recv(response_length)
            response_data = json.loads(response_bytes.decode())
            
            return CrossLanguageResponse(
                id=response_data.get("id", message.id),
                result=response_data.get("result"),
                error=response_data.get("error"),
                timestamp=time.time(),
                duration=time.time() - start_time
            )
            
        except Exception as e:
            return CrossLanguageResponse(
                id=message.id,
                result=None,
                error=str(e),
                timestamp=time.time(),
                duration=time.time() - start_time
            )
    
    def is_available(self) -> bool:
        try:
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(1)
            result = test_socket.connect_ex((self.host, self.port))
            test_socket.close()
            return result == 0
        except:
            return False

class FileBasedBridge(CrossLanguageBridge):
    """檔案基礎通訊橋接器"""
    
    def __init__(self, input_dir: str, output_dir: str):
        super().__init__("FileBased")
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    async def connect(self) -> bool:
        self.is_connected = True
        return True
    
    async def disconnect(self) -> bool:
        self.is_connected = False
        return True
    
    async def call(self, message: CrossLanguageMessage) -> CrossLanguageResponse:
        start_time = time.time()
        try:
            # 寫入請求檔案
            request_file = self.input_dir / f"{message.id}.json"
            request_data = {
                "id": message.id,
                "method": message.method,
                "params": message.params,
                "language": message.language,
                "timestamp": message.timestamp
            }
            
            with open(request_file, 'w', encoding='utf-8') as f:
                json.dump(request_data, f, indent=2)
            
            # 等待回應檔案
            response_file = self.output_dir / f"{message.id}.json"
            timeout_end = time.time() + message.timeout
            
            while time.time() < timeout_end:
                if response_file.exists():
                    with open(response_file, 'r', encoding='utf-8') as f:
                        response_data = json.load(f)
                    
                    # 清理檔案
                    request_file.unlink(missing_ok=True)
                    response_file.unlink(missing_ok=True)
                    
                    return CrossLanguageResponse(
                        id=response_data.get("id", message.id),
                        result=response_data.get("result"),
                        error=response_data.get("error"),
                        timestamp=time.time(),
                        duration=time.time() - start_time
                    )
                
                await asyncio.sleep(0.1)
            
            # 超時處理
            request_file.unlink(missing_ok=True)
            raise TimeoutError("File-based communication timeout")
            
        except Exception as e:
            return CrossLanguageResponse(
                id=message.id,
                result=None,
                error=str(e),
                timestamp=time.time(),
                duration=time.time() - start_time
            )
    
    def is_available(self) -> bool:
        return self.input_dir.exists() and self.output_dir.exists()

class CrossLanguageManager:
    """跨語言通訊管理器"""
    
    def __init__(self):
        self.bridges: Dict[str, CrossLanguageBridge] = {}
        self.default_bridge = None
        self.logger = logging.getLogger("CrossLanguageManager")
    
    def register_bridge(self, name: str, bridge: CrossLanguageBridge, 
                       is_default: bool = False):
        """註冊橋接器"""
        self.bridges[name] = bridge
        if is_default or not self.default_bridge:
            self.default_bridge = name
        self.logger.info(f"Bridge registered: {name}")
    
    async def connect_all(self):
        """連接所有橋接器"""
        for name, bridge in self.bridges.items():
            if bridge.is_available():
                success = await bridge.connect()
                self.logger.info(f"Bridge {name}: {'Connected' if success else 'Failed'}")
    
    async def disconnect_all(self):
        """斷開所有橋接器"""
        for name, bridge in self.bridges.items():
            if bridge.is_connected:
                await bridge.disconnect()
                self.logger.info(f"Bridge {name}: Disconnected")
    
    async def call(self, method: str, params: Dict[str, Any], 
                  language: str, bridge_name: Optional[str] = None,
                  timeout: float = 30.0) -> CrossLanguageResponse:
        """發送跨語言調用"""
        
        # 選擇橋接器
        if bridge_name:
            if bridge_name not in self.bridges:
                raise ValueError(f"Bridge not found: {bridge_name}")
            bridge = self.bridges[bridge_name]
        else:
            if not self.default_bridge:
                raise ValueError("No default bridge set")
            bridge = self.bridges[self.default_bridge]
        
        # 建立訊息
        message = CrossLanguageMessage(
            id=str(uuid.uuid4()),
            method=method,
            params=params,
            language=language,
            timestamp=time.time(),
            timeout=timeout
        )
        
        # 發送調用
        if not bridge.is_connected:
            await bridge.connect()
        
        return await bridge.call(message)
    
    def get_available_bridges(self) -> List[str]:
        """獲取可用的橋接器列表"""
        return [name for name, bridge in self.bridges.items() 
                if bridge.is_available()]
    
    def get_bridge_status(self) -> Dict[str, Dict[str, Any]]:
        """獲取所有橋接器狀態"""
        status = {}
        for name, bridge in self.bridges.items():
            status[name] = {
                "available": bridge.is_available(),
                "connected": bridge.is_connected,
                "type": bridge.__class__.__name__
            }
        return status

# 工廠函數
def create_aiva_cross_language_manager() -> CrossLanguageManager:
    """建立 AIVA 專用的跨語言管理器"""
    manager = CrossLanguageManager()
    
    # 註冊各種橋接器
    
    # 1. Go 模組 - 子進程方式
    go_bridge = SubprocessBridge(
        executable="C:/D/fold7/AIVA-git/services/features/function_sca_go/worker.exe",
        working_dir="C:/D/fold7/AIVA-git/services/features/function_sca_go"
    )
    manager.register_bridge("go_sca", go_bridge)
    
    # 2. Rust 模組 - 子進程方式
    rust_bridge = SubprocessBridge(
        executable="C:/D/fold7/AIVA-git/services/features/function_sast_rust/target/debug/function_sast_rust.exe",
        working_dir="C:/D/fold7/AIVA-git/services/features/function_sast_rust"
    )
    manager.register_bridge("rust_sast", rust_bridge)
    
    # 3. WebSocket 橋接器 (用於實時通訊)
    ws_bridge = WebSocketBridge("ws://localhost:8765")
    manager.register_bridge("websocket", ws_bridge)
    
    # 4. ZMQ 橋接器 (用於高性能通訊)
    zmq_bridge = ZMQBridge("tcp://localhost:5555")
    manager.register_bridge("zmq", zmq_bridge)
    
    # 5. TCP Socket 橋接器
    tcp_bridge = TCPSocketBridge("localhost", 9999)
    manager.register_bridge("tcp", tcp_bridge)
    
    # 6. 檔案基礎橋接器 (最可靠的備案)
    file_bridge = FileBasedBridge(
        input_dir="C:/D/fold7/AIVA-git/temp/cross_lang_input",
        output_dir="C:/D/fold7/AIVA-git/temp/cross_lang_output"
    )
    manager.register_bridge("file", file_bridge, is_default=True)
    
    return manager

# 使用範例
async def demo_cross_language_communication():
    """示範跨語言通訊"""
    manager = create_aiva_cross_language_manager()
    
    try:
        # 連接所有橋接器
        await manager.connect_all()
        
        # 顯示狀態
        status = manager.get_bridge_status()
        print("橋接器狀態:")
        for name, info in status.items():
            print(f"  {name}: {'可用' if info['available'] else '不可用'} | "
                  f"{'已連接' if info['connected'] else '未連接'}")
        
        # 測試調用
        print("\n測試跨語言調用:")
        
        # 調用 Go 模組
        response = await manager.call(
            method="scan_vulnerabilities",
            params={"target": "test.go", "rules": ["all"]},
            language="go",
            bridge_name="file"  # 使用檔案橋接器作為示範
        )
        
        print(f"Go 模組回應: {response.result}")
        if response.error:
            print(f"錯誤: {response.error}")
        
    finally:
        await manager.disconnect_all()

if __name__ == "__main__":
    # 執行示範
    asyncio.run(demo_cross_language_communication())