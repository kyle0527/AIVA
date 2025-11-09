"""
AIVA MCP (Model Context Protocol) Support v2.0
模型上下文協議支援 - 標準化模組間通訊

實現2025年新興的Model Context Protocol標準，為所有AIVA模組提供
標準化的通訊介面，支援無縫的模組間集成和互操作性。

Author: AIVA Team
Created: 2025-11-09
Version: 2.0.0
"""

import asyncio
import json
import time
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union, Callable, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# 導入事件系統
from ..event_system.event_bus import AIEvent, AIEventBus, EventPriority
from ..controller.strangler_fig_controller import AIRequest, AIResponse, MessageType

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 常量定義
MIME_TYPE_JSON = "application/json"
MCP_TOOLS_CALL_METHOD = 'tools/call'

# ==================== MCP 核心協議定義 ====================

class MCPVersion(Enum):
    """MCP協議版本"""
    V1_0 = "1.0"
    V2_0 = "2.0"

class MCPCapability(Enum):
    """MCP能力標識"""
    RESOURCES = "resources"           # 資源管理
    TOOLS = "tools"                  # 工具調用
    PROMPTS = "prompts"              # 提示模板
    LOGGING = "logging"              # 日誌記錄
    SAMPLING = "sampling"            # 採樣配置

class MCPMessageType(Enum):
    """MCP消息類型"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    CAPABILITY_ANNOUNCEMENT = "capability_announcement"
    RESOURCE_UPDATE = "resource_update"

@dataclass
class MCPMessage:
    """MCP標準消息格式"""
    
    # 基礎欄位
    jsonrpc: str = "2.0"
    id: Optional[str] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    
    # MCP擴展欄位
    mcp_version: str = MCPVersion.V2_0.value
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_module: Optional[str] = None
    target_module: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            'jsonrpc': self.jsonrpc,
            'id': self.id,
            'method': self.method,
            'params': self.params,
            'result': self.result,
            'error': self.error,
            'mcp_version': self.mcp_version,
            'timestamp': self.timestamp,
            'source_module': self.source_module,
            'target_module': self.target_module,
            'capabilities': self.capabilities,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPMessage':
        """從字典創建"""
        return cls(
            jsonrpc=data.get('jsonrpc', '2.0'),
            id=data.get('id'),
            method=data.get('method'),
            params=data.get('params'),
            result=data.get('result'),
            error=data.get('error'),
            mcp_version=data.get('mcp_version', MCPVersion.V2_0.value),
            timestamp=data.get('timestamp', datetime.now(timezone.utc).isoformat()),
            source_module=data.get('source_module'),
            target_module=data.get('target_module'),
            capabilities=data.get('capabilities', []),
            metadata=data.get('metadata', {})
        )

@dataclass
class MCPResource:
    """MCP資源定義"""
    
    uri: str
    name: str
    text: str  
    description: Optional[str] = None
    mime_type: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    # AIVA擴展
    module_source: Optional[str] = None
    access_level: str = "read"  # read, write, execute
    version: str = "1.0"
    last_modified: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class MCPTool:
    """MCP工具定義"""
    
    name: str
    description: str
    input_schema: Dict[str, Any]
    
    # AIVA擴展
    module_source: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    execution_time_estimate: float = 1.0  # 秒
    confidence_level: float = 1.0

# ==================== MCP介面協議 ====================

class MCPServerProtocol(Protocol):
    """MCP伺服器協議介面"""
    
    async def handle_request(self, message: MCPMessage) -> MCPMessage:
        """處理MCP請求"""
        ...
    
    async def list_resources(self) -> List[MCPResource]:
        """列出可用資源"""
        ...
    
    async def get_resource(self, uri: str) -> Optional[Dict[str, Any]]:
        """獲取特定資源"""
        ...
    
    async def list_tools(self) -> List[MCPTool]:
        """列出可用工具"""
        ...
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """調用工具"""
        ...

class MCPClientProtocol(Protocol):
    """MCP客戶端協議介面"""
    
    async def send_request(self, message: MCPMessage) -> MCPMessage:
        """發送MCP請求"""
        ...
    
    async def send_notification(self, message: MCPMessage) -> None:
        """發送通知"""
        ...

# ==================== MCP 適配器 ====================

class AIVAMCPAdapter:
    """AIVA模組MCP適配器"""
    
    def __init__(self, module_name: str, module_version: str, 
                 event_bus: Optional[AIEventBus] = None):
        self.module_name = module_name
        self.module_version = module_version
        self.event_bus = event_bus
        
        # MCP狀態
        self.capabilities = [MCPCapability.RESOURCES.value, MCPCapability.TOOLS.value]
        self.resources = {}
        self.tools = {}
        self.active_sessions = {}
        
        # 統計資訊
        self.stats = {
            'requests_processed': 0,
            'responses_sent': 0,
            'errors_handled': 0,
            'avg_response_time': 0.0
        }
        
        logger.info(f"MCP適配器初始化完成 - 模組: {module_name}")
    
    def register_resource(self, resource: MCPResource) -> bool:
        """註冊MCP資源"""
        try:
            resource.module_source = self.module_name
            self.resources[resource.uri] = resource
            logger.info(f"註冊MCP資源: {resource.uri}")
            return True
        except Exception as e:
            logger.error(f"註冊資源失敗: {str(e)}")
            return False
    
    def register_tool(self, tool: MCPTool) -> bool:
        """註冊MCP工具"""
        try:
            tool.module_source = self.module_name
            self.tools[tool.name] = tool
            logger.info(f"註冊MCP工具: {tool.name}")
            return True
        except Exception as e:
            logger.error(f"註冊工具失敗: {str(e)}")
            return False
    
    async def handle_mcp_request(self, message: MCPMessage) -> MCPMessage:
        """處理MCP請求"""
        start_time = time.time()
        
        try:
            method = message.method
            params = message.params or {}
            
            if method == "resources/list":
                result = self._handle_list_resources()
            elif method == "resources/read":
                result = self._handle_read_resource(params)
            elif method == "tools/list":
                result = self._handle_list_tools()
            elif method == MCP_TOOLS_CALL_METHOD:
                result = await self._handle_call_tool(params)
            elif method == "capabilities/announce":
                result = self._handle_capability_announcement()
            else:
                raise ValueError(f"不支援的MCP方法: {method}")
            
            # 創建回應消息
            response = MCPMessage(
                id=message.id,
                result=result,
                source_module=self.module_name,
                target_module=message.source_module,
                capabilities=self.capabilities
            )
            
            # 更新統計
            processing_time = (time.time() - start_time) * 1000
            self._update_stats('request', processing_time, True)
            
            # 發布事件
            if self.event_bus:
                await self._publish_mcp_event('mcp.request.processed', {
                    'method': method,
                    'processing_time': processing_time,
                    'source_module': message.source_module
                })
            
            return response
            
        except Exception as e:
            # 錯誤回應
            error_response = MCPMessage(
                id=message.id,
                error={
                    'code': -1,
                    'message': str(e),
                    'data': {'module': self.module_name}
                },
                source_module=self.module_name,
                target_module=message.source_module
            )
            
            processing_time = (time.time() - start_time) * 1000
            self._update_stats('request', processing_time, False)
            
            logger.error(f"MCP請求處理錯誤: {str(e)}")
            
            return error_response
    
    def _handle_list_resources(self) -> Dict[str, Any]:
        """處理資源列表請求"""
        resource_list = []
        
        for resource in self.resources.values():
            resource_list.append({
                'uri': resource.uri,
                'name': resource.name,
                'description': resource.description,
                'mimeType': resource.mimeType,
                'annotations': resource.annotations,
                'module_source': resource.module_source,
                'access_level': resource.access_level,
                'version': resource.version,
                'last_modified': resource.last_modified
            })
        
        return {
            'resources': resource_list,
            'total_count': len(resource_list),
            'module': self.module_name
        }
    
    def _handle_read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """處理讀取資源請求"""
        uri = params.get('uri')
        
        if not uri:
            raise ValueError("缺少資源URI參數")
        
        resource = self.resources.get(uri)
        
        if not resource:
            raise ValueError(f"資源不存在: {uri}")
        
        # 簡化實現返回資源元數據
        return {
            'uri': resource.uri,
            'name': resource.name,
            'description': resource.description,
            'mimeType': resource.mime_type,
            'content': f"Resource content for {resource.name}",  # 簡化內容
            'metadata': {
                'module_source': resource.module_source,
                'access_level': resource.access_level,
                'version': resource.version,
                'last_modified': resource.last_modified
            }
        }
    
    def _handle_list_tools(self) -> Dict[str, Any]:
        """處理工具列表請求"""
        tool_list = []
        
        for tool in self.tools.values():
            tool_list.append({
                'name': tool.name,
                'description': tool.description,
                'inputSchema': tool.inputSchema,
                'module_source': tool.module_source,
                'capabilities': tool.capabilities,
                'execution_time_estimate': tool.execution_time_estimate,
                'confidence_level': tool.confidence_level
            })
        
        return {
            'tools': tool_list,
            'total_count': len(tool_list),
            'module': self.module_name
        }
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """處理工具調用請求"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        if not tool_name:
            raise ValueError("缺少工具名稱參數")
        
        tool = self.tools.get(tool_name)
        
        if not tool:
            raise ValueError(f"工具不存在: {tool_name}")
        
        # 簡化實現返回模擬結果
        execution_start = time.time()
        
        # 模擬工具執行
        await asyncio.sleep(0.1)  # 模擬處理時間
        
        execution_time = (time.time() - execution_start) * 1000
        
        return {
            'tool_name': tool_name,
            'arguments': arguments,
            'result': {
                'status': 'success',
                'output': f"Tool {tool_name} executed successfully",
                'data': arguments  # 簡化返回輸入參數
            },
            'execution_metadata': {
                'execution_time_ms': execution_time,
                'module_source': tool.module_source,
                'confidence': tool.confidence_level
            }
        }
    
    def _handle_capability_announcement(self) -> Dict[str, Any]:
        """處理能力公告請求"""
        return {
            'module_name': self.module_name,
            'module_version': self.module_version,
            'supported_capabilities': self.capabilities,
            'mcp_version': MCPVersion.V2_0.value,
            'resources_count': len(self.resources),
            'tools_count': len(self.tools),
            'status': 'active'
        }
    
    def send_mcp_request(self, target_module: str, method: str, 
                        params: Optional[Dict[str, Any]] = None) -> MCPMessage:
        """發送MCP請求"""
        
        request = MCPMessage(
            method=method,
            params=params,
            source_module=self.module_name,
            target_module=target_module,
            capabilities=self.capabilities
        )
        
        # 簡化實現，直接返回模擬回應
        response = MCPMessage(
            id=request.id,
            result={
                'status': 'success',
                'message': f'Request {method} processed',
                'target_module': target_module
            },
            source_module=target_module,
            target_module=self.module_name
        )
        
        self._update_stats('response', 0, True)
        
        return response
    
    async def _publish_mcp_event(self, event_type: str, data: Dict[str, Any]):
        """發布MCP事件"""
        if not self.event_bus:
            return
        
        event = AIEvent(
            event_type=event_type,
            source_module=self.module_name,
            source_version=self.module_version,
            data={
                **data,
                'mcp_version': MCPVersion.V2_0.value,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )
        
        await self.event_bus.publish(event)
    
    def _update_stats(self, operation: str, processing_time: float, success: bool):
        """更新統計資訊"""
        if operation == 'request':
            self.stats['requests_processed'] += 1
        elif operation == 'response':
            self.stats['responses_sent'] += 1
        
        if not success:
            self.stats['errors_handled'] += 1
        
        # 更新平均回應時間
        if self.stats['avg_response_time'] == 0:
            self.stats['avg_response_time'] = processing_time
        else:
            alpha = 0.1
            self.stats['avg_response_time'] = (
                alpha * processing_time + 
                (1 - alpha) * self.stats['avg_response_time']
            )

# ==================== MCP 管理器 ====================

class MCPManager:
    """MCP協議管理器"""
    
    def __init__(self, event_bus: Optional[AIEventBus] = None):
        self.event_bus = event_bus
        
        # 註冊的模組和適配器
        self.adapters = {}
        self.module_capabilities = {}
        
        # 全域資源和工具註冊表
        self.global_resources = {}
        self.global_tools = {}
        
        # 統計資訊
        self.stats = {
            'total_modules': 0,
            'total_resources': 0,
            'total_tools': 0,
            'message_exchanges': 0,
            'last_activity': None
        }
        
        logger.info("MCP管理器初始化完成")
    
    def register_module(self, module_name: str, module_version: str) -> AIVAMCPAdapter:
        """註冊模組並創建MCP適配器"""
        
        adapter = AIVAMCPAdapter(module_name, module_version, self.event_bus)
        self.adapters[module_name] = adapter
        
        self.module_capabilities[module_name] = {
            'version': module_version,
            'capabilities': adapter.capabilities,
            'registered_at': datetime.now(timezone.utc).isoformat(),
            'status': 'active'
        }
        
        self.stats['total_modules'] += 1
        logger.info(f"註冊模組: {module_name} v{module_version}")
        
        return adapter
    
    async def route_mcp_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """路由MCP消息"""
        target_module = message.target_module
        
        if not target_module:
            logger.error("MCP消息缺少目標模組")
            return None
        
        adapter = self.adapters.get(target_module)
        
        if not adapter:
            logger.error(f"目標模組不存在: {target_module}")
            return None
        
        try:
            response = await adapter.handle_mcp_request(message)
            self.stats['message_exchanges'] += 1
            self.stats['last_activity'] = datetime.now(timezone.utc).isoformat()
            
            return response
            
        except Exception as e:
            logger.error(f"路由MCP消息錯誤: {str(e)}")
            return None
    
    async def discover_capabilities(self) -> Dict[str, Any]:
        """發現所有模組的能力"""
        
        capabilities = {}
        
        for module_name, adapter in self.adapters.items():
            # 發送能力公告請求
            capability_request = MCPMessage(
                method="capabilities/announce",
                source_module="mcp_manager",
                target_module=module_name
            )
            
            response = await adapter.handle_mcp_request(capability_request)
            
            if response.result:
                capabilities[module_name] = response.result
        
        return {
            'module_capabilities': capabilities,
            'discovery_timestamp': datetime.now(timezone.utc).isoformat(),
            'total_modules': len(capabilities)
        }
    
    async def aggregate_resources(self) -> Dict[str, Any]:
        """聚合所有模組的資源"""
        
        aggregated_resources = []
        
        for module_name, adapter in self.adapters.items():
            resource_request = MCPMessage(
                method="resources/list",
                source_module="mcp_manager",
                target_module=module_name
            )
            
            response = await adapter.handle_mcp_request(resource_request)
            
            if response.result and 'resources' in response.result:
                for resource in response.result['resources']:
                    resource['source_module'] = module_name
                    aggregated_resources.append(resource)
        
        return {
            'resources': aggregated_resources,
            'total_resources': len(aggregated_resources),
            'aggregation_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def aggregate_tools(self) -> Dict[str, Any]:
        """聚合所有模組的工具"""
        
        aggregated_tools = []
        
        for module_name, adapter in self.adapters.items():
            tools_request = MCPMessage(
                method="tools/list",
                source_module="mcp_manager",
                target_module=module_name
            )
            
            response = await adapter.handle_mcp_request(tools_request)
            
            if response.result and 'tools' in response.result:
                for tool in response.result['tools']:
                    tool['source_module'] = module_name
                    aggregated_tools.append(tool)
        
        return {
            'tools': aggregated_tools,
            'total_tools': len(aggregated_tools),
            'aggregation_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def execute_cross_module_workflow(self, workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """執行跨模組工作流程"""
        
        workflow_results = []
        workflow_id = str(uuid.uuid4())
        
        logger.info(f"開始執行跨模組工作流程: {workflow_id}")
        
        for step_idx, step in enumerate(workflow):
            step_start_time = time.time()
            
            target_module = step.get('target_module')
            method = step.get('method')
            params = step.get('params', {})
            
            if not target_module or not method:
                step_result = {
                    'step_index': step_idx,
                    'status': 'error',
                    'error': 'Missing target_module or method'
                }
            else:
                # 創建MCP請求
                request = MCPMessage(
                    method=method,
                    params=params,
                    source_module="mcp_manager",
                    target_module=target_module
                )
                
                # 路由請求
                response = await self.route_mcp_message(request)
                
                step_execution_time = (time.time() - step_start_time) * 1000
                
                if response:
                    step_result = {
                        'step_index': step_idx,
                        'status': 'success',
                        'result': response.result,
                        'execution_time_ms': step_execution_time,
                        'target_module': target_module,
                        'method': method
                    }
                else:
                    step_result = {
                        'step_index': step_idx,
                        'status': 'error',
                        'error': 'Failed to route message',
                        'execution_time_ms': step_execution_time
                    }
            
            workflow_results.append(step_result)
            
            # 簡單的錯誤處理：如果步驟失敗且標記為必需，停止工作流程
            if (step_result['status'] == 'error' and 
                step.get('required', True)):
                logger.error(f"工作流程步驟 {step_idx} 失敗，停止執行")
                break
        
        return {
            'workflow_id': workflow_id,
            'total_steps': len(workflow),
            'completed_steps': len(workflow_results),
            'results': workflow_results,
            'overall_status': 'success' if all(r['status'] == 'success' for r in workflow_results) else 'partial_failure',
            'completion_timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_mcp_status(self) -> Dict[str, Any]:
        """獲取MCP系統狀態"""
        
        active_modules = len([name for name, info in self.module_capabilities.items() 
                             if info['status'] == 'active'])
        
        return {
            'mcp_version': MCPVersion.V2_0.value,
            'manager_status': 'active',
            'registered_modules': list(self.adapters.keys()),
            'active_modules': active_modules,
            'total_modules': self.stats['total_modules'],
            'system_capabilities': list(self.module_capabilities.keys()),
            'statistics': self.stats,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }

# ==================== 測試和示例 ====================

async def test_mcp_system():
    """測試MCP系統"""
    
    print("🔗 測試 MCP (Model Context Protocol) 系統")
    print("=" * 60)
    
    # 創建MCP管理器
    mcp_manager = MCPManager()
    
    # 註冊測試模組
    print("\n📝 註冊測試模組...")
    
    # 註冊感知模組
    perception_adapter = mcp_manager.register_module("perception", "v2.0")
    
    # 註冊感知模組資源
    perception_adapter.register_resource(MCPResource(
        uri="aiva://perception/scan_results",
        name="Scan Results",
        description="Current system scan analysis results",
        mime_type=MIME_TYPE_JSON,
        text="",
        access_level="read"
    ))
    
    # 註冊感知模組工具
    perception_adapter.register_tool(MCPTool(
        name="analyze_system",
        description="Analyze system state and generate insights",
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "depth": {"type": "integer", "default": 1}
            },
            "required": ["target"]
        }
    ))
    
    # 註冊認知模組
    cognition_adapter = mcp_manager.register_module("cognition", "v2.0")
    
    # 註冊認知模組資源
    cognition_adapter.register_resource(MCPResource(
        uri="aiva://cognition/capabilities",
        name="Cognitive Capabilities",
        description="Available cognitive analysis capabilities",
        mime_type=MIME_TYPE_JSON,
        text="",
        access_level="read"
    ))
    
    # 註冊認知模組工具
    cognition_adapter.register_tool(MCPTool(
        name="assess_capability",
        description="Assess system capability for specific tasks",
        input_schema={
            "type": "object",
            "properties": {
                "task_description": {"type": "string"},
                "complexity": {"type": "string", "enum": ["low", "medium", "high"]}
            },
            "required": ["task_description"]
        }
    ))
    
    # 註冊知識模組
    knowledge_adapter = mcp_manager.register_module("knowledge", "v2.0")
    
    # 註冊知識模組資源
    knowledge_adapter.register_resource(MCPResource(
        uri="aiva://knowledge/documents",
        name="Knowledge Documents",
        description="Stored knowledge base documents",
        mime_type=MIME_TYPE_JSON,
        text="",
        access_level="read"
    ))
    
    # 註冊知識模組工具
    knowledge_adapter.register_tool(MCPTool(
        name="semantic_search",
        description="Perform semantic search in knowledge base",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    ))
    
    print(f"✅ 已註冊 {len(mcp_manager.adapters)} 個模組")
    
    # 測試能力發現
    print("\n🔍 測試能力發現...")
    capabilities = await mcp_manager.discover_capabilities()
    print(f"✅ 發現能力：{capabilities['total_modules']} 個模組")
    
    for module, caps in capabilities['module_capabilities'].items():
        print(f"   📦 {module}: {len(caps.get('supported_capabilities', []))} 個能力")
    
    # 測試資源聚合
    print("\n📚 測試資源聚合...")
    resources = await mcp_manager.aggregate_resources()
    print(f"✅ 聚合資源：{resources['total_resources']} 個資源")
    
    for resource in resources['resources']:
        print(f"   📄 {resource['name']} ({resource['source_module']})")
    
    # 測試工具聚合
    print("\n🔧 測試工具聚合...")
    tools = await mcp_manager.aggregate_tools()
    print(f"✅ 聚合工具：{tools['total_tools']} 個工具")
    
    for tool in tools['tools']:
        print(f"   🛠️ {tool['name']} ({tool['source_module']})")
    
    # 測試跨模組工作流程
    print("\n🔄 測試跨模組工作流程...")
    
    workflow = [
        {
            'target_module': 'perception',
            'method': MCP_TOOLS_CALL_METHOD,
            'params': {
                'name': 'analyze_system',
                'arguments': {'target': 'system_state', 'depth': 2}
            },
            'required': True
        },
        {
            'target_module': 'cognition',
            'method': MCP_TOOLS_CALL_METHOD,
            'params': {
                'name': 'assess_capability',
                'arguments': {'task_description': 'analyze performance', 'complexity': 'medium'}
            },
            'required': True
        },
        {
            'target_module': 'knowledge',
            'method': MCP_TOOLS_CALL_METHOD,
            'params': {
                'name': 'semantic_search',
                'arguments': {'query': 'performance optimization', 'max_results': 3}
            },
            'required': False
        }
    ]
    
    workflow_result = await mcp_manager.execute_cross_module_workflow(workflow)
    print(f"✅ 工作流程執行完成：{workflow_result['overall_status']}")
    print(f"   📊 {workflow_result['completed_steps']}/{workflow_result['total_steps']} 步驟完成")
    
    # 測試MCP消息路由
    print("\n📨 測試MCP消息路由...")
    
    test_message = MCPMessage(
        method="resources/read",
        params={"uri": "aiva://perception/scan_results"},
        source_module="test_client",
        target_module="perception"
    )
    
    response = await mcp_manager.route_mcp_message(test_message)
    
    if response and response.result:
        print(f"✅ 消息路由成功：{response.result.get('name', 'Unknown')}")
    else:
        print("❌ 消息路由失敗")
    
    # 獲取MCP系統狀態
    mcp_status = mcp_manager.get_mcp_status()
    print(f"\n💚 MCP系統狀態：{mcp_status['manager_status']}")
    print(f"📈 活躍模組：{mcp_status['active_modules']}/{mcp_status['total_modules']}")
    print(f"🔄 消息交換次數：{mcp_status['statistics']['message_exchanges']}")

if __name__ == "__main__":
    asyncio.run(test_mcp_system())