"""統一工具系統 - 整合所有AI代理工具

將原本的 tools.py 中的工具類別重新組織到 tools/ 目錄下
提供統一的工具接口和管理系統，消除重複實現
"""

from abc import ABC, abstractmethod
from typing import Any

# 導入所有整合後的工具
from .command_executor import CommandExecutor
from .code_reader import CodeReader
from .code_writer import CodeWriter
from .code_analyzer import CodeAnalyzer
from .shell_command_tool import ShellCommandTool
from .system_status_tool import SystemStatusTool

__all__ = [
    "Tool", 
    "CommandExecutor", 
    "CodeReader", 
    "CodeWriter", 
    "CodeAnalyzer", 
    "ShellCommandTool", 
    "SystemStatusTool"
]


class Tool(ABC):
    """工具基礎抽象類別 - 所有AI工具的統一接口"""

    def __init__(self, name: str, description: str) -> None:
        """初始化工具
        
        Args:
            name: 工具名稱
            description: 工具描述
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """執行工具
        
        Args:
            **kwargs: 工具參數
            
        Returns:
            執行結果字典，包含:
            - status: 'success' | 'error' 
            - 其他工具特定的返回數據
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def get_info(self) -> dict[str, str]:
        """獲取工具基本信息"""
        return {
            "name": self.name,
            "description": self.description,
            "class": self.__class__.__name__
        }


# 工具管理器 - 提供統一的工具訪問點
class ToolManager:
    """工具管理器 - 管理所有AI工具實例"""
    
    def __init__(self, codebase_path: str):
        """初始化工具管理器
        
        Args:
            codebase_path: 程式碼庫根目錄
        """
        self.codebase_path = codebase_path
        self._tools = {}
        self._initialize_tools()
    
    def _initialize_tools(self) -> None:
        """初始化所有工具實例"""
        self._tools = {
            "code_reader": CodeReader(self.codebase_path),
            "code_writer": CodeWriter(self.codebase_path),
            "code_analyzer": CodeAnalyzer(self.codebase_path),
            "command_executor": CommandExecutor(self.codebase_path),
            "shell_command": ShellCommandTool(cwd=self.codebase_path),
            "system_status": SystemStatusTool(),
        }
    
    def get_tool(self, tool_name: str) -> Tool | None:
        """獲取指定工具實例
        
        Args:
            tool_name: 工具名稱
            
        Returns:
            工具實例或 None
        """
        return self._tools.get(tool_name)
    
    def list_tools(self) -> dict[str, dict[str, str]]:
        """列出所有可用工具
        
        Returns:
            工具信息字典
        """
        return {name: tool.get_info() for name, tool in self._tools.items()}
    
    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """執行指定工具
        
        Args:
            tool_name: 工具名稱
            **kwargs: 工具參數
            
        Returns:
            執行結果
        """
        tool = self.get_tool(tool_name)
        if tool is None:
            return {
                "status": "error",
                "error": f"工具 '{tool_name}' 不存在"
            }
        
        return tool.execute(**kwargs)