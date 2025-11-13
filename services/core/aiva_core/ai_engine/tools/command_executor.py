"""統一命令執行工具 - 整合所有命令執行功能

將原本的 CommandExecutor 和 ShellCommandTool 整合為統一接口
提供完整的命令執行能力，包含安全性、超時控制、輸出限制等特性
"""

import logging
from pathlib import Path
from typing import Any

from . import Tool
from .shell_command_tool import ShellCommandTool

logger = logging.getLogger(__name__)


class CommandExecutor(Tool):
    """統一命令執行工具 - 基於 ShellCommandTool 實現
    
    提供安全的系統命令執行能力，包含：
    - 命令白名單驗證
    - 超時控制 
    - 輸出大小限制
    - 工作目錄支持
    - 完整的錯誤處理
    
    這是原本 CommandExecutor 和 ShellCommandTool 的統一版本
    保留最佳功能，淘汰重複實現
    """
    
    def __init__(self, codebase_path: str) -> None:
        """初始化命令執行器
        
        Args:
            codebase_path: 程式碼庫根目錄，用作預設工作目錄
        """
        super().__init__(
            name="CommandExecutor",
            description="安全執行系統命令，支援白名單驗證和超時控制"
        )
        self.codebase_path = Path(codebase_path)
        
        # 使用增強的 ShellCommandTool 作為底層實現
        self.shell_tool = ShellCommandTool(
            cwd=str(self.codebase_path),
            timeout_sec=30,  # 保持原 CommandExecutor 的 30 秒超時
            max_output_size=1048576  # 1MB 輸出限制
        )
    
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """執行系統命令 - 統一接口
        
        Args:
            **kwargs: 工具參數
                command (str): 要執行的命令（必須）
                args (list): 命令參數列表（可選）
                env (dict): 環境變數字典（可選）
                shell (bool): 是否使用 shell 模式（可選，預設 False）
                
        Returns:
            執行結果字典：
            - status: 'success' | 'error'
            - command: 執行的完整命令
            - stdout: 標準輸出
            - stderr: 標準錯誤（如果分離）
            - returncode: 退出代碼
            - error: 錯誤信息（如果有）
        """
        command = kwargs.get("command", "")
        args = kwargs.get("args", [])
        env = kwargs.get("env", None)
        shell = kwargs.get("shell", False)
        
        if not command:
            return {
                "status": "error",
                "error": "缺少必需參數: command"
            }
        
        # 處理命令格式 - 使用正確的 shell 風格解析
        if isinstance(command, str) and " " in command and not args:
            # 使用 shlex 進行正確的 shell 風格解析，正確處理引號
            import shlex
            try:
                parts = shlex.split(command)
                cmd = parts[0] if parts else ""
                cmd_args = parts[1:] if len(parts) > 1 else []
            except ValueError as e:
                # shlex 解析失敗時的降級處理
                logger.warning(f"Shell 解析失敗，使用簡單分割: {e}")
                parts = command.split()
                cmd = parts[0]
                cmd_args = parts[1:] if len(parts) > 1 else []
        else:
            # 如果是單個命令加參數列表
            cmd = command
            cmd_args = args if isinstance(args, list) else []
        
        # 使用 ShellCommandTool 執行
        result = self.shell_tool.exec(cmd, cmd_args, env, shell)
        
        # 轉換結果格式以匹配原 CommandExecutor 接口
        if result["ok"]:
            return {
                "status": "success",
                "command": " ".join(result["command"]),
                "stdout": result.get("output", ""),
                "stderr": "",  # ShellCommandTool 合併了 stdout/stderr
                "returncode": result.get("exit_code", 0)
            }
        else:
            return {
                "status": "error", 
                "command": " ".join(result.get("command", [cmd] + cmd_args)),
                "error": result.get("error", "Unknown error"),
                "returncode": result.get("exit_code", -1)
            }
    
    def is_command_safe(self, command: str) -> bool:
        """檢查命令是否安全（在白名單中）
        
        Args:
            command: 命令名稱
            
        Returns:
            是否為安全命令
        """
        # 提取基礎命令名
        cmd = command.split()[0] if " " in command else command
        return self.shell_tool.is_safe_command(cmd)
    
    def get_safe_commands(self) -> set[str]:
        """獲取安全命令白名單
        
        Returns:
            允許執行的命令集合
        """
        return self.shell_tool.SAFE_BIN_ALLOWLIST.copy()


# 向後兼容別名
UnifiedCommandExecutor = CommandExecutor

__all__ = ["CommandExecutor", "UnifiedCommandExecutor"]