"""Shell 命令執行工具

提供安全的系統命令執行能力，用於 AI 生成和執行系統層 CLI 指令。
包含白名單機制、超時控制、輸出限制等安全特性。
"""
import os
from subprocess import PIPE, STDOUT, Popen, TimeoutExpired
from typing import Any


class ShellCommandTool:
    """輕量殼層執行器（受控）

    特性：
    - 命令白名單：只允許執行預定義的安全命令
    - 超時控制：防止命令無限期執行
    - 輸出限制：限制輸出大小，防止記憶體溢出
    - 工作目錄：支援自定義工作目錄
    - 錯誤處理：完整的異常捕獲和錯誤報告

    安全考量：
    - 不預設提權，使用當前用戶權限
    - 自動拼接參數，防止命令注入
    - 返回詳細的執行結果（stdout、stderr、exit_code）

    使用範例：
        >>> tool = ShellCommandTool()
        >>> result = tool.exec("python", ["-c", "print('hello')"])
        >>> print(result['ok'], result['output'])
    """

    # 安全命令白名單
    SAFE_BIN_ALLOWLIST = {
        # Python 相關
        "python",
        "python3",
        "pip",
        "pip3",
        # Node.js 相關
        "node",
        "npm",
        "npx",
        "yarn",
        # Shell 相關
        "bash",
        "sh",
        "zsh",
        # 網路工具
        "curl",
        "wget",
        "httpx",
        "nmap",
        # 資料處理
        "jq",
        "grep",
        "awk",
        "sed",
        # Bug Bounty 工具
        "gau",
        "subfinder",
        "nuclei",
        "ffuf",
        "sqlmap",
        "nikto",
        "dirsearch",
        # 系統工具
        "ls",
        "cat",
        "echo",
        "pwd",
        "whoami",
        # Git 相關
        "git",
    }

    def __init__(
        self,
        cwd: str | None = None,
        timeout_sec: int = 120,
        max_output_size: int = 1048576,  # 1MB
    ):
        """初始化命令執行工具

        Args:
            cwd: 工作目錄路徑，預設為當前目錄
            timeout_sec: 命令超時時間（秒），預設 120 秒
            max_output_size: 最大輸出大小（bytes），預設 1MB
        """
        self.cwd = cwd or os.getcwd()
        self.timeout_sec = timeout_sec
        self.max_output_size = max_output_size

    def build(self, cmd: str, args: list[str] | None = None) -> list[str]:
        """構建命令列表

        Args:
            cmd: 命令名稱
            args: 命令參數列表

        Returns:
            完整的命令列表
        """
        parts = [cmd]
        if args:
            parts.extend(args)
        return parts

    def is_safe_command(self, cmd: str) -> bool:
        """檢查命令是否在白名單中

        Args:
            cmd: 命令名稱

        Returns:
            是否為安全命令
        """
        # 提取基礎命令名（去除路徑）
        base_cmd = os.path.basename(cmd)
        return base_cmd in self.SAFE_BIN_ALLOWLIST

    def exec(
        self,
        cmd: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        shell: bool = False,
    ) -> dict[str, Any]:
        """執行系統命令

        Args:
            cmd: 命令名稱
            args: 命令參數列表
            env: 環境變數字典
            shell: 是否使用 shell 模式（預設 False，更安全）

        Returns:
            執行結果字典，包含：
            - ok (bool): 是否成功執行
            - exit_code (int): 退出代碼
            - output (str): 標準輸出和錯誤輸出
            - error (str): 錯誤信息（如果有）
            - command (list): 執行的命令
            - timeout (bool): 是否超時
        """
        # 安全檢查
        if not self.is_safe_command(cmd):
            return {
                "ok": False,
                "error": f"命令 `{cmd}` 不在安全白名單中",
                "command": [cmd] + (args or []),
                "exit_code": -1,
            }

        command_list = self.build(cmd, args)

        # 準備環境變數
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        try:
            # 執行命令
            proc = Popen(
                command_list if not shell else " ".join(command_list),
                cwd=self.cwd,
                stdout=PIPE,
                stderr=STDOUT,
                text=True,
                env=exec_env,
                shell=shell,
            )

            try:
                # 等待命令完成，設置超時
                output, _ = proc.communicate(timeout=self.timeout_sec)

                # 限制輸出大小
                if len(output) > self.max_output_size:
                    output = output[: self.max_output_size] + "\n... [輸出已截斷] ..."

                return {
                    "ok": proc.returncode == 0,
                    "exit_code": proc.returncode,
                    "output": output,
                    "command": command_list,
                    "timeout": False,
                }

            except TimeoutExpired:
                # 超時處理
                proc.kill()
                proc.communicate()  # 清理子進程

                return {
                    "ok": False,
                    "error": f"命令執行超時（>{self.timeout_sec}秒）",
                    "command": command_list,
                    "exit_code": -1,
                    "timeout": True,
                }

        except FileNotFoundError:
            return {
                "ok": False,
                "error": f"命令 `{cmd}` 未找到，請確認是否已安裝",
                "command": command_list,
                "exit_code": -1,
            }

        except PermissionError:
            return {
                "ok": False,
                "error": f"沒有執行 `{cmd}` 的權限",
                "command": command_list,
                "exit_code": -1,
            }

        except Exception as e:
            return {
                "ok": False,
                "error": f"執行命令時發生錯誤: {str(e)}",
                "command": command_list,
                "exit_code": -1,
            }

    def exec_multiple(
        self, commands: list[dict[str, Any]], stop_on_error: bool = True
    ) -> list[dict[str, Any]]:
        """批次執行多個命令

        Args:
            commands: 命令列表，每個命令是包含 cmd 和 args 的字典
            stop_on_error: 遇到錯誤是否停止，預設 True

        Returns:
            所有命令的執行結果列表
        """
        results = []

        for cmd_dict in commands:
            cmd = cmd_dict.get("cmd")
            args = cmd_dict.get("args")
            env = cmd_dict.get("env")

            result = self.exec(cmd, args, env)
            results.append(result)

            # 如果設置了遇錯停止，且當前命令失敗
            if stop_on_error and not result["ok"]:
                break

        return results

    def test_command_available(self, cmd: str) -> bool:
        """測試命令是否可用

        Args:
            cmd: 命令名稱

        Returns:
            命令是否可用
        """
        result = self.exec(cmd, ["--version"])
        return result["ok"]


# 便捷函數
def run_command(cmd: str, args: list[str] | None = None, **kwargs) -> dict[str, Any]:
    """快速執行命令的便捷函數

    Args:
        cmd: 命令名稱
        args: 參數列表
        **kwargs: 傳遞給 ShellCommandTool 的其他參數

    Returns:
        執行結果
    """
    tool = ShellCommandTool(**kwargs)
    return tool.exec(cmd, args)


__all__ = ["ShellCommandTool", "run_command"]
