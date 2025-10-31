"""
HackingTool SQL 工具管理器
提供 HackingTool SQL 工具的安裝、配置和狀態管理功能

這個管理器負責：
1. 檢查工具安裝狀態
2. 自動安裝缺失的工具
3. 配置工具執行環境
4. 監控工具執行狀態
"""

import asyncio
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from services.aiva_common.utils.logging import get_logger
from services.aiva_common.utils.ids import new_id

from .hackingtool_config import (
    HACKINGTOOL_SQL_CONFIGS, HackingToolSQLConfig,
    SQLToolType, sql_integrator
)

logger = get_logger(__name__)


class HackingToolSQLManager:
    """HackingTool SQL 工具管理器"""
    
    def __init__(self, tools_dir: Optional[Path] = None):
        self.tools_dir = tools_dir or Path.cwd() / "hackingtool_sql_tools"
        self.configs = HACKINGTOOL_SQL_CONFIGS
        self.trace_id = new_id("hackingtool_manager")
        
        # 確保工具目錄存在
        self.tools_dir.mkdir(exist_ok=True)
        
        logger.info("HackingTool SQL 管理器已初始化", 
                   extra={"tools_dir": str(self.tools_dir), "trace_id": self.trace_id})
    
    async def check_all_tools_status(self) -> Dict[str, Dict[str, Any]]:
        """檢查所有工具的狀態"""
        status_report = {}
        
        for tool_name, config in self.configs.items():
            tool_status = await self._check_tool_status(tool_name, config)
            status_report[tool_name] = tool_status
        
        return status_report
    
    async def _check_tool_status(self, tool_name: str, config: HackingToolSQLConfig) -> Dict[str, Any]:
        """檢查單個工具的狀態"""
        status = {
            "name": tool_name,
            "title": config.title,
            "installed": False,
            "dependencies_met": False,
            "executable": False,
            "last_check": datetime.now().isoformat(),
            "install_path": None,
            "missing_dependencies": [],
            "error": None
        }
        
        try:
            # 檢查依賴
            missing_deps = []
            for dep in config.dependencies:
                if not shutil.which(dep):
                    missing_deps.append(dep)
            
            status["missing_dependencies"] = missing_deps
            status["dependencies_met"] = len(missing_deps) == 0
            
            # 檢查安裝路徑
            tool_path = self.tools_dir / tool_name
            status["install_path"] = str(tool_path)
            status["installed"] = tool_path.exists()
            
            # 檢查是否可執行（簡單測試）
            if status["installed"] and status["dependencies_met"]:
                executable = await self._test_tool_executable(tool_name, config)
                status["executable"] = executable
            
        except Exception as e:
            status["error"] = str(e)
            logger.error(f"檢查工具 {tool_name} 狀態時發生錯誤: {e}", trace_id=self.trace_id)
        
        return status
    
    async def _test_tool_executable(self, tool_name: str, config: HackingToolSQLConfig) -> bool:
        """測試工具是否可執行"""
        if not config.run_commands:
            return False
        
        try:
            # 使用簡單的版本檢查或幫助命令測試
            test_cmd = config.run_commands[0].replace("{target}", "").strip()
            if "sqlmap" in tool_name.lower():
                test_cmd = f"cd {self.tools_dir / tool_name} && python3 sqlmap.py --version"
            elif test_cmd.endswith("--help") or test_cmd.endswith("-h"):
                pass  # 保持原命令
            else:
                test_cmd += " --help"
            
            process = await asyncio.create_subprocess_shell(
                test_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.tools_dir
            )
            
            _, _ = await asyncio.wait_for(process.communicate(), timeout=10)
            
            # 如果命令執行成功或返回預期的幫助信息，認為工具可執行
            return process.returncode in [0, 1, 2]  # 很多工具的 --help 返回非零
            
        except asyncio.TimeoutError:
            logger.warning(f"測試工具 {tool_name} 超時", trace_id=self.trace_id)
            return False
        except Exception as e:
            logger.warning(f"測試工具 {tool_name} 可執行性失敗: {e}", trace_id=self.trace_id)
            return False
    
    async def install_tool(self, tool_name: str, force_reinstall: bool = False) -> Dict[str, Any]:
        """安裝指定工具"""
        if tool_name not in self.configs:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        config = self.configs[tool_name]
        install_path = self.tools_dir / tool_name
        
        result = {
            "tool": tool_name,
            "success": False,
            "install_path": str(install_path),
            "steps_completed": [],
            "error": None,
            "duration": 0
        }
        
        start_time = datetime.now()
        
        try:
            # 檢查是否已安裝
            if install_path.exists() and not force_reinstall:
                result["success"] = True
                result["error"] = "Tool already installed (use force_reinstall=True to reinstall)"
                return result
            
            # 清理舊安裝
            if install_path.exists():
                shutil.rmtree(install_path)
                result["steps_completed"].append("cleaned_old_installation")
            
            # 創建工具目錄
            install_path.mkdir(exist_ok=True)
            result["steps_completed"].append("created_directory")
            
            # 執行安裝命令
            for i, cmd in enumerate(config.install_commands):
                logger.info(f"執行安裝命令 {i+1}/{len(config.install_commands)}: {cmd}")
                
                process = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self.tools_dir
                )
                
                _, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=config.timeout_seconds
                )
                
                if process.returncode != 0:
                    error_msg = f"Install command failed: {cmd}\nStderr: {stderr.decode()}"
                    result["error"] = error_msg
                    logger.error(error_msg, trace_id=self.trace_id)
                    return result
                
                result["steps_completed"].append(f"install_command_{i+1}")
            
            # 驗證安裝
            tool_status = await self._check_tool_status(tool_name, config)
            if tool_status["installed"]:
                result["success"] = True
                result["steps_completed"].append("installation_verified")
            else:
                result["error"] = "Installation completed but tool not found"
            
        except asyncio.TimeoutError:
            result["error"] = f"Installation timeout after {config.timeout_seconds} seconds"
        except Exception as e:
            result["error"] = f"Installation failed: {str(e)}"
            logger.error(f"安裝工具 {tool_name} 時發生錯誤: {e}", trace_id=self.trace_id)
        
        finally:
            result["duration"] = (datetime.now() - start_time).total_seconds()
        
        return result
    
    async def install_all_tools(self, skip_existing: bool = True) -> Dict[str, Dict[str, Any]]:
        """安裝所有工具"""
        results = {}
        
        logger.info("開始批量安裝 HackingTool SQL 工具", trace_id=self.trace_id)
        
        for tool_name in self.configs:
            logger.info(f"安裝工具: {tool_name}")
            result = await self.install_tool(tool_name, force_reinstall=not skip_existing)
            results[tool_name] = result
            
            if result["success"]:
                logger.info(f"工具 {tool_name} 安裝成功")
            else:
                logger.error(f"工具 {tool_name} 安裝失敗: {result.get('error', 'Unknown error')}")
        
        # 生成安裝報告
        successful = sum(1 for r in results.values() if r["success"])
        total = len(results)
        
        logger.info(f"批量安裝完成: {successful}/{total} 工具安裝成功", trace_id=self.trace_id)
        
        return results
    
    def uninstall_tool(self, tool_name: str) -> Dict[str, Any]:
        """卸載指定工具"""
        if tool_name not in self.configs:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        install_path = self.tools_dir / tool_name
        
        try:
            if install_path.exists():
                shutil.rmtree(install_path)
                logger.info(f"工具 {tool_name} 已卸載", trace_id=self.trace_id)
                return {"success": True, "message": f"Tool {tool_name} uninstalled"}
            else:
                return {"success": True, "message": f"Tool {tool_name} was not installed"}
        
        except Exception as e:
            error_msg = f"Failed to uninstall {tool_name}: {str(e)}"
            logger.error(error_msg, trace_id=self.trace_id)
            return {"success": False, "error": error_msg}
    
    async def get_tool_recommendations(self, target_type: str = "web") -> List[str]:
        """根據目標類型推薦工具"""
        recommendations = []
        
        # 基於目標類型和工具優先級的推薦邏輯
        if target_type.lower() in ["web", "webapp", "website"]:
            # Web 應用推薦順序
            priority_tools = [
                ("sqlmap", "最全面的 SQL 注入工具"),
                ("dsss", "快速輕量级掃描"),
                ("nosqlmap", "NoSQL 數據庫檢測")
            ]
        elif target_type.lower() in ["api", "rest", "graphql"]:
            # API 推薦順序
            priority_tools = [
                ("sqlmap", "支援各種 API 格式"),
                ("nosqlmap", "適合現代 API 後端"),
                ("blisqy", "時間盲注檢測")
            ]
        else:
            # 通用推薦
            priority_tools = [
                ("sqlmap", "業界標準工具"),
                ("dsss", "快速掃描")
            ]
        
        # 檢查推薦工具的可用性
        for tool_name, reason in priority_tools:
            if tool_name in self.configs:
                tool_status = await self._check_tool_status(tool_name, self.configs[tool_name])
                recommendations.append({
                    "tool": tool_name,
                    "reason": reason,
                    "available": tool_status["installed"] and tool_status["executable"],
                    "priority": self.configs[tool_name].priority
                })
        
        # 按優先級排序
        recommendations.sort(key=lambda x: x["priority"])
        
        return recommendations
    
    def get_installation_script(self, tool_names: Optional[List[str]] = None) -> str:
        """生成安裝腳本"""
        if tool_names is None:
            tool_names = list(self.configs.keys())
        
        script_lines = [
            "#!/bin/bash",
            "# HackingTool SQL 工具自動安裝腳本",
            f"# 生成時間: {datetime.now().isoformat()}",
            "",
            f"TOOLS_DIR=\"{self.tools_dir}\"",
            "mkdir -p \"$TOOLS_DIR\"",
            "cd \"$TOOLS_DIR\"",
            ""
        ]
        
        for tool_name in tool_names:
            if tool_name not in self.configs:
                continue
            
            config = self.configs[tool_name]
            script_lines.extend([
                f"# 安裝 {config.title}",
                f"echo '正在安裝 {tool_name}...'",
                ""
            ])
            
            for cmd in config.install_commands:
                script_lines.append(cmd)
            
            script_lines.extend(["", f"echo '{tool_name} 安裝完成'", ""])
        
        script_lines.append("echo '所有工具安裝完成！'")
        
        return "\n".join(script_lines)
    
    async def generate_status_report(self) -> Dict[str, Any]:
        """生成詳細的狀態報告"""
        status_data = await self.check_all_tools_status()
        
        # 統計資訊
        total_tools = len(status_data)
        installed_tools = sum(1 for s in status_data.values() if s["installed"])
        executable_tools = sum(1 for s in status_data.values() if s["executable"])
        
        # 按類型分類
        by_type = {}
        for tool_name, config in self.configs.items():
            tool_type = config.tool_type.value
            if tool_type not in by_type:
                by_type[tool_type] = []
            by_type[tool_type].append({
                "name": tool_name,
                "status": status_data[tool_name]
            })
        
        report = {
            "summary": {
                "total_tools": total_tools,
                "installed_tools": installed_tools,
                "executable_tools": executable_tools,
                "installation_rate": f"{installed_tools}/{total_tools} ({installed_tools/total_tools*100:.1f}%)",
                "executable_rate": f"{executable_tools}/{total_tools} ({executable_tools/total_tools*100:.1f}%)"
            },
            "by_type": by_type,
            "detailed_status": status_data,
            "tools_directory": str(self.tools_dir),
            "generated_at": datetime.now().isoformat()
        }
        
        return report


# 全域管理器實例
sql_tool_manager = HackingToolSQLManager()