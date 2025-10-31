#!/usr/bin/env python3
"""
AIVA 工具生命週期管理器
基於 HackingTool tool_manager.py 的設計模式

功能:
- 自動化工具安裝
- 工具更新管理
- 工具卸載清理
- 依賴關係管理
- 生命週期監控
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import shutil
import tempfile
import aiofiles

from aiva_common.utils.logging import get_logger
from aiva_common.utils.ids import new_id
from aiva_common.enums import ProgrammingLanguage

from .models import CapabilityRecord, CapabilityStatus
from .registry import CapabilityRegistry
from .toolkit import CapabilityToolkit


logger = get_logger(__name__)

# 常數定義
AIVA_TOOLS_DIR = ".aiva"


@dataclass
class ToolLifecycleEvent:
    """工具生命週期事件記錄"""
    event_id: str
    capability_id: str
    event_type: str  # install, update, uninstall, health_check
    timestamp: datetime
    status: str  # success, failed, in_progress
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class InstallationResult:
    """安裝結果"""
    success: bool
    capability_id: str
    installation_path: Optional[str] = None
    installed_version: Optional[str] = None
    dependencies_installed: Optional[List[str]] = field(default_factory=list)
    error_message: Optional[str] = None
    installation_time_seconds: float = 0.0


class ToolLifecycleManager:
    """
    工具生命週期管理器
    基於 HackingTool 的 UpdateTool 和 UninstallTool 設計模式
    """
    
    def __init__(self):
        self.registry = CapabilityRegistry()
        self.toolkit = CapabilityToolkit()
        self.trace_id = new_id("lifecycle")
        self.events: List[ToolLifecycleEvent] = []
        
        # 安裝路徑配置
        self.installation_paths = {
            ProgrammingLanguage.PYTHON: Path.home() / AIVA_TOOLS_DIR / "tools" / "python",
            ProgrammingLanguage.GO: Path.home() / AIVA_TOOLS_DIR / "tools" / "go",
            ProgrammingLanguage.RUST: Path.home() / AIVA_TOOLS_DIR / "tools" / "rust",
            ProgrammingLanguage.RUBY: Path.home() / AIVA_TOOLS_DIR / "tools" / "ruby",
            ProgrammingLanguage.JAVASCRIPT: Path.home() / AIVA_TOOLS_DIR / "tools" / "nodejs",
            ProgrammingLanguage.PHP: Path.home() / AIVA_TOOLS_DIR / "tools" / "php",
        }
        
        # 確保安裝路徑存在
        for path in self.installation_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"工具生命週期管理器已初始化 (trace_id: {self.trace_id})")
    
    def _record_event(
        self, 
        capability_id: str, 
        event_type: str, 
        status: str,
        details: Dict[str, Any],
        error_message: Optional[str] = None
    ) -> None:
        """記錄生命週期事件"""
        event = ToolLifecycleEvent(
            event_id=new_id("event"),
            capability_id=capability_id,
            event_type=event_type,
            timestamp=datetime.now(),
            status=status,
            details=details,
            error_message=error_message
        )
        self.events.append(event)
        
        logger.info(
            f"記錄生命週期事件: {event_type} - {capability_id} - {status} (trace_id: {self.trace_id})"
        )
    
    async def install_tool(
        self, 
        capability_id: str,
        force_reinstall: bool = False
    ) -> InstallationResult:
        """
        自動安裝工具
        基於 HackingTool 的安裝模式
        """
        start_time = datetime.now()
        
        try:
            # 獲取能力記錄
            capability = await self.registry.get_capability(capability_id)
            if not capability:
                error_msg = f"能力 '{capability_id}' 不存在"
                self._record_event(capability_id, "install", "failed", {}, error_msg)
                return InstallationResult(False, capability_id, error_message=error_msg)
            
            # 檢查是否已安裝 (除非強制重裝)
            if not force_reinstall and self._is_tool_installed(capability):
                self._record_event(capability_id, "install", "skipped", 
                                 {"reason": "already_installed"})
                return InstallationResult(
                    True, capability_id, 
                    installation_path=str(self._get_installation_path(capability))
                )
            
            logger.info(f"開始安裝工具: {capability.name} (capability_id: {capability_id})")
            
            # 根據語言類型選擇安裝方法
            installation_result = await self._install_by_language(capability)
            
            if installation_result.success:
                # 更新能力狀態
                self._update_capability_status(capability_id, CapabilityStatus.HEALTHY)
                
                # 記錄成功事件
                self._record_event(
                    capability_id, "install", "success",
                    {
                        "installation_path": installation_result.installation_path,
                        "version": installation_result.installed_version,
                        "dependencies": installation_result.dependencies_installed or []
                    }
                )
                
                logger.info(f"工具安裝成功: {capability.name} (capability_id: {capability_id})")
            else:
                # 記錄失敗事件
                self._record_event(
                    capability_id, "install", "failed", {},
                    installation_result.error_message
                )
                
                logger.error(
                    f"工具安裝失敗: {capability.name} (capability_id: {capability_id}) - {installation_result.error_message}"
                )
            
            # 計算安裝時間
            installation_result.installation_time_seconds = (
                datetime.now() - start_time
            ).total_seconds()
            
            return installation_result
            
        except Exception as e:
            error_msg = f"安裝過程中出現異常: {str(e)}"
            self._record_event(capability_id, "install", "failed", {}, error_msg)
            
            logger.error(
                f"工具安裝異常 (capability_id: {capability_id}) - {error_msg}",
                exc_info=True
            )
            
            return InstallationResult(False, capability_id, error_message=error_msg)
    
    async def _install_by_language(self, capability: CapabilityRecord) -> InstallationResult:
        """根據程式語言選擇安裝方法"""
        
        if capability.language == ProgrammingLanguage.PYTHON:
            return await self._install_python_tool(capability)
        elif capability.language == ProgrammingLanguage.GO:
            return await self._install_go_tool(capability)
        elif capability.language == ProgrammingLanguage.RUST:
            return await self._install_rust_tool(capability)
        elif capability.language == ProgrammingLanguage.RUBY:
            return await self._install_ruby_tool(capability)
        elif capability.language == ProgrammingLanguage.JAVASCRIPT:
            return await self._install_nodejs_tool(capability)
        elif capability.language == ProgrammingLanguage.PHP:
            return await self._install_php_tool(capability)
        else:
            return InstallationResult(
                False, capability.id,
                error_message=f"不支援的程式語言: {capability.language.value}"
            )
    
    async def _install_python_tool(self, capability: CapabilityRecord) -> InstallationResult:
        """安裝 Python 工具"""
        try:
            installation_path = self._get_installation_path(capability)
            
            # 創建虛擬環境
            venv_path = installation_path / "venv"
            if not venv_path.exists():
                await self._run_command([
                    sys.executable, "-m", "venv", str(venv_path)
                ])
            
            # 確定 pip 路徑
            if os.name == 'nt':  # Windows
                pip_path = venv_path / "Scripts" / "pip.exe"
            else:  # Unix/Linux/macOS
                pip_path = venv_path / "bin" / "pip"
            
            installed_packages = []
            
            # 安裝依賴
            if capability.dependencies:
                for dep in capability.dependencies:
                    await self._run_command([str(pip_path), "install", dep])
                    installed_packages.append(dep)
            
            # 如果有特定的安裝命令，執行它
            if capability.entrypoint and capability.entrypoint.endswith('.py'):
                # 如果是 Python 腳本，複製到安裝目錄
                source_path = Path(capability.entrypoint)
                if source_path.exists():
                    dest_path = installation_path / source_path.name
                    shutil.copy2(source_path, dest_path)
            
            return InstallationResult(
                True, capability.id,
                installation_path=str(installation_path),
                installed_version="latest",
                dependencies_installed=installed_packages
            )
            
        except Exception as e:
            return InstallationResult(
                False, capability.id,
                error_message=f"Python 工具安裝失敗: {str(e)}"
            )
    
    async def _install_go_tool(self, capability: CapabilityRecord) -> InstallationResult:
        """安裝 Go 工具"""
        try:
            installation_path = self._get_installation_path(capability)
            
            # 檢查是否有 go.mod 文件
            if capability.entrypoint and "go.mod" in capability.entrypoint:
                module_dir = Path(capability.entrypoint).parent
                
                # 複製模組到安裝目錄
                tool_dir = installation_path / capability.id
                if tool_dir.exists():
                    shutil.rmtree(tool_dir)
                shutil.copytree(module_dir, tool_dir)
                
                # 在安裝目錄中構建
                await self._run_command(["go", "mod", "tidy"], cwd=str(tool_dir))
                await self._run_command(["go", "build", "."], cwd=str(tool_dir))
                
                return InstallationResult(
                    True, capability.id,
                    installation_path=str(tool_dir),
                    installed_version="latest"
                )
            else:
                return InstallationResult(
                    False, capability.id,
                    error_message="找不到 go.mod 文件"
                )
                
        except Exception as e:
            return InstallationResult(
                False, capability.id,
                error_message=f"Go 工具安裝失敗: {str(e)}"
            )
    
    async def _install_rust_tool(self, capability: CapabilityRecord) -> InstallationResult:
        """安裝 Rust 工具"""
        try:
            installation_path = self._get_installation_path(capability)
            
            # 檢查是否有 Cargo.toml 文件
            if capability.entrypoint and "Cargo.toml" in capability.entrypoint:
                project_dir = Path(capability.entrypoint).parent
                
                # 複製專案到安裝目錄
                tool_dir = installation_path / capability.id
                if tool_dir.exists():
                    shutil.rmtree(tool_dir)
                shutil.copytree(project_dir, tool_dir)
                
                # 在安裝目錄中構建
                await self._run_command(["cargo", "build", "--release"], cwd=str(tool_dir))
                
                return InstallationResult(
                    True, capability.id,
                    installation_path=str(tool_dir),
                    installed_version="latest"
                )
            else:
                return InstallationResult(
                    False, capability.id,
                    error_message="找不到 Cargo.toml 文件"
                )
                
        except Exception as e:
            return InstallationResult(
                False, capability.id,
                error_message=f"Rust 工具安裝失敗: {str(e)}"
            )
    
    async def _install_ruby_tool(self, capability: CapabilityRecord) -> InstallationResult:
        """安裝 Ruby 工具"""
        try:
            installation_path = self._get_installation_path(capability)
            
            installed_gems = []
            
            # 安裝依賴 gems
            if capability.dependencies:
                for dep in capability.dependencies:
                    await self._run_command(["gem", "install", dep])
                    installed_gems.append(dep)
            
            # 如果有 Ruby 腳本，複製到安裝目錄
            if capability.entrypoint and capability.entrypoint.endswith('.rb'):
                source_path = Path(capability.entrypoint)
                if source_path.exists():
                    dest_path = installation_path / source_path.name
                    shutil.copy2(source_path, dest_path)
            
            return InstallationResult(
                True, capability.id,
                installation_path=str(installation_path),
                installed_version="latest",
                dependencies_installed=installed_gems
            )
            
        except Exception as e:
            return InstallationResult(
                False, capability.id,
                error_message=f"Ruby 工具安裝失敗: {str(e)}"
            )
    
    async def _install_nodejs_tool(self, capability: CapabilityRecord) -> InstallationResult:
        """安裝 Node.js 工具"""
        try:
            installation_path = self._get_installation_path(capability)
            
            # 創建 package.json 如果不存在
            package_json = installation_path / "package.json"
            if not package_json.exists():
                package_data = {
                    "name": capability.id,
                    "version": "1.0.0",
                    "description": capability.description,
                    "dependencies": {}
                }
                
                async with aiofiles.open(package_json, 'w') as f:
                    await f.write(json.dumps(package_data, indent=2))
            
            installed_packages = []
            
            # 安裝依賴
            if capability.dependencies:
                for dep in capability.dependencies:
                    await self._run_command(
                        ["npm", "install", dep], 
                        cwd=str(installation_path)
                    )
                    installed_packages.append(dep)
            
            # 複製 JavaScript 文件
            if capability.entrypoint and capability.entrypoint.endswith('.js'):
                source_path = Path(capability.entrypoint)
                if source_path.exists():
                    dest_path = installation_path / source_path.name
                    shutil.copy2(source_path, dest_path)
            
            return InstallationResult(
                True, capability.id,
                installation_path=str(installation_path),
                installed_version="latest",
                dependencies_installed=installed_packages
            )
            
        except Exception as e:
            return InstallationResult(
                False, capability.id,
                error_message=f"Node.js 工具安裝失敗: {str(e)}"
            )
    
    async def _install_php_tool(self, capability: CapabilityRecord) -> InstallationResult:
        """安裝 PHP 工具"""
        try:
            installation_path = self._get_installation_path(capability)
            
            installed_packages = []
            
            # 如果有 composer.json，使用 Composer 安裝依賴
            if capability.dependencies:
                composer_json = installation_path / "composer.json"
                if not composer_json.exists():
                    composer_data = {
                        "name": f"aiva/{capability.id}",
                        "description": capability.description,
                        "require": {}
                    }
                    
                    for dep in capability.dependencies:
                        composer_data["require"][dep] = "*"
                    
                    async with aiofiles.open(composer_json, 'w') as f:
                        await f.write(json.dumps(composer_data, indent=2))
                
                await self._run_command(
                    ["composer", "install"], 
                    cwd=str(installation_path)
                )
                installed_packages = capability.dependencies
            
            # 複製 PHP 文件
            if capability.entrypoint and capability.entrypoint.endswith('.php'):
                source_path = Path(capability.entrypoint)
                if source_path.exists():
                    dest_path = installation_path / source_path.name
                    shutil.copy2(source_path, dest_path)
            
            return InstallationResult(
                True, capability.id,
                installation_path=str(installation_path),
                installed_version="latest",
                dependencies_installed=installed_packages
            )
            
        except Exception as e:
            return InstallationResult(
                False, capability.id,
                error_message=f"PHP 工具安裝失敗: {str(e)}"
            )
    
    async def update_tool(self, capability_id: str) -> bool:
        """
        更新工具
        基於 HackingTool 的 UpdateTool.update_ht() 模式
        """
        try:
            capability = await self.registry.get_capability(capability_id)
            if not capability:
                logger.error(f"能力 '{capability_id}' 不存在")
                return False
            
            logger.info(f"開始更新工具: {capability.name} (capability_id: {capability_id})")
            
            # 記錄更新開始
            self._record_event(capability_id, "update", "in_progress", {})
            
            # 檢查工具是否已安裝
            if not self._is_tool_installed(capability):
                # 如果未安裝，執行安裝
                result = await self.install_tool(capability_id)
                return result.success
            
            # 執行更新邏輯
            update_success = await self._perform_update(capability)
            
            if update_success:
                # 更新能力的最後更新時間
                self._update_capability_timestamp(capability_id)
                
                self._record_event(capability_id, "update", "success", {})
                logger.info(f"工具更新成功: {capability.name} (capability_id: {capability_id})")
            else:
                self._record_event(capability_id, "update", "failed", {})
                logger.error(f"工具更新失敗: {capability.name} (capability_id: {capability_id})")
            
            return update_success
            
        except Exception as e:
            error_msg = f"更新過程中出現異常: {str(e)}"
            self._record_event(capability_id, "update", "failed", {}, error_msg)
            
            logger.error(
                f"工具更新異常 (capability_id: {capability_id}) - {error_msg}",
                exc_info=True
            )
            
            return False
    
    async def _perform_update(self, capability: CapabilityRecord) -> bool:
        """執行具體的更新操作"""
        try:
            if capability.language == ProgrammingLanguage.PYTHON:
                return await self._update_python_tool(capability)
            elif capability.language == ProgrammingLanguage.GO:
                return await self._update_go_tool(capability)
            elif capability.language == ProgrammingLanguage.RUST:
                return await self._update_rust_tool(capability)
            elif capability.language == ProgrammingLanguage.RUBY:
                return await self._update_ruby_tool(capability)
            elif capability.language == ProgrammingLanguage.JAVASCRIPT:
                return await self._update_nodejs_tool(capability)
            elif capability.language == ProgrammingLanguage.PHP:
                return await self._update_php_tool(capability)
            else:
                logger.warning(f"不支援更新的語言: {capability.language.value}")
                return False
                
        except Exception as e:
            logger.error(f"執行更新失敗: {str(e)}", exc_info=True)
            return False
    
    async def _update_python_tool(self, capability: CapabilityRecord) -> bool:
        """更新 Python 工具"""
        installation_path = self._get_installation_path(capability)
        venv_path = installation_path / "venv"
        
        if os.name == 'nt':
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = venv_path / "bin" / "pip"
        
        try:
            # 更新 pip 本身
            await self._run_command([str(pip_path), "install", "--upgrade", "pip"])
            
            # 更新所有依賴
            if capability.dependencies:
                for dep in capability.dependencies:
                    await self._run_command([str(pip_path), "install", "--upgrade", dep])
            
            return True
        except Exception as e:
            logger.error(f"Python 工具更新失敗: {str(e)}")
            return False
    
    async def _update_go_tool(self, capability: CapabilityRecord) -> bool:
        """更新 Go 工具"""
        installation_path = self._get_installation_path(capability)
        tool_dir = installation_path / capability.id
        
        try:
            # 更新模組依賴
            await self._run_command(["go", "get", "-u", "all"], cwd=str(tool_dir))
            await self._run_command(["go", "mod", "tidy"], cwd=str(tool_dir))
            
            # 重新構建
            await self._run_command(["go", "build", "."], cwd=str(tool_dir))
            
            return True
        except Exception as e:
            logger.error(f"Go 工具更新失敗: {str(e)}")
            return False
    
    async def _update_rust_tool(self, capability: CapabilityRecord) -> bool:
        """更新 Rust 工具"""
        installation_path = self._get_installation_path(capability)
        tool_dir = installation_path / capability.id
        
        try:
            # 更新依賴並重新構建
            await self._run_command(["cargo", "update"], cwd=str(tool_dir))
            await self._run_command(["cargo", "build", "--release"], cwd=str(tool_dir))
            
            return True
        except Exception as e:
            logger.error(f"Rust 工具更新失敗: {str(e)}")
            return False
    
    async def _update_ruby_tool(self, capability: CapabilityRecord) -> bool:
        """更新 Ruby 工具"""
        try:
            # 更新所有 gems
            if capability.dependencies:
                for dep in capability.dependencies:
                    await self._run_command(["gem", "update", dep])
            
            return True
        except Exception as e:
            logger.error(f"Ruby 工具更新失敗: {str(e)}")
            return False
    
    async def _update_nodejs_tool(self, capability: CapabilityRecord) -> bool:
        """更新 Node.js 工具"""
        installation_path = self._get_installation_path(capability)
        
        try:
            # 更新 npm 包
            await self._run_command(["npm", "update"], cwd=str(installation_path))
            
            return True
        except Exception as e:
            logger.error(f"Node.js 工具更新失敗: {str(e)}")
            return False
    
    async def _update_php_tool(self, capability: CapabilityRecord) -> bool:
        """更新 PHP 工具"""
        installation_path = self._get_installation_path(capability)
        
        try:
            # 使用 Composer 更新依賴
            await self._run_command(["composer", "update"], cwd=str(installation_path))
            
            return True
        except Exception as e:
            logger.error(f"PHP 工具更新失敗: {str(e)}")
            return False
    
    async def uninstall_tool(self, capability_id: str, remove_dependencies: bool = False) -> bool:
        """
        卸載工具
        基於 HackingTool 的 UninstallTool.uninstall() 模式
        """
        try:
            capability = await self.registry.get_capability(capability_id)
            if not capability:
                logger.error(f"能力 '{capability_id}' 不存在")
                return False
            
            logger.info(f"開始卸載工具: {capability.name} (capability_id: {capability_id})")
            
            # 記錄卸載開始
            self._record_event(capability_id, "uninstall", "in_progress", {})
            
            # 執行卸載
            uninstall_success = await self._perform_uninstall(capability, remove_dependencies)
            
            if uninstall_success:
                # 更新能力狀態
                self._update_capability_status(capability_id, CapabilityStatus.UNAVAILABLE)
                
                self._record_event(capability_id, "uninstall", "success", {})
                logger.info(f"工具卸載成功: {capability.name} (capability_id: {capability_id})")
            else:
                self._record_event(capability_id, "uninstall", "failed", {})
                logger.error(f"工具卸載失敗: {capability.name} (capability_id: {capability_id})")
            
            return uninstall_success
            
        except Exception as e:
            error_msg = f"卸載過程中出現異常: {str(e)}"
            self._record_event(capability_id, "uninstall", "failed", {}, error_msg)
            
            logger.error(
                f"工具卸載異常 (capability_id: {capability_id}) - {error_msg}",
                exc_info=True
            )
            
            return False
    
    async def _perform_uninstall(self, capability: CapabilityRecord, remove_dependencies: bool) -> bool:
        """執行具體的卸載操作"""
        try:
            installation_path = self._get_installation_path(capability)
            
            # 刪除安裝目錄
            if installation_path.exists():
                shutil.rmtree(installation_path)
                logger.info(f"已刪除安裝目錄: {installation_path}")
            
            # 如果需要移除依賴，根據語言類型執行相應操作
            if remove_dependencies:
                await self._remove_dependencies(capability)
            
            return True
            
        except Exception as e:
            logger.error(f"執行卸載失敗: {str(e)}", exc_info=True)
            return False
    
    async def _remove_dependencies(self, capability: CapabilityRecord) -> None:
        """移除工具依賴"""
        try:
            if capability.language == ProgrammingLanguage.PYTHON:
                # Python 依賴在虛擬環境中，刪除目錄即可
                pass
            elif capability.language == ProgrammingLanguage.RUBY:
                # 卸載 Ruby gems (謹慎操作)
                if capability.dependencies:
                    for dep in capability.dependencies:
                        try:
                            await self._run_command(["gem", "uninstall", dep, "-x"])
                        except Exception:
                            pass  # 忽略卸載失敗
            elif capability.language == ProgrammingLanguage.JAVASCRIPT:
                # Node.js 依賴在本地目錄中，刪除目錄即可
                pass
            elif capability.language == ProgrammingLanguage.PHP:
                # PHP 依賴在本地目錄中，刪除目錄即可
                pass
        except Exception as e:
            logger.warning(f"移除依賴時出現錯誤: {str(e)}")
    
    async def health_check_tool(self, capability_id: str) -> Dict[str, Any]:
        """工具健康檢查"""
        try:
            capability = await self.registry.get_capability(capability_id)
            if not capability:
                return {
                    "success": False,
                    "error": f"能力 '{capability_id}' 不存在"
                }
            
            # 使用現有的 toolkit 進行連接性測試
            evidence = await self.toolkit.test_capability_connectivity(capability)
            
            health_info = {
                "success": evidence.success,
                "capability_id": capability_id,
                "capability_name": capability.name,
                "language": capability.language.value,
                "status": capability.status.value,
                "is_installed": self._is_tool_installed(capability),
                "installation_path": str(self._get_installation_path(capability)),
                "latency_ms": evidence.latency_ms,
                "last_check": evidence.timestamp.isoformat(),
                "error_message": evidence.error_message
            }
            
            # 記錄健康檢查事件
            self._record_event(
                capability_id, "health_check",
                "success" if evidence.success else "failed",
                health_info,
                evidence.error_message
            )
            
            return health_info
            
        except Exception as e:
            error_msg = f"健康檢查異常: {str(e)}"
            self._record_event(capability_id, "health_check", "failed", {}, error_msg)
            
            return {
                "success": False,
                "capability_id": capability_id,
                "error": error_msg
            }
    
    async def batch_health_check(self, capability_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """批量健康檢查"""
        if capability_ids is None:
            # 檢查所有已註冊的能力
            capabilities = await self.registry.list_capabilities()
            capability_ids = [cap.id for cap in capabilities]
        
        results = {}
        healthy_count = 0
        total_count = len(capability_ids)
        
        logger.info(f"開始批量健康檢查，共 {total_count} 個工具")
        
        # 並發執行健康檢查
        tasks = []
        for cap_id in capability_ids:
            task = asyncio.create_task(self.health_check_tool(cap_id))
            tasks.append((cap_id, task))
        
        for cap_id, task in tasks:
            try:
                result = await task
                results[cap_id] = result
                if result.get("success"):
                    healthy_count += 1
            except Exception as e:
                results[cap_id] = {
                    "success": False,
                    "capability_id": cap_id,
                    "error": f"檢查過程異常: {str(e)}"
                }
        
        summary = {
            "total_tools": total_count,
            "healthy_tools": healthy_count,
            "unhealthy_tools": total_count - healthy_count,
            "health_rate": healthy_count / total_count if total_count > 0 else 0,
            "results": results,
            "check_time": datetime.now().isoformat()
        }
        
        logger.info(
            f"批量健康檢查完成: 總計 {total_count} 個工具，健康 {healthy_count} 個，健康率 {summary['health_rate']:.2%}"
        )
        
        return summary
    
    def get_lifecycle_events(
        self, 
        capability_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ToolLifecycleEvent]:
        """獲取生命週期事件歷史"""
        events = self.events.copy()
        
        # 過濾條件
        if capability_id:
            events = [e for e in events if e.capability_id == capability_id]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # 按時間降序排序
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return events[:limit]
    
    def _get_installation_path(self, capability: CapabilityRecord) -> Path:
        """獲取工具的安裝路徑"""
        base_path = self.installation_paths.get(capability.language)
        if not base_path:
            raise ValueError(f"不支援的程式語言: {capability.language.value}")
        
        return base_path / capability.id
    
    def _is_tool_installed(self, capability: CapabilityRecord) -> bool:
        """檢查工具是否已安裝"""
        installation_path = self._get_installation_path(capability)
        return installation_path.exists()
    
    async def _run_command(self, cmd: List[str], cwd: Optional[str] = None) -> str:
        """執行命令行命令"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                raise RuntimeError(f"命令執行失敗 (返回碼: {process.returncode}): {error_msg}")
            
            return stdout.decode('utf-8', errors='ignore')
            
        except Exception as e:
            logger.error(f"執行命令失敗: {' '.join(cmd)} - {str(e)}")
            raise
    
    def _update_capability_status(self, capability_id: str, status: CapabilityStatus) -> None:
        """更新能力狀態"""
        # 注意：這裡需要擴展 CapabilityRegistry 以支援狀態更新
        # 目前先記錄日誌
        logger.info(f"更新能力狀態: {capability_id} -> {status.value}")
    
    def _update_capability_timestamp(self, capability_id: str) -> None:
        """更新能力的最後更新時間"""
        # 注意：這裡需要擴展 CapabilityRegistry 以支援時間戳更新
        # 目前先記錄日誌
        logger.info(f"更新能力時間戳: {capability_id}")