#!/usr/bin/env python3
"""
AIVA Reverse Engineering Tools - Task 18 (Direct Port from HackingTool)
Mobile app reverse engineering and analysis tools for Android applications
For authorized security testing and educational purposes only
"""

import asyncio
import logging
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.theme import Theme

# Local imports
from ...core.base_capability import BaseCapability
from ...core.registry import CapabilityRegistry

# Setup theme and console
_theme = Theme({"purple": "#7B61FF"})
console = Console(theme=_theme)
logger = logging.getLogger(__name__)


@dataclass
class ReverseEngineeringResult:
    """Reverse engineering analysis result data structure"""
    tool_name: str
    command: str
    start_time: str
    end_time: str
    duration: float
    success: bool
    output: str = ""
    error_details: Optional[str] = None
    target_file: Optional[str] = None
    output_directory: Optional[str] = None
    analysis_type: Optional[str] = None


class ReverseEngineeringTool:
    """Base reverse engineering tool class - equivalent to HackingTool"""
    
    def __init__(self):
        self.TITLE = ""
        self.DESCRIPTION = ""
        self.INSTALL_COMMANDS = []
        self.RUN_COMMANDS = []
        self.PROJECT_URL = ""
        self.installable = True
        self.runnable = True
    
    def is_installed(self) -> bool:
        """Check if tool is installed"""
        if not self.installable:
            return True
            
        # For reverse engineering tools, check specific binaries
        if self.TITLE == "Androguard":
            try:
                result = subprocess.run(["python3", "-c", "import androguard"], 
                                      capture_output=True, timeout=5)
                return result.returncode == 0
            except Exception:
                return False
        elif self.TITLE == "Apk2Gold":
            try:
                result = subprocess.run(["which", "apk2gold"], capture_output=True, timeout=5)
                return result.returncode == 0
            except Exception:
                return False
        elif self.TITLE == "JadX":
            # Check for jadx installation
            if Path("/usr/local/bin/jadx").exists() or Path("jadx/build/jadx/bin/jadx").exists():
                return True
            try:
                result = subprocess.run(["jadx", "--help"], capture_output=True, timeout=5)
                return result.returncode == 0
            except Exception:
                return False
        
        return True
    
    def install(self) -> bool:
        """Install tool"""
        if not self.installable:
            console.print(f"[yellow]{self.TITLE} is not installable via this interface[/yellow]")
            return False
            
        console.print(f"[cyan]Installing {self.TITLE}...[/cyan]")
        console.print("[yellow]⚠️  For authorized reverse engineering analysis only![/yellow]")
        
        if not Confirm.ask(f"Confirm install {self.TITLE}?"):
            return False
        
        success = True
        for cmd in self.INSTALL_COMMANDS:
            try:
                console.print(f"[yellow]Executing: {cmd}[/yellow]")
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    timeout=600,  # Longer timeout for builds
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    console.print(f"[red]Install command failed: {result.stderr}[/red]")
                    success = False
                    break
                    
            except subprocess.TimeoutExpired:
                console.print(f"[red]Install timeout: {cmd}[/red]")
                success = False
                break
            except Exception as e:
                console.print(f"[red]Install error: {e}[/red]")
                success = False
                break
        
        if success:
            console.print(f"[green]✅ {self.TITLE} installed successfully[/green]")
        else:
            console.print(f"[red]❌ {self.TITLE} installation failed[/red]")
        
        return success
    
    def run(self) -> ReverseEngineeringResult:
        """Run tool"""
        console.print(f"[bold green]🔍 Running {self.TITLE}[/bold green]")
        console.print("[yellow]⚠️  For authorized reverse engineering analysis only![/yellow]")
        
        if not self.runnable:
            console.print(f"[yellow]{self.TITLE} requires manual setup[/yellow]")
            return ReverseEngineeringResult(
                tool_name=self.TITLE,
                command="not_runnable",
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration=0,
                success=False,
                error_details="Tool not runnable via CLI"
            )
        
        if not Confirm.ask(f"Confirm run {self.TITLE}?"):
            return ReverseEngineeringResult(
                tool_name=self.TITLE,
                command="cancelled",
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration=0,
                success=False,
                error_details="User cancelled"
            )
        
        start_time = datetime.now()
        
        try:
            # Run custom method if exists, otherwise run standard commands
            if hasattr(self, 'custom_run'):
                result = self.custom_run()
                if isinstance(result, ReverseEngineeringResult):
                    return result
            else:
                for cmd in self.RUN_COMMANDS:
                    console.print(f"[yellow]Executing: {cmd}[/yellow]")
                    subprocess.run(cmd, shell=True)
        
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return ReverseEngineeringResult(
                tool_name=self.TITLE,
                command=str(self.RUN_COMMANDS),
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                success=False,
                error_details=str(e)
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return ReverseEngineeringResult(
            tool_name=self.TITLE,
            command=str(self.RUN_COMMANDS),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration=duration,
            success=True
        )


class AndroGuard(ReverseEngineeringTool):
    """Androguard - Direct port from HackingTool"""
    
    def __init__(self):
        super().__init__()
        self.TITLE = "Androguard"
        self.DESCRIPTION = "Androguard is a Reverse engineering, Malware and goodware " \
                          "analysis of Android applications and more"
        self.INSTALL_COMMANDS = ["sudo pip3 install -U androguard"]
        self.PROJECT_URL = "https://github.com/androguard/androguard"
        self.runnable = False  # Same as HackingTool


class Apk2Gold(ReverseEngineeringTool):
    """Apk2Gold - Direct port from HackingTool"""
    
    def __init__(self):
        super().__init__()
        self.TITLE = "Apk2Gold"
        self.DESCRIPTION = "Apk2Gold is a CLI tool for decompiling Android apps to Java"
        self.INSTALL_COMMANDS = [
            "sudo git clone https://github.com/lxdvs/apk2gold.git",
            "cd apk2gold && sudo bash make.sh"
        ]
        self.PROJECT_URL = "https://github.com/lxdvs/apk2gold"
    
    def custom_run(self) -> ReverseEngineeringResult:
        """Custom run method - same as HackingTool"""
        start_time = datetime.now()
        
        # Get APK file input
        apk_file = Prompt.ask("Enter APK filename (.apk)")
        
        if not apk_file.endswith('.apk'):
            return ReverseEngineeringResult(
                tool_name=self.TITLE,
                command="invalid_input",
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_details="Input must be .apk file"
            )
        
        if not Path(apk_file).exists():
            return ReverseEngineeringResult(
                tool_name=self.TITLE,
                command="file_not_found",
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_details=f"APK file not found: {apk_file}"
            )
        
        try:
            cmd = ["sudo", "apk2gold", apk_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            return ReverseEngineeringResult(
                tool_name=self.TITLE,
                command=" ".join(cmd),
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=result.returncode == 0,
                output=result.stdout,
                error_details=result.stderr if result.returncode != 0 else None,
                target_file=apk_file,
                analysis_type="APK decompilation"
            )
            
        except subprocess.TimeoutExpired:
            return ReverseEngineeringResult(
                tool_name=self.TITLE,
                command="timeout",
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_details="Command timeout after 300 seconds"
            )
        except Exception as e:
            return ReverseEngineeringResult(
                tool_name=self.TITLE,
                command="error",
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_details=str(e)
            )


class Jadx(ReverseEngineeringTool):
    """JadX - Direct port from HackingTool"""
    
    def __init__(self):
        super().__init__()
        self.TITLE = "JadX"
        self.DESCRIPTION = "Jadx is Dex to Java decompiler.\n" \
                          "[*] decompile Dalvik bytecode to java classes from APK, dex," \
                          " aar and zip files\n" \
                          "[*] decode AndroidManifest.xml and other resources from " \
                          "resources.arsc"
        self.INSTALL_COMMANDS = [
            "sudo git clone https://github.com/skylot/jadx.git",
            "cd jadx && ./gradlew dist"
        ]
        self.PROJECT_URL = "https://github.com/skylot/jadx"
        self.runnable = False  # Same as HackingTool


class ReverseEngineeringManager:
    """Reverse engineering tools manager - equivalent to HackingToolsCollection"""
    
    def __init__(self):
        self.TITLE = "Reverse Engineering Tools"
        self.DESCRIPTION = "Mobile app reverse engineering and analysis tools"
        self.TOOLS = [
            AndroGuard(),
            Apk2Gold(),
            Jadx()
        ]
        self.re_results = []
    
    def _get_attr(self, obj, *names, default=""):
        """Get attribute with fallback"""
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return default

    def pretty_print(self):
        """Display tools table - same as HackingTool"""
        table = Table(title="Reverse Engineering Tools", show_lines=True, expand=True)
        table.add_column("Title", style="purple", no_wrap=True)
        table.add_column("Description", style="purple")
        table.add_column("Project URL", style="purple", no_wrap=True)

        for t in self.TOOLS:
            title = self._get_attr(t, "TITLE", "Title", "title", default=t.__class__.__name__)
            desc = self._get_attr(t, "DESCRIPTION", "Description", "description", default="").strip().replace("\n", " ")
            url = self._get_attr(t, "PROJECT_URL", "PROJECT_URL", "project_url", "projectUrl", default="")
            table.add_row(str(title), str(desc or "—"), str(url))

        panel = Panel(table, title="[purple]Available Tools[/purple]", border_style="purple")
        console.print(panel)

    def show_options(self, parent=None):
        """Interactive menu - same as HackingTool"""
        console.print("\n")
        panel = Panel.fit("[bold magenta]🔍 AIVA Reverse Engineering Tools[/bold magenta]\n"
                          "Mobile app reverse engineering and analysis toolkit\n"
                          "⚠️  For authorized reverse engineering analysis only!",
                          border_style="purple")
        console.print(panel)

        table = Table(title="[bold cyan]Available Tools[/bold cyan]", show_lines=True, expand=True)
        table.add_column("Index", justify="center", style="bold yellow")
        table.add_column("Tool Name", justify="left", style="bold green")
        table.add_column("Description", justify="left", style="white")
        table.add_column("Status", justify="center", style="cyan")

        for i, tool in enumerate(self.TOOLS):
            title = self._get_attr(tool, "TITLE", "Title", "title", default=tool.__class__.__name__)
            desc = self._get_attr(tool, "DESCRIPTION", "Description", "description", default="—")
            status = "✅" if tool.is_installed() else "❌"
            runnable = "🔧" if not tool.runnable else "🏃"
            short_desc = (desc[:50] + "...") if len(desc) > 50 else desc
            table.add_row(str(i + 1), title, short_desc or "—", f"{status} {runnable}")

        table.add_row("[cyan]88[/cyan]", "[bold cyan]Show Details[/bold cyan]", "Show detailed tool information", "—")
        table.add_row("[yellow]77[/yellow]", "[bold yellow]Analysis Results[/bold yellow]", "View reverse engineering history", "—")
        table.add_row("[red]99[/red]", "[bold red]Exit[/bold red]", "Return to main menu", "—")
        console.print(table)

        try:
            choice = int(Prompt.ask("[bold cyan]Select a tool to run[/bold cyan]", default="99"))
            
            if 1 <= choice <= len(self.TOOLS):
                selected = self.TOOLS[choice - 1]
                self._handle_tool_selection(selected)
            elif choice == 88:
                self.pretty_print()
            elif choice == 77:
                self._show_re_results()
            elif choice == 99:
                return 99
            else:
                console.print("[bold red]Invalid choice. Try again.[/bold red]")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")
        
        return self.show_options(parent=parent)
    
    def _handle_tool_selection(self, tool: ReverseEngineeringTool):
        """Handle tool selection"""
        self._display_tool_info(tool)
        
        if not self._ensure_tool_installed(tool):
            return
        
        self._execute_tool_if_confirmed(tool)
    
    def _display_tool_info(self, tool: ReverseEngineeringTool):
        """Display tool information"""
        console.print(f"\n[bold green]Selected: {tool.TITLE}[/bold green]")
        console.print(f"[cyan]Description: {tool.DESCRIPTION}[/cyan]")
        console.print(f"[blue]Project URL: {getattr(tool, 'PROJECT_URL', 'N/A')}[/blue]")
        
        if not tool.runnable:
            console.print("[yellow]⚠️  This tool requires manual setup and cannot be run directly[/yellow]")
    
    def _ensure_tool_installed(self, tool: ReverseEngineeringTool) -> bool:
        """Ensure tool is installed"""
        if not tool.installable or tool.is_installed():
            return True
        
        console.print(f"[yellow]{tool.TITLE} is not installed[/yellow]")
        if not Confirm.ask("Install now?"):
            return False
        
        if tool.install():
            console.print(f"[green]{tool.TITLE} installed successfully![/green]")
            return True
        else:
            console.print(f"[red]{tool.TITLE} installation failed![/red]")
            return False
    
    def _execute_tool_if_confirmed(self, tool: ReverseEngineeringTool):
        """Execute tool if confirmed"""
        if not tool.runnable:
            console.print(f"[yellow]{tool.TITLE} is not runnable via CLI interface[/yellow]")
            console.print(f"[cyan]Please use {tool.TITLE} manually after installation[/cyan]")
            return
        
        if not Confirm.ask(f"Run {tool.TITLE}?"):
            return
        
        result = tool.run()
        self.re_results.append(result)
        
        if result.success:
            console.print(f"[green]✅ {tool.TITLE} completed![/green]")
            if result.target_file:
                console.print(f"[blue]Target file: {result.target_file}[/blue]")
        else:
            console.print(f"[red]❌ {tool.TITLE} failed: {result.error_details}[/red]")
    
    def _show_re_results(self):
        """Show reverse engineering analysis results"""
        if not self.re_results:
            console.print("[yellow]No reverse engineering analysis results available[/yellow]")
            return
        
        table = Table(title="🔍 Reverse Engineering Analysis Results")
        table.add_column("Tool", style="cyan")
        table.add_column("Result", style="green")
        table.add_column("Duration", style="blue")
        table.add_column("Time", style="magenta")
        table.add_column("Target File", style="yellow")
        table.add_column("Analysis Type", style="red")
        
        for result in self.re_results:
            status = "✅ Success" if result.success else "❌ Failed"
            start_time = result.start_time.split('T')[1][:8]
            target = result.target_file or "N/A"
            analysis = result.analysis_type or "N/A"
            
            table.add_row(
                result.tool_name,
                status,
                f"{result.duration:.1f}s",
                start_time,
                target[:15] + "..." if len(target) > 15 else target,
                analysis[:15] + "..." if len(analysis) > 15 else analysis
            )
        
        console.print(table)


class ReverseEngineeringCapability(BaseCapability):
    """Reverse engineering analysis capability - AIVA integration"""
    
    def __init__(self):
        super().__init__()
        self.name = "reverse_engineering_tools"
        self.version = "1.0.0"
        self.description = "Reverse Engineering Toolkit - Direct port from HackingTool"
        self.dependencies = ["python3-pip", "git", "openjdk-8-jdk", "gradle"]
        self.manager = ReverseEngineeringManager()
    
    async def initialize(self) -> bool:
        """Initialize capability"""
        try:
            console.print("[yellow]Initializing reverse engineering toolkit...[/yellow]")
            console.print("[red]⚠️  For authorized reverse engineering analysis only![/red]")
            console.print("[cyan]Mobile app reverse engineering and analysis tools[/cyan]")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    async def execute(self, command: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command"""
        try:
            if command == "interactive_menu":
                self.manager.show_options()
                return {"success": True, "message": "Interactive menu completed"}
            
            elif command == "list_tools":
                tools_info = []
                for tool in self.manager.TOOLS:
                    tools_info.append({
                        "title": tool.TITLE,
                        "description": tool.DESCRIPTION,
                        "project_url": getattr(tool, "PROJECT_URL", ""),
                        "installed": tool.is_installed(),
                        "runnable": tool.runnable
                    })
                return {"success": True, "data": {"tools": tools_info}}
            
            elif command == "show_details":
                self.manager.pretty_print()
                return {"success": True, "message": "Tool details displayed"}
                
            elif command == "run_tool":
                tool_name = parameters.get('tool_name')
                if not tool_name:
                    return {"success": False, "error": "Missing tool_name parameter"}
                
                tool = next((t for t in self.manager.TOOLS if t.TITLE == tool_name), None)
                if not tool:
                    return {"success": False, "error": f"Tool {tool_name} not found"}
                
                result = tool.run()
                self.manager.re_results.append(result)
                return {"success": True, "data": asdict(result)}
            
            else:
                return {"success": False, "error": f"Unknown command: {command}"}
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def cleanup(self) -> bool:
        """Cleanup resources"""
        try:
            self.manager.re_results.clear()
            return True
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False


# Register capability
CapabilityRegistry.register("reverse_engineering_tools", ReverseEngineeringCapability)


if __name__ == "__main__":
    # Test case - same as HackingTool
    async def test_reverse_engineering_tools():
        capability = ReverseEngineeringCapability()
        await capability.initialize()
        
        console.print("[bold red]⚠️  Reverse engineering toolkit - Direct port from HackingTool![/bold red]")
        console.print("[yellow]For authorized reverse engineering analysis only![/yellow]")
        
        # Show tools and start interactive menu
        capability.manager.pretty_print()
        capability.manager.show_options()
        
        await capability.cleanup()
    
    # Run test
    asyncio.run(test_reverse_engineering_tools())