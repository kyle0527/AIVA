#!/usr/bin/env python3
"""
AIVA Forensic Tools - Task 16 (Direct Port from HackingTool)
Digital forensics and incident response tools for evidence collection and analysis
For authorized security testing and educational purposes only
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import logging
import os
import subprocess
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Local imports - 使用正確的導入路徑
from .models import BaseCapability

# Setup theme and console
_theme = Theme({"purple": "#7B61FF"})
console = Console(theme=_theme)
logger = logging.getLogger(__name__)
PURPLE_STYLE = "bold magenta"


@dataclass
class ForensicResult:
    """Forensic analysis result data structure"""
    tool_name: str
    command: str
    start_time: str
    end_time: str
    duration: float
    success: bool
    output: str = ""
    error_details: str | None = None
    evidence_path: str | None = None


class ForensicTool:
    """Base forensic tool class - equivalent to HackingTool"""

    def __init__(self):
        self.name = ""
        self.description = ""
        self.install_commands: list[str] = []
        self.run_commands: list[str] = []
        self.project_url = ""
        self.installable = True
        self.runnable = True

    def custom_run(self) -> None:
        """Override this method in subclasses for custom execution logic"""

    def is_installed(self) -> bool:
        """Check if tool is installed"""
        if not self.run_commands:
            return False

        # Extract main command
        main_cmd = self.run_commands[0].split()[0]
        if main_cmd.startswith("sudo "):
            main_cmd = main_cmd.replace("sudo ", "")

        try:
            result = subprocess.run(
                ["which", main_cmd],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def install(self) -> bool:
        """Install tool"""
        if not self.installable:
            console.print(f"[yellow]{self.name} is not installable via this interface[/yellow]")
            return False

        console.print(f"[cyan]Installing {self.name}...[/cyan]")
        console.print("[yellow]⚠️  For authorized forensic analysis only![/yellow]")

        if not Confirm.ask(f"Confirm install {self.name}?"):
            return False

        success = True
        for cmd in self.install_commands:
            try:
                console.print(f"[yellow]Executing: {cmd}[/yellow]")
                result = subprocess.run(
                    cmd,
                    shell=True,
                    timeout=300,
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
            console.print(f"[green]✅ {self.name} installed successfully[/green]")
        else:
            console.print(f"[red]❌ {self.name} installation failed[/red]")

        return success

    def run(self) -> ForensicResult:
        """Run tool"""
        console.print(f"[bold green]🔍 Running {self.name}[/bold green]")
        console.print("[yellow]⚠️  For authorized forensic analysis only![/yellow]")

        if not self.runnable:
            console.print(f"[yellow]{self.name} requires manual setup or GUI interface[/yellow]")
            return ForensicResult(
                tool_name=self.name,
                command="not_runnable",
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration=0,
                success=False,
                error_details="Tool not runnable via CLI"
            )

        if not Confirm.ask(f"Confirm run {self.name}?"):
            return ForensicResult(
                tool_name=self.name,
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
                self.custom_run()
            else:
                for cmd in self.run_commands:
                    console.print(f"[yellow]Executing: {cmd}[/yellow]")
                    subprocess.run(cmd, shell=True)

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return ForensicResult(
                tool_name=self.name,
                command=str(self.run_commands),
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                duration=duration,
                success=False,
                error_details=str(e)
            )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return ForensicResult(
            tool_name=self.name,
            command=str(self.run_commands),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration=duration,
            success=True
        )


class Autopsy(ForensicTool):
    """Autopsy Digital Forensics Platform - Direct port from HackingTool"""

    def __init__(self):
        super().__init__()
        self.name = "Autopsy"
        self.description = (
            "Autopsy is a platform that is used by Cyber Investigators.\n"
            "[!] Works in any OS\n"
            "[!] Recover Deleted Files from any OS & Media\n"
            "[!] Extract Image Metadata"
        )
        self.run_commands = ["sudo autopsy"]
        self.installable = False


class Wireshark(ForensicTool):
    """Wireshark Network Protocol Analyzer - Direct port from HackingTool"""

    def __init__(self):
        super().__init__()
        self.name = "Wireshark"
        self.description = (
            "Wireshark is a network capture and analyzer\n"
            "tool to see what's happening in your network.\n"
            "And also investigate Network related incident"
        )
        self.run_commands = ["sudo wireshark"]
        self.installable = False


class BulkExtractor(ForensicTool):
    """Bulk Extractor - Direct port from HackingTool"""

    def __init__(self):
        super().__init__()
        self.name = "Bulk extractor"
        self.description = "Extract useful information without parsing the file system"
        self.project_url = "https://github.com/simsong/bulk_extractor"
        self.installable = False
        self.runnable = False

    def gui_mode(self):
        """GUI Mode implementation"""
        import subprocess
        console.print(Panel(Text(self.name, justify="center"), style=PURPLE_STYLE))
        console.print("[bold magenta]Cloning repository and attempting to run GUI...[/]")
        subprocess.run(["sudo", "git", "clone", "https://github.com/simsong/bulk_extractor.git"], shell=False)
        subprocess.run(["ls", "src/"], shell=False)
        subprocess.run(["./BEViewer"], cwd="../java_gui", shell=False)
        console.print(
            "[magenta]If you get an error after clone go to /java_gui/src/ and compile the .jar file && run ./BEViewer[/]")
        console.print(
            "[magenta]Please visit for more details about installation: https://github.com/simsong/bulk_extractor[/]")

    def cli_mode(self):
        """CLI Mode implementation"""
        import subprocess
        console.print(Panel(Text(self.name + " - CLI Mode", justify="center"), style=PURPLE_STYLE))
        subprocess.run(["sudo", "apt", "install", "-y", "bulk-extractor"], shell=False)
        console.print("[magenta]Showing bulk_extractor help and options:[/]")
        subprocess.run(["bulk_extractor", "-h"], shell=False)

        # Pipeline handling using subprocess.Popen
        p1 = subprocess.Popen(["echo", "bulk_extractor [options] imagefile"], stdout=subprocess.PIPE, shell=False)
        p2 = subprocess.Popen(["boxes", "-d", "headline"], stdin=p1.stdout, stdout=subprocess.PIPE, shell=False)
        p1.stdout.close()
        subprocess.run(["lolcat"], stdin=p2.stdout, shell=False)

    def custom_run(self):
        """Custom run with mode selection"""
        console.print(f"[bold cyan]{self.name} Mode Selection[/bold cyan]")
        console.print("1. GUI Mode (Download required)")
        console.print("2. CLI Mode")

        choice = Prompt.ask("Select mode", choices=["1", "2"], default="2")

        if choice == "1":
            self.gui_mode()
        else:
            self.cli_mode()


class Guymager(ForensicTool):
    """Guymager Forensic Imager - Direct port from HackingTool"""

    def __init__(self):
        super().__init__()
        self.name = "Disk Clone and ISO Image Acquire"
        self.description = "Guymager is a free forensic imager for media acquisition."
        self.install_commands = ["sudo apt install guymager"]
        self.run_commands = ["sudo guymager"]
        self.project_url = "https://guymager.sourceforge.io/"
        self.installable = False


class Toolsley(ForensicTool):
    """Toolsley Online Forensic Tools - Direct port from HackingTool"""

    def __init__(self):
        super().__init__()
        self.name = "Toolsley"
        self.description = (
            "Toolsley got more than ten useful tools for investigation.\n"
            "[+]File signature verifier\n"
            "[+]File identifier\n"
            "[+]Hash & Validate\n"
            "[+]Binary inspector\n"
            "[+]Encode text\n"
            "[+]Data URI generator\n"
            "[+]Password generator"
        )
        self.project_url = "https://www.toolsley.com/"
        self.installable = False
        self.runnable = False

    def custom_run(self):
        """Open Toolsley website"""
        console.print(f"[cyan]Opening {self.name} website...[/cyan]")
        console.print(f"[blue]Visit: {self.project_url}[/blue]")

        # Cross-platform URL opening
        import webbrowser
        webbrowser.open(self.project_url)


class ForensicManager:
    """Forensic tools manager - equivalent to HackingToolsCollection"""

    def __init__(self):
        self.name = "Forensic tools"
        self.description = "Digital forensics and incident response tools"
        self.tools = [
            Autopsy(),
            Wireshark(),
            BulkExtractor(),
            Guymager(),
            Toolsley()
        ]
        self.forensic_results = []

    def _get_attr(self, obj, *names, default=""):
        """獲取屬性（支持多種命名習慣）。

        嘗試多個可能的屬性名稱，以支持不同工具的命名規範。
        這是兼容性處理，不是降級邏輯。
        """
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return default

    def pretty_print(self):
        """Display tools table - same as HackingTool"""
        table = Table(title="Forensic Tools", show_lines=True, expand=True)
        table.add_column("Title", style=PURPLE_STYLE, no_wrap=True)
        table.add_column("Description", style=PURPLE_STYLE)
        table.add_column("Project URL", style=PURPLE_STYLE, no_wrap=True)

        for t in self.tools:
            title = self._get_attr(t, "TITLE", "Title", "title", default=t.__class__.__name__)
            desc = self._get_attr(t, "DESCRIPTION", "Description", "description", default="")
            url = self._get_attr(t, "PROJECT_URL", "PROJECT_URL", "PROJECT", "project_url", "projectUrl", default="")
            table.add_row(str(title), str(desc).replace("\n", " "), str(url))

        panel = Panel(table, title="[magenta]Available Tools[/magenta]", border_style=PURPLE_STYLE)
        console.print(panel)

    def show_options(self, parent=None):
        """Interactive menu - same as HackingTool"""
        console.print("\n")
        panel = Panel.fit("[bold magenta]🔍 AIVA Forensic Tools[/bold magenta]\n"
                          "Digital forensics and incident response toolkit\n"
                          "⚠️  For authorized forensic analysis only!",
                          border_style=PURPLE_STYLE)
        console.print(panel)

        table = Table(title="[bold cyan]Available Tools[/bold cyan]", show_lines=True, expand=True)
        table.add_column("Index", justify="center", style="bold yellow")
        table.add_column("Tool Name", justify="left", style="bold green")
        table.add_column("Description", justify="left", style="white")
        table.add_column("Status", justify="center", style="cyan")

        for i, tool in enumerate(self.tools):
            title = self._get_attr(tool, "TITLE", "Title", "title", default=tool.__class__.__name__)
            desc = self._get_attr(tool, "DESCRIPTION", "Description", "description", default="—")
            status = "✅" if tool.is_installed() else "❌"
            short_desc = (desc[:50] + "...") if len(desc) > 50 else desc
            table.add_row(str(i + 1), title, short_desc or "—", status)

        table.add_row("[cyan]88[/cyan]", "[bold cyan]Show Details[/bold cyan]", "Show detailed tool information", "—")
        table.add_row("[yellow]77[/yellow]", "[bold yellow]Analysis Results[/bold yellow]", "View forensic analysis history", "—")
        table.add_row("[red]99[/red]", "[bold red]Exit[/bold red]", "Return to main menu", "—")
        console.print(table)

        try:
            choice = Prompt.ask("[bold cyan]Select a tool to run[/bold cyan]", default="99")
            choice = int(choice)

            if 1 <= choice <= len(self.tools):
                selected = self.tools[choice - 1]
                self._handle_tool_selection(selected)
            elif choice == 88:
                self.pretty_print()
            elif choice == 77:
                self._show_forensic_results()
            elif choice == 99:
                return 99
            else:
                console.print("[bold red]Invalid choice. Try again.[/bold red]")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")

        return self.show_options(parent=parent)

    def _handle_tool_selection(self, tool: ForensicTool):
        """Handle tool selection"""
        console.print(f"\n[bold green]Selected: {tool.name}[/bold green]")
        console.print(f"[cyan]Description: {tool.description}[/cyan]")
        console.print(f"[blue]Project URL: {getattr(tool, 'PROJECT_URL', 'N/A')}[/blue]")

        if tool.installable and not tool.is_installed():
            console.print(f"[yellow]{tool.name} is not installed[/yellow]")
            if Confirm.ask("Install now?"):
                if tool.install():
                    console.print(f"[green]{tool.name} installed successfully![/green]")
                else:
                    console.print(f"[red]{tool.name} installation failed![/red]")
                    return
            else:
                return

        if Confirm.ask(f"Run {tool.name}?"):
            result = tool.run()
            self.forensic_results.append(result)

            if result.success:
                console.print(f"[green]✅ {tool.name} completed![/green]")
            else:
                console.print(f"[red]❌ {tool.name} failed: {result.error_details}[/red]")

    def _show_forensic_results(self):
        """Show forensic analysis results"""
        if not self.forensic_results:
            console.print("[yellow]No forensic analysis results available[/yellow]")
            return

        table = Table(title="🔍 Forensic Analysis Results")
        table.add_column("Tool", style="cyan")
        table.add_column("Result", style="green")
        table.add_column("Duration", style="blue")
        table.add_column("Time", style="magenta")
        table.add_column("Evidence", style="yellow")

        for result in self.forensic_results:
            status = "✅ Success" if result.success else "❌ Failed"
            start_time = result.start_time.split('T')[1][:8]
            evidence = result.evidence_path or "N/A"

            table.add_row(
                result.tool_name,
                status,
                f"{result.duration:.1f}s",
                start_time,
                evidence[:20] + "..." if len(evidence) > 20 else evidence
            )

        console.print(table)


class ForensicCapability(BaseCapability):
    """Forensic analysis capability - AIVA integration"""

    def __init__(self):
        super().__init__()
        self.name = "forensic_tools"
        self.version = "1.0.0"
        self.description = "Digital Forensics Toolkit - Direct port from HackingTool"
        self.dependencies = ["wireshark", "autopsy", "bulk-extractor"]
        self.manager = ForensicManager()

    async def initialize(self) -> bool:
        """Initialize capability"""
        try:
            console.print("[yellow]Initializing forensic analysis toolkit...[/yellow]")
            console.print("[red]⚠️  For authorized forensic analysis only![/red]")
            console.print("[cyan]Digital forensics and incident response tools[/cyan]")

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def execute(self, command: str, parameters: dict[str, Any]) -> dict[str, Any]:
        """Execute command"""
        try:
            if command == "interactive_menu":
                self.manager.show_options()
                return {"success": True, "message": "Interactive menu completed"}

            elif command == "list_tools":
                tools_info = []
                for tool in self.manager.tools:
                    tools_info.append({
                        "title": tool.name,
                        "description": tool.description,
                        "project_url": getattr(tool, "PROJECT_URL", ""),
                        "installed": tool.is_installed()
                    })
                return {"success": True, "data": {"tools": tools_info}}

            elif command == "show_details":
                self.manager.pretty_print()
                return {"success": True, "message": "Tool details displayed"}

            elif command == "run_tool":
                tool_name = parameters.get('tool_name')
                if not tool_name:
                    return {"success": False, "error": "Missing tool_name parameter"}

                tool = next((t for t in self.manager.tools if t.name == tool_name), None)
                if not tool:
                    return {"success": False, "error": f"Tool {tool_name} not found"}

                result = tool.run()
                self.manager.forensic_results.append(result)
                return {"success": True, "data": asdict(result)}

            else:
                return {"success": False, "error": f"Unknown command: {command}"}

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"success": False, "error": str(e)}

    async def cleanup(self) -> bool:
        """Cleanup resources"""
        try:
            self.manager.forensic_results.clear()
            return True
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False


# Note: Registration is now handled via async registration function
# See register_standardized_capabilities.py for proper registration flow
# CapabilityRegistry.register("forensic_tools", ForensicCapability)


if __name__ == "__main__":
    # Test case - same as HackingTool
    async def test_forensic_tools():
        capability = ForensicCapability()
        await capability.initialize()

        console.print("[bold red]⚠️  Digital forensics toolkit - Direct port from HackingTool![/bold red]")
        console.print("[yellow]For authorized forensic analysis only![/yellow]")

        # Show tools and start interactive menu
        capability.manager.pretty_print()
        capability.manager.show_options()

        await capability.cleanup()

    # Run test
    asyncio.run(test_forensic_tools())
