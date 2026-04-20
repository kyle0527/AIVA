#!/usr/bin/env python3
"""
AIVA Steganography Tools - Task 17 (Direct Port from HackingTool)
Steganography detection and exploitation tools for data hiding in multimedia
For authorized security testing and educational purposes only
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime
import logging
import subprocess
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.theme import Theme

# Setup theme and console
_theme = Theme({"purple": "#7B61FF"})
console = Console(theme=_theme)
logger = logging.getLogger(__name__)


@dataclass
class SteganographyResult:
    """Steganography analysis result data structure"""
    tool_name: str
    command: str
    start_time: str
    end_time: str
    duration: float
    success: bool
    output: str = ""
    error_details: str | None = None
    target_file: str | None = None
    hidden_data: str | None = None


def validate_input(choice, valid_choices):
    """Validate user input"""
    try:
        choice = int(choice)
        if choice in valid_choices:
            return choice
    except ValueError:
        pass
    return None


class SteganographyTool:
    """Base steganography tool class - equivalent to HackingTool"""

    def __init__(self):
        self.name = ""
        self.description = ""
        self.install_commands = []
        self.run_commands = []
        self.project_url = ""
        self.installable = True
        self.runnable = True

    def custom_run(self) -> SteganographyResult | None:
        """Override in subclass for custom run behavior"""
        return None

    def is_installed(self) -> bool:
        """Check if tool is installed"""
        if not self.installable:
            return True  # Assume non-installable tools are available

        # For steganography tools, check specific commands
        if self.name == "SteganoHide":
            try:
                result = subprocess.run(["which", "steghide"], capture_output=True, timeout=5)
                return result.returncode == 0
            except Exception:
                return False
        elif self.name == "StegnoCracker":
            try:
                result = subprocess.run(["stegcracker", "--help"], capture_output=True, timeout=5)
                return result.returncode == 0
            except Exception:
                return False

        return True

    def install(self) -> bool:
        """Install tool"""
        if not self.installable:
            console.print(f"[yellow]{self.name} is not installable via this interface[/yellow]")
            return False

        console.print(f"[cyan]Installing {self.name}...[/cyan]")
        console.print("[yellow]⚠️  For authorized steganography analysis only![/yellow]")

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

    def run(self) -> SteganographyResult:
        """Run tool"""
        console.print(f"[bold green]🎭 Running {self.name}[/bold green]")
        console.print("[yellow]⚠️  For authorized steganography analysis only![/yellow]")

        if not self.runnable:
            console.print(f"[yellow]{self.name} requires manual setup[/yellow]")
            return SteganographyResult(
                tool_name=self.name,
                command="not_runnable",
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration=0,
                success=False,
                error_details="Tool not runnable via CLI"
            )

        if not Confirm.ask(f"Confirm run {self.name}?"):
            return SteganographyResult(
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
                result = self.custom_run()
                if isinstance(result, SteganographyResult):
                    return result
            else:
                for cmd in self.run_commands:
                    console.print(f"[yellow]Executing: {cmd}[/yellow]")
                    subprocess.run(cmd, shell=True)

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return SteganographyResult(
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

        return SteganographyResult(
            tool_name=self.name,
            command=str(self.run_commands),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration=duration,
            success=True
        )


class SteganoHide(SteganographyTool):
    """SteganoHide - Direct port from HackingTool"""

    def __init__(self):
        super().__init__()
        self.name = "SteganoHide"
        self.description = "Hide and extract data in/from image and audio files using steganography"
        self.install_commands = ["sudo apt-get install steghide -y"]

    def custom_run(self) -> SteganographyResult:
        """Custom run method - same as HackingTool"""
        start_time = datetime.now()

        console.print("[bold cyan]SteganoHide Operations[/bold cyan]")
        console.print("[1] Hide data in file")
        console.print("[2] Extract data from file")
        console.print("[99] Cancel")

        choice = Prompt.ask("Select operation", choices=["1", "2", "99"], default="99")

        if choice == "99":
            return SteganographyResult(
                tool_name=self.name,
                command="cancelled",
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_details="User cancelled"
            )

        try:
            if choice == "1":
                file_hide = Prompt.ask("Enter filename to embed (e.g., secret.txt)")
                file_to_be_hide = Prompt.ask("Enter cover filename (e.g., image.jpg)")

                cmd = ["steghide", "embed", "-cf", file_to_be_hide, "-ef", file_hide]
                result = subprocess.run(cmd, capture_output=True, text=True)

                return SteganographyResult(
                    tool_name=self.name,
                    command=" ".join(cmd),
                    start_time=start_time.isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration=(datetime.now() - start_time).total_seconds(),
                    success=result.returncode == 0,
                    output=result.stdout,
                    error_details=result.stderr if result.returncode != 0 else None,
                    target_file=file_to_be_hide,
                    hidden_data=file_hide
                )

            elif choice == "2":
                from_file = Prompt.ask("Enter filename to extract data from")

                cmd = ["steghide", "extract", "-sf", from_file]
                result = subprocess.run(cmd, capture_output=True, text=True)

                return SteganographyResult(
                    tool_name=self.name,
                    command=" ".join(cmd),
                    start_time=start_time.isoformat(),
                    end_time=datetime.now().isoformat(),
                    duration=(datetime.now() - start_time).total_seconds(),
                    success=result.returncode == 0,
                    output=result.stdout,
                    error_details=result.stderr if result.returncode != 0 else None,
                    target_file=from_file
                )

        except Exception as e:
            return SteganographyResult(
                tool_name=self.name,
                command="error",
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_details=str(e)
            )

        # Fallback return (should not reach here due to choices constraint)
        return SteganographyResult(
            tool_name=self.name,
            command="unknown",
            start_time=start_time.isoformat(),
            end_time=datetime.now().isoformat(),
            duration=(datetime.now() - start_time).total_seconds(),
            success=False,
            error_details="Unknown operation"
        )


class StegnoCracker(SteganographyTool):
    """StegnoCracker - Direct port from HackingTool"""

    def __init__(self):
        super().__init__()
        self.name = "StegnoCracker"
        self.description = "SteganoCracker uncovers hidden data inside files using brute-force utility"
        self.install_commands = ["pip3 install stegcracker && pip3 install stegcracker -U --force-reinstall"]

    def custom_run(self) -> SteganographyResult:
        """Custom run method - same as HackingTool"""
        start_time = datetime.now()

        filename = Prompt.ask("Enter filename to crack")
        passfile = Prompt.ask("Enter wordlist filename")

        try:
            cmd = ["stegcracker", filename, passfile]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            return SteganographyResult(
                tool_name=self.name,
                command=" ".join(cmd),
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=result.returncode == 0,
                output=result.stdout,
                error_details=result.stderr if result.returncode != 0 else None,
                target_file=filename
            )

        except subprocess.TimeoutExpired:
            return SteganographyResult(
                tool_name=self.name,
                command=" ".join(cmd),
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_details="Command timeout after 300 seconds"
            )
        except Exception as e:
            return SteganographyResult(
                tool_name=self.name,
                command="error",
                start_time=start_time.isoformat(),
                end_time=datetime.now().isoformat(),
                duration=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_details=str(e)
            )


class StegoCracker(SteganographyTool):
    """StegoCracker - Direct port from HackingTool"""

    def __init__(self):
        super().__init__()
        self.name = "StegoCracker"
        self.description = "StegoCracker lets you hide and retrieve data in image or audio files"
        self.install_commands = [
            "sudo git clone https://github.com/W1LDN16H7/StegoCracker.git",
            "sudo chmod -R 755 StegoCracker"
        ]
        self.run_commands = [
            "cd StegoCracker && python3 -m pip install -r requirements.txt",
            "./install.sh"
        ]
        self.project_url = "https://github.com/W1LDN16H7/StegoCracker"


class Whitespace(SteganographyTool):
    """Whitespace Steganography - Direct port from HackingTool"""

    def __init__(self):
        super().__init__()
        self.name = "Whitespace"
        self.description = "Use whitespace and unicode characters for steganography"
        self.install_commands = [
            "sudo git clone https://github.com/beardog108/snow10.git",
            "sudo chmod -R 755 snow10"
        ]
        self.run_commands = ["cd snow10 && ./install.sh"]
        self.project_url = "https://github.com/beardog108/snow10"


class SteganographyManager:
    """Steganography tools manager - equivalent to HackingToolsCollection"""

    def __init__(self):
        self.name = "Steganography Tools"
        self.description = "Data hiding and steganography analysis tools"
        self.tools = [
            SteganoHide(),
            StegnoCracker(),
            StegoCracker(),
            Whitespace()
        ]
        self.steganography_results = []

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
        table = Table(title="Steganography Tools", show_lines=True, expand=True)
        table.add_column("Title", style="purple", no_wrap=True)
        table.add_column("Description", style="purple")
        table.add_column("Project URL", style="purple", no_wrap=True)

        for t in self.tools:
            title = self._get_attr(t, "TITLE", "Title", "title", default=t.__class__.__name__)
            desc = self._get_attr(t, "DESCRIPTION", "Description", "description", default="").strip().replace("\n", " ")
            url = self._get_attr(t, "PROJECT_URL", "PROJECT_URL", "project_url", "projectUrl", default="")
            table.add_row(str(title), str(desc or "—"), str(url))

        panel = Panel(table, title="[purple]Available Tools[/purple]", border_style="purple")
        console.print(panel)

    def show_options(self, parent=None):
        """Interactive menu - same as HackingTool"""
        console.print("\n")
        panel = Panel.fit("[bold magenta]🎭 AIVA Steganography Tools[/bold magenta]\n"
                          "Data hiding and steganography analysis toolkit\n"
                          "⚠️  For authorized steganography analysis only!",
                          border_style="purple")
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
        table.add_row("[yellow]77[/yellow]", "[bold yellow]Analysis Results[/bold yellow]", "View steganography analysis history", "—")
        table.add_row("[red]99[/red]", "[bold red]Exit[/bold red]", "Return to main menu", "—")
        console.print(table)

        try:
            choice = int(Prompt.ask("[bold cyan]Select a tool to run[/bold cyan]", default="99"))

            if 1 <= choice <= len(self.tools):
                selected = self.tools[choice - 1]
                self._handle_tool_selection(selected)
            elif choice == 88:
                self.pretty_print()
            elif choice == 77:
                self._show_steganography_results()
            elif choice == 99:
                return 99
            else:
                console.print("[bold red]Invalid choice. Try again.[/bold red]")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")

        return self.show_options(parent=parent)

    def _handle_tool_selection(self, tool: SteganographyTool):
        """Handle tool selection"""
        self._display_tool_info(tool)

        if not self._ensure_tool_installed(tool):
            return

        self._execute_tool_if_confirmed(tool)

    def _display_tool_info(self, tool: SteganographyTool):
        """Display tool information"""
        console.print(f"\n[bold green]Selected: {tool.name}[/bold green]")
        console.print(f"[cyan]Description: {tool.description}[/cyan]")
        console.print(f"[blue]Project URL: {getattr(tool, 'PROJECT_URL', 'N/A')}[/blue]")

    def _ensure_tool_installed(self, tool: SteganographyTool) -> bool:
        """Ensure tool is installed"""
        if not tool.installable or tool.is_installed():
            return True

        console.print(f"[yellow]{tool.name} is not installed[/yellow]")
        if not Confirm.ask("Install now?"):
            return False

        if tool.install():
            console.print(f"[green]{tool.name} installed successfully![/green]")
            return True
        else:
            console.print(f"[red]{tool.name} installation failed![/red]")
            return False

    def _execute_tool_if_confirmed(self, tool: SteganographyTool):
        """Execute tool if confirmed"""
        if not Confirm.ask(f"Run {tool.name}?"):
            return

        result = tool.run()
        self.steganography_results.append(result)

        if result.success:
            console.print(f"[green]✅ {tool.name} completed![/green]")
            if result.hidden_data:
                console.print(f"[blue]Hidden data: {result.hidden_data}[/blue]")
        else:
            console.print(f"[red]❌ {tool.name} failed: {result.error_details}[/red]")

    def _show_steganography_results(self):
        """Show steganography analysis results"""
        if not self.steganography_results:
            console.print("[yellow]No steganography analysis results available[/yellow]")
            return

        table = Table(title="🎭 Steganography Analysis Results")
        table.add_column("Tool", style="cyan")
        table.add_column("Result", style="green")
        table.add_column("Duration", style="blue")
        table.add_column("Time", style="magenta")
        table.add_column("Target File", style="yellow")
        table.add_column("Hidden Data", style="red")

        for result in self.steganography_results:
            status = "✅ Success" if result.success else "❌ Failed"
            start_time = result.start_time.split('T')[1][:8]
            target = result.target_file or "N/A"
            hidden = result.hidden_data or "N/A"

            table.add_row(
                result.tool_name,
                status,
                f"{result.duration:.1f}s",
                start_time,
                target[:15] + "..." if len(target) > 15 else target,
                hidden[:15] + "..." if len(hidden) > 15 else hidden
            )

        console.print(table)


if __name__ == "__main__":
    # Test case - standalone test
    async def test_steganography_tools():
        console.print("[yellow]Initializing steganography analysis toolkit...[/yellow]")
        console.print("[red]⚠️  For authorized steganography analysis only![/red]")
        console.print("[cyan]Data hiding and steganography detection tools[/cyan]")

        manager = SteganographyManager()

        console.print("[bold red]⚠️  Steganography toolkit - Direct port from HackingTool![/bold red]")
        console.print("[yellow]For authorized steganography analysis only![/yellow]")

        # Show tools and start interactive menu
        manager.pretty_print()
        manager.show_options()


    # Run test
    asyncio.run(test_steganography_tools())
