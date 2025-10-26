"""
現代化 CLI 工具
基於 Click 和 Rich 的最佳實踐
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import click
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    
from pydantic import BaseModel


class CLIConfig(BaseModel):
    """CLI 配置"""
    debug: bool = False
    verbose: bool = False
    output_format: str = "table"  # table, json, yaml
    color: bool = True
    width: Optional[int] = None


class CLIContext:
    """CLI 上下文"""
    
    def __init__(self, config: Optional[CLIConfig] = None):
        self.config = config or CLIConfig()
        if RICH_AVAILABLE:
            self.console = Console(
                width=self.config.width,
                force_terminal=self.config.color
            )
        else:
            self.console = None
    
    def print(self, *args, **kwargs):
        """打印輸出"""
        if self.console:
            self.console.print(*args, **kwargs)
        else:
            print(*args, **kwargs)
    
    def print_json(self, data: Any, indent: int = 2):
        """打印 JSON 格式"""
        if self.config.output_format == "json":
            if self.console:
                self.console.print_json(json.dumps(data, indent=indent, ensure_ascii=False))
            else:
                print(json.dumps(data, indent=indent, ensure_ascii=False))
        else:
            self.print(data)
    
    def print_table(self, data: List[Dict[str, Any]], title: Optional[str] = None):
        """打印表格"""
        if not data:
            self.print("No data to display")
            return
        
        if self.config.output_format == "json":
            self.print_json(data)
            return
        
        if not self.console:
            # 簡單的文本表格
            if title:
                print(f"\n{title}")
                print("=" * len(title))
            
            headers = list(data[0].keys())
            print("\t".join(headers))
            print("-" * 40)
            for row in data:
                print("\t".join(str(row.get(h, "")) for h in headers))
            return
        
        # Rich 表格
        table = Table(title=title, show_header=True, header_style="bold magenta")
        
        if data:
            headers = list(data[0].keys())
            for header in headers:
                table.add_column(header, style="cyan")
            
            for row in data:
                table.add_row(*[str(row.get(h, "")) for h in headers])
        
        self.console.print(table)
    
    def print_panel(self, content: str, title: Optional[str] = None, style: str = "blue"):
        """打印面板"""
        if self.console:
            panel = Panel(content, title=title, border_style=style)
            self.console.print(panel)
        else:
            if title:
                print(f"\n[{title}]")
            print(content)
    
    def print_error(self, message: str):
        """打印錯誤信息"""
        if self.console:
            self.console.print(f"[bold red]Error:[/bold red] {message}")
        else:
            print(f"Error: {message}", file=sys.stderr)
    
    def print_warning(self, message: str):
        """打印警告信息"""
        if self.console:
            self.console.print(f"[bold yellow]Warning:[/bold yellow] {message}")
        else:
            print(f"Warning: {message}")
    
    def print_success(self, message: str):
        """打印成功信息"""
        if self.console:
            self.console.print(f"[bold green]Success:[/bold green] {message}")
        else:
            print(f"Success: {message}")
    
    def print_info(self, message: str):
        """打印信息"""
        if self.console:
            self.console.print(f"[bold blue]Info:[/bold blue] {message}")
        else:
            print(f"Info: {message}")


class CLIProgressBar:
    """CLI 進度條"""
    
    def __init__(self, ctx: CLIContext):
        self.ctx = ctx
        self._progress = None
        self._task_id = None
    
    def __enter__(self):
        if self.ctx.console and RICH_AVAILABLE:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.ctx.console,
                transient=True
            )
            self._progress.__enter__()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._progress:
            self._progress.__exit__(exc_type, exc_val, exc_tb)
    
    def start_task(self, description: str, total: Optional[int] = None):
        """開始任務"""
        if self._progress:
            self._task_id = self._progress.add_task(description, total=total)
        else:
            self.ctx.print(f"Starting: {description}")
    
    def update_task(self, advance: int = 1, description: Optional[str] = None):
        """更新任務進度"""
        if self._progress and self._task_id is not None:
            kwargs = {"advance": advance}
            if description:
                kwargs["description"] = description
            self._progress.update(self._task_id, **kwargs)
    
    def complete_task(self, description: Optional[str] = None):
        """完成任務"""
        if self._progress and self._task_id is not None:
            if description:
                self._progress.update(self._task_id, description=description)
            self._progress.update(self._task_id, completed=True)
        else:
            self.ctx.print(f"Completed: {description or 'Task'}")


def create_cli_group(name: str = "aiva", help_text: str = "AIVA Command Line Interface"):
    """創建 CLI 組"""
    if not click:
        raise ImportError("Click is required for CLI functionality")
    
    @click.group(name=name, help=help_text)
    @click.option('--debug', is_flag=True, help='Enable debug mode')
    @click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
    @click.option('--output-format', '-f', 
                  type=click.Choice(['table', 'json', 'yaml']),
                  default='table', help='Output format')
    @click.option('--no-color', is_flag=True, help='Disable colored output')
    @click.option('--width', type=int, help='Console width')
    @click.pass_context
    def cli(ctx, debug, verbose, output_format, no_color, width):
        """AIVA 命令行介面"""
        config = CLIConfig(
            debug=debug,
            verbose=verbose,
            output_format=output_format,
            color=not no_color,
            width=width
        )
        ctx.obj = CLIContext(config)
    
    return cli


def add_common_options(func):
    """添加通用選項的裝飾器"""
    func = click.option('--config', '-c', type=click.Path(exists=True), 
                       help='Configuration file path')(func)
    func = click.option('--dry-run', is_flag=True, 
                       help='Show what would be done without executing')(func)
    return func


def validate_path(ctx, param, value):
    """路徑驗證回調"""
    if value and not Path(value).exists():
        raise click.BadParameter(f"Path does not exist: {value}")
    return value


def validate_json(ctx, param, value):
    """JSON 驗證回調"""
    if value:
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise click.BadParameter(f"Invalid JSON: {e}")
    return value


class CLICommand:
    """CLI 命令基類"""
    
    def __init__(self, ctx: CLIContext):
        self.ctx = ctx
    
    def execute(self, *args, **kwargs):
        """執行命令"""
        raise NotImplementedError
    
    def validate_input(self, **kwargs):
        """驗證輸入"""
        pass
    
    def handle_error(self, error: Exception):
        """處理錯誤"""
        if self.ctx.config.debug:
            import traceback
            self.ctx.print_error(f"{error}\n{traceback.format_exc()}")
        else:
            self.ctx.print_error(str(error))


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".aiva" / "config.json"
        self._config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self):
        """加載配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load config: {e}")
    
    def save_config(self):
        """保存配置"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """獲取配置值"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        """設置配置值"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def delete(self, key: str) -> bool:
        """刪除配置值"""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                return False
            config = config[k]
        
        if keys[-1] in config:
            del config[keys[-1]]
            return True
        return False


def create_aiva_cli():
    """創建 AIVA CLI"""
    cli = create_cli_group()
    
    @cli.command()
    @click.pass_obj
    def version(ctx):
        """顯示版本信息"""
        try:
            from ..version import __version__
            ctx.print_info(f"AIVA Common v{__version__}")
        except ImportError:
            ctx.print_info("AIVA Common (version unknown)")
    
    @cli.group()
    @click.pass_obj
    def config(ctx):
        """配置管理"""
        pass
    
    @config.command('get')
    @click.argument('key')
    @click.pass_obj
    def config_get(ctx, key):
        """獲取配置值"""
        manager = ConfigManager()
        value = manager.get(key)
        if value is not None:
            if ctx.config.output_format == 'json':
                ctx.print_json({key: value})
            else:
                ctx.print(f"{key}: {value}")
        else:
            ctx.print_error(f"Configuration key '{key}' not found")
    
    @config.command('set')
    @click.argument('key')
    @click.argument('value')
    @click.pass_obj
    def config_set(ctx, key, value):
        """設置配置值"""
        manager = ConfigManager()
        # 嘗試解析為 JSON
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        
        manager.set(key, parsed_value)
        manager.save_config()
        ctx.print_success(f"Set {key} = {parsed_value}")
    
    @config.command('list')
    @click.pass_obj
    def config_list(ctx):
        """列出所有配置"""
        manager = ConfigManager()
        if ctx.config.output_format == 'json':
            ctx.print_json(manager._config)
        else:
            def print_config(config, prefix=""):
                for key, value in config.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        print_config(value, full_key)
                    else:
                        ctx.print(f"{full_key}: {value}")
            
            if manager._config:
                print_config(manager._config)
            else:
                ctx.print("No configuration found")
    
    return cli


# 輔助函數
def run_cli_command(command_func, *args, **kwargs):
    """運行 CLI 命令"""
    try:
        return command_func(*args, **kwargs)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


# 全域實例
default_config_manager = ConfigManager()


__all__ = [
    "CLIConfig",
    "CLIContext", 
    "CLIProgressBar",
    "CLICommand",
    "ConfigManager",
    "create_cli_group",
    "create_aiva_cli",
    "add_common_options",
    "validate_path",
    "validate_json",
    "run_cli_command",
    "default_config_manager",
]