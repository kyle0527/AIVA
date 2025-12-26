"""AIVA 統一 CLI 入口點 - 基於 Manifest 的函數調用系統"""
import click
import json
from pathlib import Path
from typing import Any, Dict

from services.core.aiva_core.cognitive_core.manifest.manifest_loader import ManifestLoader
from services.core.aiva_core.task_planning.command_builder import CommandBuilder


@click.group()
def aiva():
    """AIVA - AI-powered Vulnerability Analysis System
    
    基於 Manifest 架構的統一命令接口，支援 840 flows 的函數調用鏈。
    """
    pass


@aiva.command()
@click.argument('flow_id', type=int)
@click.option('--context', '-c', help='上下文數據 (JSON 格式)', default='{}')
@click.option('--intensity', '-i', type=float, default=0.5, help='AI 強度 (0.0-1.0)')
@click.option('--dry-run', is_flag=True, help='僅顯示將執行的操作，不實際執行')
def run(flow_id: int, context: str, intensity: float, dry_run: bool):
    """執行指定 Flow
    
    Examples:
        aiva run 0 -c '{"query": "SQL injection"}' -i 0.8
        aiva run 4 -c '{"training_data_path": "/data/model.npz"}' -i 0.6
    """
    try:
        # 解析上下文
        context_data = json.loads(context)
        
        # 載入 Manifest
        loader = ManifestLoader()
        manifest = loader.get_by_flow_id(flow_id)
        
        if not manifest:
            click.echo(f"❌ 找不到 Flow {flow_id} 的 Manifest", err=True)
            return 1
        
        click.echo(f"🚀 執行 Flow {flow_id}: {manifest.meta.tool_name}")
        click.echo(f"   模組: {manifest.meta.module}")
        click.echo(f"   強度: {intensity}")
        
        if dry_run:
            click.echo("\n🔍 Dry Run 模式 - 預覽操作:\n")
            builder = CommandBuilder()
            preview = builder.preview_parameters(flow_id, context_data, [0.2, 0.5, 0.9])
            click.echo(json.dumps(preview, indent=2, ensure_ascii=False))
            return 0
        
        # 實際執行 Flow
        from services.core.aiva_core.core_capabilities.manifests.flow_executor import FlowExecutor
        executor = FlowExecutor()
        result = executor.execute_flow(flow_id, context_data, intensity)
        
        click.echo(f"\n✅ 執行完成")
        click.echo(f"結果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
    except json.JSONDecodeError:
        click.echo("❌ 無效的 JSON 格式", err=True)
        return 1
    except Exception as e:
        click.echo(f"❌ 執行失敗: {e}", err=True)
        return 1


@aiva.command()
@click.argument('query_text')
@click.option('--intensity', '-i', type=float, default=0.5)
def query(query_text: str, intensity: float):
    """內部查詢 (Flow 0 別名)
    
    Example: aiva query "SQL injection vulnerabilities" -i 0.8
    """
    context = json.dumps({"query": query_text})
    ctx = click.get_current_context()
    ctx.invoke(run, flow_id=0, context=context, intensity=intensity, dry_run=False)


@aiva.command()
@click.argument('data_path')
@click.option('--intensity', '-i', type=float, default=0.5)
def train(data_path: str, intensity: float):
    """模型訓練 (Flow 4 別名)
    
    Example: aiva train /data/dataset.npz -i 0.7
    """
    context = json.dumps({"training_data_path": data_path})
    ctx = click.get_current_context()
    ctx.invoke(run, flow_id=4, context=context, intensity=intensity, dry_run=False)


@aiva.command()
@click.argument('target_url')
@click.option('--intensity', '-i', type=float, default=0.5)
def scan(target_url: str, intensity: float):
    """攻擊面掃描 (Flow 8 別名)
    
    Example: aiva scan https://example.com -i 0.6
    """
    context = json.dumps({"target_url": target_url})
    ctx = click.get_current_context()
    ctx.invoke(run, flow_id=8, context=context, intensity=intensity, dry_run=False)


@aiva.command()
@click.argument('scan_id')
def status(scan_id: str):
    """查詢掃描狀態 (Flow 2 別名)
    
    Example: aiva status scan_20231225_001
    """
    context = json.dumps({"scan_id": scan_id})
    ctx = click.get_current_context()
    ctx.invoke(run, flow_id=2, context=context, intensity=0.0, dry_run=False)


@aiva.command()
def health():
    """系統健康檢查 (Flow 1 別名)
    
    Example: aiva health
    """
    ctx = click.get_current_context()
    ctx.invoke(run, flow_id=1, context='{}', intensity=0.0, dry_run=False)


@aiva.command()
def list():
    """列出所有可用的 Flows"""
    loader = ManifestLoader()
    loader.load_all()
    stats = loader.get_stats()
    
    click.echo(f"\n📋 可用的 Flows ({stats['total_manifests']} 個):\n")
    
    all_manifests = loader._manifests
    for flow_id in sorted(all_manifests.keys()):
        manifest = all_manifests[flow_id]
        click.echo(f"Flow {flow_id:2d}: {manifest.meta.tool_name}")
        click.echo(f"         模組: {manifest.meta.module}")
        click.echo(f"         風險: {manifest.ai_cognitive.risk_level}")
        click.echo(f"         描述: {manifest.meta.description}")
        click.echo()


if __name__ == "__main__":
    aiva()
