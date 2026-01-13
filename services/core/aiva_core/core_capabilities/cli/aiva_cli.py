"""AIVA 統一 CLI 入口點 - 基於動態 Flow 的函數調用系統"""
import click
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# 統一使用 FlowExecutor（來自 internal_exploration）
# ManifestLoader 已棄用 - manifests/capabilities 目錄已移除


def load_flow_definitions() -> List[Dict[str, Any]]:
    """從 latest_classification.json 讀取所有 flows
    
    Returns:
        List[Dict]: Flow 定義列表
    """
    possible_paths = [
        Path("C:/Users/User/Downloads/data/internal_exploration/latest_classification.json"),
        Path("C:/D/fold7/AIVA-git/data/internal_exploration/latest_classification.json"),
        Path("C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/latest_classification.json"),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                    flows = data.get('flows', [])
                    if flows:
                        click.echo(f"📋 已載入 {len(flows)} 個 flows (來源: {path.name})")
                        return flows
            except Exception as e:
                click.echo(f"⚠️  無法讀取 {path}: {e}", err=True)
                continue
    
    click.echo("⚠️  未找到 flow 定義文件", err=True)
    return []


def create_flow_command(flow_id: int, flow_info: Dict[str, Any]):
    """為指定 flow_id 創建命令函數
    
    Args:
        flow_id: Flow ID
        flow_info: Flow 定義信息（從 JSON 讀取）
    
    Returns:
        Click command 函數
    """
    # 提取 flow 信息用於 help 文本
    path_preview = " -> ".join(flow_info.get('path', [])[:3])
    if len(flow_info.get('path', [])) > 3:
        path_preview += " ..."
    primary_module = flow_info.get('primary_module', 'unknown')
    length = flow_info.get('length', 0)
    
    @click.option('--target', '-t', help='目標 URL/路徑/對象')
    @click.option('--data', '-d', help='數據路徑')
    @click.option('--query', '-q', help='查詢字串')
    @click.option('--param', '-p', multiple=True, help='額外參數 (key=value 格式)')
    @click.option('--intensity', '-i', type=float, default=0.5, 
                  help='AI 強度 (0.0-1.0)')
    @click.option('--dry-run', is_flag=True, help='預覽模式，不實際執行')
    def flow_command(target, data, query, param, intensity, dry_run):
        """執行指定 Flow
        
        根據 flow_id 執行對應的 Flow 工作流程。
        
        Examples:
            aiva flow0 --target https://example.com -i 0.8
            aiva flow1 --data /path/to/data.npz -i 0.6
            aiva flow2 --param key1=value1 --param key2=value2
        """
        # 構建 context_data
        context_data = {}
        
        if target:
            context_data['target'] = target
            context_data['target_url'] = target  # 兼容性
            context_data['url'] = target
        
        if data:
            context_data['data_path'] = data
            context_data['training_data_path'] = data  # 兼容性
            context_data['input_path'] = data
        
        if query:
            context_data['query'] = query
        
        # 解析 --param key=value
        for p in param:
            if '=' in p:
                k, v = p.split('=', 1)
                context_data[k.strip()] = v.strip()
            else:
                click.echo(f"⚠️  忽略無效參數: {p} (應為 key=value 格式)", err=True)
        
        # 顯示執行信息
        click.echo(f"\n🚀 執行 Flow {flow_id}")
        click.echo(f"   模組: {primary_module}")
        click.echo(f"   路徑: {path_preview}")
        click.echo(f"   強度: {intensity}")
        if context_data:
            click.echo(f"   參數: {json.dumps(context_data, ensure_ascii=False)}")
        
        if dry_run:
            click.echo("\n🔍 Dry Run 模式 - 僅預覽，不執行")
            return
        
        # 執行 Flow
        try:
            from services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation import FlowExecutor
            
            executor = FlowExecutor()
            click.echo("\n⏳ 執行中...")
            executor.execute_flow(flow_id, context_data=context_data, dry_run=False)
            
            click.echo("\n✅ 執行完成")
            return 0  # 成功退出碼
            
        except ImportError as e:
            click.echo(f"\n❌ 依賴缺失: {e}", err=True)
            return 2  # 系統錯誤退出碼
        except Exception as e:
            click.echo(f"\n❌ 執行失敗: {e}", err=True)
            import traceback
            traceback.print_exc()
            return 1  # 邏輯錯誤退出碼
    
    return flow_command


def register_all_flow_commands(cli_group):
    """為所有 flows 注冊動態命令
    
    Args:
        cli_group: Click group 對象
    """
    flows = load_flow_definitions()
    
    if not flows:
        return
    
    registered_count = 0
    for flow in flows:
        flow_id = flow.get('id')
        if flow_id is None:
            continue
        
        try:
            # 創建命令
            cmd = create_flow_command(flow_id, flow)
            
            # 注冊到 CLI group
            cli_group.command(name=f"flow{flow_id}")(cmd)
            registered_count += 1
        except Exception as e:
            click.echo(f"⚠️  無法注冊 flow{flow_id}: {e}", err=True)
    
    if registered_count > 0:
        click.echo(f"✅ 已注冊 {registered_count} 個動態命令 (flow0 - flow{registered_count-1})\n")


@click.group()
def aiva():
    """AIVA - AI-powered Vulnerability Analysis System
    
    基於動態 Flow 架構的統一命令接口，支援自動分析生成的 flows 命令。
    
    使用方式:
        aiva flow<ID> --target <目標> -i <強度>
        aiva flow<ID> --data <路徑> --dry-run
        aiva list-flows  # 查看所有可用 flows
    """
    pass


@aiva.command()
@click.argument('flow_id', type=int)
@click.option('--context', '-c', help='上下文數據 (JSON 格式)', default='{}')
@click.option('--intensity', '-i', type=float, default=0.5, help='AI 強度 (0.0-1.0)')
@click.option('--dry-run', is_flag=True, help='僅顯示將執行的操作，不實際執行')
def run(flow_id: int, context: str, intensity: float, dry_run: bool):
    """執行指定 Flow（基於 flow_id）
    
    統一使用 FlowExecutor 執行，不再依賴已棄用的 ManifestLoader。
    
    Examples:
        aiva run 0 -c '{"query": "find vulnerabilities"}' -i 0.8
        aiva run 8 -c '{"target_url": "http://example.com"}' -i 0.6
        aiva run 4 -c '{"training_data_path": "/data/train.npz"}' --dry-run
    """
    try:
        # 解析上下文
        context_data = json.loads(context)
        
        # 使用 FlowExecutor 執行
        from services.core.aiva_core.internal_exploration.python_tools.aiva_cli_implementation import FlowExecutor
        
        executor = FlowExecutor()
        flow = executor.get_flow_by_id(flow_id)
        
        if not flow:
            click.echo(f"❌ 找不到 Flow {flow_id}", err=True)
            return 1
        
        # 顯示 flow 資訊
        path_preview = " -> ".join(flow.get('path', [])[:3])
        if len(flow.get('path', [])) > 3:
            path_preview += " ..."
        
        click.echo(f"🚀 執行 Flow {flow_id}")
        click.echo(f"   路徑: {path_preview}")
        click.echo(f"   強度: {intensity}")
        
        if dry_run:
            click.echo("\n🔍 Dry Run 模式 - 預覽執行流程")
            executor.execute_flow(flow_id, context_data=context_data, dry_run=True)
            return 0
        
        # 實際執行
        click.echo("\n⏳ 執行中...")
        executor.execute_flow(flow_id, context_data=context_data, dry_run=False)
        click.echo("\n✅ 執行完成")
        
    except json.JSONDecodeError:
        click.echo("❌ 無效的 JSON 格式", err=True)
        return 1
    except Exception as e:
        click.echo(f"❌ 執行失敗: {e}", err=True)
        import traceback
        traceback.print_exc()
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


@aiva.command(name='list-flows')
@click.option('--limit', '-l', type=int, default=20, help='顯示數量限制')
@click.option('--module', '-m', help='過濾特定模組')
@click.option('--by-endpoint', is_flag=True, help='按終點模組分類（六大模組）')
@click.option('--stats', is_flag=True, help='顯示統計摘要')
def list_flows(limit, module, by_endpoint, stats):
    """列出所有可用的 Flows
    
    Examples:
        aiva list-flows --limit 10              # 列出前10個
        aiva list-flows --module cognitive_core # 過濾特定模組
        aiva list-flows --by-endpoint           # 按終點模組分類
        aiva list-flows --stats                 # 顯示統計
    """
    flows = load_flow_definitions()
    
    if not flows:
        click.echo("❌ 無法載入 flow 定義", err=True)
        return
    
    # 統計模式
    if stats:
        show_flow_statistics(flows)
        return
    
    # 按終點模組分類模式
    if by_endpoint:
        show_flows_by_endpoint_module(flows, limit)
        return
    
    # 過濾
    if module:
        flows = [f for f in flows if f.get('endpoint_module') == module or f.get('primary_module') == module]
        click.echo(f"\n📋 {module} 模組的 Flows ({len(flows)} 個):\n")
    else:
        click.echo(f"\n📋 可用的 Flows ({len(flows)} 個):\n")
    
    # 顯示
    for flow in flows[:limit]:
        flow_id = flow.get('id')
        path = " -> ".join(flow.get('path', [])[:3])
        if len(flow.get('path', [])) > 3:
            path += " ..."
        endpoint_module = flow.get('endpoint_module', 'unknown')
        primary_module = flow.get('primary_module', 'unknown')
        length = flow.get('length', 0)
        
        click.echo(f"flow{flow_id:3d}: {path}")
        click.echo(f"         終點: {endpoint_module} | 主模組: {primary_module} | 長度: {length}")
        click.echo()
    
    if len(flows) > limit:
        click.echo(f"... 還有 {len(flows) - limit} 個 flows")
        click.echo(f"\n使用 'aiva list-flows --limit {len(flows)}' 查看全部")
        click.echo("使用 'aiva flow<ID> --help' 查看特定 flow 的詳情")


def show_flow_statistics(flows: List[Dict[str, Any]]):
    """顯示 Flow 統計信息"""
    from collections import Counter
    
    click.echo(f"\n📊 Flow 統計報告\n{'=' * 60}\n")
    
    # 總數
    click.echo(f"總 Flows: {len(flows)}")
    
    # 按終點模組統計（六大模組）
    endpoint_modules = Counter(f.get('endpoint_module', 'unknown') for f in flows)
    click.echo(f"\n按終點模組分佈:")
    for module, count in sorted(endpoint_modules.items(), key=lambda x: -x[1]):
        percentage = (count / len(flows)) * 100
        bar = '█' * int(percentage / 2)
        click.echo(f"  {module:20s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # 按主要模組統計
    primary_modules = Counter(f.get('primary_module', 'unknown') for f in flows)
    click.echo(f"\n按主要模組分佈:")
    for module, count in sorted(primary_modules.items(), key=lambda x: -x[1]):
        percentage = (count / len(flows)) * 100
        click.echo(f"  {module:20s}: {count:4d} ({percentage:5.1f}%)")
    
    # 按路徑長度統計
    lengths = Counter(f.get('length', 0) for f in flows)
    click.echo(f"\n按路徑長度分佈:")
    for length, count in sorted(lengths.items()):
        click.echo(f"  {length} 步: {count:4d} 個 flows")
    
    # 組件類型統計
    component_types = Counter(f.get('primary_component_type', 'unknown') for f in flows)
    click.echo(f"\n按組件類型分佈:")
    for comp_type, count in component_types.items():
        percentage = (count / len(flows)) * 100
        click.echo(f"  {comp_type:15s}: {count:4d} ({percentage:5.1f}%)")
    
    click.echo()


def show_flows_by_endpoint_module(flows: List[Dict[str, Any]], limit_per_module: int):
    """按終點模組（六大模組）分類顯示 Flows"""
    from collections import defaultdict
    
    # 按終點模組分組
    flows_by_endpoint = defaultdict(list)
    for flow in flows:
        endpoint_module = flow.get('endpoint_module', 'unknown')
        flows_by_endpoint[endpoint_module].append(flow)
    
    # 五大模組順序 (2026-01-03: external_learning 整合至 cognitive_core.learning_system)
    five_modules = [
        'cognitive_core',
        'internal_exploration', 
        'task_planning',
        'core_capabilities',
        'service_backbone',
        'learning_system',  # 向後相容：新的學習子系統
        'external_learning'  # 向後相容：舊的模組名稱
    ]
    
    click.echo(f"\n📋 按終點模組分類的 Flows (五大模組)\n{'=' * 70}\n")
    
    # 按五大模組順序顯示
    for module in five_modules:
        if module in flows_by_endpoint:
            module_flows = flows_by_endpoint[module]
            click.echo(f"🔹 {module.upper()} ({len(module_flows)} 個 flows)")
            click.echo("-" * 70)
            
            for flow in module_flows[:limit_per_module]:
                flow_id = flow.get('id')
                path = " -> ".join(flow.get('path', [])[:3])
                if len(flow.get('path', [])) > 3:
                    path += " ..."
                length = flow.get('length', 0)
                
                click.echo(f"  flow{flow_id:3d}: {path}")
                click.echo(f"           長度: {length} 步")
            
            if len(module_flows) > limit_per_module:
                click.echo(f"  ... 還有 {len(module_flows) - limit_per_module} 個 flows")
            
            click.echo()
    
    # 顯示未分類的
    other_modules = set(flows_by_endpoint.keys()) - set(five_modules)
    if other_modules:
        click.echo(f"🔹 其他模組")
        click.echo("-" * 70)
        for module in sorted(other_modules):
            module_flows = flows_by_endpoint[module]
            click.echo(f"  {module}: {len(module_flows)} 個 flows")
        click.echo()


# 🔑 關鍵：在 CLI 啟動時注冊所有動態命令
register_all_flow_commands(aiva)


# ============================================
# CLI 退出碼規範
# ============================================
# 0: 成功
# 1: 邏輯/業務錯誤 (如：找不到 Flow、參數無效)
# 2: 系統/運行時錯誤 (如：連接失敗、依賴缺失)
# 130: 用戶中斷 (Ctrl+C)

EXIT_SUCCESS = 0
EXIT_LOGIC_ERROR = 1
EXIT_SYSTEM_ERROR = 2
EXIT_USER_INTERRUPT = 130


if __name__ == "__main__":
    import sys
    
    try:
        result = aiva(standalone_mode=False)
        # Click 命令返回值可能是 None 或整數
        exit_code = result if isinstance(result, int) else EXIT_SUCCESS
        sys.exit(exit_code)
    except click.Abort:
        # 用戶中斷或 raise click.Abort()
        sys.exit(EXIT_USER_INTERRUPT)
    except click.ClickException as e:
        # Click 框架錯誤
        e.show()
        sys.exit(EXIT_LOGIC_ERROR)
    except KeyboardInterrupt:
        # Ctrl+C
        click.echo("\n⚠️ 用戶中斷", err=True)
        sys.exit(EXIT_USER_INTERRUPT)
    except Exception as e:
        # 未預期的系統錯誤
        click.echo(f"\n❌ 系統錯誤: {e}", err=True)
        sys.exit(EXIT_SYSTEM_ERROR)
