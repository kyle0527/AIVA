#!/usr/bin/env python3
"""
獨立CLI驗證器 - 無需完整 aiva_core 依賴
用於快速驗證和測試 AIVA CLI 命令

特性：
- 直接導入目標模組，繞過 __init__.py 導入鏈
- 最小依賴：僅需 Python 標準庫 + pydantic
- 快速啟動：<1秒
- 支持 676 個能力的參數驗證

Usage:
    python standalone_cli_validator.py --flow 51
    python standalone_cli_validator.py --list-capabilities
    python standalone_cli_validator.py --validate-all
"""

import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import importlib.util

# ==================== 路徑配置 ====================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.parent.parent
AIVA_CORE = PROJECT_ROOT / "services/core/aiva_core"
DATA_DIR = PROJECT_ROOT / "data/internal_exploration/analysis_history/v1"

# 添加專案路徑到 sys.path
sys.path.insert(0, str(PROJECT_ROOT))

# ==================== 最小依賴檢查 ====================
REQUIRED_PACKAGES = {
    "json": "標準庫",
    "pathlib": "標準庫",
    "datetime": "標準庫",
}

OPTIONAL_PACKAGES = {
    "pydantic": "用於數據驗證",
}

def check_dependencies():
    """檢查最小依賴"""
    missing = []
    for pkg, desc in REQUIRED_PACKAGES.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"{pkg} ({desc})")
    
    if missing:
        print(f"❌ 缺少必需依賴: {', '.join(missing)}")
        return False
    
    print("✅ 所有必需依賴已滿足")
    return True

# ==================== 直接模組導入（繞過 __init__.py）====================
def load_module_directly(module_name: str, file_path: Path) -> Any:
    """
    直接從檔案路徑載入模組，繞過 __init__.py 導入鏈
    
    Args:
        module_name: 模組名稱（用於 sys.modules）
        file_path: 模組檔案絕對路徑
    
    Returns:
        載入的模組對象
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"無法創建模組規範: {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"❌ 載入模組失敗 {module_name}: {e}")
        return None

def load_ai_capabilities() -> Dict[str, Any]:
    """
    載入 AI 內部能力模組（無需完整 aiva_core）
    
    Returns:
        能力模組字典: {
            'internal_loop_connector': module,
            'rl_models': module,
            'neural_network': module,
        }
    """
    capabilities = {}
    
    # 1. 載入 internal_loop_connector
    ilc_path = AIVA_CORE / "cognitive_core/internal_loop_connector.py"
    if ilc_path.exists():
        ilc_module = load_module_directly("internal_loop_connector", ilc_path)
        if ilc_module:
            capabilities['internal_loop_connector'] = ilc_module
            print(f"✅ 載入成功: internal_loop_connector")
    
    # 2. 載入 rl_models
    rl_path = AIVA_CORE / "ai_models/rl_models.py"
    if rl_path.exists():
        rl_module = load_module_directly("rl_models", rl_path)
        if rl_module:
            capabilities['rl_models'] = rl_module
            print(f"✅ 載入成功: rl_models")
    
    # 3. 載入 neural_network
    nn_path = AIVA_CORE / "ai_models/neural_network.py"
    if nn_path.exists():
        nn_module = load_module_directly("neural_network", nn_path)
        if nn_module:
            capabilities['neural_network'] = nn_module
            print(f"✅ 載入成功: neural_network")
    
    return capabilities

# ==================== 數據載入 ====================
def load_classification_data() -> Optional[Dict[str, Any]]:
    """載入分類數據"""
    data_path = DATA_DIR / "classification_data.json"
    if not data_path.exists():
        print(f"❌ 找不到分類數據: {data_path}")
        return None
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 載入分類數據: {len(data.get('flows', []))} 個 flows")
        return data
    except Exception as e:
        print(f"❌ 載入分類數據失敗: {e}")
        return None

def load_analysis_results() -> Optional[Dict[str, Any]]:
    """載入分析結果"""
    data_path = DATA_DIR / "analysis_results.json"
    if not data_path.exists():
        print(f"❌ 找不到分析結果: {data_path}")
        return None
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 載入分析結果: {len(data.get('flows', []))} 個 flows")
        return data
    except Exception as e:
        print(f"❌ 載入分析結果失敗: {e}")
        return None

# ==================== CLI 驗證功能 ====================
def validate_flow_parameters(flow_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    驗證單一 flow 的參數配置
    
    Args:
        flow_data: Flow 數據字典
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # 檢查必需欄位
    required_fields = ['flow_number', 'start_point', 'end_point', 'cli_command']
    for field in required_fields:
        if field not in flow_data:
            errors.append(f"缺少必需欄位: {field}")
    
    # 檢查 CLI 命令格式
    cli_cmd = flow_data.get('cli_command', '')
    if not cli_cmd.startswith('aiva '):
        errors.append(f"CLI 命令格式錯誤: {cli_cmd}")
    
    # 檢查參數
    params = flow_data.get('parameters', {})
    if not isinstance(params, dict):
        errors.append(f"參數格式錯誤: {type(params)}")
    
    # 檢查必需參數
    for param_name, param_info in params.items():
        if isinstance(param_info, dict):
            if param_info.get('required', False):
                # 必需參數應該沒有預設值
                if param_info.get('default') not in [None, 'None', '', 'NO_DEFAULT']:
                    errors.append(f"必需參數 {param_name} 不應有預設值: {param_info.get('default')}")
    
    return len(errors) == 0, errors

def list_all_capabilities(classification_data: Dict[str, Any]) -> None:
    """列出所有能力"""
    flows = classification_data.get('flows', [])
    
    print(f"\n{'='*80}")
    print(f"AIVA 能力清單 (共 {len(flows)} 個)")
    print(f"{'='*80}\n")
    
    # 按模組分組
    by_module = {}
    for flow in flows:
        module = flow.get('module_name', 'unknown')
        if module not in by_module:
            by_module[module] = []
        by_module[module].append(flow)
    
    # 按 AI 類型分組
    ai_internal = []
    ai_external = []
    non_ai = []
    
    for flow in flows:
        ai_type = flow.get('ai_capability_type', 'non_ai')
        if ai_type == 'AI內部能力':
            ai_internal.append(flow)
        elif ai_type == 'AI對外能力':
            ai_external.append(flow)
        else:
            non_ai.append(flow)
    
    print(f"📊 按 AI 類型分類:")
    print(f"  - AI內部能力: {len(ai_internal)} 個")
    print(f"  - AI對外能力: {len(ai_external)} 個")
    print(f"  - 非AI能力: {len(non_ai)} 個\n")
    
    print(f"📦 按模組分類:")
    for module, module_flows in sorted(by_module.items()):
        print(f"  - {module}: {len(module_flows)} 個能力")
    
    print(f"\n{'='*80}")

def validate_single_flow(flow_number: int, classification_data: Dict[str, Any]) -> None:
    """驗證單一 flow"""
    flows = classification_data.get('flows', [])
    
    # 找到對應的 flow
    target_flow = None
    for flow in flows:
        if flow.get('flow_number') == flow_number:
            target_flow = flow
            break
    
    if not target_flow:
        print(f"❌ 找不到 Flow {flow_number}")
        return
    
    print(f"\n{'='*80}")
    print(f"驗證 Flow {flow_number}")
    print(f"{'='*80}\n")
    
    # 顯示基本信息
    print(f"模組: {target_flow.get('module_name', 'N/A')}")
    print(f"AI類型: {target_flow.get('ai_capability_type', 'N/A')}")
    print(f"起點: {target_flow.get('start_point', 'N/A')}")
    print(f"終點: {target_flow.get('end_point', 'N/A')}")
    print(f"CLI命令: {target_flow.get('cli_command', 'N/A')}\n")
    
    # 驗證參數
    is_valid, errors = validate_flow_parameters(target_flow)
    
    if is_valid:
        print("✅ 參數驗證通過")
    else:
        print("❌ 參數驗證失敗:")
        for error in errors:
            print(f"  - {error}")
    
    # 顯示參數詳情
    params = target_flow.get('parameters', {})
    if params:
        print(f"\n參數詳情 ({len(params)} 個):")
        for param_name, param_info in params.items():
            if isinstance(param_info, dict):
                print(f"  - {param_name}:")
                print(f"      類型: {param_info.get('type', 'N/A')}")
                print(f"      必需: {param_info.get('required', False)}")
                print(f"      預設: {param_info.get('default', 'N/A')}")
                if param_info.get('docstring'):
                    print(f"      說明: {param_info.get('docstring')}")
    
    print(f"\n{'='*80}")

def validate_all_flows(classification_data: Dict[str, Any]) -> None:
    """驗證所有 flows"""
    flows = classification_data.get('flows', [])
    
    print(f"\n{'='*80}")
    print(f"驗證所有 Flows (共 {len(flows)} 個)")
    print(f"{'='*80}\n")
    
    valid_count = 0
    invalid_count = 0
    invalid_flows = []
    
    for flow in flows:
        flow_num = flow.get('flow_number', 'N/A')
        is_valid, errors = validate_flow_parameters(flow)
        
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            invalid_flows.append((flow_num, errors))
    
    print(f"✅ 有效: {valid_count} 個")
    print(f"❌ 無效: {invalid_count} 個")
    
    if invalid_flows:
        print(f"\n無效 Flows 詳情:")
        for flow_num, errors in invalid_flows[:10]:  # 只顯示前 10 個
            print(f"\n  Flow {flow_num}:")
            for error in errors:
                print(f"    - {error}")
        
        if len(invalid_flows) > 10:
            print(f"\n  ... 還有 {len(invalid_flows) - 10} 個無效 flows")
    
    print(f"\n{'='*80}")

def test_ai_capability(capability_name: str, capabilities: Dict[str, Any]) -> None:
    """測試 AI 能力"""
    if capability_name not in capabilities:
        print(f"❌ 能力 {capability_name} 未載入")
        return
    
    module = capabilities[capability_name]
    
    print(f"\n{'='*80}")
    print(f"測試 AI 能力: {capability_name}")
    print(f"{'='*80}\n")
    
    try:
        if capability_name == 'internal_loop_connector':
            # 測試 CapabilityScopeClassifier
            classifier = module.CapabilityScopeClassifier()
            print(f"✅ 成功創建 CapabilityScopeClassifier")
            
            # 測試分類
            test_file = "services/core/aiva_core/cognitive_core/internal_loop_connector.py"
            scope, visibility = classifier.classify_scope(test_file)
            print(f"✅ 測試分類: scope={scope}, visibility={visibility}")
        
        elif capability_name == 'rl_models':
            # 測試 DQN
            if hasattr(module, 'DQN'):
                # 注意: 需要 torch，這裡只能檢查類是否存在
                print(f"✅ DQN 類存在 (需要 torch 才能實例化)")
            
            if hasattr(module, 'PPO'):
                print(f"✅ PPO 類存在 (需要 torch 才能實例化)")
        
        elif capability_name == 'neural_network':
            # 測試 neural_network
            if hasattr(module, 'NeuralNetwork'):
                print(f"✅ NeuralNetwork 類存在 (需要 torch 才能實例化)")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
    
    print(f"\n{'='*80}")

# ==================== 主程序 ====================
def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AIVA CLI 驗證器 - 快速驗證和測試 CLI 命令',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python standalone_cli_validator.py --list-capabilities
  python standalone_cli_validator.py --flow 51
  python standalone_cli_validator.py --validate-all
  python standalone_cli_validator.py --test-ai internal_loop_connector
        """
    )
    
    parser.add_argument('--list-capabilities', action='store_true',
                       help='列出所有能力')
    parser.add_argument('--flow', type=int, metavar='N',
                       help='驗證指定的 Flow N')
    parser.add_argument('--validate-all', action='store_true',
                       help='驗證所有 flows')
    parser.add_argument('--test-ai', type=str, metavar='NAME',
                       choices=['internal_loop_connector', 'rl_models', 'neural_network'],
                       help='測試指定的 AI 能力')
    parser.add_argument('--check-deps', action='store_true',
                       help='檢查依賴')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"AIVA CLI 驗證器 v1.0")
    print(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 檢查依賴
    if args.check_deps or not any(vars(args).values()):
        check_dependencies()
        if args.check_deps:
            return
    
    # 載入數據
    classification_data = load_classification_data()
    if not classification_data:
        return
    
    # 執行命令
    if args.list_capabilities:
        list_all_capabilities(classification_data)
    
    elif args.flow:
        validate_single_flow(args.flow, classification_data)
    
    elif args.validate_all:
        validate_all_flows(classification_data)
    
    elif args.test_ai:
        capabilities = load_ai_capabilities()
        test_ai_capability(args.test_ai, capabilities)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
