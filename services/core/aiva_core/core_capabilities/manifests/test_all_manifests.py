"""測試所有 Manifest（Flow 0-4）"""
import json
from pathlib import Path

# 路徑配置
manifests_dir = Path(__file__).parent / "capabilities"

def load_manifest(flow_id):
    """載入指定的 Manifest"""
    # 找到對應的 JSON 文件
    json_files = list(manifests_dir.glob(f"{flow_id:02d}_*.json"))
    if not json_files:
        raise FileNotFoundError(f"找不到 Flow {flow_id} 的 Manifest")
    
    with open(json_files[0], 'r', encoding='utf-8') as f:
        return json.load(f)

def map_linear(intensity, min_val, max_val):
    """線性映射"""
    result = min_val + intensity * (max_val - min_val)
    return int(result) if isinstance(min_val, int) else round(result, 6)

def map_exponential(intensity, min_val, max_val):
    """指數映射"""
    import math
    if min_val == 0:
        return int(max_val * (intensity ** 2))
    ratio = max_val / min_val
    result = min_val * (ratio ** intensity)
    return int(result) if isinstance(min_val, int) else round(result, 6)

def map_logarithmic(intensity, min_val, max_val):
    """對數映射"""
    import math
    result = min_val + (max_val - min_val) * math.log10(1 + intensity * 9)
    return int(result) if isinstance(min_val, int) else round(result, 6)

def build_command(flow_id, context_data, intensity):
    """構建命令"""
    manifest = load_manifest(flow_id)
    template = manifest['execution']['command_template']
    
    # A-class: 上下文映射
    for param, ctx_key in manifest['parameters']['context_mapping'].items():
        value = context_data.get(ctx_key, '')
        template = template.replace(f'{{{param}}}', str(value))
    
    # B-class: 可調參數
    for param, config in manifest['parameters']['tunable'].items():
        strategy = config['mapping_strategy']
        min_val = config['min']
        max_val = config['max']
        
        if strategy == 'linear':
            value = map_linear(intensity, min_val, max_val)
        elif strategy == 'exponential':
            value = map_exponential(intensity, min_val, max_val)
        elif strategy == 'logarithmic':
            value = map_logarithmic(intensity, min_val, max_val)
        else:
            value = min_val
        
        template = template.replace(f'{{{param}}}', str(value))
    
    return template

def test_all_manifests():
    """測試所有 Manifest"""
    print("=" * 80)
    print("🚀 測試所有 Manifest（Flow 0-4）")
    print("=" * 80)
    
    # Flow 0: Internal Loop Connector
    print("\n[Flow 0] Internal Loop Connector:")
    for intensity in [0.2, 0.5, 0.9]:
        cmd = build_command(0, {"query": "SQL injection"}, intensity)
        print(f"  強度 {intensity}: {cmd}")
    
    # Flow 1: Health Check
    print("\n[Flow 1] Health Check:")
    cmd = build_command(1, {}, 0.0)
    print(f"  {cmd}")
    
    # Flow 2: Scan Status
    print("\n[Flow 2] Scan Status:")
    cmd = build_command(2, {"scan_id": "scan_20231225_001"}, 0.0)
    print(f"  {cmd}")
    
    # Flow 3: Session State Manager
    print("\n[Flow 3] Session State Manager:")
    for intensity in [0.2, 0.9]:
        cmd = build_command(3, {"scan_id": "scan_20231225_001"}, intensity)
        print(f"  強度 {intensity}: {cmd}")
    
    # Flow 4: ScalableBio Trainer
    print("\n[Flow 4] ScalableBio Trainer:")
    for intensity in [0.2, 0.5, 0.9]:
        cmd = build_command(4, {"training_data_path": "/data/test.npz"}, intensity)
        print(f"  強度 {intensity}: {cmd}")
    
    print("\n" + "=" * 80)
    print("✅ 所有測試完成！已驗證 5 個 Manifest")
    print("=" * 80)
    
    # 統計資訊
    print("\n📊 Manifest 統計:")
    for flow_id in range(5):
        manifest = load_manifest(flow_id)
        print(f"  Flow {flow_id}: {manifest['meta']['tool_name']}")
        print(f"    - 模組: {manifest['meta']['module']}")
        print(f"    - 風險: {manifest['ai_cognitive']['risk_level']}")
        print(f"    - 可調參數: {len(manifest['parameters']['tunable'])}")

if __name__ == "__main__":
    test_all_manifests()
