"""直接測試 Rust Bridge"""
import sys
sys.path.insert(0, 'C:\\D\\fold7\\AIVA-git')

from services.scan.engines.rust_engine.python_bridge import rust_info_gatherer

# 測試1: 使用 deep_analysis
print("="*80)
print("Test 1: mode = 'deep_analysis'")
print("="*80)

result = rust_info_gatherer.scan_target(
    "http://localhost:3000",
    {"mode": "deep_analysis", "timeout": 30}
)

print(f"Success: {result.get('success')}")
print(f"Error: {result.get('error')}")
if result.get('results'):
    print(f"Endpoints: {len(result['results'].get('targets', [{}])[0].get('endpoints', []))}")

# 測試2: 使用 fast_discovery  
print("\n" + "="*80)
print("Test 2: mode = 'fast_discovery'")
print("="*80)

result = rust_info_gatherer.scan_target(
    "http://localhost:3000",
    {"mode": "fast_discovery", "timeout": 30}
)

print(f"Success: {result.get('success')}")
print(f"Error: {result.get('error')}")
if result.get('results'):
    print(f"Endpoints: {len(result['results'].get('targets', [{}])[0].get('endpoints', []))}")

# 測試3: 沒有 mode (應該失敗)
print("\n" + "="*80)
print("Test 3: 沒有 mode 參數")
print("="*80)

result = rust_info_gatherer.scan_target(
    "http://localhost:3000",
    {"timeout": 30}
)

print(f"Success: {result.get('success')}")
print(f"Error: {result.get('error')}")
