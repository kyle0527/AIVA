#!/usr/bin/env python3
"""
檢查數據文件的生成時間和版本信息
"""

import json
from pathlib import Path
from datetime import datetime

# 可能的數據路徑
possible_paths = [
    Path("C:/Users/User/Downloads/data/internal_exploration/latest_classification.json"),
    Path("C:/D/fold7/AIVA-git/data/internal_exploration/latest_classification.json"),
    Path("C:/D/fold7/AIVA-git/services/integration/data/internal_exploration/latest_classification.json"),
]

print("\n" + "="*70)
print("📊 檢查所有可能的分類數據文件")
print("="*70)

for path in possible_paths:
    print(f"\n路徑: {path}")
    if path.exists():
        print("  ✅ 文件存在")
        
        # 讀取文件
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 顯示元數據
        metadata = data.get('metadata', {})
        print(f"  生成時間: {metadata.get('generated_at', 'N/A')}")
        print(f"  總流程數: {metadata.get('total_flows', 'N/A')}")
        
        # 檢查版本信息
        version = data.get('version', metadata.get('version', 'N/A'))
        print(f"  版本: {version}")
        
        # 檢查模組分佈
        module_dist = metadata.get('module_distribution', {})
        if module_dist:
            print(f"  模組分佈 (元數據):")
            for module, count in sorted(module_dist.items(), key=lambda x: x[1], reverse=True):
                print(f"    {module:25} : {count:4}")
        
        # 實際計算終點模組分佈
        flows = data.get('flows', [])
        from collections import Counter
        endpoint_modules = Counter()
        for flow in flows:
            modules = flow.get('modules', [])
            if modules:
                endpoint_modules[modules[-1]] += 1
        
        print(f"  終點模組分佈 (實際計算):")
        for module, count in sorted(endpoint_modules.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(flows) * 100 if flows else 0
            print(f"    {module:25} : {count:4} ({percentage:5.1f}%)")
        
        # 文件修改時間
        import os
        mtime = os.path.getmtime(path)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  文件修改時間: {mtime_str}")
        
    else:
        print("  ❌ 文件不存在")

print("\n" + "="*70)
print("\n❓ 問題診斷:")
print("-" * 70)
print("""
如果生成時間是 2025-12-15，說明這是修復前的舊數據。
如果生成時間是 2026-01-01，說明這是修復後的新數據。

CHANGELOG_CLI.md 記錄的 v3.2 修復日期是 2026-01-01。
""")
print("="*70)
