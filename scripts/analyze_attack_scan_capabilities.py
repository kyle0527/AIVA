#!/usr/bin/env python3
"""
分析攻擊和掃描相關能力
識別哪些終點可以執行掃描和攻擊操作
"""

import json
from pathlib import Path
from collections import defaultdict

def main():
    base_path = Path(r"C:\D\fold7\AIVA-git")
    classification_file = base_path / "services" / "integration" / "data" / "internal_exploration" / "latest_classification.json"
    
    print("\n" + "="*70)
    print("🎯 攻擊與掃描能力分析")
    print("="*70)
    
    with open(classification_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flows = data.get('flows', [])
    
    # 定義掃描和攻擊關鍵字
    scan_keywords = [
        'scan', 'scanner', 'reconnaissance', 'recon', 
        'probe', 'discovery', 'enumeration', 'fingerprint',
        'initial_surface', 'port_scan', 'network_scan'
    ]
    
    attack_keywords = [
        'attack', 'exploit', 'executor', 'injection',
        'xss', 'sqli', 'rce', 'payload', 'bizlogic',
        'attack_chain', 'attack_plan', 'pwn'
    ]
    
    # 定義能力數據類型
    from typing import Dict, List, Set, Any
    
    CapabilityData = Dict[str, List[str] | str | Set[str]]
    
    # 收集相關能力 - 使用明確的工廠函數
    def create_capability_entry() -> Dict[str, Any]:
        return {
            'paths': [],
            'module': '',
            'full_path': '',
            'descriptions': set()
        }
    
    scan_capabilities: Dict[str, Dict[str, Any]] = defaultdict(create_capability_entry)
    attack_capabilities: Dict[str, Dict[str, Any]] = defaultdict(create_capability_entry)
    
    for flow in flows:
        classifications = flow.get('classifications', [])
        if not classifications:
            continue
        
        full_path = flow.get('full_path', [])
        if len(full_path) < 2:
            continue
        
        end_path = full_path[-1]
        end_classification = classifications[-1]
        end_name = flow.get('end', '').lower()
        module = end_classification.get('module', 'unknown')
        description = end_classification.get('description', '')
        
        # 檢查是否為掃描能力
        is_scan = any(keyword in end_name for keyword in scan_keywords) or \
                  any(keyword in end_path.lower() for keyword in scan_keywords)
        
        # 檢查是否為攻擊能力
        is_attack = any(keyword in end_name for keyword in attack_keywords) or \
                    any(keyword in end_path.lower() for keyword in attack_keywords)
        
        if is_scan:
            scan_capabilities[end_path]['paths'].append(full_path[0])
            scan_capabilities[end_path]['module'] = module
            scan_capabilities[end_path]['full_path'] = end_path
            scan_capabilities[end_path]['descriptions'].add(description)
        
        if is_attack:
            attack_capabilities[end_path]['paths'].append(full_path[0])
            attack_capabilities[end_path]['module'] = module
            attack_capabilities[end_path]['full_path'] = end_path
            attack_capabilities[end_path]['descriptions'].add(description)
    
    # 統計並輸出掃描能力
    print("\n" + "="*70)
    print("🔍 掃描相關能力")
    print("="*70)
    print(f"\n找到 {len(scan_capabilities)} 個掃描相關能力\n")
    
    scan_by_module = defaultdict(list)
    for cap_path, cap_data in scan_capabilities.items():
        scan_by_module[cap_data['module']].append((cap_path, cap_data))
    
    for module in sorted(scan_by_module.keys()):
        caps = scan_by_module[module]
        print(f"\n📦 {module} ({len(caps)} 個能力)")
        print("-" * 70)
        
        for cap_path, cap_data in sorted(caps, key=lambda x: len(x[1]['paths']), reverse=True):
            cap_name = cap_path.split('\\')[-1].replace('.py', '') if '\\' in cap_path else cap_path.split('/')[-1].replace('.py', '')
            unique_paths = len(set(cap_data['paths']))
            print(f"\n  🎯 {cap_name}")
            print(f"     路徑數: {unique_paths}")
            if cap_data['descriptions']:
                desc = list(cap_data['descriptions'])[0]
                if desc:
                    print(f"     說明: {desc}")
            print(f"     位置: ...{cap_path[-60:]}")
    
    # 統計並輸出攻擊能力
    print("\n\n" + "="*70)
    print("⚔️  攻擊執行相關能力")
    print("="*70)
    print(f"\n找到 {len(attack_capabilities)} 個攻擊執行相關能力\n")
    
    attack_by_module = defaultdict(list)
    for cap_path, cap_data in attack_capabilities.items():
        attack_by_module[cap_data['module']].append((cap_path, cap_data))
    
    for module in sorted(attack_by_module.keys()):
        caps = attack_by_module[module]
        print(f"\n📦 {module} ({len(caps)} 個能力)")
        print("-" * 70)
        
        for cap_path, cap_data in sorted(caps, key=lambda x: len(x[1]['paths']), reverse=True):
            cap_name = cap_path.split('\\')[-1].replace('.py', '') if '\\' in cap_path else cap_path.split('/')[-1].replace('.py', '')
            unique_paths = len(set(cap_data['paths']))
            print(f"\n  ⚔️  {cap_name}")
            print(f"     路徑數: {unique_paths}")
            if cap_data['descriptions']:
                desc = list(cap_data['descriptions'])[0]
                if desc:
                    print(f"     說明: {desc}")
            print(f"     位置: ...{cap_path[-60:]}")
    
    # 整體統計
    print("\n\n" + "="*70)
    print("📊 整體統計")
    print("="*70)
    
    total_scan_paths = sum(len(set(cap['paths'])) for cap in scan_capabilities.values())
    total_attack_paths = sum(len(set(cap['paths'])) for cap in attack_capabilities.values())
    
    print(f"\n掃描能力:")
    print(f"  • 獨特終點數: {len(scan_capabilities)}")
    print(f"  • 總路徑數: {total_scan_paths}")
    print(f"  • 平均路徑數: {total_scan_paths/len(scan_capabilities):.2f}" if scan_capabilities else "  • 平均路徑數: 0")
    
    print(f"\n攻擊能力:")
    print(f"  • 獨特終點數: {len(attack_capabilities)}")
    print(f"  • 總路徑數: {total_attack_paths}")
    print(f"  • 平均路徑數: {total_attack_paths/len(attack_capabilities):.2f}" if attack_capabilities else "  • 平均路徑數: 0")
    
    # 找出同時具備掃描和攻擊功能的能力
    common_capabilities = set(scan_capabilities.keys()) & set(attack_capabilities.keys())
    if common_capabilities:
        print(f"\n混合能力 (同時具備掃描和攻擊):")
        print(f"  • 數量: {len(common_capabilities)}")
        for cap in sorted(common_capabilities):
            cap_name = cap.split('\\')[-1].replace('.py', '') if '\\' in cap else cap.split('/')[-1].replace('.py', '')
            print(f"    - {cap_name}")
    
    # 保存報告
    output_file = base_path / "docs" / "ATTACK_SCAN_CAPABILITIES_ANALYSIS.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 攻擊與掃描能力分析\n\n")
        f.write("**分析日期**: 2026-01-01\n\n")
        
        f.write("## 分析目的\n\n")
        f.write("識別 AIVA 系統中可以執行掃描和攻擊操作的能力終點，\n")
        f.write("分析這些能力的分布和訪問路徑。\n\n")
        
        f.write("## 掃描相關能力\n\n")
        f.write(f"共找到 **{len(scan_capabilities)}** 個掃描相關能力\n\n")
        
        for module in sorted(scan_by_module.keys()):
            caps = scan_by_module[module]
            f.write(f"### {module} ({len(caps)} 個)\n\n")
            
            for cap_path, cap_data in sorted(caps, key=lambda x: len(x[1]['paths']), reverse=True):
                cap_name = cap_path.split('\\')[-1].replace('.py', '') if '\\' in cap_path else cap_path.split('/')[-1].replace('.py', '')
                unique_paths = len(set(cap_data['paths']))
                f.write(f"#### {cap_name}\n\n")
                f.write(f"- **路徑數**: {unique_paths}\n")
                if cap_data['descriptions']:
                    desc = list(cap_data['descriptions'])[0]
                    if desc:
                        f.write(f"- **說明**: {desc}\n")
                f.write(f"- **位置**: `{cap_path}`\n\n")
        
        f.write("## 攻擊執行相關能力\n\n")
        f.write(f"共找到 **{len(attack_capabilities)}** 個攻擊執行相關能力\n\n")
        
        for module in sorted(attack_by_module.keys()):
            caps = attack_by_module[module]
            f.write(f"### {module} ({len(caps)} 個)\n\n")
            
            for cap_path, cap_data in sorted(caps, key=lambda x: len(x[1]['paths']), reverse=True):
                cap_name = cap_path.split('\\')[-1].replace('.py', '') if '\\' in cap_path else cap_path.split('/')[-1].replace('.py', '')
                unique_paths = len(set(cap_data['paths']))
                f.write(f"#### {cap_name}\n\n")
                f.write(f"- **路徑數**: {unique_paths}\n")
                if cap_data['descriptions']:
                    desc = list(cap_data['descriptions'])[0]
                    if desc:
                        f.write(f"- **說明**: {desc}\n")
                f.write(f"- **位置**: `{cap_path}`\n\n")
        
        f.write("## 統計摘要\n\n")
        f.write("| 類型 | 獨特終點數 | 總路徑數 | 平均路徑數 |\n")
        f.write("|------|-----------|---------|----------|\n")
        f.write(f"| 掃描能力 | {len(scan_capabilities)} | {total_scan_paths} | {total_scan_paths/len(scan_capabilities):.2f} |\n" if scan_capabilities else "| 掃描能力 | 0 | 0 | 0 |\n")
        f.write(f"| 攻擊能力 | {len(attack_capabilities)} | {total_attack_paths} | {total_attack_paths/len(attack_capabilities):.2f} |\n" if attack_capabilities else "| 攻擊能力 | 0 | 0 | 0 |\n")
        
        if common_capabilities:
            f.write(f"\n### 混合能力\n\n")
            f.write(f"以下 {len(common_capabilities)} 個能力同時具備掃描和攻擊功能：\n\n")
            for cap in sorted(common_capabilities):
                cap_name = cap.split('\\')[-1].replace('.py', '') if '\\' in cap else cap.split('/')[-1].replace('.py', '')
                f.write(f"- `{cap_name}`\n")
        
        f.write("\n## 關鍵發現\n\n")
        f.write(f"1. **掃描能力分布**: {len(scan_by_module)} 個模組提供掃描功能\n")
        f.write(f"2. **攻擊能力分布**: {len(attack_by_module)} 個模組提供攻擊功能\n")
        f.write(f"3. **能力覆蓋率**: {(len(scan_capabilities) + len(attack_capabilities))/98*100:.1f}% 的獨特終點與攻擊或掃描相關\n")
        f.write(f"4. **路徑彈性**: 攻擊和掃描能力平均有 {(total_scan_paths + total_attack_paths)/(len(scan_capabilities) + len(attack_capabilities)):.2f} 條訪問路徑\n")
    
    print(f"\n✅ 詳細報告已保存至: {output_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
