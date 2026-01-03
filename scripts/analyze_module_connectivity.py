#!/usr/bin/env python3
"""
分析模組間連結完整性
檢查跨模組調用、孤立模組、連接斷裂等問題
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

def get_module(path: str) -> str:
    """從路徑提取模組名稱"""
    if 'services/core/aiva_core/' in path or 'services\\core\\aiva_core\\' in path:
        parts = path.replace('\\', '/').split('services/core/aiva_core/')[1].split('/')
        return parts[0] if parts else 'unknown'
    return 'unknown'

def main():
    base_path = Path(r"C:\D\fold7\AIVA-git")
    classification_file = base_path / "services" / "integration" / "data" / "internal_exploration" / "latest_classification.json"
    
    print("\n" + "="*70)
    print("🔗 模組間連結完整性分析")
    print("="*70)
    
    with open(classification_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    flows = data.get('flows', [])
    
    # 定義六大模組
    six_modules = [
        'cognitive_core',
        'internal_exploration',
        'task_planning',
        'external_learning',
        'core_capabilities',
        'service_backbone'
    ]
    
    # 收集數據
    module_internal_flows = defaultdict(int)  # 模組內部流
    module_cross_flows = defaultdict(lambda: defaultdict(int))  # 跨模組流
    module_endpoints = defaultdict(set)  # 每個模組的終點
    module_startpoints = defaultdict(set)  # 每個模組的起點
    
    cross_module_flows = []  # 所有跨模組的流
    
    for flow in flows:
        full_path = flow.get('full_path', [])
        if len(full_path) < 2:
            continue
        
        start_path = full_path[0]
        end_path = full_path[-1]
        
        start_module = get_module(start_path)
        end_module = get_module(end_path)
        
        # 記錄起點和終點
        module_startpoints[start_module].add(start_path)
        module_endpoints[end_module].add(end_path)
        
        if start_module == end_module:
            # 模組內部流
            module_internal_flows[start_module] += 1
        else:
            # 跨模組流
            module_cross_flows[start_module][end_module] += 1
            cross_module_flows.append({
                'from': start_module,
                'to': end_module,
                'start': start_path,
                'end': end_path,
                'path': full_path
            })
    
    # 分析結果
    print("\n📊 模組流量統計:\n")
    print(f"{'模組':<25} {'內部流':<10} {'跨模組流(出)':<15} {'跨模組流(入)':<15}")
    print("-" * 70)
    
    incoming_flows = defaultdict(int)
    for from_module, targets in module_cross_flows.items():
        for to_module, count in targets.items():
            incoming_flows[to_module] += count
    
    for module in six_modules:
        internal = module_internal_flows.get(module, 0)
        outgoing = sum(module_cross_flows.get(module, {}).values())
        incoming = incoming_flows.get(module, 0)
        
        print(f"{module:<25} {internal:<10} {outgoing:<15} {incoming:<15}")
    
    # 分析跨模組連接矩陣
    print("\n\n🔗 跨模組連接矩陣:")
    print("   (行 → 列，數字表示 flow 數量)\n")
    
    # 打印表頭
    print(f"{'從 \\ 到':<20}", end='')
    for module in six_modules:
        print(f"{module[:12]:<13}", end='')
    print()
    print("-" * (20 + 13 * len(six_modules)))
    
    # 打印矩陣
    for from_module in six_modules:
        print(f"{from_module:<20}", end='')
        for to_module in six_modules:
            if from_module == to_module:
                count = module_internal_flows.get(from_module, 0)
                print(f"[{count}]".center(13), end='')
            else:
                count = module_cross_flows.get(from_module, {}).get(to_module, 0)
                if count > 0:
                    print(f"{count}".center(13), end='')
                else:
                    print("-".center(13), end='')
        print()
    
    # 檢測問題
    print("\n\n🔍 連接問題檢測:\n")
    
    issues = []
    
    # 1. 檢測孤立模組（沒有任何跨模組連接）
    isolated_modules = []
    for module in six_modules:
        outgoing = sum(module_cross_flows.get(module, {}).values())
        incoming = incoming_flows.get(module, 0)
        if outgoing == 0 and incoming == 0:
            isolated_modules.append(module)
    
    if isolated_modules:
        issues.append(f"❌ 孤立模組: {', '.join(isolated_modules)}")
        print(f"❌ 孤立模組 ({len(isolated_modules)} 個):")
        for module in isolated_modules:
            print(f"   • {module} - 沒有任何跨模組連接")
    else:
        print(f"✅ 無孤立模組 - 所有模組都有跨模組連接")
    
    # 2. 檢測單向連接模組（只有入或只有出）
    print("\n")
    oneway_modules = []
    for module in six_modules:
        outgoing = sum(module_cross_flows.get(module, {}).values())
        incoming = incoming_flows.get(module, 0)
        if (outgoing > 0 and incoming == 0) or (outgoing == 0 and incoming > 0):
            oneway_modules.append((module, 'out' if outgoing > 0 else 'in'))
    
    if oneway_modules:
        issues.append(f"⚠️  單向連接模組: {len(oneway_modules)} 個")
        print(f"⚠️  單向連接模組 ({len(oneway_modules)} 個):")
        for module, direction in oneway_modules:
            dir_text = "只能調用其他模組" if direction == 'out' else "只能被其他模組調用"
            print(f"   • {module} - {dir_text}")
    else:
        print(f"✅ 無單向連接模組 - 所有模組都有雙向連接能力")
    
    # 3. 檢測連接稀疏的模組對（核心模組間應該有連接）
    print("\n")
    core_modules = ['cognitive_core', 'task_planning', 'core_capabilities']
    missing_connections = []
    
    for i, mod1 in enumerate(core_modules):
        for mod2 in core_modules[i+1:]:
            conn1 = module_cross_flows.get(mod1, {}).get(mod2, 0)
            conn2 = module_cross_flows.get(mod2, {}).get(mod1, 0)
            if conn1 == 0 and conn2 == 0:
                missing_connections.append((mod1, mod2))
    
    if missing_connections:
        issues.append(f"⚠️  核心模組間缺少連接: {len(missing_connections)} 對")
        print(f"⚠️  核心模組間缺少連接 ({len(missing_connections)} 對):")
        for mod1, mod2 in missing_connections:
            print(f"   • {mod1} ↔ {mod2} - 無直接連接")
    else:
        print(f"✅ 核心模組間連接完整")
    
    # 4. 檢查關鍵功能的跨模組可達性
    print("\n")
    print("🎯 關鍵功能跨模組可達性:")
    
    # 檢查攻擊能力
    attack_keywords = ['attack', 'exploit', 'executor', 'payload']
    scan_keywords = ['scan', 'scanner', 'reconnaissance']
    
    attack_endpoints = []
    scan_endpoints = []
    
    for flow in cross_module_flows:
        end_name = flow['end'].lower()
        if any(kw in end_name for kw in attack_keywords):
            attack_endpoints.append(flow)
        if any(kw in end_name for kw in scan_keywords):
            scan_endpoints.append(flow)
    
    print(f"\n   攻擊能力跨模組調用:")
    if attack_endpoints:
        attack_from_modules = set(f['from'] for f in attack_endpoints)
        attack_to_modules = set(f['to'] for f in attack_endpoints)
        print(f"   ✅ 可從 {len(attack_from_modules)} 個模組發起")
        print(f"      發起模組: {', '.join(sorted(attack_from_modules))}")
        print(f"   ✅ 可到達 {len(attack_to_modules)} 個模組")
        print(f"      目標模組: {', '.join(sorted(attack_to_modules))}")
    else:
        issues.append("❌ 攻擊能力無跨模組調用")
        print(f"   ❌ 無跨模組攻擊能力調用")
    
    print(f"\n   掃描能力跨模組調用:")
    if scan_endpoints:
        scan_from_modules = set(f['from'] for f in scan_endpoints)
        scan_to_modules = set(f['to'] for f in scan_endpoints)
        print(f"   ✅ 可從 {len(scan_from_modules)} 個模組發起")
        print(f"      發起模組: {', '.join(sorted(scan_from_modules))}")
        print(f"   ✅ 可到達 {len(scan_to_modules)} 個模組")
        print(f"      目標模組: {', '.join(sorted(scan_to_modules))}")
    else:
        issues.append("❌ 掃描能力無跨模組調用")
        print(f"   ❌ 無跨模組掃描能力調用")
    
    # 5. 計算模組間連接密度
    print("\n\n📈 模組連接密度分析:")
    
    total_possible_connections = len(six_modules) * (len(six_modules) - 1)  # 不包括自己
    actual_connections = sum(
        1 for from_m in six_modules 
        for to_m in six_modules 
        if from_m != to_m and module_cross_flows.get(from_m, {}).get(to_m, 0) > 0
    )
    
    density = actual_connections / total_possible_connections * 100
    
    print(f"\n   可能的模組對連接數: {total_possible_connections}")
    print(f"   實際存在的連接數: {actual_connections}")
    print(f"   連接密度: {density:.1f}%")
    
    if density < 30:
        issues.append(f"⚠️  連接密度過低: {density:.1f}%")
        print(f"\n   ⚠️  連接密度低於 30%，模組間耦合較松散")
    elif density > 70:
        print(f"\n   ⚠️  連接密度高於 70%，模組間耦合較緊密")
    else:
        print(f"\n   ✅ 連接密度適中，模組間耦合度合理")
    
    # 6. 找出連接最多和最少的模組
    print("\n\n🔝 連接度排名:")
    
    module_total_connections = {}
    for module in six_modules:
        outgoing = sum(module_cross_flows.get(module, {}).values())
        incoming = incoming_flows.get(module, 0)
        module_total_connections[module] = outgoing + incoming
    
    sorted_modules = sorted(module_total_connections.items(), key=lambda x: x[1], reverse=True)
    
    print("\n   跨模組連接數 (出+入):")
    for i, (module, count) in enumerate(sorted_modules, 1):
        outgoing = sum(module_cross_flows.get(module, {}).values())
        incoming = incoming_flows.get(module, 0)
        print(f"   {i}. {module:<25} {count:3} (出: {outgoing:3}, 入: {incoming:3})")
    
    # 總結
    print("\n\n" + "="*70)
    print("📋 總結報告")
    print("="*70)
    
    print(f"\n✅ 檢測完成:")
    print(f"   • 總 Flows: {len(flows)}")
    print(f"   • 模組內部流: {sum(module_internal_flows.values())}")
    print(f"   • 跨模組流: {len(cross_module_flows)}")
    print(f"   • 連接密度: {density:.1f}%")
    
    if issues:
        print(f"\n⚠️  發現 {len(issues)} 個潛在問題:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print(f"\n✅ 未發現連接問題，模組間連結完整")
    
    # 建議
    print("\n💡 建議:")
    
    if isolated_modules:
        print(f"   • 檢查孤立模組是否為設計意圖")
    
    if oneway_modules:
        print(f"   • 考慮為單向模組添加反向連接以提高彈性")
    
    if missing_connections:
        print(f"   • 考慮在核心模組間建立直接連接以提高效率")
    
    if density < 30:
        print(f"   • 考慮增加模組間的連接路徑以提高系統彈性")
    
    if not issues:
        print(f"   • 當前模組間連結設計合理，保持現狀")
    
    # 保存報告
    output_file = base_path / "docs" / "MODULE_CONNECTIVITY_ANALYSIS.md"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 模組間連結完整性分析\n\n")
        f.write("**分析日期**: 2026-01-01\n\n")
        
        f.write("## 分析目的\n\n")
        f.write("檢查六大模組間的連結完整性，識別孤立模組、單向連接、連接斷裂等問題。\n\n")
        
        f.write("## 模組流量統計\n\n")
        f.write("| 模組 | 內部流 | 跨模組流(出) | 跨模組流(入) |\n")
        f.write("|------|--------|------------|------------|\n")
        for module in six_modules:
            internal = module_internal_flows.get(module, 0)
            outgoing = sum(module_cross_flows.get(module, {}).values())
            incoming = incoming_flows.get(module, 0)
            f.write(f"| {module} | {internal} | {outgoing} | {incoming} |\n")
        
        f.write("\n## 跨模組連接矩陣\n\n")
        f.write("行 → 列，數字表示 flow 數量，[數字] 表示內部流\n\n")
        f.write("| 從 \\ 到 | " + " | ".join([m[:12] for m in six_modules]) + " |\n")
        f.write("|" + "---|" * (len(six_modules) + 1) + "\n")
        
        for from_module in six_modules:
            row = [from_module]
            for to_module in six_modules:
                if from_module == to_module:
                    count = module_internal_flows.get(from_module, 0)
                    row.append(f"[{count}]")
                else:
                    count = module_cross_flows.get(from_module, {}).get(to_module, 0)
                    row.append(str(count) if count > 0 else "-")
            f.write("| " + " | ".join(row) + " |\n")
        
        f.write("\n## 連接問題檢測\n\n")
        
        if isolated_modules:
            f.write(f"### ❌ 孤立模組 ({len(isolated_modules)} 個)\n\n")
            for module in isolated_modules:
                f.write(f"- **{module}**: 沒有任何跨模組連接\n")
            f.write("\n")
        else:
            f.write("### ✅ 無孤立模組\n\n所有模組都有跨模組連接\n\n")
        
        if oneway_modules:
            f.write(f"### ⚠️ 單向連接模組 ({len(oneway_modules)} 個)\n\n")
            for module, direction in oneway_modules:
                dir_text = "只能調用其他模組" if direction == 'out' else "只能被其他模組調用"
                f.write(f"- **{module}**: {dir_text}\n")
            f.write("\n")
        else:
            f.write("### ✅ 無單向連接模組\n\n所有模組都有雙向連接能力\n\n")
        
        if missing_connections:
            f.write(f"### ⚠️ 核心模組間缺少連接 ({len(missing_connections)} 對)\n\n")
            for mod1, mod2 in missing_connections:
                f.write(f"- **{mod1}** ↔ **{mod2}**: 無直接連接\n")
            f.write("\n")
        else:
            f.write("### ✅ 核心模組間連接完整\n\n")
        
        f.write("## 關鍵功能跨模組可達性\n\n")
        
        f.write("### 攻擊能力\n\n")
        if attack_endpoints:
            attack_from_modules = set(f['from'] for f in attack_endpoints)
            attack_to_modules = set(f['to'] for f in attack_endpoints)
            f.write(f"- ✅ 可從 {len(attack_from_modules)} 個模組發起: {', '.join(sorted(attack_from_modules))}\n")
            f.write(f"- ✅ 可到達 {len(attack_to_modules)} 個模組: {', '.join(sorted(attack_to_modules))}\n\n")
        else:
            f.write("- ❌ 無跨模組攻擊能力調用\n\n")
        
        f.write("### 掃描能力\n\n")
        if scan_endpoints:
            scan_from_modules = set(f['from'] for f in scan_endpoints)
            scan_to_modules = set(f['to'] for f in scan_endpoints)
            f.write(f"- ✅ 可從 {len(scan_from_modules)} 個模組發起: {', '.join(sorted(scan_from_modules))}\n")
            f.write(f"- ✅ 可到達 {len(scan_to_modules)} 個模組: {', '.join(sorted(scan_to_modules))}\n\n")
        else:
            f.write("- ❌ 無跨模組掃描能力調用\n\n")
        
        f.write("## 模組連接密度\n\n")
        f.write(f"- **可能的模組對連接數**: {total_possible_connections}\n")
        f.write(f"- **實際存在的連接數**: {actual_connections}\n")
        f.write(f"- **連接密度**: {density:.1f}%\n\n")
        
        if density < 30:
            f.write("⚠️ 連接密度低於 30%，模組間耦合較松散\n\n")
        elif density > 70:
            f.write("⚠️ 連接密度高於 70%，模組間耦合較緊密\n\n")
        else:
            f.write("✅ 連接密度適中，模組間耦合度合理\n\n")
        
        f.write("## 連接度排名\n\n")
        f.write("| 排名 | 模組 | 總連接數 | 出站 | 入站 |\n")
        f.write("|------|------|---------|------|------|\n")
        for i, (module, count) in enumerate(sorted_modules, 1):
            outgoing = sum(module_cross_flows.get(module, {}).values())
            incoming = incoming_flows.get(module, 0)
            f.write(f"| {i} | {module} | {count} | {outgoing} | {incoming} |\n")
        
        f.write("\n## 總結\n\n")
        f.write(f"- **總 Flows**: {len(flows)}\n")
        f.write(f"- **模組內部流**: {sum(module_internal_flows.values())} ({sum(module_internal_flows.values())/len(flows)*100:.1f}%)\n")
        f.write(f"- **跨模組流**: {len(cross_module_flows)} ({len(cross_module_flows)/len(flows)*100:.1f}%)\n")
        f.write(f"- **連接密度**: {density:.1f}%\n\n")
        
        if issues:
            f.write(f"### ⚠️ 發現 {len(issues)} 個潛在問題\n\n")
            for issue in issues:
                f.write(f"- {issue}\n")
        else:
            f.write("### ✅ 未發現連接問題\n\n模組間連結完整\n")
        
        f.write("\n## 建議\n\n")
        
        if isolated_modules:
            f.write("- 檢查孤立模組是否為設計意圖\n")
        
        if oneway_modules:
            f.write("- 考慮為單向模組添加反向連接以提高彈性\n")
        
        if missing_connections:
            f.write("- 考慮在核心模組間建立直接連接以提高效率\n")
        
        if density < 30:
            f.write("- 考慮增加模組間的連接路徑以提高系統彈性\n")
        
        if not issues:
            f.write("- 當前模組間連結設計合理，保持現狀\n")
    
    print(f"\n✅ 詳細報告已保存至: {output_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
