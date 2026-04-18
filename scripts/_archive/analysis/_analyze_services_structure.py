#!/usr/bin/env python3


"""分析 services 目錄中的 MD 文件並建議重組方案"""

import os
import json
from pathlib import Path
from typing import Dict, List
import re
project_root = Path(__file__).resolve().parent.parent.parent.parent

def analyze_services():
    """分析 services 目錄"""
    services_dir = (project_root / "services")
    
    # 掃描所有 MD 文件
    md_files = list(services_dir.rglob("*.md"))
    
    # 排除 node_modules
    md_files = [f for f in md_files if 'node_modules' not in str(f)]
    
    print(f"📁 找到 {len(md_files)} 個 MD 文件 (排除 node_modules)\n")
    
    # 分類
    classification = {
        'readme': [],
        'usage_guides': [],
        'api_docs': [],
        'development_standards': [],
        'architecture': []
    }
    
    for md_file in md_files:
        rel_path = md_file.relative_to(services_dir)
        name = md_file.name.lower()
        
        # 讀取文件內容
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read(1000)
                title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
                title = title_match.group(1) if title_match else ''
        except:
            title = ''
            content = ''
        
        info = {
            'path': str(rel_path),
            'full_path': str(md_file),
            'name': md_file.name,
            'title': title,
            'size': md_file.stat().st_size
        }
        
        # 分類
        if 'usage_guide' in name or 'usage' in name:
            classification['usage_guides'].append(info)
        elif 'dependencies_guide' in name:
            classification['usage_guides'].append(info)
        elif 'development_standards' in name or 'standards' in name:
            classification['development_standards'].append(info)
        elif 'readme.md' == name:
            classification['readme'].append(info)
        elif 'docs/' in str(rel_path) and name == 'readme.md':
            classification['api_docs'].append(info)
        else:
            classification['readme'].append(info)
    
    # 顯示分類結果
    print("="*80)
    print("📊 分類結果")
    print("="*80)
    
    for category, files in classification.items():
        if files:
            print(f"\n{category.upper()} ({len(files)} 個):")
            for f in sorted(files, key=lambda x: x['path']):
                print(f"  • {f['path']}")
                if f['title']:
                    print(f"    標題: {f['title']}")
    
    # 生成移動建議
    print("\n" + "="*80)
    print("🔄 重組建議")
    print("="*80)
    
    moves = []
    
    # USAGE_GUIDE 應該移到 docs/guides/services/
    for f in classification['usage_guides']:
        if 'USAGE_GUIDE' in f['name'] or 'DEPENDENCIES_GUIDE' in f['name']:
            parent = Path(f['path']).parent.name
            new_path = f"docs/guides/services/{parent}_{f['name']}"
            moves.append({
                'from': f'services/{f["path"]}',
                'to': new_path,
                'reason': '服務使用指南應統一放在 docs/guides/services/'
            })
    
    # DEVELOPMENT_STANDARDS 應該移到 docs/development/
    for f in classification['development_standards']:
        new_path = f"docs/development/services_{f['name']}"
        moves.append({
            'from': f'services/{f["path"]}',
            'to': new_path,
            'reason': '開發標準文檔應放在 docs/development/'
        })
    
    if moves:
        print("\n建議移動的文件:\n")
        for m in moves:
            print(f"  FROM: {m['from']}")
            print(f"    TO: {m['to']}")
            print(f"    原因: {m['reason']}")
            print()
    else:
        print("\n✅ 沒有需要移動的文件，所有文件都在正確位置！")
    
    # 生成報告
    report = []
    report.append("# 📊 Services 目錄 MD 文件分析報告\n\n")
    report.append("---\n")
    report.append(f"**分析時間**: 2025年11月27日\n")
    report.append(f"**總文件數**: {len(md_files)} (排除 node_modules)\n\n")
    
    report.append("## 📑 目錄\n\n")
    report.append("- [文件分類](#文件分類)\n")
    report.append("- [重組建議](#重組建議)\n")
    report.append("- [執行計劃](#執行計劃)\n")
    report.append("- [內部連結檢查](#內部連結檢查)\n\n")
    
    report.append("---\n\n")
    report.append("## 📂 文件分類\n\n")
    
    for category, files in classification.items():
        if files:
            report.append(f"### {category.replace('_', ' ').upper()} ({len(files)} 個)\n\n")
            for f in sorted(files, key=lambda x: x['path']):
                report.append(f"- `{f['path']}`\n")
                if f['title']:
                    report.append(f"  - 標題: *{f['title']}*\n")
                report.append(f"  - 大小: {f['size']:,} bytes\n")
            report.append("\n")
    
    report.append("## 🔄 重組建議\n\n")
    
    if moves:
        report.append("根據 AIVA 專案的文件分類標準，建議進行以下調整:\n\n")
        
        # 按類型分組
        by_target = {}
        for m in moves:
            target_dir = m['to'].split('/')[0] + '/' + m['to'].split('/')[1]
            if target_dir not in by_target:
                by_target[target_dir] = []
            by_target[target_dir].append(m)
        
        for target_dir, move_list in by_target.items():
            report.append(f"### 移動到 `{target_dir}/`\n\n")
            for m in move_list:
                report.append(f"**{m['from']}**\n")
                report.append(f"  - → `{m['to']}`\n")
                report.append(f"  - 原因: {m['reason']}\n\n")
    else:
        report.append("✅ **所有文件都在正確位置，無需重組！**\n\n")
        report.append("Services 目錄中的 MD 文件組織良好:\n\n")
        report.append("- README.md 文件保留在各服務模組內作為入口\n")
        report.append("- 文檔結構清晰，符合模組化設計原則\n")
        report.append("- 無需進行目錄重組\n\n")
    
    report.append("## 📋 執行計劃\n\n")
    
    if moves:
        report.append("### 1. 創建目標目錄\n\n")
        report.append("```powershell\n")
        target_dirs = set()
        for m in moves:
            target_dir = str(Path(m['to']).parent)
            target_dirs.add(target_dir)
        for target_dir in sorted(target_dirs):
            report.append(f'New-Item -ItemType Directory -Force -Path "{project_root}/{target_dir}"\n')
        report.append("```\n\n")
        
        report.append("### 2. 移動文件\n\n")
        report.append("```powershell\n")
        for m in moves:
            report.append(f'Move-Item -Path "{project_root}/{m["from"]}" -Destination "{project_root}/{m["to"]}" -Force\n')
        report.append("```\n\n")
    else:
        report.append("✅ 無需執行移動操作\n\n")
    
    report.append("## 🔗 內部連結檢查\n\n")
    
    if moves:
        report.append("移動文件後需要檢查和修正以下位置的連結:\n\n")
        report.append("1. **主 README**: `README.md` 中引用 services 文檔的連結\n")
        report.append("2. **服務 README**: `services/*/README.md` 中的相對連結\n")
        report.append("3. **文檔索引**: `docs/` 和 `guides/` 中的索引文件\n")
        report.append("4. **報告文件**: `reports/` 中引用服務文檔的連結\n\n")
        
        report.append("### 連結搜索命令\n\n")
        report.append("```powershell\n")
        report.append("# 搜索所有引用 services/ 的連結\n")
        report.append('Get-ChildItem -Path . -Filter "*.md" -Recurse | Select-String -Pattern "\\]\\(.*services/" | Select-Object Path, LineNumber, Line\n')
        report.append("```\n\n")
    else:
        report.append("✅ 由於沒有文件需要移動，無需檢查內部連結\n\n")
    
    report.append("---\n\n")
    report.append("## 📊 統計摘要\n\n")
    report.append(f"- **總 MD 文件數**: {len(md_files)}\n")
    report.append(f"- **README 文件**: {len(classification['readme'])}\n")
    report.append(f"- **使用指南**: {len(classification['usage_guides'])}\n")
    report.append(f"- **開發標準**: {len(classification['development_standards'])}\n")
    report.append(f"- **需要移動**: {len(moves)}\n")
    report.append(f"- **保持原位**: {len(md_files) - len(moves)}\n\n")
    
    report.append("---\n\n")
    report.append("*報告生成時間: 2025年11月27日*\n")
    
    # 保存報告
    report_file = (project_root / "SERVICES_MD_REORGANIZATION_PLAN.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print(f"\n✅ 報告已保存: {report_file}")
    
    # 保存 JSON
    json_data = {
        'total_files': len(md_files),
        'classification': {k: [f['path'] for f in v] for k, v in classification.items()},
        'moves': moves
    }
    
    json_file = (project_root / "_services_reorganization.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 數據已保存: {json_file}")
    
    return moves

if __name__ == "__main__":
    moves = analyze_services()
    print("\n" + "="*80)
    print("✨ 分析完成！")
    print("="*80)
