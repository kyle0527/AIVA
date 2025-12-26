#!/usr/bin/env python3
"""分析所有 MD 檔案內容並分類"""
import re
from pathlib import Path
from collections import defaultdict
import json

def analyze_content(file_path, content):
    """分析檔案內容並判斷性質"""
    rel_path = str(file_path.relative_to('.')).replace('\\', '/')
    
    # 基本資訊
    info = {
        'path': rel_path,
        'name': file_path.name,
        'dir': str(file_path.parent.relative_to('.')).replace('\\', '/'),
        'size': len(content),
        'lines': content.count('\n')
    }
    
    # 內容分析
    content_lower = content.lower()
    
    # 檢查關鍵字判斷文件類型
    if 'readme' in file_path.name.lower():
        info['type'] = 'readme'
    elif any(kw in content_lower for kw in ['報告', 'report', '分析', 'analysis', '完成']):
        if any(kw in content_lower for kw in ['architecture', '架構', 'design', '設計']):
            info['type'] = 'architecture_report'
        elif any(kw in content_lower for kw in ['implementation', '實作', '實現', 'completed']):
            info['type'] = 'implementation_report'
        elif any(kw in content_lower for kw in ['test', '測試', 'validation', '驗證']):
            info['type'] = 'testing_report'
        elif any(kw in content_lower for kw in ['fix', '修復', 'bug', 'issue']):
            info['type'] = 'bugfix_report'
        else:
            info['type'] = 'general_report'
    elif any(kw in content_lower for kw in ['guide', '指南', 'manual', '手冊', 'tutorial']):
        if any(kw in content_lower for kw in ['user', '使用者', '用戶']):
            info['type'] = 'user_guide'
        elif any(kw in content_lower for kw in ['development', '開發', 'developer']):
            info['type'] = 'dev_guide'
        else:
            info['type'] = 'general_guide'
    elif any(kw in content_lower for kw in ['api', 'interface', '接口']):
        info['type'] = 'api_doc'
    elif any(kw in content_lower for kw in ['changelog', 'history', '更新']):
        info['type'] = 'changelog'
    elif any(kw in content_lower for kw in ['plan', '計劃', '規劃', 'roadmap']):
        info['type'] = 'plan'
    elif any(kw in content_lower for kw in ['standard', '標準', 'specification', '規範']):
        info['type'] = 'standard'
    else:
        info['type'] = 'other'
    
    # 提取標題作為摘要
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if h1_match:
        info['title'] = h1_match.group(1).strip()
    else:
        info['title'] = file_path.stem
    
    return info

def main():
    # 掃描所有主要目錄的 MD 檔案
    dirs_to_scan = ['reports', 'docs', 'guides', 'services', 'scripts', 'tests']
    all_files = []
    
    for dir_name in dirs_to_scan:
        dir_path = Path(dir_name)
        if dir_path.exists():
            all_files.extend(dir_path.rglob('*.md'))
    
    print(f"找到 {len(all_files)} 個 MD 檔案\n")
    
    # 分析所有檔案
    analysis_results = []
    type_stats = defaultdict(list)
    dir_stats = defaultdict(list)
    
    for i, file_path in enumerate(all_files, 1):
        try:
            # 跳過 node_modules
            if 'node_modules' in str(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content) < 100:  # 跳過太小的檔案
                continue
            
            info = analyze_content(file_path, content)
            analysis_results.append(info)
            type_stats[info['type']].append(info)
            
            # 取得第一層目錄
            parts = info['dir'].split('/')
            first_dir = parts[0] if parts else 'root'
            dir_stats[first_dir].append(info)
            
        except Exception as e:
            print(f"錯誤: {file_path}: {e}")
    
    print(f"成功分析 {len(analysis_results)} 個檔案\n")
    
    # 輸出統計
    print("=" * 80)
    print("檔案類型分佈")
    print("=" * 80)
    
    type_names = {
        'readme': 'README 檔案',
        'architecture_report': '架構報告',
        'implementation_report': '實作報告',
        'testing_report': '測試報告',
        'bugfix_report': '修復報告',
        'general_report': '一般報告',
        'user_guide': '使用者指南',
        'dev_guide': '開發指南',
        'general_guide': '一般指南',
        'api_doc': 'API 文件',
        'changelog': '變更記錄',
        'plan': '計劃文件',
        'standard': '標準規範',
        'other': '其他'
    }
    
    for type_key in sorted(type_stats.keys(), key=lambda x: len(type_stats[x]), reverse=True):
        count = len(type_stats[type_key])
        name = type_names.get(type_key, type_key)
        print(f"  {name:20s}: {count:3d} 個")
        
        # 顯示範例
        for item in type_stats[type_key][:2]:
            print(f"    - {item['name']} ({item['dir']})")
    
    print()
    print("=" * 80)
    print("目錄分佈")
    print("=" * 80)
    
    for dir_name in sorted(dir_stats.keys(), key=lambda x: len(dir_stats[x]), reverse=True):
        count = len(dir_stats[dir_name])
        print(f"  {dir_name:15s}: {count:3d} 個")
    
    # 儲存分析結果
    output_file = '_md_files_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n詳細分析結果已儲存至: {output_file}")
    
    # 建議移動方案
    print()
    print("=" * 80)
    print("建議整理方案")
    print("=" * 80)
    print()
    
    # 檢查錯位的檔案
    misplaced = []
    
    for item in analysis_results:
        current_dir = item['dir'].split('/')[0]
        
        # 報告應該在 reports/
        if item['type'].endswith('_report') and current_dir != 'reports':
            misplaced.append({
                'file': item['path'],
                'current': current_dir,
                'suggested': f"reports/{item['type'].replace('_report', '')}",
                'reason': f"{type_names[item['type']]}應該在 reports/ 目錄"
            })
        
        # 指南應該在 guides/
        elif item['type'].endswith('_guide') and current_dir != 'guides':
            misplaced.append({
                'file': item['path'],
                'current': current_dir,
                'suggested': f"guides/{item['type'].replace('_guide', '')}",
                'reason': f"{type_names[item['type']]}應該在 guides/ 目錄"
            })
        
        # README 可以留在原處
        elif item['type'] == 'readme':
            pass
        
        # 文件應該在 docs/
        elif item['type'] in ['api_doc', 'standard', 'changelog'] and current_dir != 'docs':
            misplaced.append({
                'file': item['path'],
                'current': current_dir,
                'suggested': f"docs/{item['type']}",
                'reason': f"{type_names[item['type']]}應該在 docs/ 目錄"
            })
    
    if misplaced:
        print(f"發現 {len(misplaced)} 個可能需要移動的檔案:")
        print()
        
        # 按建議目錄分組
        by_suggestion = defaultdict(list)
        for item in misplaced:
            by_suggestion[item['suggested']].append(item)
        
        for suggested_dir in sorted(by_suggestion.keys()):
            files = by_suggestion[suggested_dir]
            print(f"  建議移至 {suggested_dir}/ ({len(files)} 個):")
            for item in files[:5]:
                print(f"    - {item['file']}")
            if len(files) > 5:
                print(f"    ... 還有 {len(files) - 5} 個")
            print()
    else:
        print("✅ 所有檔案位置合理，無需移動")
    
    print()

if __name__ == '__main__':
    main()
