#!/usr/bin/env python3
"""根據分析結果生成檔案移動計劃並執行"""
import json
import shutil
from pathlib import Path
from collections import defaultdict

def load_analysis():
    """載入分析結果"""
    with open('_md_files_analysis.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_move_plan(analysis_results):
    """生成移動計劃"""
    move_plan = []
    
    for item in analysis_results:
        current_path = Path(item['path'])
        current_dir = item['dir'].split('/')[0]
        
        # 判斷是否需要移動
        target_dir = None
        
        # 報告類
        if item['type'] == 'architecture_report' and current_dir != 'reports':
            target_dir = 'reports/architecture'
        elif item['type'] == 'implementation_report' and current_dir != 'reports':
            target_dir = 'reports/implementation'
        elif item['type'] == 'testing_report' and current_dir != 'reports':
            target_dir = 'reports/testing'
        elif item['type'] == 'bugfix_report' and current_dir != 'reports':
            target_dir = 'reports/fixes'
        elif item['type'] == 'general_report' and current_dir != 'reports':
            target_dir = 'reports/analysis'
        
        # 指南類
        elif item['type'] == 'user_guide' and current_dir != 'guides':
            target_dir = 'guides/user'
        elif item['type'] == 'dev_guide' and current_dir != 'guides':
            target_dir = 'guides/development'
        elif item['type'] == 'general_guide' and current_dir != 'guides':
            target_dir = 'guides/general'
        
        # 文件類
        elif item['type'] == 'api_doc' and current_dir != 'docs':
            target_dir = 'docs/api'
        elif item['type'] == 'standard' and current_dir != 'docs':
            target_dir = 'docs/standards'
        elif item['type'] == 'plan' and current_dir != 'docs':
            target_dir = 'docs/planning'
        elif item['type'] == 'changelog' and current_dir != 'docs':
            target_dir = 'docs/changelog'
        
        if target_dir:
            target_path = Path(target_dir) / current_path.name
            move_plan.append({
                'from': str(current_path),
                'to': str(target_path),
                'type': item['type'],
                'title': item['title']
            })
    
    return move_plan

def execute_move_plan(move_plan, dry_run=True):
    """執行移動計劃"""
    print(f"{'=' * 80}")
    print(f"檔案移動計劃 ({'預覽模式' if dry_run else '執行模式'})")
    print(f"{'=' * 80}\n")
    
    # 按目標目錄分組
    by_target = defaultdict(list)
    for item in move_plan:
        target_dir = str(Path(item['to']).parent)
        by_target[target_dir].append(item)
    
    total_files = len(move_plan)
    print(f"共需移動 {total_files} 個檔案\n")
    
    for target_dir in sorted(by_target.keys()):
        items = by_target[target_dir]
        print(f"📁 {target_dir}/ ({len(items)} 個檔案)")
        
        if not dry_run:
            # 建立目標目錄
            Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        for item in items:
            from_path = Path(item['from'])
            to_path = Path(item['to'])
            
            # 顯示前5個
            if success_count < 5:
                print(f"  {from_path.name}")
                print(f"    {item['from']} →")
                print(f"    {item['to']}")
            
            if not dry_run:
                try:
                    # 檢查目標是否已存在
                    if to_path.exists():
                        print(f"    ⚠️  目標已存在，跳過")
                        continue
                    
                    # 移動檔案
                    shutil.move(str(from_path), str(to_path))
                    success_count += 1
                    
                except Exception as e:
                    print(f"    ❌ 錯誤: {e}")
            else:
                success_count += 1
        
        if len(items) > 5:
            print(f"  ... 還有 {len(items) - 5} 個")
        
        if not dry_run:
            print(f"  ✅ 成功移動 {success_count} 個檔案")
        print()
    
    if dry_run:
        print(f"{'=' * 80}")
        print(f"這是預覽模式，未實際移動檔案")
        print(f"要執行實際移動，請使用參數: --execute")
        print(f"{'=' * 80}\n")
    else:
        print(f"{'=' * 80}")
        print(f"✅ 檔案移動完成！")
        print(f"{'=' * 80}\n")

def main():
    import sys
    
    # 載入分析結果
    print("載入分析結果...")
    analysis = load_analysis()
    print(f"已載入 {len(analysis)} 個檔案的分析資料\n")
    
    # 生成移動計劃
    print("生成移動計劃...")
    move_plan = generate_move_plan(analysis)
    print(f"生成 {len(move_plan)} 個移動操作\n")
    
    # 儲存移動計劃
    with open('_move_plan.json', 'w', encoding='utf-8') as f:
        json.dump(move_plan, f, ensure_ascii=False, indent=2)
    print("移動計劃已儲存至: _move_plan.json\n")
    
    # 執行移動（預覽或實際執行）
    dry_run = '--execute' not in sys.argv
    execute_move_plan(move_plan, dry_run=dry_run)

if __name__ == '__main__':
    main()
