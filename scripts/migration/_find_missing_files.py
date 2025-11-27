#!/usr/bin/env python3
"""找出遺漏的文件並重新完整提取"""
import json
from pathlib import Path

def main():
    print("=== 尋找遺漏的文件 ===\n")
    
    # 1. 獲取所有實際文件
    node_modules = Path('services/scan/engines/typescript_engine/node_modules')
    actual_files = {str(f.relative_to(node_modules)): f for f in node_modules.rglob('*.md')}
    print(f"實際文件數: {len(actual_files)}")
    
    # 2. 獲取已提取的文件
    with open('_node_modules_md_content.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    extracted_paths = set()
    for category in ['core', 'dev_tools', 'dependencies', 'licenses', 'changelogs', 'other']:
        for item in data[category]:
            # 從完整路徑提取相對路徑
            rel_path = item['file'].replace('services/scan/engines/typescript_engine/node_modules/', '')
            extracted_paths.add(rel_path)
    
    print(f"已提取文件數: {len(extracted_paths)}")
    print()
    
    # 3. 找出遺漏的
    missing = set(actual_files.keys()) - extracted_paths
    
    if missing:
        print(f"⚠️  遺漏 {len(missing)} 個文件:\n")
        for path in sorted(missing):
            print(f"  - {path}")
            # 檢查文件內容
            file_path = actual_files[path]
            try:
                content = file_path.read_text(encoding='utf-8')
                print(f"    大小: {len(content)} bytes")
            except:
                print(f"    (無法讀取)")
        
        print("\n=== 重新完整提取 ===\n")
        
        # 重新提取所有文件
        all_data = {
            'core': [],
            'dev_tools': [],
            'dependencies': [],
            'licenses': [],
            'changelogs': [],
            'other': []
        }
        
        # 定義核心和開發工具套件
        core_packages = {
            'playwright', 'amqplib', 'pino', 'pino-pretty', 'pino-std-serializers',
            'pino-abstract-transport', 'atomic-sleep', 'on-exit-leak-free',
            'process-warning', 'quick-format-unescaped', 'real-require',
            'secure-json-parse', 'sonic-boom', 'thread-stream'
        }
        
        dev_packages = {
            'typescript', '@types/node', '@typescript-eslint/eslint-plugin',
            '@typescript-eslint/parser', 'eslint', 'eslint-config-prettier',
            'eslint-plugin-prettier', 'prettier', 'tsx', 'vitest', '@vitest/ui',
            'esbuild', 'vite', 'rollup', '@esbuild/win32-x64'
        }
        
        for rel_path, file_path in actual_files.items():
            try:
                content = file_path.read_text(encoding='utf-8')
                size = len(content)
                
                # 提取套件名稱
                parts = rel_path.split('/')
                if parts[0].startswith('@'):
                    package_name = f"{parts[0]}/{parts[1]}"
                else:
                    package_name = parts[0]
                
                # 分類文件
                filename = file_path.name.lower()
                
                item = {
                    'file': str(file_path),
                    'package': package_name,
                    'filename': file_path.name,
                    'size': size,
                    'content': content
                }
                
                # 根據文件名分類
                if 'license' in filename or 'licence' in filename:
                    all_data['licenses'].append(item)
                elif filename in ['history.md', 'changelog.md', 'changes.md']:
                    all_data['changelogs'].append(item)
                elif filename == 'readme.md':
                    # README 根據套件重要性分類
                    if package_name in core_packages:
                        all_data['core'].append(item)
                    elif package_name in dev_packages:
                        all_data['dev_tools'].append(item)
                    else:
                        all_data['dependencies'].append(item)
                else:
                    all_data['other'].append(item)
                    
            except Exception as e:
                print(f"⚠️  無法讀取 {rel_path}: {e}")
        
        # 統計
        total = sum(len(all_data[cat]) for cat in all_data)
        print(f"重新提取完成!")
        print(f"  核心套件: {len(all_data['core'])}")
        print(f"  開發工具: {len(all_data['dev_tools'])}")
        print(f"  傳遞依賴: {len(all_data['dependencies'])}")
        print(f"  授權文件: {len(all_data['licenses'])}")
        print(f"  變更記錄: {len(all_data['changelogs'])}")
        print(f"  其他文檔: {len(all_data['other'])}")
        print(f"  總計: {total}")
        
        if total == len(actual_files):
            print(f"\n✅ 完美！已提取全部 {total} 個文件")
            
            # 儲存完整數據
            with open('_node_modules_md_content_complete.json', 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 完整數據已儲存: _node_modules_md_content_complete.json")
        else:
            print(f"\n⚠️  仍有差異: {len(actual_files)} vs {total}")
    else:
        print("✅ 沒有遺漏文件！")

if __name__ == '__main__':
    main()
