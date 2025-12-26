#!/usr/bin/env python3
"""為所有沒有目錄的 MD 文件批量添加目錄"""
import re
from pathlib import Path
from collections import defaultdict

def extract_headers(content):
    """提取文件中的所有標題"""
    lines = content.split('\n')
    headers = []
    
    for i, line in enumerate(lines):
        # 匹配 markdown 標題
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            # 移除標題中的表情符號和特殊字符（保留用於 anchor）
            clean_title = re.sub(r'[^\w\s\u4e00-\u9fff-]', '', title)
            headers.append({
                'level': level,
                'title': title,
                'clean_title': clean_title,
                'line': i
            })
    
    return headers

def generate_toc(headers):
    """生成目錄"""
    if not headers:
        return []
    
    toc_lines = ['## 📑 目錄', '']
    
    # 只包含 H2 和 H3 標題
    for header in headers:
        if header['level'] == 2:
            # 生成 anchor
            anchor = header['clean_title'].lower().replace(' ', '-')
            toc_lines.append(f"- [{header['title']}](#{anchor})")
        elif header['level'] == 3:
            anchor = header['clean_title'].lower().replace(' ', '-')
            toc_lines.append(f"  - [{header['title']}](#{anchor})")
    
    toc_lines.append('')
    toc_lines.append('---')
    toc_lines.append('')
    
    return toc_lines

def has_toc(content):
    """檢查是否已有目錄"""
    patterns = [
        r'##\s*目錄',
        r'##\s*📑\s*目錄',
        r'##\s*Table of Contents',
        r'##\s*Contents',
    ]
    
    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return True
    return False

def add_toc_to_file(file_path):
    """為單個文件添加目錄"""
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # 檢查是否已有目錄
        if has_toc(content):
            return False, "已有目錄"
        
        # 提取標題
        headers = extract_headers(content)
        
        # 過濾出 H2 和 H3
        relevant_headers = [h for h in headers if h['level'] in [2, 3]]
        
        if len(relevant_headers) < 2:
            return False, "標題太少（< 2）"
        
        # 生成目錄
        toc_lines = generate_toc(headers)
        
        # 找到插入位置（第一個 H1 之後，或第一個 H2 之前）
        lines = content.split('\n')
        insert_pos = 0
        
        # 找到第一個 H1
        for i, line in enumerate(lines):
            if line.startswith('# '):
                insert_pos = i + 1
                # 跳過空行
                while insert_pos < len(lines) and not lines[insert_pos].strip():
                    insert_pos += 1
                break
        
        # 如果沒有 H1，找第一個 H2
        if insert_pos == 0:
            for i, line in enumerate(lines):
                if line.startswith('## '):
                    insert_pos = i
                    break
        
        # 插入目錄
        new_lines = lines[:insert_pos] + toc_lines + lines[insert_pos:]
        new_content = '\n'.join(new_lines)
        
        # 寫回文件
        file_path.write_text(new_content, encoding='utf-8')
        
        return True, f"成功（{len(relevant_headers)} 個標題）"
        
    except Exception as e:
        return False, f"錯誤: {str(e)}"

def main():
    print("=== 批量為 MD 文件添加目錄 ===\n")
    
    # 讀取檢查報告找出沒有目錄的文件
    report_path = Path('MD_FILES_COMPLETE_CHECK_REPORT.md')
    
    if not report_path.exists():
        print("❌ 找不到檢查報告！")
        return
    
    # 從報告中提取沒有目錄的文件列表
    content = report_path.read_text(encoding='utf-8')
    
    # 解析文件路徑
    no_toc_files = []
    in_no_toc_section = False
    
    for line in content.split('\n'):
        if '## 沒有目錄的文件' in line:
            in_no_toc_section = True
            continue
        elif in_no_toc_section and line.startswith('## '):
            break
        
        if in_no_toc_section:
            # 匹配文件路徑
            match = re.match(r'\s*-\s*`(.+\.md)`', line)
            if match:
                no_toc_files.append(match.group(1))
    
    print(f"找到 {len(no_toc_files)} 個需要添加目錄的文件\n")
    
    # 按目錄分組
    by_dir = defaultdict(list)
    for file_path in no_toc_files:
        if file_path.startswith('reports'):
            by_dir['reports'].append(file_path)
        elif file_path.startswith('services'):
            by_dir['services'].append(file_path)
        elif file_path.startswith('_out'):
            by_dir['_out'].append(file_path)
        elif file_path.startswith('_archive'):
            by_dir['_archive'].append(file_path)
        else:
            by_dir['root_other'].append(file_path)
    
    # 統計
    total_files = len(no_toc_files)
    processed = 0
    success = 0
    skipped = 0
    failed = 0
    
    # 處理每個文件
    for dir_name, files in sorted(by_dir.items()):
        print(f"\n### {dir_name.upper()} ({len(files)} 個)")
        print("-" * 60)
        
        for file_path_str in sorted(files):
            file_path = Path(file_path_str)
            
            if not file_path.exists():
                print(f"  ⚠️  {file_path_str} - 文件不存在")
                skipped += 1
                processed += 1
                continue
            
            result, message = add_toc_to_file(file_path)
            
            if result:
                print(f"  ✅ {file_path_str} - {message}")
                success += 1
            else:
                if "已有目錄" in message or "標題太少" in message:
                    print(f"  ⏭️  {file_path_str} - {message}")
                    skipped += 1
                else:
                    print(f"  ❌ {file_path_str} - {message}")
                    failed += 1
            
            processed += 1
            
            # 進度提示
            if processed % 20 == 0:
                print(f"\n進度: {processed}/{total_files} ({processed/total_files*100:.1f}%)\n")
    
    # 總結
    print("\n" + "=" * 60)
    print("📊 處理總結")
    print("=" * 60)
    print(f"總文件數: {total_files}")
    print(f"  ✅ 成功添加目錄: {success}")
    print(f"  ⏭️  跳過: {skipped}")
    print(f"  ❌ 失敗: {failed}")
    print(f"  完成率: {success/total_files*100:.1f}%")
    print()

if __name__ == '__main__':
    main()
