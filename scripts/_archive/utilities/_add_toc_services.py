#!/usr/bin/env python3
"""為 services/ 目錄的所有 MD 檔案生成目錄"""
import re
from pathlib import Path

def github_anchor(text):
    """GitHub 標準 anchor 生成"""
    text = re.sub(r'[*_`]', '', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    text = text.lower().replace(' ', '-')
    text = re.sub(r'-+', '-', text).strip('-')
    return text

def extract_headings(content):
    """提取所有二級以上標題"""
    headings = []
    seen_titles = {}
    
    for match in re.finditer(r'^(#{2,6})\s+(.+)$', content, re.MULTILINE):
        level = len(match.group(1))
        title = match.group(2).strip()
        
        if re.match(r'^[📋📑]?\s*目錄\s*$', title):
            continue
        
        anchor = github_anchor(title)
        
        if anchor in seen_titles:
            seen_titles[anchor] += 1
            anchor = f"{anchor}-{seen_titles[anchor]}"
        else:
            seen_titles[anchor] = 0
        
        headings.append({
            'level': level,
            'title': title,
            'anchor': anchor
        })
    
    return headings

def generate_toc(headings):
    """生成目錄內容"""
    if not headings:
        return None
    
    toc_lines = []
    for h in headings:
        indent = '  ' * (h['level'] - 2)
        toc_lines.append(f"{indent}- [{h['title']}](#{h['anchor']})")
    
    return '\n'.join(toc_lines)

def add_toc_to_file(file_path):
    """為單個檔案添加目錄"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 檢查是否已有目錄
        if re.search(r'## [📑📋] 目錄', content):
            return False, "已有目錄"
        
        # 提取標題
        headings = extract_headings(content)
        if len(headings) < 2:
            return False, "標題太少"
        
        # 生成目錄
        toc_content = generate_toc(headings)
        if not toc_content:
            return False, "無法生成目錄"
        
        # 找到第一個 H1 標題後插入，如果沒有則在開頭插入
        h1_match = re.search(r'^#\s+.+$', content, re.MULTILINE)
        
        if h1_match:
            # 有 H1，在 H1 後插入
            insert_pos = h1_match.end()
            toc_block = f"\n\n## 📑 目錄\n\n{toc_content}\n\n---\n"
            new_content = content[:insert_pos] + toc_block + content[insert_pos:]
        else:
            # 無 H1，在檔案開頭插入（包含 H1 標題）
            file_name = file_path.stem.replace('_', ' ').replace('-', ' ').title()
            toc_block = f"# {file_name}\n\n## 📑 目錄\n\n{toc_content}\n\n---\n\n"
            new_content = toc_block + content
        
        # 寫回檔案
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True, "成功"
        
    except Exception as e:
        return False, f"錯誤: {str(e)}"

def main():
    services_path = Path('services')
    md_files = list(services_path.rglob('*.md'))
    
    print(f"找到 {len(md_files)} 個 MD 檔案")
    print()
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, file_path in enumerate(md_files, 1):
        rel_path = file_path.relative_to('.')
        success, msg = add_toc_to_file(file_path)
        
        if success:
            success_count += 1
            if success_count <= 10:
                print(f"✅ [{i}/{len(md_files)}] {rel_path.name}")
        elif "已有目錄" in msg or "標題太少" in msg:
            skip_count += 1
        else:
            fail_count += 1
            if fail_count <= 5:
                print(f"❌ [{i}/{len(md_files)}] {rel_path.name}: {msg}")
        
        if i % 50 == 0:
            print(f"進度: {i}/{len(md_files)} ({success_count} 成功, {skip_count} 略過, {fail_count} 失敗)")
    
    print()
    print(f"{'='*60}")
    print(f"處理完成!")
    print(f"  成功: {success_count} 個")
    print(f"  略過: {skip_count} 個 (已有目錄或內容太少)")
    print(f"  失敗: {fail_count} 個")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
