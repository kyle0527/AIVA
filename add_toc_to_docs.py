#!/usr/bin/env python3
"""
为 docs 目录中的 markdown 文件自动添加目录
"""
import os
import re
from pathlib import Path

def extract_headers(content):
    """提取 markdown 文件中的标题"""
    headers = []
    lines = content.split('\n')
    
    for line in lines:
        # 匹配 # 开头的标题
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            # 生成锚点链接
            anchor = title.lower()
            anchor = re.sub(r'[^\w\s\u4e00-\u9fff-]', '', anchor)  # 保留中文、英文、数字
            anchor = re.sub(r'\s+', '-', anchor)
            headers.append((level, title, anchor))
    
    return headers

def generate_toc(headers, max_level=4):
    """生成目录"""
    if not headers:
        return ""
    
    toc_lines = ["## 📑 目錄\n"]
    
    for level, title, anchor in headers:
        if level <= max_level and level > 1:  # 跳过一级标题
            indent = "  " * (level - 2)
            toc_lines.append(f"{indent}- [{title}](#{anchor})")
    
    toc_lines.append("")  # 空行
    return "\n".join(toc_lines)

def has_toc(content):
    """检查文件是否已有目录"""
    return bool(re.search(r'##\s*📑?\s*目[錄录]', content, re.IGNORECASE))

def add_toc_to_file(filepath):
    """为单个文件添加目录"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 如果已有目录，跳过
        if has_toc(content):
            return False, "已有目錄"
        
        # 提取标题
        headers = extract_headers(content)
        
        if len(headers) <= 2:  # 如果标题太少，不添加目录
            return False, "標題太少"
        
        # 生成目录
        toc = generate_toc(headers)
        
        if not toc:
            return False, "無法生成目錄"
        
        # 找到第一个标题后插入目录
        lines = content.split('\n')
        first_header_idx = -1
        
        for i, line in enumerate(lines):
            if re.match(r'^#\s+', line):
                first_header_idx = i
                break
        
        if first_header_idx == -1:
            return False, "未找到標題"
        
        # 在第一个标题后插入目录
        lines.insert(first_header_idx + 1, '\n' + toc + '\n---\n')
        new_content = '\n'.join(lines)
        
        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return True, f"成功添加 {len(headers)} 個標題"
    
    except Exception as e:
        return False, f"錯誤: {str(e)}"

def main():
    """主函数"""
    docs_dir = Path(r"C:\D\fold7\AIVA-git\docs")
    
    if not docs_dir.exists():
        print(f"❌ 目錄不存在: {docs_dir}")
        return
    
    md_files = list(docs_dir.glob("*.md"))
    
    print(f"📁 找到 {len(md_files)} 個 Markdown 文件\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for filepath in sorted(md_files):
        filename = filepath.name
        success, message = add_toc_to_file(filepath)
        
        if success:
            print(f"✅ {filename:50s} - {message}")
            success_count += 1
        elif "已有目錄" in message:
            print(f"⏭️  {filename:50s} - {message}")
            skip_count += 1
        else:
            print(f"⚠️  {filename:50s} - {message}")
            error_count += 1
    
    print(f"\n{'='*70}")
    print(f"📊 統計:")
    print(f"  ✅ 成功添加目錄: {success_count}")
    print(f"  ⏭️  跳過: {skip_count}")
    print(f"  ⚠️  錯誤/無需添加: {error_count}")
    print(f"  📄 總計: {len(md_files)}")

if __name__ == "__main__":
    main()
