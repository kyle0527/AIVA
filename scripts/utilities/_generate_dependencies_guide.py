#!/usr/bin/env python3
"""生成 TypeScript Engine 依賴套件完整使用指南"""
import json
from pathlib import Path

def load_extracted_data():
    """載入提取的內容"""
    with open('_node_modules_md_content.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_key_info(content):
    """從 README 提取關鍵資訊"""
    lines = content.split('\n')
    
    # 提取標題
    title = None
    for line in lines[:10]:
        if line.startswith('# '):
            title = line[2:].strip()
            break
    
    # 提取簡介（第一段文字）
    description = []
    in_description = False
    for line in lines:
        if line.strip() and not line.startswith('#') and not line.startswith('```'):
            if not in_description:
                in_description = True
            description.append(line.strip())
            if len(description) >= 3:
                break
        elif in_description and not line.strip():
            break
    
    # 尋找安裝指令
    install_cmd = None
    for i, line in enumerate(lines):
        if 'npm install' in line.lower() or 'yarn add' in line.lower():
            install_cmd = line.strip().replace('```', '').replace('$', '').strip()
            break
    
    # 尋找使用範例
    usage_example = []
    in_code_block = False
    code_lines = []
    for i, line in enumerate(lines):
        if '```' in line:
            if in_code_block:
                if code_lines:
                    usage_example.append('\n'.join(code_lines))
                    if len(usage_example) >= 1:
                        break
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
        elif in_code_block:
            code_lines.append(line)
    
    return {
        'title': title,
        'description': ' '.join(description) if description else None,
        'install': install_cmd,
        'example': usage_example[0] if usage_example else None
    }

def generate_report(data):
    """生成完整報告"""
    
    lines = []
    
    # 標題
    lines.append("# TypeScript Engine 依賴套件完整使用指南")
    lines.append("")
    lines.append("本文檔整合自 node_modules/ 中 439 個 Markdown 文件的內容。")
    lines.append("")
    lines.append("生成時間: 2025-11-27")
    lines.append("")
    lines.append("## 📑 目錄")
    lines.append("")
    lines.append("- [概述](#概述)")
    lines.append("- [1. 核心運行時依賴](#1-核心運行時依賴)")
    lines.append("- [2. 開發工具依賴](#2-開發工具依賴)")
    lines.append("- [3. 主要傳遞依賴](#3-主要傳遞依賴)")
    lines.append("- [4. 快速參考](#4-快速參考)")
    lines.append("- [附錄: 完整套件清單](#附錄-完整套件清單)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 概述
    lines.append("## 概述")
    lines.append("")
    lines.append("TypeScript Engine 使用 Node.js 生態系統，依賴於以下套件：")
    lines.append("")
    lines.append(f"- **總套件數**: 235 個")
    lines.append(f"- **直接依賴**: 13 個 (4 個運行時 + 9 個開發)")
    lines.append(f"- **傳遞依賴**: ~220 個 (自動安裝)")
    lines.append(f"- **總大小**: ~100 MB")
    lines.append("")
    lines.append("本指南涵蓋所有核心套件的功能、使用方法和範例。")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 1. 核心運行時依賴
    lines.append("## 1. 核心運行時依賴")
    lines.append("")
    lines.append("這些套件是程式執行時必需的，缺少任何一個都無法運行。")
    lines.append("")
    
    core_packages = data['core']
    for i, item in enumerate(core_packages, 1):
        pkg_name = item['package']
        content = item['content']
        
        info = extract_key_info(content)
        
        lines.append(f"### {i}. {pkg_name}")
        lines.append("")
        
        if info['title'] and info['title'] != pkg_name:
            lines.append(f"**{info['title']}**")
            lines.append("")
        
        if info['description']:
            desc = info['description'][:500]
            lines.append(desc)
            lines.append("")
        
        if info['install']:
            lines.append("**安裝**:")
            lines.append("```bash")
            lines.append(info['install'])
            lines.append("```")
            lines.append("")
        
        if info['example']:
            lines.append("**使用範例**:")
            lines.append("```javascript")
            example = info['example'][:800]
            lines.append(example)
            if len(info['example']) > 800:
                lines.append("// ... (更多範例請見套件文檔)")
            lines.append("```")
            lines.append("")
        
        lines.append(f"**文檔大小**: {item['size']} bytes")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # 2. 開發工具依賴
    lines.append("## 2. 開發工具依賴")
    lines.append("")
    lines.append("這些套件用於開發過程，不影響程式運行。")
    lines.append("")
    
    dev_packages = data['dev_tools']
    for i, item in enumerate(dev_packages, 1):
        pkg_name = item['package']
        content = item['content']
        
        info = extract_key_info(content)
        
        lines.append(f"### {i}. {pkg_name}")
        lines.append("")
        
        if info['description']:
            desc = info['description'][:300]
            lines.append(desc)
            lines.append("")
        
        if info['install']:
            lines.append(f"**安裝**: `{info['install']}`")
            lines.append("")
        
        lines.append(f"**文檔大小**: {item['size']} bytes")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # 3. 主要傳遞依賴 (僅列舉重要的)
    lines.append("## 3. 主要傳遞依賴")
    lines.append("")
    lines.append("這些套件由上述直接依賴自動引入，提供底層功能。")
    lines.append("")
    lines.append("僅列出重要的傳遞依賴（完整清單見附錄）：")
    lines.append("")
    
    # 挑選重要的依賴套件
    important_deps = [
        'ws', 'jpeg-js', 'pngjs',  # Playwright 相關
        'chalk', 'picocolors',      # 終端輸出
        'glob', 'minimatch',        # 檔案匹配
        'semver',                   # 版本管理
    ]
    
    dep_count = 0
    for item in data['dependencies']:
        if any(imp in item['package'] for imp in important_deps):
            pkg_name = item['package']
            info = extract_key_info(item['content'])
            
            lines.append(f"### {pkg_name}")
            if info['description']:
                desc = info['description'][:200]
                lines.append(desc)
            lines.append("")
            
            dep_count += 1
            if dep_count >= 10:
                break
    
    lines.append(f"**其他傳遞依賴**: {len(data['dependencies']) - dep_count} 個")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 4. 快速參考
    lines.append("## 4. 快速參考")
    lines.append("")
    lines.append("### 安裝所有依賴")
    lines.append("```bash")
    lines.append("cd services/scan/engines/typescript_engine")
    lines.append("npm install")
    lines.append("```")
    lines.append("")
    lines.append("### 核心套件用途速查")
    lines.append("")
    lines.append("| 套件 | 用途 | 必要性 |")
    lines.append("|------|------|--------|")
    
    quick_ref = [
        ('playwright', '瀏覽器自動化', '✅ 絕對必要'),
        ('amqplib', 'RabbitMQ 客戶端', '✅ 架構必需'),
        ('pino', '日誌記錄', '⚠️ 建議保留'),
        ('pino-pretty', '日誌美化', '❌ 僅開發用'),
        ('typescript', 'TS 編譯器', '✅ 絕對必要'),
        ('@types/node', 'Node.js 型別', '✅ 強烈建議'),
        ('tsx', 'TS 執行器', '⚠️ 開發便利'),
        ('eslint', '程式碼檢查', '❌ 可選'),
        ('prettier', '程式碼格式化', '❌ 可選'),
    ]
    
    for pkg, purpose, necessity in quick_ref:
        lines.append(f"| `{pkg}` | {purpose} | {necessity} |")
    
    lines.append("")
    lines.append("### 常用命令")
    lines.append("")
    lines.append("```bash")
    lines.append("# 開發模式（熱重載）")
    lines.append("npm run dev")
    lines.append("")
    lines.append("# 編譯 TypeScript")
    lines.append("npm run build")
    lines.append("")
    lines.append("# 執行編譯後的程式")
    lines.append("npm start")
    lines.append("")
    lines.append("# 程式碼檢查")
    lines.append("npm run lint")
    lines.append("")
    lines.append("# 程式碼格式化")
    lines.append("npm run format")
    lines.append("")
    lines.append("# 執行測試")
    lines.append("npm test")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 附錄
    lines.append("## 附錄: 完整套件清單")
    lines.append("")
    lines.append("### 核心運行時依賴")
    lines.append("")
    for item in core_packages:
        lines.append(f"- **{item['package']}** ({item['size']} bytes)")
    lines.append("")
    
    lines.append("### 開發工具依賴")
    lines.append("")
    for item in dev_packages:
        lines.append(f"- **{item['package']}** ({item['size']} bytes)")
    lines.append("")
    
    lines.append("### 傳遞依賴 (Top 50)")
    lines.append("")
    # 按大小排序
    sorted_deps = sorted(data['dependencies'], key=lambda x: x['size'], reverse=True)[:50]
    for item in sorted_deps:
        lines.append(f"- {item['package']} ({item['size']} bytes)")
    lines.append("")
    lines.append(f"**其他依賴**: {len(data['dependencies']) - 50} 個")
    lines.append("")
    
    # 授權資訊
    if data['licenses']:
        lines.append("### 授權資訊")
        lines.append("")
        lines.append("主要套件的授權類型：")
        lines.append("")
        for lic in data['licenses'][:10]:
            lines.append(f"- **{lic['package']}**: {lic['size']} bytes 授權文件")
        lines.append("")
    
    # 結尾
    lines.append("---")
    lines.append("")
    lines.append("## 📝 備註")
    lines.append("")
    lines.append("- 本文檔整合自 node_modules/ 中的 MD 文件")
    lines.append("- 詳細的 API 文檔請參考各套件官方網站")
    lines.append("- node_modules/ 已在 .gitignore 中，不會提交到版本控制")
    lines.append("- 需要時執行 `npm install` 可完全重建")
    lines.append("")
    lines.append("## 🔗 相關資源")
    lines.append("")
    lines.append("- [Playwright 官方文檔](https://playwright.dev/)")
    lines.append("- [RabbitMQ 教學](https://www.rabbitmq.com/tutorials/tutorial-one-javascript.html)")
    lines.append("- [Pino 文檔](https://getpino.io/)")
    lines.append("- [TypeScript 手冊](https://www.typescriptlang.org/docs/)")
    lines.append("- [NPM 官方註冊表](https://www.npmjs.com/)")
    lines.append("")
    
    return '\n'.join(lines)

def main():
    print("生成整合使用指南...")
    print()
    
    # 載入數據
    data = load_extracted_data()
    
    # 生成報告
    report = generate_report(data)
    
    # 儲存
    output_file = 'services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 使用指南已生成: {output_file}")
    print()
    print(f"內容統計:")
    print(f"  - 核心套件: {len(data['core'])} 個")
    print(f"  - 開發工具: {len(data['dev_tools'])} 個")
    print(f"  - 傳遞依賴: {len(data['dependencies'])} 個")
    print(f"  - 文檔總大小: {len(report):,} bytes")
    print()

if __name__ == '__main__':
    main()
