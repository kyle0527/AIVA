#!/usr/bin/env python3
"""完整重新提取並生成包含所有439個文件的使用指南"""
import json
from pathlib import Path
from collections import defaultdict

def main():
    print("=== 完整提取所有 MD 文件 ===\n")
    
    node_modules = Path('services/scan/engines/typescript_engine/node_modules')
    
    # 收集所有文件
    all_files = []
    for md_file in node_modules.rglob('*.md'):
        try:
            content = md_file.read_text(encoding='utf-8', errors='ignore')
            rel_path = str(md_file.relative_to(node_modules))
            
            # 提取套件名稱
            parts = rel_path.split('\\')
            if parts[0].startswith('@'):
                package_name = f"{parts[0]}/{parts[1]}"
            else:
                package_name = parts[0]
            
            all_files.append({
                'path': rel_path,
                'package': package_name,
                'filename': md_file.name,
                'size': len(content),
                'content': content
            })
        except Exception as e:
            print(f"⚠️  無法讀取 {md_file}: {e}")
    
    print(f"✅ 成功提取 {len(all_files)} 個文件\n")
    
    # 按套件分組
    by_package = defaultdict(list)
    for f in all_files:
        by_package[f['package']].append(f)
    
    print(f"涉及 {len(by_package)} 個套件\n")
    
    # 生成完整使用指南
    print("=== 生成完整使用指南 ===\n")
    
    lines = []
    
    # 標題和目錄
    lines.append("# TypeScript Engine 依賴套件完整使用指南")
    lines.append("")
    lines.append(f"本文檔整合自 node_modules/ 中 **{len(all_files)} 個 Markdown 文件**的內容。")
    lines.append("")
    lines.append("生成時間: 2025-11-27")
    lines.append("")
    lines.append("## 📑 目錄")
    lines.append("")
    lines.append("- [概述](#概述)")
    lines.append("- [核心運行時依賴](#核心運行時依賴)")
    lines.append("- [開發工具依賴](#開發工具依賴)")
    lines.append("- [傳遞依賴套件](#傳遞依賴套件)")
    lines.append("- [快速參考](#快速參考)")
    lines.append("- [完整套件清單](#完整套件清單)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 概述
    lines.append("## 概述")
    lines.append("")
    lines.append("TypeScript Engine 的依賴結構：")
    lines.append("")
    lines.append(f"- **總套件數**: {len(by_package)} 個")
    lines.append(f"- **總文檔數**: {len(all_files)} 個 Markdown 文件")
    lines.append(f"- **總大小**: ~100 MB")
    lines.append("")
    
    # 定義核心套件
    core_packages = ['playwright', 'amqplib', 'pino', 'pino-pretty']
    dev_packages = ['typescript', '@types/node', 'eslint', 'prettier', 'tsx', 'vitest']
    
    lines.append("### 套件分類")
    lines.append("")
    lines.append("#### 核心運行時依賴 (4個)")
    lines.append("- `playwright` - 瀏覽器自動化框架")
    lines.append("- `amqplib` - RabbitMQ 客戶端")
    lines.append("- `pino` - 高性能日誌記錄")
    lines.append("- `pino-pretty` - 日誌美化輸出")
    lines.append("")
    lines.append("#### 開發工具依賴 (6個)")
    lines.append("- `typescript` - TypeScript 編譯器")
    lines.append("- `@types/node` - Node.js 型別定義")
    lines.append("- `eslint` - 程式碼檢查工具")
    lines.append("- `prettier` - 程式碼格式化")
    lines.append("- `tsx` - TypeScript 執行器")
    lines.append("- `vitest` - 測試框架")
    lines.append("")
    lines.append(f"#### 傳遞依賴 ({len(by_package) - len(core_packages) - len(dev_packages)}個)")
    lines.append("由上述直接依賴自動引入的底層套件。")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 核心運行時依賴
    lines.append("## 核心運行時依賴")
    lines.append("")
    lines.append("這些套件是程式執行時必需的。")
    lines.append("")
    
    for pkg_name in core_packages:
        if pkg_name in by_package:
            files = by_package[pkg_name]
            lines.append(f"### {pkg_name}")
            lines.append("")
            lines.append(f"**文檔數量**: {len(files)} 個")
            lines.append("")
            
            # 顯示主要 README
            readme = next((f for f in files if f['filename'].lower() == 'readme.md'), None)
            if readme:
                content_preview = readme['content'][:500].replace('\n', ' ')
                lines.append(f"**簡介**: {content_preview}...")
                lines.append("")
            
            lines.append("**相關文檔**:")
            for f in files[:5]:  # 只列前5個
                lines.append(f"- `{f['filename']}` ({f['size']} bytes)")
            if len(files) > 5:
                lines.append(f"- ... 還有 {len(files) - 5} 個文檔")
            lines.append("")
            lines.append("---")
            lines.append("")
    
    # 開發工具依賴
    lines.append("## 開發工具依賴")
    lines.append("")
    lines.append("這些套件用於開發過程。")
    lines.append("")
    
    for pkg_name in dev_packages:
        if pkg_name in by_package:
            files = by_package[pkg_name]
            lines.append(f"### {pkg_name}")
            lines.append(f"**文檔數量**: {len(files)} 個")
            lines.append("")
            
            # 列出文檔
            for f in files[:3]:
                lines.append(f"- `{f['filename']}` ({f['size']} bytes)")
            if len(files) > 3:
                lines.append(f"- ... 還有 {len(files) - 3} 個文檔")
            lines.append("")
            lines.append("---")
            lines.append("")
    
    # 傳遞依賴（按文檔數量排序，列出前20個）
    lines.append("## 傳遞依賴套件")
    lines.append("")
    lines.append("由直接依賴自動引入的套件。以下列出文檔最多的前 20 個套件：")
    lines.append("")
    
    other_packages = [(pkg, files) for pkg, files in by_package.items() 
                      if pkg not in core_packages and pkg not in dev_packages]
    other_packages.sort(key=lambda x: len(x[1]), reverse=True)
    
    for i, (pkg_name, files) in enumerate(other_packages[:20], 1):
        lines.append(f"### {i}. {pkg_name}")
        lines.append(f"**文檔數量**: {len(files)} 個")
        
        # 列出文檔名稱
        doc_list = ', '.join([f['filename'] for f in files[:5]])
        lines.append(f"**文檔**: {doc_list}")
        if len(files) > 5:
            lines.append(f" ... 還有 {len(files) - 5} 個")
        lines.append("")
    
    lines.append(f"**其他傳遞依賴**: {len(other_packages) - 20} 個套件")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 快速參考
    lines.append("## 快速參考")
    lines.append("")
    lines.append("### 安裝依賴")
    lines.append("```bash")
    lines.append("cd services/scan/engines/typescript_engine")
    lines.append("npm install")
    lines.append("```")
    lines.append("")
    lines.append("### 核心套件用途")
    lines.append("")
    lines.append("| 套件 | 用途 | 必要性 |")
    lines.append("|------|------|--------|")
    lines.append("| `playwright` | 瀏覽器自動化 | ✅ 絕對必要 |")
    lines.append("| `amqplib` | RabbitMQ 客戶端 | ✅ 架構必需 |")
    lines.append("| `pino` | 日誌記錄 | ⚠️ 建議保留 |")
    lines.append("| `pino-pretty` | 日誌美化 | ❌ 僅開發用 |")
    lines.append("| `typescript` | TS 編譯器 | ✅ 絕對必要 |")
    lines.append("| `@types/node` | Node.js 型別 | ✅ 強烈建議 |")
    lines.append("| `eslint` | 程式碼檢查 | ❌ 可選 |")
    lines.append("| `prettier` | 程式碼格式化 | ❌ 可選 |")
    lines.append("| `tsx` | TS 執行器 | ⚠️ 開發便利 |")
    lines.append("| `vitest` | 測試框架 | ⚠️ 建議保留 |")
    lines.append("")
    lines.append("### 常用命令")
    lines.append("```bash")
    lines.append("npm run dev      # 開發模式")
    lines.append("npm run build    # 編譯")
    lines.append("npm start        # 執行")
    lines.append("npm run lint     # 檢查")
    lines.append("npm run format   # 格式化")
    lines.append("npm test         # 測試")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # 完整套件清單
    lines.append("## 完整套件清單")
    lines.append("")
    lines.append(f"共 {len(by_package)} 個套件，{len(all_files)} 個文檔：")
    lines.append("")
    
    # 按字母排序
    for pkg_name in sorted(by_package.keys()):
        files = by_package[pkg_name]
        lines.append(f"### {pkg_name}")
        lines.append(f"文檔數: {len(files)} 個")
        for f in files:
            lines.append(f"- `{f['filename']}` ({f['size']} bytes)")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("## 📝 備註")
    lines.append("")
    lines.append(f"- 本文檔整合自 **{len(all_files)} 個** Markdown 文件")
    lines.append(f"- 涵蓋 **{len(by_package)} 個**套件的完整文檔")
    lines.append("- node_modules/ 已在 .gitignore 中，不會提交")
    lines.append("- 執行 `npm install` 可完全重建")
    lines.append("")
    lines.append("## 🔗 相關資源")
    lines.append("")
    lines.append("- [Playwright](https://playwright.dev/)")
    lines.append("- [RabbitMQ](https://www.rabbitmq.com/)")
    lines.append("- [Pino](https://getpino.io/)")
    lines.append("- [TypeScript](https://www.typescriptlang.org/)")
    lines.append("- [NPM](https://www.npmjs.com/)")
    lines.append("")
    
    # 儲存
    content = '\n'.join(lines)
    output_file = Path('services/scan/engines/typescript_engine/DEPENDENCIES_GUIDE.md')
    output_file.write_text(content, encoding='utf-8')
    
    print(f"✅ 使用指南已生成: {output_file}")
    print(f"   文檔大小: {len(content):,} bytes")
    print(f"   總行數: {len(lines):,} 行")
    print(f"   涵蓋文件: {len(all_files)} 個")
    print(f"   涵蓋套件: {len(by_package)} 個")
    print()
    
    # 同時儲存完整數據供參考
    json_data = {
        'total_files': len(all_files),
        'total_packages': len(by_package),
        'by_package': {pkg: [{'filename': f['filename'], 'size': f['size']} 
                             for f in files] 
                      for pkg, files in by_package.items()}
    }
    
    json_file = Path('_node_modules_complete_inventory.json')
    json_file.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"✅ 完整清單已儲存: {json_file}")

if __name__ == '__main__':
    main()
