#!/usr/bin/env python3
"""分析 typescript_engine 對 node_modules 的實際使用情況"""
import json
import re
from pathlib import Path
from collections import defaultdict

def load_package_json():
    """載入 package.json"""
    package_file = Path('services/scan/engines/typescript_engine/package.json')
    with open(package_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_imports_in_source():
    """分析源碼中的 import 語句"""
    src_path = Path('services/scan/engines/typescript_engine/src')
    imports = defaultdict(set)
    
    if not src_path.exists():
        return imports
    
    for ts_file in src_path.rglob('*.ts'):
        try:
            with open(ts_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 匹配 import 語句
            # import { ... } from 'package'
            # import * as ... from 'package'
            # import ... from 'package'
            import_patterns = [
                r"import\s+.*?\s+from\s+['\"]([^'\"./][^'\"]*)['\"]",
                r"import\s+['\"]([^'\"./][^'\"]*)['\"]",
                r"require\s*\(['\"]([^'\"./][^'\"]*)['\"]\)"
            ]
            
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    # 提取套件名稱（處理 @scope/package 格式）
                    package_name = match.split('/')[0]
                    if match.startswith('@') and '/' in match:
                        package_name = '/'.join(match.split('/')[:2])
                    
                    imports[package_name].add(str(ts_file.relative_to('.')))
        
        except Exception as e:
            print(f"讀取 {ts_file} 時出錯: {e}")
    
    return imports

def analyze_node_modules():
    """分析 node_modules 目錄結構"""
    node_modules_path = Path('services/scan/engines/typescript_engine/node_modules')
    
    if not node_modules_path.exists():
        return None, None
    
    packages = []
    total_size = 0
    md_files = []
    
    for item in node_modules_path.iterdir():
        if item.is_dir():
            package_name = item.name
            
            # 處理 @scope 套件
            if package_name.startswith('@'):
                for sub_item in item.iterdir():
                    if sub_item.is_dir():
                        full_name = f"{package_name}/{sub_item.name}"
                        size = sum(f.stat().st_size for f in sub_item.rglob('*') if f.is_file())
                        packages.append({
                            'name': full_name,
                            'path': sub_item,
                            'size': size
                        })
                        total_size += size
                        
                        # 統計 MD 檔案
                        md_count = len(list(sub_item.rglob('*.md')))
                        if md_count > 0:
                            md_files.append({'name': full_name, 'count': md_count})
            else:
                size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                packages.append({
                    'name': package_name,
                    'path': item,
                    'size': size
                })
                total_size += size
                
                # 統計 MD 檔案
                md_count = len(list(item.rglob('*.md')))
                if md_count > 0:
                    md_files.append({'name': package_name, 'count': md_count})
    
    return packages, total_size, md_files

def main():
    print("=" * 80)
    print("TypeScript Engine Node_modules 使用分析")
    print("=" * 80)
    print()
    
    # 1. 載入 package.json
    print("📦 載入 package.json...")
    pkg = load_package_json()
    
    dependencies = pkg.get('dependencies', {})
    dev_dependencies = pkg.get('devDependencies', {})
    
    print(f"  運行依賴 (dependencies): {len(dependencies)} 個")
    for name, version in dependencies.items():
        print(f"    - {name}: {version}")
    
    print(f"\n  開發依賴 (devDependencies): {len(dev_dependencies)} 個")
    for name, version in dev_dependencies.items():
        print(f"    - {name}: {version}")
    
    total_declared = len(dependencies) + len(dev_dependencies)
    print(f"\n  總聲明依賴: {total_declared} 個")
    print()
    
    # 2. 分析源碼實際使用
    print("=" * 80)
    print("🔍 分析源碼中的實際 import")
    print("=" * 80)
    print()
    
    imports = analyze_imports_in_source()
    
    if imports:
        print(f"找到 {len(imports)} 個套件被實際使用:")
        for package, files in sorted(imports.items()):
            print(f"  📦 {package}")
            for file in sorted(files):
                print(f"      {file}")
        print()
    else:
        print("⚠️  未找到 src/ 目錄或沒有 import 語句")
        print()
    
    # 3. 分析 node_modules 實際安裝情況
    print("=" * 80)
    print("📂 分析 node_modules 目錄")
    print("=" * 80)
    print()
    
    packages, total_size, md_files = analyze_node_modules()
    
    if packages:
        print(f"已安裝套件: {len(packages)} 個")
        print(f"總大小: {total_size / 1024 / 1024:.2f} MB")
        print()
        
        # MD 檔案統計
        total_md = sum(item['count'] for item in md_files)
        print(f"MD 檔案分佈: {total_md} 個檔案分散在 {len(md_files)} 個套件中")
        print()
        
        print("前 20 個含有 MD 檔案的套件:")
        for item in sorted(md_files, key=lambda x: x['count'], reverse=True)[:20]:
            print(f"  {item['name']}: {item['count']} 個 MD 檔案")
        print()
        
        # 最大的套件
        print("佔用空間最大的前 10 個套件:")
        sorted_packages = sorted(packages, key=lambda x: x['size'], reverse=True)[:10]
        for pkg in sorted_packages:
            size_mb = pkg['size'] / 1024 / 1024
            print(f"  {pkg['name']}: {size_mb:.2f} MB")
        print()
    else:
        print("⚠️  node_modules 目錄不存在")
        print()
    
    # 4. 對比分析
    print("=" * 80)
    print("📊 對比分析")
    print("=" * 80)
    print()
    
    print(f"聲明的直接依賴: {total_declared} 個")
    print(f"實際安裝的套件: {len(packages) if packages else 0} 個")
    print(f"源碼中實際使用: {len(imports)} 個")
    print()
    
    if packages:
        ratio = len(packages) / total_declared if total_declared > 0 else 0
        print(f"📈 依賴膨脹率: {ratio:.1f}x")
        print(f"   (每 1 個聲明的依賴，實際安裝了 {ratio:.1f} 個套件)")
        print()
    
    # 5. 分析原因
    print("=" * 80)
    print("🔬 100+ MD 檔案的原因分析")
    print("=" * 80)
    print()
    
    if md_files:
        print("原因 1: 依賴樹深度")
        print(f"  - 聲明 {total_declared} 個直接依賴")
        print(f"  - 實際安裝 {len(packages)} 個套件（包含傳遞依賴）")
        print(f"  - 平均每個直接依賴帶來 {len(packages) / total_declared:.1f} 個間接依賴")
        print()
        
        print("原因 2: 文件分散")
        print(f"  - {len(md_files)} 個套件包含 MD 檔案")
        print(f"  - 平均每個套件 {total_md / len(md_files):.1f} 個 MD 檔案")
        print(f"  - 大多是 README.md, LICENSE.md, CHANGELOG.md 等")
        print()
        
        # 找出 MD 檔案最多的套件
        top_md_package = max(md_files, key=lambda x: x['count'])
        print(f"原因 3: 個別套件文檔過多")
        print(f"  - 最多的套件: {top_md_package['name']} ({top_md_package['count']} 個 MD)")
        print()
    
    # 6. 刪除建議
    print("=" * 80)
    print("💡 處理建議")
    print("=" * 80)
    print()
    
    print("✅ node_modules/ 完全可以刪除，原因:")
    print()
    print("  1️⃣  可重現性 (Reproducible)")
    print("     - package.json 和 package-lock.json 保存完整依賴資訊")
    print("     - 執行 'npm install' 可完全重建")
    print()
    
    print("  2️⃣  標準實踐 (Best Practice)")
    print("     - node_modules/ 已在 .gitignore 中")
    print("     - Node.js 生態系統標準做法")
    print("     - CI/CD 每次都會重新安裝")
    print()
    
    print("  3️⃣  空間效益 (Space Efficiency)")
    if packages and total_size:
        print(f"     - 目前佔用: {total_size / 1024 / 1024:.2f} MB")
        print(f"     - 包含 {total_md} 個 MD 檔案（無實際用途）")
        print(f"     - 刪除後可隨時重建: npm install ({total_declared} 個套件)")
    print()
    
    print("  4️⃣  快速安裝 (Fast Installation)")
    print("     - 現代 npm (v10+) 安裝速度極快")
    print("     - 使用 npm cache 可加速重複安裝")
    print("     - 通常 30 秒內完成")
    print()
    
    print("🎯 建議操作:")
    print()
    print("  方案 A: 僅刪除 MD 檔案 (釋放約 3 MB)")
    print("    python _delete_node_modules_md.py --execute")
    print()
    print("  方案 B: 刪除整個 node_modules/ (釋放約 100 MB)")
    print("    Remove-Item -Recurse -Force services/scan/engines/typescript_engine/node_modules")
    print("    # 需要時重建: cd services/scan/engines/typescript_engine; npm install")
    print()
    print("  推薦: 方案 B (完全刪除)")
    print("    理由: 已在 .gitignore，不應提交到版本控制")
    print()

if __name__ == '__main__':
    main()
