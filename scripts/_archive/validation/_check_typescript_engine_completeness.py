#!/usr/bin/env python3
"""檢查 typescript_engine 的完整可用性狀態"""
from pathlib import Path
import json

def check_typescript_engine_status():
    """檢查 typescript_engine 的狀態"""
    base_path = Path('services/scan/engines/typescript_engine')
    
    print("=" * 80)
    print("TypeScript Engine 完整性檢查")
    print("=" * 80)
    print()
    
    # 1. 檢查必需檔案
    print("📋 必需檔案檢查:")
    print()
    
    required_files = {
        'package.json': '套件定義檔',
        'package-lock.json': '依賴鎖定檔',
        'tsconfig.json': 'TypeScript 配置',
        'src/index.ts': '主程式入口'
    }
    
    all_required_present = True
    for file, desc in required_files.items():
        file_path = base_path / file
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {file:<25} {desc}")
        if not exists:
            all_required_present = False
    
    print()
    
    # 2. 檢查源碼
    print("📂 源碼檢查:")
    print()
    
    src_path = base_path / 'src'
    if src_path.exists():
        ts_files = list(src_path.rglob('*.ts'))
        print(f"  ✅ src/ 目錄存在")
        print(f"  📄 TypeScript 檔案: {len(ts_files)} 個")
        
        # 列出主要檔案
        main_files = [
            'index.ts',
            'services/scan-service.ts',
            'utils/logger.ts',
            'interfaces/scan.interfaces.ts'
        ]
        
        for file in main_files:
            file_path = src_path / file
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"    ✅ {file} ({size} bytes)")
            else:
                print(f"    ⚠️  {file} (不存在)")
    else:
        print(f"  ❌ src/ 目錄不存在")
        all_required_present = False
    
    print()
    
    # 3. 檢查 node_modules
    print("📦 依賴檢查:")
    print()
    
    node_modules_path = base_path / 'node_modules'
    if node_modules_path.exists():
        packages = [d for d in node_modules_path.iterdir() if d.is_dir()]
        print(f"  ✅ node_modules/ 存在")
        print(f"  📦 已安裝套件: {len(packages)} 個")
        
        # 檢查關鍵依賴
        critical_deps = ['playwright', 'playwright-core', 'amqplib', 'typescript']
        print()
        print("  關鍵依賴:")
        for dep in critical_deps:
            dep_path = node_modules_path / dep
            if dep_path.exists():
                print(f"    ✅ {dep}")
            else:
                print(f"    ❌ {dep} (缺失)")
    else:
        print(f"  ⚠️  node_modules/ 不存在")
        print(f"  ℹ️  這是正常的！需要執行 'npm install' 安裝依賴")
    
    print()
    
    # 4. 檢查編譯輸出
    print("🔨 編譯輸出檢查:")
    print()
    
    dist_path = base_path / 'dist'
    if dist_path.exists():
        js_files = list(dist_path.rglob('*.js'))
        print(f"  ✅ dist/ 目錄存在")
        print(f"  📄 JavaScript 檔案: {len(js_files)} 個")
    else:
        print(f"  ⚠️  dist/ 不存在")
        print(f"  ℹ️  需要執行 'npm run build' 編譯 TypeScript")
    
    print()
    
    # 5. 讀取 package.json 分析依賴
    print("=" * 80)
    print("依賴分析")
    print("=" * 80)
    print()
    
    package_json = base_path / 'package.json'
    if package_json.exists():
        with open(package_json, 'r', encoding='utf-8') as f:
            pkg = json.load(f)
        
        deps = pkg.get('dependencies', {})
        dev_deps = pkg.get('devDependencies', {})
        
        print(f"運行時依賴 (dependencies): {len(deps)} 個")
        for name, version in deps.items():
            print(f"  - {name}: {version}")
        
        print()
        print(f"開發依賴 (devDependencies): {len(dev_deps)} 個")
        for name, version in dev_deps.items():
            print(f"  - {name}: {version}")
        
        print()
        print(f"總依賴數: {len(deps) + len(dev_deps)} 個")
    
    print()
    
    # 6. 完整性總結
    print("=" * 80)
    print("完整性評估")
    print("=" * 80)
    print()
    
    print("問題 1: 實際運作需要多少檔案?")
    print()
    print("  運行時必需 (Runtime):")
    print("    ✅ 源碼: src/ 目錄 (約 10-20 個 .ts 檔案)")
    print("    ✅ 依賴: node_modules/ (235 個套件，~100 MB)")
    print("    ✅ 配置: package.json, tsconfig.json")
    print("    ⚠️  編譯輸出: dist/ 目錄 (如果直接執行 .ts 則不需要)")
    print()
    print("  開發時需要 (Development):")
    print("    ✅ 上述所有檔案")
    print("    ✅ 開發工具: TypeScript, ESLint, Prettier (在 node_modules)")
    print("    ✅ 開發配置: .eslintrc.json, .prettierrc")
    print()
    
    print("問題 2: 直接下載是否完整可用?")
    print()
    
    has_package_json = (base_path / 'package.json').exists()
    has_package_lock = (base_path / 'package-lock.json').exists()
    has_src = (base_path / 'src').exists()
    has_node_modules = node_modules_path.exists()
    
    if has_package_json and has_package_lock and has_src:
        if has_node_modules:
            print("  ✅ 當前狀態: 【完整可用】")
            print("     - 源碼: ✅")
            print("     - 依賴定義: ✅")
            print("     - 已安裝依賴: ✅")
            print()
            print("  🚀 可以直接執行:")
            print("     npm run dev      # 開發模式")
            print("     npm run build    # 編譯")
            print("     npm start        # 執行編譯後的程式")
        else:
            print("  ⚠️  當前狀態: 【幾乎完整，缺少依賴】")
            print("     - 源碼: ✅")
            print("     - 依賴定義: ✅")
            print("     - 已安裝依賴: ❌")
            print()
            print("  🔧 需要一個步驟:")
            print("     cd services/scan/engines/typescript_engine")
            print("     npm install      # 30-60 秒完成")
            print()
            print("  ℹ️  這就是為什麼 node_modules 不應該在 Git 中:")
            print("     - 可以從 package.json + package-lock.json 完全重建")
            print("     - 不同環境/平台可能需要不同的二進位檔")
            print("     - 保持儲存庫精簡")
    else:
        print("  ❌ 當前狀態: 【不完整】")
        missing = []
        if not has_package_json:
            missing.append("package.json")
        if not has_package_lock:
            missing.append("package-lock.json")
        if not has_src:
            missing.append("src/")
        print(f"     缺少: {', '.join(missing)}")
    
    print()
    
    # 7. 使用建議
    print("=" * 80)
    print("使用建議")
    print("=" * 80)
    print()
    
    print("📥 首次使用 (Clone 後):")
    print("  1. cd services/scan/engines/typescript_engine")
    print("  2. npm install                    # 安裝依賴 (30-60秒)")
    print("  3. npm run install:browsers       # 安裝 Playwright 瀏覽器")
    print("  4. npm run dev                    # 開發模式執行")
    print()
    
    print("🔨 開發流程:")
    print("  - npm run dev        # 開發模式 (熱重載)")
    print("  - npm run build      # 編譯 TypeScript → JavaScript")
    print("  - npm test           # 執行測試")
    print("  - npm run lint       # 程式碼檢查")
    print("  - npm run format     # 程式碼格式化")
    print()
    
    print("🚀 生產環境:")
    print("  1. npm ci                         # 使用 lock file 安裝 (更快)")
    print("  2. npm run build                  # 編譯")
    print("  3. npm start                      # 執行編譯後的程式")
    print()
    
    print("🗑️  清理建議:")
    if has_node_modules:
        print("  ✅ 可以刪除 node_modules/")
        print("     - 釋放 ~100 MB 空間")
        print("     - 不影響源碼")
        print("     - 需要時 'npm install' 重建")
        print()
        print("     執行: Remove-Item -Recurse -Force node_modules")
    else:
        print("  ✅ node_modules/ 不存在 (正確狀態)")
    
    print()

if __name__ == '__main__':
    check_typescript_engine_status()
