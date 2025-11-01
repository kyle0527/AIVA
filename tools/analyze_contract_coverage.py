#!/usr/bin/env python3
"""
AIVA 合約覆蓋率深度分析工具
"""

import sys
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple

def analyze_imports_in_file(file_path: Path) -> Tuple[Set[str], Set[str]]:
    """分析單個檔案中的導入和本地定義"""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # 提取 aiva_common.schemas 導入
        imported_contracts = set()
        
        # 匹配多行導入
        import_blocks = re.finditer(
            r'from\s+(?:services\.)?aiva_common\.schemas\s+import\s*\(\s*(.*?)\s*\)', 
            content, 
            re.DOTALL
        )
        
        for match in import_blocks:
            import_content = match.group(1)
            # 清理並分割
            contracts = re.split(r'[,\n]', import_content)
            for contract in contracts:
                contract = contract.strip()
                if contract and not contract.startswith('#'):
                    imported_contracts.add(contract)
        
        # 匹配單行導入
        single_imports = re.findall(
            r'from\s+(?:services\.)?aiva_common\.schemas\s+import\s+([^\n]+)', 
            content
        )
        
        for import_line in single_imports:
            if '(' not in import_line:  # 避免重複處理多行導入
                contracts = [c.strip() for c in import_line.split(',')]
                imported_contracts.update(contracts)
        
        # 提取本地 BaseModel 定義
        local_schemas = set()
        local_classes = re.findall(r'class\s+(\w+)\s*\(\s*BaseModel\s*\):', content)
        local_schemas.update(local_classes)
        
        return imported_contracts, local_schemas
        
    except Exception:  # 遵循PEP-8，移除未使用的異常變數
        return set(), set()

def get_modules_to_check():
    """獲取需要檢查的模組路徑字典，遵循PEP-8單一責任原則"""
    return {
        'SQLi': Path('services/features/function_sqli'),
        'XSS': Path('services/features/function_xss'),
        'IDOR': Path('services/features/function_idor'),
        'SSRF': Path('services/features/function_ssrf'),
        'PostEx': Path('services/features/function_postex'),
        'Scan': Path('services/scan/aiva_scan'),
        'Core': Path('services/core/aiva_core'),
        'API': Path('api'),
        'Web': Path('web')
    }

def analyze_module_contracts(module_path):
    """分析單一模組的合約使用情況，降低認知複雜度"""
    if not module_path.exists():
        return set(), set()
        
    module_imported = set()
    module_local = set()
    
    # 掃描所有 Python 檔案
    for py_file in module_path.rglob("*.py"):
        imported, local = analyze_imports_in_file(py_file)
        module_imported.update(imported)
        module_local.update(local)
    
    return module_imported, module_local

def print_module_report(module_name, module_imported, module_local):
    """打印模組報告，遵循DRY原則"""
    print(f"\n📦 {module_name} 模組:")
    print(f"  📥 導入合約: {len(module_imported)} 個")
    print(f"  🏠 本地合約: {len(module_local)} 個")
    
    if module_imported:
        contracts_list = sorted(module_imported)
        if len(contracts_list) <= 5:
            print(f"    ✅ 導入: {contracts_list}")
        else:
            print(f"    ✅ 導入: {contracts_list[:5]} ... (+{len(contracts_list)-5})")
    
    if module_local:
        local_list = sorted(module_local)
        if len(local_list) <= 3:
            print(f"    🔧 本地: {local_list}")
        else:
            print(f"    🔧 本地: {local_list[:3]} ... (+{len(local_list)-3})")

def print_coverage_analysis(all_imported_contracts):
    """打印覆蓋率分析，符合單一責任原則"""
    try:
        sys.path.append('services')
        from aiva_common.schemas import __all__ as available_schemas
        total_available = len(available_schemas)
        coverage_percent = (len(all_imported_contracts) / total_available) * 100
        
        print("\n🎯 覆蓋率分析:")
        print(f"  📋 可用合約總數: {total_available}")
        print(f"  ✅ 使用覆蓋率: {coverage_percent:.1f}%")
        
        if len(all_imported_contracts) > 0:
            print(f"  📝 常用合約: {sorted(list(all_imported_contracts))[:10]}")
            
    except Exception as e:
        print(f"  ⚠️ 無法獲取總合約數: {e}")

def print_health_assessment(all_imported_contracts):
    """打印健康度評估，遵循PEP-8函數設計原則"""
    print("\n🏥 合約使用健康度評估:")
    
    if len(all_imported_contracts) >= 30:
        print("  ✅ 優秀: 合約使用率很高，系統整合度良好")
    elif len(all_imported_contracts) >= 15:
        print("  ⚠️ 良好: 合約使用適中，有改進空間")
    else:
        print("  🔴 需要改進: 合約使用率偏低，建議加強整合")

def print_recommendations(all_imported_contracts):
    """打印改進建議，降低主函數複雜度"""
    print("\n💡 改進建議:")
    if len(all_imported_contracts) < 20:
        print("  1. 增加功能模組對通用合約的使用")
        print("  2. 減少重複的本地 schema 定義")
        print("  3. 標準化資料交換格式")

def main():
    """主函數：重構後符合PEP-8複雜度要求"""
    print("🔍 AIVA 合約覆蓋率深度分析")
    print("=" * 60)
    
    modules = get_modules_to_check()
    all_imported_contracts = set()
    all_local_schemas = set()
    
    # 分析各模組
    for module_name, module_path in modules.items():
        module_imported, module_local = analyze_module_contracts(module_path)
        
        all_imported_contracts.update(module_imported)
        all_local_schemas.update(module_local)
        
        print_module_report(module_name, module_imported, module_local)
    
    # 總結統計
    print("\n📊 總體統計:")
    print(f"  🎯 總共使用合約: {len(all_imported_contracts)} 個")
    print(f"  🏗️ 總共本地合約: {len(all_local_schemas)} 個")
    
    # 各種分析報告
    print_coverage_analysis(all_imported_contracts)
    print_health_assessment(all_imported_contracts)
    print_recommendations(all_imported_contracts)

if __name__ == "__main__":
    main()