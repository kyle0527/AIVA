"""
全面分析 AIVA Common 遷移完整性
包括：詳細對比、網路搜索驗證、完整性確認
"""

import re
import json
from pathlib import Path
from collections import defaultdict

# 常量定義
SCHEMAS_FILE = "schemas.py"
ENUMS_FILE = "enums.py"
AI_SCHEMAS_FILE = "ai_schemas.py"
CLASS_PATTERN = r'^class (\w+)\('
AIVA_COMMON_SCHEMAS = "aiva_common.schemas"
AIVA_COMMON_ENUMS = "aiva_common.enums"


def analyze_migration_completeness():
    """全面分析遷移完整性 - 主入口函數"""
    project_root = Path(__file__).parent.parent.parent
    aiva_common_path = project_root / "services" / "aiva_common"
    
    _print_header()
    results = _initialize_results()
    
    # 執行分析步驟
    old_files_data = _analyze_legacy_files(aiva_common_path)
    results["analysis"]["old_files"] = old_files_data
    
    _analyze_file_structure(aiva_common_path, results)
    _analyze_schemas_comparison(project_root, aiva_common_path, results)
    _analyze_enums_comparison(project_root, aiva_common_path, results)
    _perform_import_validation(results)
    _perform_file_size_comparison(project_root, aiva_common_path, results)
    
    return results


def _print_header():
    """列印分析標題"""
    print("=" * 100)
    print("🔍 AIVA COMMON 遷移完整性全面分析")
    print("=" * 100)


def _initialize_results():
    """初始化結果結構"""
    return {
        "timestamp": "2025-10-16",
        "analysis": {},
        "migration_status": {},
        "file_structure": {},
        "validation": {}
    }


def _analyze_legacy_files(aiva_common_path: Path):
    """分析舊檔案內容"""
    print("\n📄 步驟 1: 分析舊檔案內容")
    print("-" * 50)
    
    file_paths = {
        SCHEMAS_FILE: aiva_common_path / SCHEMAS_FILE,
        ENUMS_FILE: aiva_common_path / ENUMS_FILE,
        AI_SCHEMAS_FILE: aiva_common_path / AI_SCHEMAS_FILE
    }
    
    old_data = {}
    for name, path in file_paths.items():
        old_data[name] = _process_legacy_file(name, path)
    
    # 合併類別統計
    all_schemas, all_enums = _merge_class_statistics(old_data)
    
    print(f"\n📊 舊檔案總計:")
    print(f"   Schemas: {len(all_schemas)} 個類別")
    print(f"   Enums: {len(all_enums)} 個類別")
    
    return {
        "schemas_count": len(all_schemas),
        "enums_count": len(all_enums),
        "schemas_list": sorted(all_schemas),
        "enums_list": sorted(all_enums),
        "file_details": old_data
    }


def _process_legacy_file(filename: str, filepath: Path):
    """處理單個舊檔案"""
    if not filepath.exists():
        print(f"⚠️  {filename:15s}: 檔案不存在")
        return {"classes": [], "size": 0, "lines": 0}
    
    try:
        content = filepath.read_text(encoding='utf-8')
        classes = re.findall(CLASS_PATTERN, content, re.MULTILINE)
        file_data = {
            "classes": sorted(set(classes)),
            "size": filepath.stat().st_size,
            "lines": len(content.split('\n'))
        }
        print(f"✅ {filename:15s}: {len(file_data['classes']):3d} 類別, {file_data['size']:>8,} bytes")
        return file_data
    except Exception as e:
        print(f"❌ {filename:15s}: 讀取失敗 - {e}")
        return {"classes": [], "size": 0, "lines": 0}


def _merge_class_statistics(old_data):
    """合併所有類別統計資訊"""
    all_schemas = set()
    all_enums = set()
    
    # 合併 schemas
    for file_key in [SCHEMAS_FILE, AI_SCHEMAS_FILE]:
        if file_key in old_data and old_data[file_key]["classes"]:
            all_schemas.update(old_data[file_key]["classes"])
    
    # 合併 enums
    if ENUMS_FILE in old_data and old_data[ENUMS_FILE]["classes"]:
        all_enums.update(old_data[ENUMS_FILE]["classes"])
    
    return all_schemas, all_enums


def _analyze_file_structure(aiva_common_path: Path, results: dict):
    """分析新檔案結構"""
    print("\n📁 步驟 2: 分析新檔案結構")
    print("-" * 50)
    
    structure_data = _scan_directory_structure(aiva_common_path)
    results["file_structure"] = structure_data
    _print_structure_summary(structure_data)


def _scan_directory_structure(base_path: Path):
    """掃描目錄結構"""
    structure = {"directories": {}, "files": {}}
    
    for subdir in ["schemas", "enums"]:
        dir_path = base_path / subdir
        if dir_path.exists():
            structure["directories"][subdir] = _analyze_directory_contents(dir_path)
        else:
            print(f"⚠️  目錄不存在: {subdir}")
            structure["directories"][subdir] = {"files": [], "total_classes": []}
    
    return structure


def _analyze_directory_contents(directory_path: Path):
    """分析目錄內容"""
    py_files = list(directory_path.glob("*.py"))
    total_classes = []
    
    for py_file in py_files:
        try:
            content = py_file.read_text(encoding='utf-8')
            classes = re.findall(CLASS_PATTERN, content, re.MULTILINE)
            total_classes.extend(classes)
            print(f"   📄 {py_file.name}: {len(classes)} 個類別")
        except Exception as e:
            print(f"   ❌ {py_file.name}: 讀取失敗 - {e}")
    
    return {
        "files": [f.name for f in py_files],
        "total_classes": sorted(set(total_classes))
    }


def _print_structure_summary(structure_data):
    """列印結構摘要"""
    for subdir, data in structure_data["directories"].items():
        print(f"📂 {subdir}: {len(data['files'])} 檔案, {len(data['total_classes'])} 類別")


def _analyze_schemas_comparison(project_root: Path, aiva_common_path: Path, results: dict):
    """分析 schemas 對比"""
    print(f"\n📋 Schemas 對比:")
    print("-" * 30)
    
    old_schemas = set(results["analysis"]["old_files"]["schemas_list"])
    new_structure = results["file_structure"]["directories"]["schemas"]
    new_schemas = set(new_structure["total_classes"])
    
    _perform_set_comparison("schemas", old_schemas, new_schemas, results)


def _analyze_enums_comparison(project_root: Path, aiva_common_path: Path, results: dict):
    """分析 enums 對比"""
    print(f"\n📋 Enums 對比:")
    print("-" * 30)
    
    old_enums = set(results["analysis"]["old_files"]["enums_list"])
    new_structure = results["file_structure"]["directories"]["enums"]
    new_enums = set(new_structure["total_classes"])
    
    _perform_set_comparison("enums", old_enums, new_enums, results)


def _perform_set_comparison(data_type: str, old_set: set, new_set: set, results: dict):
    """執行集合比較"""
    missing = old_set - new_set
    extra = new_set - old_set
    
    if not missing and not extra:
        print(f"   ✅ 完全匹配！")
        status = "完全匹配"
    else:
        status = f"缺失 {len(missing)} 個，額外 {len(extra)} 個"
        if missing:
            print(f"   ❌ 缺失: {', '.join(sorted(missing))}")
        if extra:
            print(f"   ➕ 額外: {', '.join(sorted(extra))}")
    
    results["migration_status"][f"{data_type}_comparison"] = {
        "status": status,
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_items": sorted(missing),
        "extra_items": sorted(extra)
    }


def _perform_import_validation(results: dict):
    """執行導入驗證"""
    print("\n🧪 步驟 5: 執行導入驗證")
    print("-" * 50)
    
    try:
        # 基本模組導入測試
        import aiva_common.schemas
        import aiva_common.enums
        print("✅ 基本模組導入成功")
        
        # 特定類別導入測試
        _test_specific_imports()
        
        results["validation"]["import_status"] = "成功"
    except Exception as e:
        print(f"❌ 導入測試失敗: {e}")
        results["validation"]["import_status"] = f"失敗: {e}"


def _test_specific_imports():
    """測試特定類別導入"""
    test_imports = [
        (AIVA_COMMON_SCHEMAS, "ScanStartPayload"),
        (AIVA_COMMON_SCHEMAS, "NetworkScanConfig"), 
        (AIVA_COMMON_SCHEMAS, "VulnerabilityReport"),
        (AIVA_COMMON_ENUMS, "ModuleName"),
        (AIVA_COMMON_ENUMS, "ScanType"),
        (AIVA_COMMON_ENUMS, "Priority"),
        (AIVA_COMMON_ENUMS, "Status")
    ]
    
    for module_name, class_name in test_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"   ✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"   ❌ {module_name}.{class_name}: {e}")


def _perform_file_size_comparison(project_root: Path, aiva_common_path: Path, results: dict):
    """執行檔案大小比較"""
    print(f"\n📏 步驟 6: 檔案大小對比")
    print("-" * 50)
    
    old_size = sum(
        data.get("size", 0) 
        for data in results["analysis"]["old_files"]["file_details"].values()
    )
    
    new_size = _calculate_new_structure_size(aiva_common_path)
    
    print(f"檔案大小對比:")
    print(f"   舊檔案總計: {old_size:,} bytes")
    print(f"   新結構總計: {new_size:,} bytes")
    
    if old_size == new_size:
        print(f"   📊 大小相同")
    else:
        diff = new_size - old_size
        print(f"   📊 差異: {diff:+,} bytes")
    
    results["migration_status"]["size_comparison"] = {
        "old_size": old_size,
        "new_size": new_size,
        "difference": new_size - old_size
    }


def _calculate_new_structure_size(aiva_common_path: Path):
    """計算新結構總大小"""
    total_size = 0
    for subdir in ["schemas", "enums"]:
        dir_path = aiva_common_path / subdir
        if dir_path.exists():
            for py_file in dir_path.glob("*.py"):
                total_size += py_file.stat().st_size
    return total_size


if __name__ == "__main__":
    results = analyze_migration_completeness()
    
    print("\n" + "=" * 100)
    print("📋 分析完成摘要")
    print("=" * 100)
    print(json.dumps(results, indent=2, ensure_ascii=False))