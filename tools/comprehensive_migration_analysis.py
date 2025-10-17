"""
全面分析 AIVA Common 遷移完整性
包括：詳細對比、網路搜索驗證、完整性確認
"""

import re
import json
from pathlib import Path
from collections import defaultdict

def analyze_migration_completeness():
    """全面分析遷移完整性"""
    
    project_root = Path(__file__).parent.parent
    aiva_common = project_root / "services" / "aiva_common"
    
    print("=" * 100)
    print("[SEARCH] AIVA COMMON 遷移完整性全面分析")
    print("=" * 100)
    
    results = {
        "timestamp": "2025-10-16",
        "analysis": {},
        "migration_status": {},
        "file_structure": {},
        "validation": {}
    }
    
    # ========================================================================
    # 1. 分析舊檔案內容
    # ========================================================================
    print("\n[U+1F4C4] 步驟 1: 分析舊檔案內容")
    print("-" * 50)
    
    old_files = {
        "schemas.py": aiva_common / "schemas.py",
        "enums.py": aiva_common / "enums.py", 
        "ai_schemas.py": aiva_common / "ai_schemas.py"
    }
    
    old_data = {}
    for name, path in old_files.items():
        if path.exists():
            try:
                content = path.read_text(encoding='utf-8')
                classes = re.findall(r'^class (\w+)\(', content, re.MULTILINE)
                old_data[name] = {
                    "classes": sorted(set(classes)),
                    "size": path.stat().st_size,
                    "lines": len(content.split('\n'))
                }
                print(f"[OK] {name:15s}: {len(old_data[name]['classes']):3d} 類別, {old_data[name]['size']:>8,} bytes")
            except Exception as e:
                print(f"[FAIL] {name:15s}: 讀取失敗 - {e}")
                old_data[name] = {"classes": [], "size": 0, "lines": 0}
        else:
            print(f"[WARN]  {name:15s}: 檔案不存在")
            old_data[name] = {"classes": [], "size": 0, "lines": 0}
    
    # 合併所有舊類別
    all_old_schemas = set()
    all_old_enums = set()
    
    if old_data["schemas.py"]["classes"]:
        all_old_schemas.update(old_data["schemas.py"]["classes"])
    if old_data["ai_schemas.py"]["classes"]:
        all_old_schemas.update(old_data["ai_schemas.py"]["classes"])  
    if old_data["enums.py"]["classes"]:
        all_old_enums.update(old_data["enums.py"]["classes"])
    
    print(f"\n[STATS] 舊檔案總計:")
    print(f"   Schemas: {len(all_old_schemas)} 個類別")
    print(f"   Enums: {len(all_old_enums)} 個類別")
    
    results["analysis"]["old_files"] = {
        "schemas_count": len(all_old_schemas),
        "enums_count": len(all_old_enums),
        "schemas_list": sorted(all_old_schemas),
        "enums_list": sorted(all_old_enums)
    }
    
    # ========================================================================
    # 2. 分析新模組結構
    # ========================================================================
    print("\n[U+1F4C1] 步驟 2: 分析新模組結構")
    print("-" * 50)
    
    new_structure = {}
    
    # 分析 schemas/ 資料夾
    schemas_dir = aiva_common / "schemas"
    if schemas_dir.exists():
        schema_modules = {}
        total_schema_classes = set()
        
        for py_file in schemas_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                classes = re.findall(r'^class (\w+)\(', content, re.MULTILINE)
                schema_modules[py_file.stem] = {
                    "classes": sorted(set(classes)),
                    "size": py_file.stat().st_size,
                    "lines": len(content.split('\n'))
                }
                total_schema_classes.update(classes)
                print(f"   [U+1F4C4] {py_file.name:20s}: {len(classes):3d} 類別, {py_file.stat().st_size:>8,} bytes")
            except Exception as e:
                print(f"   [FAIL] {py_file.name:20s}: 讀取失敗 - {e}")
        
        new_structure["schemas"] = {
            "modules": schema_modules,
            "total_classes": sorted(total_schema_classes),
            "count": len(total_schema_classes)
        }
        print(f"   [STATS] Schemas 總計: {len(total_schema_classes)} 個類別")
    
    # 分析 enums/ 資料夾
    enums_dir = aiva_common / "enums"
    if enums_dir.exists():
        enum_modules = {}
        total_enum_classes = set()
        
        for py_file in enums_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                classes = re.findall(r'^class (\w+)\(', content, re.MULTILINE)
                enum_modules[py_file.stem] = {
                    "classes": sorted(set(classes)),
                    "size": py_file.stat().st_size,
                    "lines": len(content.split('\n'))
                }
                total_enum_classes.update(classes)
                print(f"   [U+1F4C4] {py_file.name:20s}: {len(classes):3d} 類別, {py_file.stat().st_size:>8,} bytes")
            except Exception as e:
                print(f"   [FAIL] {py_file.name:20s}: 讀取失敗 - {e}")
        
        new_structure["enums"] = {
            "modules": enum_modules,
            "total_classes": sorted(total_enum_classes),
            "count": len(total_enum_classes)
        }
        print(f"   [STATS] Enums 總計: {len(total_enum_classes)} 個類別")
    
    results["file_structure"] = new_structure
    
    # ========================================================================
    # 3. 對比分析 - 檢查遷移完整性
    # ========================================================================
    print("\n[RELOAD] 步驟 3: 遷移完整性對比")
    print("-" * 50)
    
    # Schemas 對比
    if "schemas" in new_structure:
        new_schemas = set(new_structure["schemas"]["total_classes"])
        missing_schemas = all_old_schemas - new_schemas
        extra_schemas = new_schemas - all_old_schemas
        
        print(f"[LIST] Schemas 對比:")
        print(f"   舊檔案: {len(all_old_schemas)} 個")
        print(f"   新模組: {len(new_schemas)} 個")
        
        if missing_schemas:
            print(f"   [WARN]  缺失: {len(missing_schemas)} 個")
            for cls in sorted(missing_schemas)[:10]:  # 只顯示前10個
                print(f"      - {cls}")
            if len(missing_schemas) > 10:
                print(f"      ... 還有 {len(missing_schemas) - 10} 個")
        
        if extra_schemas:
            print(f"   [U+2795] 新增: {len(extra_schemas)} 個")
            for cls in sorted(extra_schemas)[:5]:
                print(f"      - {cls}")
        
        if not missing_schemas and not extra_schemas:
            print(f"   [OK] 完全匹配！")
        
        results["migration_status"]["schemas"] = {
            "old_count": len(all_old_schemas),
            "new_count": len(new_schemas),
            "missing": sorted(missing_schemas),
            "extra": sorted(extra_schemas),
            "complete": len(missing_schemas) == 0
        }
    
    # Enums 對比
    if "enums" in new_structure:
        new_enums = set(new_structure["enums"]["total_classes"])
        missing_enums = all_old_enums - new_enums
        extra_enums = new_enums - all_old_enums
        
        print(f"\n[LIST] Enums 對比:")
        print(f"   舊檔案: {len(all_old_enums)} 個")
        print(f"   新模組: {len(new_enums)} 個")
        
        if missing_enums:
            print(f"   [WARN]  缺失: {len(missing_enums)} 個")
            for cls in sorted(missing_enums):
                print(f"      - {cls}")
        
        if extra_enums:
            print(f"   [U+2795] 新增: {len(extra_enums)} 個")
            for cls in sorted(extra_enums):
                print(f"      - {cls}")
        
        if not missing_enums and not extra_enums:
            print(f"   [OK] 完全匹配！")
        
        results["migration_status"]["enums"] = {
            "old_count": len(all_old_enums),
            "new_count": len(new_enums),
            "missing": sorted(missing_enums),
            "extra": sorted(extra_enums),
            "complete": len(missing_enums) == 0
        }
    
    # ========================================================================
    # 4. 實際導入測試
    # ========================================================================
    print("\n[TEST] 步驟 4: 實際導入測試")
    print("-" * 50)
    
    import sys
    sys.path.insert(0, str(project_root / "services"))
    
    import_tests = []
    
    # 測試基本模組導入
    try:
        from aiva_common import schemas, enums
        schema_exported = len([n for n in dir(schemas) if not n.startswith('_') and n[0].isupper()])
        enum_exported = len([n for n in dir(enums) if not n.startswith('_') and n[0].isupper()])
        
        import_tests.append({
            "test": "基本模組導入",
            "status": "[OK] PASS",
            "detail": f"schemas: {schema_exported}, enums: {enum_exported}"
        })
        print(f"[OK] 基本模組導入成功")
        print(f"   導出 schemas: {schema_exported} 個")  
        print(f"   導出 enums: {enum_exported} 個")
        
    except Exception as e:
        import_tests.append({
            "test": "基本模組導入", 
            "status": "[FAIL] FAIL",
            "detail": str(e)
        })
        print(f"[FAIL] 基本模組導入失敗: {e}")
    
    # 測試具體類別導入
    test_classes = [
        ("aiva_common.schemas", "ScanStartPayload"),
        ("aiva_common.schemas", "FunctionTaskPayload"),  
        ("aiva_common.schemas", "FindingPayload"),
        ("aiva_common.schemas", "MessageHeader"),
        ("aiva_common.enums", "ModuleName"),
        ("aiva_common.enums", "VulnerabilityType"),
        ("aiva_common.enums", "Severity"),
        ("aiva_common.enums", "Topic"),
    ]
    
    for module, class_name in test_classes:
        try:
            exec(f"from {module} import {class_name}")
            import_tests.append({
                "test": f"{module}.{class_name}",
                "status": "[OK] PASS", 
                "detail": "導入成功"
            })
            print(f"[OK] {module}.{class_name}")
        except Exception as e:
            import_tests.append({
                "test": f"{module}.{class_name}",
                "status": "[FAIL] FAIL",
                "detail": str(e)
            })
            print(f"[FAIL] {module}.{class_name}: {e}")
    
    results["validation"]["import_tests"] = import_tests
    
    # ========================================================================
    # 5. 檔案大小和效率分析
    # ========================================================================
    print("\n[STATS] 步驟 5: 檔案大小和效率分析")
    print("-" * 50)
    
    old_total_size = sum(data["size"] for data in old_data.values())
    
    new_total_size = 0
    if schemas_dir.exists():
        new_total_size += sum(f.stat().st_size for f in schemas_dir.glob("*.py"))
    if enums_dir.exists():
        new_total_size += sum(f.stat().st_size for f in enums_dir.glob("*.py"))
    
    print(f"檔案大小對比:")
    print(f"   舊檔案總大小: {old_total_size:>10,} bytes ({old_total_size/1024:.1f} KB)")
    print(f"   新檔案總大小: {new_total_size:>10,} bytes ({new_total_size/1024:.1f} KB)")
    
    size_diff = new_total_size - old_total_size
    if size_diff > 0:
        print(f"   [U+1F4C8] 增加: {size_diff:>10,} bytes ({size_diff/1024:.1f} KB)")
    elif size_diff < 0:
        print(f"   [U+1F4C9] 減少: {abs(size_diff):>10,} bytes ({abs(size_diff)/1024:.1f} KB)")
    else:
        print(f"   [STATS] 大小相同")
    
    # ========================================================================
    # 6. 總結報告
    # ========================================================================
    print("\n[TARGET] 步驟 6: 遷移完整性總結")
    print("=" * 100)
    
    schemas_complete = results["migration_status"].get("schemas", {}).get("complete", False)
    enums_complete = results["migration_status"].get("enums", {}).get("complete", False)
    
    passed_tests = len([t for t in import_tests if "[OK]" in t["status"]])
    total_tests = len(import_tests)
    
    print(f"""
[SEARCH] 遷移分析結果:

[U+1F4C4] Schemas 遷移:
   - 舊檔案類別數: {len(all_old_schemas)}
   - 新模組類別數: {len(new_structure.get('schemas', {}).get('total_classes', []))}
   - 遷移狀態: {'[OK] 完整' if schemas_complete else '[WARN] 不完整'}

[U+1F4C4] Enums 遷移:  
   - 舊檔案類別數: {len(all_old_enums)}
   - 新模組類別數: {len(new_structure.get('enums', {}).get('total_classes', []))}
   - 遷移狀態: {'[OK] 完整' if enums_complete else '[WARN] 不完整'}

[TEST] 導入測試:
   - 通過: {passed_tests}/{total_tests}
   - 狀態: {'[OK] 全部通過' if passed_tests == total_tests else '[WARN] 部分失敗'}

[STATS] 整體評估:
   - 結構完整性: {'[OK] 良好' if schemas_complete and enums_complete else '[WARN] 需要關注'}
   - 功能可用性: {'[OK] 良好' if passed_tests >= total_tests * 0.8 else '[WARN] 需要修復'}
   - 建議行動: {'[SUCCESS] 可以刪除舊檔案' if schemas_complete and enums_complete and passed_tests == total_tests else '[CONFIG] 需要進一步修復'}
""")
    
    # 儲存詳細報告
    report_file = project_root / "_out" / "migration_completeness_report.json"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"[LIST] 詳細報告已儲存至: {report_file}")
    print("=" * 100)
    
    return {
        "schemas_complete": schemas_complete,
        "enums_complete": enums_complete, 
        "tests_passed": passed_tests == total_tests,
        "ready_for_cleanup": schemas_complete and enums_complete and passed_tests == total_tests
    }

if __name__ == "__main__":
    result = analyze_migration_completeness()