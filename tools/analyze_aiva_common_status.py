"""
全面分析 aiva_common 重構後的狀況
包括：
1. 現有結構分析
2. 導入路徑檢查
3. 相依性分析
4. 潛在問題識別
5. 未來需求評估
"""

import sys
import ast
from pathlib import Path
from collections import defaultdict
import json

# 設定路徑
aiva_common_path = Path(__file__).parent.parent / "services" / "aiva_common"
sys.path.insert(0, str(aiva_common_path.parent))

print("=" * 80)
print("AIVA Common 重構分析報告")
print("=" * 80)

# ============================================================================
# 1. 結構分析
# ============================================================================
print("\n" + "=" * 80)
print("1[U+FE0F][U+20E3]  目前結構分析")
print("=" * 80)

structure = {
    "schemas/": {
        "files": list((aiva_common_path / "schemas").glob("*.py")) if (aiva_common_path / "schemas").exists() else [],
        "purpose": "Schema 定義 - 數據驗證和序列化"
    },
    "enums/": {
        "files": list((aiva_common_path / "enums").glob("*.py")) if (aiva_common_path / "enums").exists() else [],
        "purpose": "枚舉定義 - 常量和類型"
    }
}

for folder, info in structure.items():
    print(f"\n[U+1F4C1] {folder}")
    print(f"   用途: {info['purpose']}")
    print(f"   檔案數量: {len(info['files'])}")
    for f in info['files']:
        size = f.stat().st_size
        print(f"   - {f.name:30s} ({size:>6,} bytes)")

# ============================================================================
# 2. 導入測試 - 檢查向後相容性
# ============================================================================
print("\n" + "=" * 80)
print("2[U+FE0F][U+20E3]  導入路徑相容性檢查")
print("=" * 80)

import_tests = {
    "schemas 基本導入": [
        "from aiva_common.schemas import TaskSchema",
        "from aiva_common.schemas import FindingSchema",
        "from aiva_common.schemas import MessageSchema",
    ],
    "enums 基本導入": [
        "from aiva_common.enums import ModuleName",
        "from aiva_common.enums import Severity",
        "from aiva_common.enums import Topic",
    ],
    "schemas 子模組導入": [
        "from aiva_common.schemas.tasks import TaskSchema",
        "from aiva_common.schemas.findings import FindingSchema",
    ],
    "enums 子模組導入": [
        "from aiva_common.enums.modules import ModuleName",
        "from aiva_common.enums.common import Severity",
    ],
    "混合導入": [
        "from aiva_common.schemas import TaskSchema",
        "from aiva_common.enums import TaskStatus",
    ]
}

results = {"passed": 0, "failed": 0, "details": []}

for category, imports in import_tests.items():
    print(f"\n[SEARCH] 測試類別: {category}")
    for import_stmt in imports:
        try:
            exec(import_stmt)
            print(f"   [OK] {import_stmt}")
            results["passed"] += 1
            results["details"].append({"test": import_stmt, "status": "[OK] PASS"})
        except Exception as e:
            print(f"   [FAIL] {import_stmt}")
            print(f"      錯誤: {str(e)[:60]}")
            results["failed"] += 1
            results["details"].append({"test": import_stmt, "status": f"[FAIL] FAIL: {str(e)[:60]}"})

print(f"\n[STATS] 測試結果: {results['passed']} 通過, {results['failed']} 失敗")

# ============================================================================
# 3. 分析實際使用情況 - 檢查其他模組如何使用 aiva_common
# ============================================================================
print("\n" + "=" * 80)
print("3[U+FE0F][U+20E3]  實際使用情況分析")
print("=" * 80)

services_path = Path(__file__).parent.parent / "services"
usage_analysis = defaultdict(lambda: {"schemas": set(), "enums": set(), "files": set()})

print("\n掃描所有服務中的導入語句...")

for service_dir in services_path.iterdir():
    if not service_dir.is_dir() or service_dir.name == "aiva_common":
        continue
    
    # 掃描 Python 檔案
    for py_file in service_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
            
            # 檢查是否使用 aiva_common
            if "from aiva_common" in content or "import aiva_common" in content:
                # 解析 AST
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ImportFrom):
                            if node.module and "aiva_common" in node.module:
                                for alias in node.names:
                                    if "schemas" in node.module:
                                        usage_analysis[service_dir.name]["schemas"].add(alias.name)
                                    elif "enums" in node.module:
                                        usage_analysis[service_dir.name]["enums"].add(alias.name)
                                usage_analysis[service_dir.name]["files"].add(str(py_file.relative_to(services_path)))
                except SyntaxError:
                    pass  # 忽略語法錯誤
        except Exception:
            pass

if usage_analysis:
    for service, data in sorted(usage_analysis.items()):
        print(f"\n[U+1F4E6] {service}")
        print(f"   檔案數: {len(data['files'])}")
        if data['schemas']:
            print(f"   使用的 Schemas: {len(data['schemas'])} 個")
            print(f"      {', '.join(sorted(list(data['schemas']))[:5])}...")
        if data['enums']:
            print(f"   使用的 Enums: {len(data['enums'])} 個")
            print(f"      {', '.join(sorted(list(data['enums']))[:5])}...")
else:
    print("\n[WARN]  未在其他服務中找到 aiva_common 的使用")

# ============================================================================
# 4. 檢查舊檔案狀態
# ============================================================================
print("\n" + "=" * 80)
print("4[U+FE0F][U+20E3]  舊檔案狀態檢查")
print("=" * 80)

old_files_to_check = [
    "schemas.py",
    "enums.py",
    "ai_schemas.py",
]

old_files_found = []
for old_file in old_files_to_check:
    file_path = aiva_common_path / old_file
    if file_path.exists():
        size = file_path.stat().st_size
        old_files_found.append({"name": old_file, "size": size, "path": str(file_path)})
        print(f"[WARN]  {old_file:20s} 仍然存在 ({size:>10,} bytes)")
    else:
        print(f"[OK] {old_file:20s} 已刪除")

# 檢查備份檔案
backup_files = list(aiva_common_path.glob("*.backup*")) + list(aiva_common_path.glob("*_backup*"))
if backup_files:
    print(f"\n[WARN]  發現 {len(backup_files)} 個備份檔案:")
    for bf in backup_files:
        print(f"   - {bf.name} ({bf.stat().st_size:>10,} bytes)")
else:
    print("\n[OK] 沒有備份檔案")

# ============================================================================
# 5. 潛在問題識別
# ============================================================================
print("\n" + "=" * 80)
print("5[U+FE0F][U+20E3]  潛在問題識別")
print("=" * 80)

issues = []

# 檢查是否有循環導入風險
print("\n[SEARCH] 檢查循環導入風險...")
if (aiva_common_path / "schemas" / "__init__.py").exists():
    init_content = (aiva_common_path / "schemas" / "__init__.py").read_text(encoding="utf-8")
    if "from aiva_common.enums import" in init_content:
        issues.append({
            "type": "循環導入風險",
            "severity": "MEDIUM",
            "detail": "schemas/__init__.py 可能導入 enums，需確認沒有循環依賴"
        })
        print("   [WARN]  schemas 中導入了 enums")
    else:
        print("   [OK] schemas 沒有導入 enums")

# 檢查 __all__ 列表完整性
print("\n[SEARCH] 檢查 __all__ 列表...")
for module in ["schemas", "enums"]:
    init_file = aiva_common_path / module / "__init__.py"
    if init_file.exists():
        content = init_file.read_text(encoding="utf-8")
        if "__all__" not in content:
            issues.append({
                "type": "__all__ 缺失",
                "severity": "LOW",
                "detail": f"{module}/__init__.py 缺少 __all__ 列表"
            })
            print(f"   [WARN]  {module}/__init__.py 缺少 __all__")
        else:
            print(f"   [OK] {module}/__init__.py 有 __all__")

# 檢查舊檔案是否會造成衝突
if old_files_found:
    issues.append({
        "type": "舊檔案衝突",
        "severity": "HIGH",
        "detail": f"發現 {len(old_files_found)} 個舊檔案可能造成導入衝突"
    })

if issues:
    print(f"\n[WARN]  發現 {len(issues)} 個潛在問題:")
    for issue in issues:
        severity_icon = {"HIGH": "[RED]", "MEDIUM": "[YELLOW]", "LOW": "[U+1F7E2]"}
        print(f"   {severity_icon[issue['severity']]} [{issue['severity']}] {issue['type']}")
        print(f"      {issue['detail']}")
else:
    print("\n[OK] 沒有發現潛在問題")

# ============================================================================
# 6. 未來需求評估
# ============================================================================
print("\n" + "=" * 80)
print("6[U+FE0F][U+20E3]  未來需求評估")
print("=" * 80)

recommendations = []

# 根據現狀提出建議
if old_files_found:
    recommendations.append({
        "priority": "HIGH",
        "category": "清理",
        "action": f"刪除 {len(old_files_found)} 個舊檔案",
        "reason": "避免導入衝突和混淆"
    })

if results["failed"] > 0:
    recommendations.append({
        "priority": "HIGH",
        "category": "修復",
        "action": "修復失敗的導入測試",
        "reason": "確保向後相容性"
    })

# 建議新增的功能
recommendations.extend([
    {
        "priority": "MEDIUM",
        "category": "文檔",
        "action": "創建遷移指南",
        "reason": "幫助開發者了解新結構和遷移路徑"
    },
    {
        "priority": "MEDIUM",
        "category": "測試",
        "action": "建立自動化整合測試",
        "reason": "確保 schemas 和 enums 在所有服務中正常工作"
    },
    {
        "priority": "LOW",
        "category": "優化",
        "action": "考慮使用 pydantic v2 的新功能",
        "reason": "提升性能和開發體驗"
    },
    {
        "priority": "LOW",
        "category": "擴展",
        "action": "考慮新增 validators/ 模組",
        "reason": "統一管理自訂驗證邏輯"
    }
])

priority_order = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
for rec in sorted(recommendations, key=lambda x: priority_order[x["priority"]]):
    priority_icon = {"HIGH": "[RED]", "MEDIUM": "[YELLOW]", "LOW": "[U+1F7E2]"}
    print(f"\n{priority_icon[rec['priority']]} [{rec['priority']}] {rec['category']}: {rec['action']}")
    print(f"   理由: {rec['reason']}")

# ============================================================================
# 7. 生成報告檔案
# ============================================================================
print("\n" + "=" * 80)
print("7[U+FE0F][U+20E3]  生成詳細報告")
print("=" * 80)

report = {
    "timestamp": "2025-10-16",
    "structure": {k: {"file_count": len(v["files"]), "purpose": v["purpose"]} for k, v in structure.items()},
    "import_tests": results,
    "usage_analysis": {k: {
        "schemas_count": len(v["schemas"]),
        "enums_count": len(v["enums"]),
        "files_count": len(v["files"])
    } for k, v in usage_analysis.items()},
    "old_files": old_files_found,
    "issues": issues,
    "recommendations": recommendations
}

report_file = Path(__file__).parent.parent / "_out" / "aiva_common_analysis_report.json"
report_file.parent.mkdir(exist_ok=True)
with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n[OK] 詳細報告已儲存至: {report_file}")

# ============================================================================
# 總結
# ============================================================================
print("\n" + "=" * 80)
print("[LIST] 分析總結")
print("=" * 80)

print(f"""
[OK] 結構完整性: {'通過' if len(structure['schemas/']['files']) > 0 and len(structure['enums/']['files']) > 0 else '失敗'}
{'[OK]' if results['failed'] == 0 else '[WARN]'} 導入測試: {results['passed']}/{results['passed'] + results['failed']} 通過
{'[WARN]' if old_files_found else '[OK]'} 舊檔案: {len(old_files_found)} 個需要處理
{'[WARN]' if issues else '[OK]'} 潛在問題: {len(issues)} 個
[TARGET] 建議行動: {len([r for r in recommendations if r['priority'] == 'HIGH'])} 個高優先級任務

下一步建議:
1. 執行整合測試確保所有服務正常
2. 刪除舊檔案避免衝突
3. 更新文檔說明新結構
""")

print("=" * 80)
