"""
快速修復所有 schemas 模組的導入問題
"""

import re
from pathlib import Path

aiva_common = Path(__file__).parent.parent / "services" / "aiva_common"

print("=" * 80)
print("修復所有 schemas 模組導入問題")
print("=" * 80)

# 需要修復的檔案和它們缺少的導入
fixes = {
    "enhanced.py": {
        "missing_imports": [
            "HttpUrl",
            # 從其他模組導入
            "Target", "FindingEvidence", "FindingImpact", "FindingRecommendation", 
            "SARIFResult", "SARIFLocation", "EnhancedVulnerability", 
            "RiskFactor", "CVSSv3Metrics", "TaskDependency"
        ],
        "from_imports": {
            ".base": ["RiskFactor", "TaskDependency"],
            ".findings": ["Target", "FindingEvidence", "FindingImpact", "FindingRecommendation"],
            ".references": ["SARIFResult", "SARIFLocation"],
            ".ai": ["EnhancedVulnerability", "CVSSv3Metrics"]
        }
    }
}

# 檢查哪些類別存在於哪個模組
def find_class_in_modules(class_name, schemas_dir):
    for py_file in schemas_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            if re.search(rf'^class {class_name}\(', content, re.MULTILINE):
                return py_file.stem
        except:
            continue
    return None

schemas_dir = aiva_common / "schemas"

print("\n[SEARCH] 查找缺失類別的位置:")
all_missing = set()
for file_info in fixes.values():
    all_missing.update(file_info["missing_imports"])

class_locations = {}
for class_name in all_missing:
    if class_name == "HttpUrl":
        continue  # 這是 pydantic 的
    location = find_class_in_modules(class_name, schemas_dir)
    if location:
        class_locations[class_name] = location
        print(f"   {class_name:25s} -> {location}.py")
    else:
        print(f"   {class_name:25s} -> [FAIL] 找不到")

# 修復 enhanced.py
print(f"\n[CONFIG] 修復 enhanced.py...")
enhanced_file = schemas_dir / "enhanced.py"
content = enhanced_file.read_text(encoding="utf-8")

# 找到導入區塊並替換
import_section = """from pydantic import BaseModel, Field, field_validator

from ..enums import Confidence, ModuleName, Severity, TestStatus, VulnerabilityType"""

new_import_section = """from pydantic import BaseModel, Field, HttpUrl, field_validator

from ..enums import Confidence, ModuleName, Severity, TestStatus, VulnerabilityType
from .ai import CVSSv3Metrics, EnhancedVulnerability
from .base import RiskFactor, TaskDependency
from .findings import FindingEvidence, FindingImpact, FindingRecommendation, Target
from .references import SARIFLocation, SARIFResult"""

content = content.replace(import_section, new_import_section)
enhanced_file.write_text(content, encoding="utf-8")
print("   [OK] enhanced.py 導入已修復")

# 檢查其他可能有問題的檔案
print(f"\n[SEARCH] 檢查其他檔案的導入問題:")
problem_files = []

for py_file in schemas_dir.glob("*.py"):
    if py_file.name in ["__init__.py", "enhanced.py"]:
        continue
        
    try:
        content = py_file.read_text(encoding="utf-8")
        # 檢查是否有 field_validator 但沒有導入
        if "field_validator" in content and "from pydantic import" in content:
            pydantic_import = re.search(r'from pydantic import ([^\\n]+)', content)
            if pydantic_import and "field_validator" not in pydantic_import.group(1):
                problem_files.append((py_file.name, "缺少 field_validator"))
                
        # 檢查是否有 HttpUrl 但沒有導入
        if "HttpUrl" in content and "from pydantic import" in content:
            pydantic_import = re.search(r'from pydantic import ([^\\n]+)', content)
            if pydantic_import and "HttpUrl" not in pydantic_import.group(1):
                problem_files.append((py_file.name, "缺少 HttpUrl"))
                
    except Exception as e:
        print(f"   [WARN] 檢查 {py_file.name} 時出錯: {e}")

for filename, issue in problem_files:
    print(f"   [WARN] {filename}: {issue}")

# 快速修復常見問題
common_fixes = {
    "tasks.py": {
        "old": "from pydantic import BaseModel, Field, HttpUrl, field_validator",
        "new": "from pydantic import BaseModel, Field, HttpUrl, field_validator"
    },
    "findings.py": {
        "pattern": r'from pydantic import BaseModel, Field([^\\n]*)',
        "replacement": "from pydantic import BaseModel, Field, HttpUrl, field_validator"
    }
}

for filename, fix_info in common_fixes.items():
    file_path = schemas_dir / filename
    if file_path.exists():
        content = file_path.read_text(encoding="utf-8")
        
        if "pattern" in fix_info:
            content = re.sub(fix_info["pattern"], fix_info["replacement"], content)
        else:
            content = content.replace(fix_info["old"], fix_info["new"])
            
        file_path.write_text(content, encoding="utf-8")
        print(f"   [OK] {filename} 已修復")

print(f"\n[OK] 所有已知問題已修復")
print("=" * 80)