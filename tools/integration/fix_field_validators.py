"""
批量修正 Pydantic @field_validator 方法簽名問題

將所有使用 @field_validator 的方法從：
    @field_validator("field_name")
    def validate_field(self, value):
        ...

修正為：
    @field_validator("field_name")
    @classmethod
    def validate_field(cls, value):
        ...
"""

import re
from pathlib import Path

# 使用相對路徑，從項目根目錄計算
project_root = Path(__file__).parent.parent.parent
schemas_dir = project_root / "services" / "aiva_common" / "schemas"

# 受影響的檔案列表
affected_files = [
    str(schemas_dir / "ai.py"),
    str(schemas_dir / "enhanced.py"),
    str(schemas_dir / "findings.py"),
    str(schemas_dir / "system.py"),
    str(schemas_dir / "tasks.py"),
    str(schemas_dir / "telemetry.py"),
]

def fix_field_validator(file_path: str) -> tuple[int, list[str]]:
    """修正檔案中的 field_validator 方法簽名"""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    original_content = content

    # 匹配 @field_validator 後面沒有 @classmethod 的情況
    pattern = r'(@field_validator\([^)]+\))\s+def\s+(\w+)\s*\(\s*cls\s*,'

    # 替換為正確的格式（添加 @classmethod）
    replacement = r'\1\n    @classmethod\n    def \2(cls,'

    # 執行替換
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # 計算修改次數
    fixes = len(re.findall(pattern, original_content))

    # 找出所有被修正的方法名稱
    fixed_methods = re.findall(r'@field_validator.*?\n\s+def\s+(\w+)', original_content)

    if content != original_content:
        path.write_text(content, encoding="utf-8")
        return fixes, fixed_methods

    return 0, []

def main():
    print("開始修正 @field_validator 方法簽名...\n")
    
    total_fixes = 0
    for file_path in affected_files:
        fixes, methods = fix_field_validator(file_path)
        if fixes > 0:
            print(f"✅ {Path(file_path).name}: 修正了 {fixes} 個方法")
            for method in methods:
                print(f"   - {method}")
            total_fixes += fixes
        else:
            print(f"⚪ {Path(file_path).name}: 無需修正")
    
    print(f"\n總計修正: {total_fixes} 個方法")
    
    if total_fixes > 0:
        print("\n✅ 所有修正已完成！")
        print("建議執行以下命令驗證：")
        print("  python -m pylint services/aiva_common/schemas/ | grep E0213")
    else:
        print("\n⚪ 所有檔案已經符合規範")

if __name__ == "__main__":
    main()
