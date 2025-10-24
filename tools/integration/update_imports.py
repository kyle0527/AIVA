#!/usr/bin/env python3
"""
更新所有模組的 import 路徑，將 aiva_common 改為 services.aiva_common
"""

from pathlib import Path
import re


def update_import_in_file(file_path: Path) -> bool:
    """更新單個檔案中的 import 路徑"""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # 記錄原始內容
        original_content = content

        # 更新 import 語句
        # from aiva_common -> from services.aiva_common
        content = re.sub(r"from aiva_common\.", "from services.aiva_common.", content)

        # import aiva_common -> import services.aiva_common
        content = re.sub(
            r"import aiva_common\.", "import services.aiva_common.", content
        )

        # 處理 from aiva_common import 的情況
        content = re.sub(
            r"from aiva_common import", "from services.aiva_common import", content
        )

        # 如果有變更，寫回檔案
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"[已] 已更新: {file_path}")
            return True
        else:
            return False

    except Exception as e:
        print(f"[失敗] 更新失敗 {file_path}: {e}")
        return False


def main():
    """主函數"""
    project_root = Path(__file__).parent.parent.parent
    services_dir = project_root / "services"

    # 需要更新的目錄
    directories_to_update = [
        services_dir / "scan",
        services_dir / "core",
        services_dir / "function",
        services_dir / "integration",
    ]

    updated_files = []

    for directory in directories_to_update:
        if not directory.exists():
            print(f"[警告] 目錄不存在: {directory}")
            continue

        print(f"\n[目錄] 處理目錄: {directory}")

        # 遞歸查找所有 Python 檔案
        for py_file in directory.rglob("*.py"):
            if update_import_in_file(py_file):
                updated_files.append(py_file)

    print(f"\n[完成] 更新完成！共更新了 {len(updated_files)} 個檔案")

    if updated_files:
        print("\n更新的檔案列表:")
        for file_path in updated_files:
            print(f"  - {file_path}")


if __name__ == "__main__":
    main()
