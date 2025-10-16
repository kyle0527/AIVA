#!/usr/bin/env python3
"""
copy_into_plugin.py

如果你希望 **插件完全自含**（不再依賴 aiva_common.schemas），可執行本腳本：
它會將現有倉庫中的 `services/aiva_common/schemas.py` 內容複製到
`aiva_schemas_plugin` 的 `__init__.py`，並保留 `__all__` 等公開介面。

使用：
    python scripts/copy_into_plugin.py --repo-root ./services
"""
from __future__ import annotations

import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, required=True, help="專案根目錄（例如 ./services）")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    src_file = repo_root / "aiva_common" / "schemas.py"
    dst_file = Path(__file__).resolve().parent.parent / "src" / "aiva_schemas_plugin" / "__init__.py"

    if not src_file.exists():
        raise SystemExit(f"找不到來源檔：{src_file}")

    content = src_file.read_text(encoding="utf-8")

    banner = (
        '"""\\n'
        'aiva_schemas_plugin - self-contained copy\\n\\n'
        '此檔案由 scripts/copy_into_plugin.py 產生，內容取自 aiva_common/schemas.py。\\n'
        '維護時請以本插件為準。\\n'
        '"""\\n\\n'
        "from __future__ import annotations\\n\\n"
    )

    # 直接覆蓋到插件 (自含版)
    dst_file.write_text(banner + content, encoding="utf-8")
    print(f"[i] 已複製 {src_file} -> {dst_file}")

if __name__ == "__main__":
    main()
