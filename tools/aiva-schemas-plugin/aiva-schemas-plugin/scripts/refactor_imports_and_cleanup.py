#!/usr/bin/env python3
"""
refactor_imports_and_cleanup.py

目的：
1) 將專案中散落的 `schemas.py` 匯入統一改為 `aiva_schemas_plugin`；
2) 刪除「非 aiva_common 下」的其他 schemas.py 檔案。

使用：
    python scripts/refactor_imports_and_cleanup.py --repo-root ./services --dry-run
    python scripts/refactor_imports_and_cleanup.py --repo-root ./services

注意：請先把整個倉庫備份或在 Git 乾淨工作樹中執行。
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import re
import shutil
from typing import Iterable, List, Tuple

CANONICAL_IMPORT = "aiva_schemas_plugin"  # 統一對外匯入入口
AIVA_COMMON_SCHEMAS = "aiva_common.schemas"

# 這份清單來自倉庫實際掃描（壓縮檔的目錄名列表），僅刪除這些位置的 schemas.py
SCHEMAS_FILES_TO_REMOVE = [
    "services/core/aiva_core/schemas.py",
    "services/function/function_idor/aiva_func_idor/schemas.py",
    "services/function/function_postex/schemas.py",
    "services/function/function_sqli/aiva_func_sqli/schemas.py",
    "services/function/function_ssrf/aiva_func_ssrf/schemas.py",
    "services/function/function_xss/aiva_func_xss/schemas.py",
    "services/scan/aiva_scan/schemas.py",
]

# 允許的副檔名
PY_EXTS = {".py"}


def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # 排除常見的虛擬環境或快取
        if any(part in {".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"} for part in p.parts):
            continue
        yield p


class ImportRewriter(ast.NodeTransformer):
    """
    AST 轉換：把 imports 改到 aiva_schemas_plugin
    """

    def visit_Import(self, node: ast.Import) -> ast.AST:
        new_names = []
        changed = False
        for alias in node.names:
            name = alias.name  # 例如 "services.scan.schemas"
            if name.endswith(".schemas") and not name.startswith(AIVA_COMMON_SCHEMAS):
                # 轉成： import aiva_schemas_plugin as schemas
                changed = True
                asname = alias.asname or "schemas"
                new_names.append(ast.alias(name=CANONICAL_IMPORT, asname=asname))
            else:
                new_names.append(alias)
        if changed:
            node.names = new_names
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        mod = node.module  # 可能是 None（相對 import）或類似 "package.schemas"
        if mod is None:
            # 相對 import： from .schemas import X -> from aiva_schemas_plugin import X
            return ast.ImportFrom(module=CANONICAL_IMPORT, names=node.names, level=0)

        if mod.endswith(".schemas") and not mod.startswith(AIVA_COMMON_SCHEMAS):
            # from package.schemas import X -> from aiva_schemas_plugin import X
            return ast.ImportFrom(module=CANONICAL_IMPORT, names=node.names, level=0)

        return node


def rewrite_file(path: Path, backup_dir: Path) -> Tuple[bool, str]:
    original = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(original)
    except SyntaxError:
        return False, "SKIP (syntax error)"

    new_tree = ImportRewriter().visit(tree)
    ast.fix_missing_locations(new_tree)
    new_code = compile(new_tree, filename=str(path), mode="exec")
    # 反組譯成原始碼：使用 ast.unparse (py>=3.9)
    new_src = ast.unparse(new_tree)

    if new_src != original:
        # 備份
        rel = path.relative_to(backup_dir.parent)
        bkp_path = backup_dir / rel
        bkp_path.parent.mkdir(parents=True, exist_ok=True)
        bkp_path.write_text(original, encoding="utf-8")

        path.write_text(new_src, encoding="utf-8")
        return True, "REWRITTEN"
    return False, "UNCHANGED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=str, required=True, help="專案根目錄（例如 ./services）")
    parser.add_argument("--dry-run", action="store_true", help="只輸出將要修改/刪除的項目，不實際動作")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if not repo_root.exists():
        raise SystemExit(f"repo-root not found: {repo_root}")

    backup_dir = repo_root.parent / ".schemas_refactor_backup"
    print(f"[i] Repo root: {repo_root}")
    print(f"[i] Backup dir: {backup_dir}")

    changed_files: List[Path] = []
    for py in iter_py_files(repo_root):
        changed, status = rewrite_file(py, backup_dir)
        if changed:
            changed_files.append(py)
        print(f"{status:10s}  {py}")

    # 刪除多餘 schemas.py（保留 aiva_common/schemas.py）
    removed = []
    for rel in SCHEMAS_FILES_TO_REMOVE:
        target = (repo_root.parent / rel).resolve()
        if target.exists():
            if args.dry_run:
                print(f"REMOVE (dry-run): {target}")
            else:
                # 備份
                bkp_path = backup_dir / target.relative_to(repo_root.parent)
                bkp_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, bkp_path)
                target.unlink()
                removed.append(target)

    print("\\n=== SUMMARY ===")
    print(f"Rewritten files: {len(changed_files)}")
    print(f"Removed schemas.py: {len(removed)}")
    print("Done.")

if __name__ == "__main__":
    main()
