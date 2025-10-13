#!/usr/bin/env python
"""
analyze_codebase.py
-------------------
綜合程式碼分析工具 (支援多語言)
使用 CodeAnalyzer 和 py2mermaid 對整個程式碼庫進行分析
支援 Python, Go, Rust, TypeScript/JavaScript
"""

import ast
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any


class CodeAnalyzer:
    """程式碼分析器 - 獨立版本."""

    def __init__(self, codebase_path: str) -> None:
        """初始化程式碼分析器.

        Args:
            codebase_path: 程式碼庫根目錄
        """
        self.codebase_path = Path(codebase_path)

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """分析程式碼.

        Args:
            **kwargs: 工具參數
                path (str): 檔案路徑
                detailed (bool): 是否進行詳細分析 (預設 False)

        Returns:
            分析結果
        """
        path = kwargs.get("path", "")
        detailed = kwargs.get("detailed", False)

        if not path:
            return {"status": "error", "error": "缺少必需參數: path"}

        try:
            full_path = self.codebase_path / path
            content = full_path.read_text(encoding="utf-8")

            # 基本統計
            lines = content.splitlines()
            non_empty_lines = [line for line in lines if line.strip()]
            comment_lines = [
                line
                for line in lines
                if line.strip().startswith("#")
                or line.strip().startswith('"""')
                or line.strip().startswith("'''")
            ]

            result = {
                "status": "success",
                "path": path,
                "total_lines": len(lines),
                "code_lines": len(non_empty_lines),
                "comment_lines": len(comment_lines),
                "blank_lines": len(lines) - len(non_empty_lines),
            }

            # 如果要求詳細分析，使用 AST
            if detailed:
                try:
                    tree = ast.parse(content)

                    # 統計各種節點
                    imports: list[str] = []
                    functions: list[str] = []
                    classes: list[str] = []
                    async_functions: list[str] = []

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            imports.extend(alias.name for alias in node.names)
                        elif isinstance(node, ast.ImportFrom):
                            module = node.module or ""
                            imports.append(module)
                        elif isinstance(node, ast.FunctionDef):
                            functions.append(node.name)
                        elif isinstance(node, ast.AsyncFunctionDef):
                            async_functions.append(node.name)
                        elif isinstance(node, ast.ClassDef):
                            classes.append(node.name)

                    # 計算複雜度指標
                    complexity = self._calculate_complexity(tree)

                    result.update(
                        {
                            "imports": list(set(imports)),
                            "import_count": len(set(imports)),
                            "functions": functions,
                            "function_count": len(functions),
                            "async_functions": async_functions,
                            "async_function_count": len(async_functions),
                            "classes": classes,
                            "class_count": len(classes),
                            "cyclomatic_complexity": complexity,
                            "has_type_hints": self._check_type_hints(tree),
                            "has_docstrings": self._check_docstrings(tree),
                        }
                    )
                except SyntaxError as e:
                    result["syntax_error"] = str(e)
            else:
                # 簡單統計（向後兼容）
                import_count = sum(
                    1 for line in lines if line.strip().startswith("import")
                )
                function_count = sum(
                    1 for line in lines if line.strip().startswith("def ")
                )
                class_count = sum(
                    1 for line in lines if line.strip().startswith("class ")
                )

                result.update(
                    {
                        "imports": import_count,
                        "functions": function_count,
                        "classes": class_count,
                    }
                )

            return result
        except Exception as e:
            return {"status": "error", "path": path, "error": str(e)}

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """計算循環複雜度.

        Args:
            tree: AST 樹

        Returns:
            複雜度分數
        """
        complexity = 1  # 基礎複雜度

        for node in ast.walk(tree):
            # 每個決策點增加複雜度
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _check_type_hints(self, tree: ast.AST) -> bool:
        """檢查是否使用類型提示.

        Args:
            tree: AST 樹

        Returns:
            是否有類型提示
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 檢查參數類型提示
                if node.args.args:
                    for arg in node.args.args:
                        if arg.annotation is not None:
                            return True
                # 檢查返回值類型提示
                if node.returns is not None:
                    return True
        return False

    def _check_docstrings(self, tree: ast.AST) -> bool:
        """檢查是否有文檔字串.

        Args:
            tree: AST 樹

        Returns:
            是否有文檔字串
        """
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
            ):
                docstring = ast.get_docstring(node)
                if docstring:
                    return True
        return False


def analyze_directory(
    root_path: Path,
    output_dir: Path,
    ignore_patterns: list[str] | None = None,
    max_files: int = 1000,
) -> dict[str, Any]:
    """分析指定目錄下的所有 Python 檔案.

    Args:
        root_path: 要分析的根目錄
        output_dir: 輸出報告的目錄
        ignore_patterns: 要忽略的路徑模式
        max_files: 最大分析檔案數

    Returns:
        分析結果摘要
    """
    if ignore_patterns is None:
        ignore_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "_out",
            "emoji_backups",
            ".pytest_cache",
            "target",
            "build",
            "dist",
        ]

    analyzer = CodeAnalyzer(str(root_path))

    # 統計數據
    stats = {
        "total_files": 0,
        "total_lines": 0,
        "total_code_lines": 0,
        "total_comment_lines": 0,
        "total_blank_lines": 0,
        "total_functions": 0,
        "total_classes": 0,
        "total_imports": 0,
        "files_with_type_hints": 0,
        "files_with_docstrings": 0,
        "total_complexity": 0,
        "syntax_errors": [],
        "file_details": [],
        "top_complex_files": [],
        "module_analysis": defaultdict(
            lambda: {
                "files": 0,
                "lines": 0,
                "functions": 0,
                "classes": 0,
            }
        ),
    }

    # 掃描所有 Python 檔案
    py_files = []
    for py_file in root_path.rglob("*.py"):
        # 檢查是否應該忽略
        if any(pattern in str(py_file) for pattern in ignore_patterns):
            continue
        py_files.append(py_file)
        if len(py_files) >= max_files:
            break

    print(f"找到 {len(py_files)} 個 Python 檔案")

    # 分析每個檔案
    for i, py_file in enumerate(py_files, 1):
        if i % 20 == 0:
            print(f"進度: {i}/{len(py_files)}")

        rel_path = py_file.relative_to(root_path)
        result = analyzer.execute(path=str(rel_path), detailed=True)

        if result["status"] == "success":
            stats["total_files"] += 1
            stats["total_lines"] += result.get("total_lines", 0)
            stats["total_code_lines"] += result.get("code_lines", 0)
            stats["total_comment_lines"] += result.get("comment_lines", 0)
            stats["total_blank_lines"] += result.get("blank_lines", 0)
            stats["total_functions"] += result.get("function_count", 0)
            stats["total_classes"] += result.get("class_count", 0)
            stats["total_imports"] += result.get("import_count", 0)

            if result.get("has_type_hints"):
                stats["files_with_type_hints"] += 1
            if result.get("has_docstrings"):
                stats["files_with_docstrings"] += 1

            complexity = result.get("cyclomatic_complexity", 0)
            stats["total_complexity"] += complexity

            # 記錄檔案詳情
            file_detail = {
                "path": str(rel_path),
                "lines": result.get("total_lines", 0),
                "functions": result.get("function_count", 0),
                "classes": result.get("class_count", 0),
                "complexity": complexity,
                "has_type_hints": result.get("has_type_hints", False),
                "has_docstrings": result.get("has_docstrings", False),
            }
            stats["file_details"].append(file_detail)

            # 按模組統計
            parts = rel_path.parts
            if len(parts) > 0:
                module = parts[0]
                stats["module_analysis"][module]["files"] += 1
                stats["module_analysis"][module]["lines"] += result.get(
                    "total_lines", 0
                )
                stats["module_analysis"][module]["functions"] += result.get(
                    "function_count", 0
                )
                stats["module_analysis"][module]["classes"] += result.get(
                    "class_count", 0
                )

            # 檢查語法錯誤
            if "syntax_error" in result:
                stats["syntax_errors"].append(
                    {
                        "path": str(rel_path),
                        "error": result["syntax_error"],
                    }
                )

    # 找出最複雜的檔案（前 20）
    stats["file_details"].sort(key=lambda x: x["complexity"], reverse=True)
    stats["top_complex_files"] = stats["file_details"][:20]

    # 生成報告
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成 JSON 報告
    json_output = (
        output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 生成文字報告
    txt_output = (
        output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AIVA 程式碼庫分析報告\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("整體統計\n")
        f.write("-" * 80 + "\n")
        f.write(f"總檔案數: {stats['total_files']}\n")
        f.write(f"總行數: {stats['total_lines']:,}\n")
        f.write(f"  - 程式碼行: {stats['total_code_lines']:,}\n")
        f.write(f"  - 註解行: {stats['total_comment_lines']:,}\n")
        f.write(f"  - 空白行: {stats['total_blank_lines']:,}\n")
        f.write(f"總函數數: {stats['total_functions']}\n")
        f.write(f"總類別數: {stats['total_classes']}\n")
        f.write(f"總導入數: {stats['total_imports']}\n")
        f.write(
            f"平均複雜度: {stats['total_complexity'] / max(stats['total_files'], 1):.2f}\n"
        )
        f.write("\n")

        f.write("程式碼品質指標\n")
        f.write("-" * 80 + "\n")
        type_hint_pct = (
            stats["files_with_type_hints"] / max(stats["total_files"], 1)
        ) * 100
        docstring_pct = (
            stats["files_with_docstrings"] / max(stats["total_files"], 1)
        ) * 100
        f.write(
            f"類型提示覆蓋率: {type_hint_pct:.1f}% ({stats['files_with_type_hints']}/{stats['total_files']})\n"
        )
        f.write(
            f"文檔字串覆蓋率: {docstring_pct:.1f}% ({stats['files_with_docstrings']}/{stats['total_files']})\n"
        )
        f.write("\n")

        if stats["syntax_errors"]:
            f.write("語法錯誤\n")
            f.write("-" * 80 + "\n")
            for err in stats["syntax_errors"]:
                f.write(f"{err['path']}: {err['error']}\n")
            f.write("\n")

        f.write("模組分析\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'模組':<30} {'檔案數':>10} {'行數':>12} {'函數':>10} {'類別':>10}\n")
        f.write("-" * 80 + "\n")
        for module, data in sorted(stats["module_analysis"].items()):
            f.write(
                f"{module:<30} {data['files']:>10} {data['lines']:>12,} "
                f"{data['functions']:>10} {data['classes']:>10}\n"
            )
        f.write("\n")

        f.write("最複雜的檔案（前 20）\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'檔案路徑':<60} {'複雜度':>10} {'行數':>8}\n")
        f.write("-" * 80 + "\n")
        for file_info in stats["top_complex_files"]:
            path = file_info["path"]
            if len(path) > 58:
                path = "..." + path[-55:]
            f.write(
                f"{path:<60} {file_info['complexity']:>10} {file_info['lines']:>8}\n"
            )
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("報告結束\n")
        f.write("=" * 80 + "\n")

    print("\n報告已生成:")
    print(f"  JSON: {json_output}")
    print(f"  TXT: {txt_output}")

    return stats


def _analyze_file_basic(content: str, comment_prefixes: list[str]) -> dict[str, int]:
    """分析檔案基本統計資訊（統一版本）.

    Args:
        content: 檔案內容
        comment_prefixes: 註解前綴列表 (如 ["//", "/*", "#"])

    Returns:
        基本統計資訊字典，包含:
        - total_lines: 總行數
        - code_lines: 程式碼行數
        - comment_lines: 註解行數
        - blank_lines: 空白行數
    """
    lines = content.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]

    comment_lines = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in comment_prefixes):
            comment_lines.append(line)

    return {
        "total_lines": len(lines),
        "code_lines": len(non_empty_lines),
        "comment_lines": len(comment_lines),
        "blank_lines": len(lines) - len(non_empty_lines),
    }


def _count_go_elements(content: str) -> dict[str, int]:
    """計算 Go 程式碼元素.

    Args:
        content: Go 檔案內容

    Returns:
        包含 functions, structs, interfaces 計數
    """
    import re

    # 計算函數 (func xxx)
    func_pattern = r"^\s*func\s+(?:\([^)]+\)\s+)?(\w+)"
    functions = len(re.findall(func_pattern, content, re.MULTILINE))

    # 計算結構體 (type xxx struct)
    struct_pattern = r"^\s*type\s+(\w+)\s+struct\s*\{"
    structs = len(re.findall(struct_pattern, content, re.MULTILINE))

    # 計算介面 (type xxx interface)
    interface_pattern = r"^\s*type\s+(\w+)\s+interface\s*\{"
    interfaces = len(re.findall(interface_pattern, content, re.MULTILINE))

    return {
        "functions": functions,
        "structs": structs,
        "interfaces": interfaces,
    }


def _count_rust_elements(content: str) -> dict[str, int]:
    """計算 Rust 程式碼元素.

    Args:
        content: Rust 檔案內容

    Returns:
        包含 functions, structs, traits, impls 計數
    """
    import re

    # 計算函數 (fn xxx)
    func_pattern = r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)"
    functions = len(re.findall(func_pattern, content, re.MULTILINE))

    # 計算結構體 (struct xxx)
    struct_pattern = r"^\s*(?:pub\s+)?struct\s+(\w+)"
    structs = len(re.findall(struct_pattern, content, re.MULTILINE))

    # 計算 traits
    trait_pattern = r"^\s*(?:pub\s+)?trait\s+(\w+)"
    traits = len(re.findall(trait_pattern, content, re.MULTILINE))

    # 計算 impl 區塊
    impl_pattern = r"^\s*impl\s+"
    impls = len(re.findall(impl_pattern, content, re.MULTILINE))

    return {
        "functions": functions,
        "structs": structs,
        "traits": traits,
        "impls": impls,
    }


def _count_typescript_elements(content: str) -> dict[str, int]:
    """計算 TypeScript 程式碼元素.

    Args:
        content: TypeScript 檔案內容

    Returns:
        包含 functions, classes, interfaces 計數
    """
    import re

    # 計算函數 (function xxx, const xxx = () =>)
    func_pattern = r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)"
    functions = len(re.findall(func_pattern, content, re.MULTILINE))

    # 計算類別
    class_pattern = r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+(\w+)"
    classes = len(re.findall(class_pattern, content, re.MULTILINE))

    # 計算介面
    interface_pattern = r"^\s*(?:export\s+)?interface\s+(\w+)"
    interfaces = len(re.findall(interface_pattern, content, re.MULTILINE))

    # 計算類型別名
    type_pattern = r"^\s*(?:export\s+)?type\s+(\w+)\s*="
    types = len(re.findall(type_pattern, content, re.MULTILINE))

    return {
        "functions": functions,
        "classes": classes,
        "interfaces": interfaces,
        "types": types,
    }


def _generate_multilang_report(stats: dict, output_dir: Path) -> None:
    """生成多語言分析報告.

    Args:
        stats: 統計數據
        output_dir: 輸出目錄
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成 JSON 報告
    json_output = (
        output_dir
        / f"multilang_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 生成文字報告
    txt_output = (
        output_dir
        / f"multilang_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(txt_output, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("AIVA 多語言程式碼分析報告\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # 總覽
        f.write("多語言程式碼統計總覽\n")
        f.write("-" * 80 + "\n")
        for lang, data in stats.items():
            if data["total_files"] > 0:
                f.write(f"\n{lang.upper()}:\n")
                f.write(f"  檔案數: {data['total_files']}\n")
                f.write(f"  總行數: {data['total_lines']:,}\n")
                f.write(f"    - 程式碼行: {data['total_code_lines']:,}\n")
                f.write(f"    - 註解行: {data['total_comment_lines']:,}\n")
                f.write(f"    - 空白行: {data['total_blank_lines']:,}\n")

                # 語言特定統計
                if lang == "go":
                    f.write(f"  函數數: {data['total_functions']}\n")
                    f.write(f"  結構體數: {data['total_structs']}\n")
                    f.write(f"  介面數: {data['total_interfaces']}\n")
                elif lang == "rust":
                    f.write(f"  函數數: {data['total_functions']}\n")
                    f.write(f"  結構體數: {data['total_structs']}\n")
                    f.write(f"  Traits: {data['total_traits']}\n")
                    f.write(f"  Impls: {data['total_impls']}\n")
                elif lang == "typescript":
                    f.write(f"  函數數: {data['total_functions']}\n")
                    f.write(f"  類別數: {data['total_classes']}\n")
                    f.write(f"  介面數: {data['total_interfaces']}\n")
                    f.write(f"  類型別名: {data['total_types']}\n")
                elif lang == "javascript":
                    f.write(f"  函數數: {data['total_functions']}\n")
                    f.write(f"  類別數: {data['total_classes']}\n")

        # 各語言檔案詳情（前10大檔案）
        f.write("\n\n最大的檔案（各語言前 10）\n")
        f.write("-" * 80 + "\n")

        for lang, data in stats.items():
            if data["total_files"] > 0:
                f.write(f"\n{lang.upper()}:\n")
                sorted_files = sorted(
                    data["file_details"], key=lambda x: x["lines"], reverse=True
                )[:10]
                f.write(f"{'檔案路徑':<50} {'行數':>10} {'函數':>8}\n")
                f.write("-" * 70 + "\n")
                for file_info in sorted_files:
                    path = file_info["path"]
                    if len(path) > 48:
                        path = "..." + path[-45:]
                    func_key = "functions" if "functions" in file_info else "N/A"
                    func_count = file_info.get(func_key, 0) if func_key != "N/A" else 0
                    f.write(f"{path:<50} {file_info['lines']:>10} {func_count:>8}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("報告結束\n")
        f.write("=" * 80 + "\n")

    print("\n多語言分析報告已生成:")
    print(f"  JSON: {json_output}")
    print(f"  TXT: {txt_output}")


def analyze_multilang_files(
    root_path: Path,
    output_dir: Path,
    ignore_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """分析多語言程式碼檔案 (Go, Rust, TypeScript, JavaScript).

    Args:
        root_path: 要分析的根目錄
        output_dir: 輸出報告的目錄
        ignore_patterns: 要忽略的路徑模式

    Returns:
        多語言分析結果摘要 (schema 統一版本)
    """
    if ignore_patterns is None:
        ignore_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "_out",
            "emoji_backups",
            ".pytest_cache",
            "target",
            "build",
            "dist",
        ]

    # 統一 schema 結構（對齊 analyze_directory）
    multilang_stats = {
        "go": {
            "total_files": 0,
            "total_lines": 0,
            "total_code_lines": 0,
            "total_comment_lines": 0,
            "total_blank_lines": 0,
            "total_functions": 0,
            "total_structs": 0,
            "total_interfaces": 0,
            "file_details": [],
        },
        "rust": {
            "total_files": 0,
            "total_lines": 0,
            "total_code_lines": 0,
            "total_comment_lines": 0,
            "total_blank_lines": 0,
            "total_functions": 0,
            "total_structs": 0,
            "total_traits": 0,
            "total_impls": 0,
            "file_details": [],
        },
        "typescript": {
            "total_files": 0,
            "total_lines": 0,
            "total_code_lines": 0,
            "total_comment_lines": 0,
            "total_blank_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "total_interfaces": 0,
            "total_types": 0,
            "file_details": [],
        },
        "javascript": {
            "total_files": 0,
            "total_lines": 0,
            "total_code_lines": 0,
            "total_comment_lines": 0,
            "total_blank_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "file_details": [],
        },
    }

    print("開始分析多語言檔案...")

    # 掃描 Go 檔案（統一 schema）
    go_files = [
        f
        for f in root_path.rglob("*.go")
        if not any(pattern in str(f) for pattern in ignore_patterns)
    ]
    print(f"找到 {len(go_files)} 個 Go 檔案")

    for i, go_file in enumerate(go_files, 1):
        if i % 10 == 0:
            print(f"  Go 進度: {i}/{len(go_files)}")
        try:
            content = go_file.read_text(encoding="utf-8")
            rel_path = str(go_file.relative_to(root_path))

            # 基本統計
            basic_stats = _analyze_file_basic(content, ["//", "/*"])
            go_elements = _count_go_elements(content)

            # 更新總計
            multilang_stats["go"]["total_files"] += 1
            multilang_stats["go"]["total_lines"] += basic_stats["total_lines"]
            multilang_stats["go"]["total_code_lines"] += basic_stats["code_lines"]
            multilang_stats["go"]["total_comment_lines"] += basic_stats["comment_lines"]
            multilang_stats["go"]["total_blank_lines"] += basic_stats["blank_lines"]
            multilang_stats["go"]["total_functions"] += go_elements["functions"]
            multilang_stats["go"]["total_structs"] += go_elements["structs"]
            multilang_stats["go"]["total_interfaces"] += go_elements["interfaces"]

            # 記錄檔案詳情
            multilang_stats["go"]["file_details"].append(
                {
                    "path": rel_path,
                    "lines": basic_stats["total_lines"],
                    "functions": go_elements["functions"],
                    "structs": go_elements["structs"],
                    "interfaces": go_elements["interfaces"],
                }
            )
        except Exception as e:
            print(f"  警告: 無法分析 {go_file.name}: {e}")

    # 掃描 Rust 檔案（統一 schema）
    rust_files = [
        f
        for f in root_path.rglob("*.rs")
        if not any(pattern in str(f) for pattern in ignore_patterns)
    ]
    print(f"找到 {len(rust_files)} 個 Rust 檔案")

    for i, rs_file in enumerate(rust_files, 1):
        if i % 10 == 0:
            print(f"  Rust 進度: {i}/{len(rust_files)}")
        try:
            content = rs_file.read_text(encoding="utf-8")
            rel_path = str(rs_file.relative_to(root_path))

            # 基本統計
            basic_stats = _analyze_file_basic(content, ["//", "/*"])
            rust_elements = _count_rust_elements(content)

            # 更新總計
            multilang_stats["rust"]["total_files"] += 1
            multilang_stats["rust"]["total_lines"] += basic_stats["total_lines"]
            multilang_stats["rust"]["total_code_lines"] += basic_stats["code_lines"]
            multilang_stats["rust"]["total_comment_lines"] += basic_stats[
                "comment_lines"
            ]
            multilang_stats["rust"]["total_blank_lines"] += basic_stats["blank_lines"]
            multilang_stats["rust"]["total_functions"] += rust_elements["functions"]
            multilang_stats["rust"]["total_structs"] += rust_elements["structs"]
            multilang_stats["rust"]["total_traits"] += rust_elements["traits"]
            multilang_stats["rust"]["total_impls"] += rust_elements["impls"]

            # 記錄檔案詳情
            multilang_stats["rust"]["file_details"].append(
                {
                    "path": rel_path,
                    "lines": basic_stats["total_lines"],
                    "functions": rust_elements["functions"],
                    "structs": rust_elements["structs"],
                    "traits": rust_elements["traits"],
                }
            )
        except Exception as e:
            print(f"  警告: 無法分析 {rs_file.name}: {e}")

    # 掃描 TypeScript 檔案（統一 schema）
    ts_files = [
        f
        for f in root_path.rglob("*.ts")
        if not any(pattern in str(f) for pattern in ignore_patterns)
    ]
    print(f"找到 {len(ts_files)} 個 TypeScript 檔案")

    for i, ts_file in enumerate(ts_files, 1):
        if i % 10 == 0:
            print(f"  TypeScript 進度: {i}/{len(ts_files)}")
        try:
            content = ts_file.read_text(encoding="utf-8")
            rel_path = str(ts_file.relative_to(root_path))

            # 基本統計
            basic_stats = _analyze_file_basic(content, ["//", "/*"])
            ts_elements = _count_typescript_elements(content)

            # 更新總計
            multilang_stats["typescript"]["total_files"] += 1
            multilang_stats["typescript"]["total_lines"] += basic_stats["total_lines"]
            multilang_stats["typescript"]["total_code_lines"] += basic_stats[
                "code_lines"
            ]
            multilang_stats["typescript"]["total_comment_lines"] += basic_stats[
                "comment_lines"
            ]
            multilang_stats["typescript"]["total_blank_lines"] += basic_stats[
                "blank_lines"
            ]
            multilang_stats["typescript"]["total_functions"] += ts_elements["functions"]
            multilang_stats["typescript"]["total_classes"] += ts_elements["classes"]
            multilang_stats["typescript"]["total_interfaces"] += ts_elements[
                "interfaces"
            ]
            multilang_stats["typescript"]["total_types"] += ts_elements["types"]

            # 記錄檔案詳情
            multilang_stats["typescript"]["file_details"].append(
                {
                    "path": rel_path,
                    "lines": basic_stats["total_lines"],
                    "functions": ts_elements["functions"],
                    "classes": ts_elements["classes"],
                    "interfaces": ts_elements["interfaces"],
                }
            )
        except Exception as e:
            print(f"  警告: 無法分析 {ts_file.name}: {e}")

    # 掃描 JavaScript 檔案（統一 schema）
    js_files = [
        f
        for f in root_path.rglob("*.js")
        if not any(pattern in str(f) for pattern in ignore_patterns)
    ]
    print(f"找到 {len(js_files)} 個 JavaScript 檔案")

    for i, js_file in enumerate(js_files, 1):
        if i % 10 == 0:
            print(f"  JavaScript 進度: {i}/{len(js_files)}")
        try:
            content = js_file.read_text(encoding="utf-8")
            rel_path = str(js_file.relative_to(root_path))

            # 基本統計
            basic_stats = _analyze_file_basic(content, ["//", "/*"])
            # JavaScript 使用與 TypeScript 相同的元素計數
            js_elements = _count_typescript_elements(content)

            # 更新總計
            multilang_stats["javascript"]["total_files"] += 1
            multilang_stats["javascript"]["total_lines"] += basic_stats["total_lines"]
            multilang_stats["javascript"]["total_code_lines"] += basic_stats[
                "code_lines"
            ]
            multilang_stats["javascript"]["total_comment_lines"] += basic_stats[
                "comment_lines"
            ]
            multilang_stats["javascript"]["total_blank_lines"] += basic_stats[
                "blank_lines"
            ]
            multilang_stats["javascript"]["total_functions"] += js_elements["functions"]
            multilang_stats["javascript"]["total_classes"] += js_elements["classes"]

            # 記錄檔案詳情
            multilang_stats["javascript"]["file_details"].append(
                {
                    "path": rel_path,
                    "lines": basic_stats["total_lines"],
                    "functions": js_elements["functions"],
                    "classes": js_elements["classes"],
                }
            )
        except Exception as e:
            print(f"  警告: 無法分析 {js_file.name}: {e}")

    # 生成報告
    _generate_multilang_report(multilang_stats, output_dir)

    return multilang_stats


def main():
    """主函數."""
    # 設定路徑
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "_out" / "analysis"

    print("AIVA 程式碼庫分析工具")
    print("=" * 80)
    print(f"專案根目錄: {project_root}")
    print(f"輸出目錄: {output_dir}")
    print("=" * 80)
    print()

    # 執行 Python 分析
    print("📊 階段 1: 分析 Python 程式碼")
    print("-" * 80)
    stats = analyze_directory(
        root_path=project_root / "services",
        output_dir=output_dir,
        max_files=1000,
    )

    # 執行多語言分析
    print("\n📊 階段 2: 分析多語言程式碼 (Go/Rust/TypeScript/JavaScript)")
    print("-" * 80)
    multilang_stats = analyze_multilang_files(
        root_path=project_root / "services",
        output_dir=output_dir,
    )

    # 顯示 Python 摘要
    print("\n" + "=" * 80)
    print("✅ Python 分析完成！摘要:")
    print("=" * 80)
    print(f"總檔案數: {stats['total_files']}")
    print(f"總行數: {stats['total_lines']:,}")
    print(f"總函數數: {stats['total_functions']}")
    print(f"總類別數: {stats['total_classes']}")
    print(f"平均複雜度: {stats['total_complexity'] / max(stats['total_files'], 1):.2f}")

    type_hint_pct = (
        stats["files_with_type_hints"] / max(stats["total_files"], 1)
    ) * 100
    docstring_pct = (
        stats["files_with_docstrings"] / max(stats["total_files"], 1)
    ) * 100
    print(f"類型提示覆蓋率: {type_hint_pct:.1f}%")
    print(f"文檔字串覆蓋率: {docstring_pct:.1f}%")

    # 顯示多語言摘要
    print("\n" + "=" * 80)
    print("✅ 多語言分析完成！摘要:")
    print("=" * 80)
    total_multilang_files = 0
    total_multilang_lines = 0
    for lang, data in multilang_stats.items():
        if data["total_files"] > 0:
            print(
                f"{lang.upper()}: {data['total_files']} 檔案, "
                f"{data['total_lines']:,} 行, "
                f"{data['total_functions']} 函數"
            )
            total_multilang_files += data["total_files"]
            total_multilang_lines += data["total_lines"]

    # 總計
    print("\n" + "=" * 80)
    print("📈 專案總計:")
    print("=" * 80)
    grand_total_files = stats["total_files"] + total_multilang_files
    grand_total_lines = stats["total_lines"] + total_multilang_lines
    print(f"總檔案數 (所有語言): {grand_total_files}")
    print(f"總行數 (所有語言): {grand_total_lines:,}")
    print(f"  - Python: {stats['total_lines']:,} 行 ({stats['total_files']} 檔案)")
    print(f"  - 其他語言: {total_multilang_lines:,} 行 ({total_multilang_files} 檔案)")
    print("=" * 80)


if __name__ == "__main__":
    main()
