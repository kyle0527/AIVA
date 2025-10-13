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
) -> dict:
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
    stats: dict[str, Any] = {
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


def analyze_multilang_files(
    root_path: Path,
    ignore_patterns: list[str] | None = None,
) -> dict:
    """分析多語言程式碼檔案 (Go, Rust, TypeScript).

    Args:
        root_path: 要分析的根目錄
        ignore_patterns: 要忽略的路徑模式

    Returns:
        多語言分析結果
    """
    if ignore_patterns is None:
        ignore_patterns = [
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "_out",
            "target",
            "build",
            "dist",
        ]

    multilang_stats: dict[str, Any] = {
        "go": {"files": [], "total_lines": 0, "total_files": 0},
        "rust": {"files": [], "total_lines": 0, "total_files": 0},
        "typescript": {"files": [], "total_lines": 0, "total_files": 0},
        "javascript": {"files": [], "total_lines": 0, "total_files": 0},
    }

    # 掃描 Go 檔案
    for go_file in root_path.rglob("*.go"):
        if any(pattern in str(go_file) for pattern in ignore_patterns):
            continue
        try:
            content = go_file.read_text(encoding="utf-8")
            lines = len(content.splitlines())
            multilang_stats["go"]["files"].append(
                {
                    "path": str(go_file.relative_to(root_path)),
                    "lines": lines,
                }
            )
            multilang_stats["go"]["total_lines"] += lines
            multilang_stats["go"]["total_files"] += 1
        except Exception:
            pass

    # 掃描 Rust 檔案
    for rs_file in root_path.rglob("*.rs"):
        if any(pattern in str(rs_file) for pattern in ignore_patterns):
            continue
        try:
            content = rs_file.read_text(encoding="utf-8")
            lines = len(content.splitlines())
            multilang_stats["rust"]["files"].append(
                {
                    "path": str(rs_file.relative_to(root_path)),
                    "lines": lines,
                }
            )
            multilang_stats["rust"]["total_lines"] += lines
            multilang_stats["rust"]["total_files"] += 1
        except Exception:
            pass

    # 掃描 TypeScript 檔案
    for ts_file in root_path.rglob("*.ts"):
        if any(pattern in str(ts_file) for pattern in ignore_patterns):
            continue
        try:
            content = ts_file.read_text(encoding="utf-8")
            lines = len(content.splitlines())
            multilang_stats["typescript"]["files"].append(
                {
                    "path": str(ts_file.relative_to(root_path)),
                    "lines": lines,
                }
            )
            multilang_stats["typescript"]["total_lines"] += lines
            multilang_stats["typescript"]["total_files"] += 1
        except Exception:
            pass

    # 掃描 JavaScript 檔案
    for js_file in root_path.rglob("*.js"):
        if any(pattern in str(js_file) for pattern in ignore_patterns):
            continue
        try:
            content = js_file.read_text(encoding="utf-8")
            lines = len(content.splitlines())
            multilang_stats["javascript"]["files"].append(
                {
                    "path": str(js_file.relative_to(root_path)),
                    "lines": lines,
                }
            )
            multilang_stats["javascript"]["total_lines"] += lines
            multilang_stats["javascript"]["total_files"] += 1
        except Exception:
            pass

    return multilang_stats


def main():
    """主函數."""
    # 設定路徑
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "_out" / "analysis"

    print("AIVA 程式碼庫分析工具 (多語言支援)")
    print("=" * 80)
    print(f"專案根目錄: {project_root}")
    print(f"輸出目錄: {output_dir}")
    print("=" * 80)
    print()

    # 執行 Python 分析
    print("📊 分析 Python 程式碼...")
    stats = analyze_directory(
        root_path=project_root / "services",
        output_dir=output_dir,
        max_files=1000,
    )

    # 執行多語言分析
    print("\n📊 分析多語言程式碼...")
    multilang_stats = analyze_multilang_files(
        root_path=project_root / "services",
    )

    # 將多語言統計加入主報告
    stats["multilang"] = multilang_stats

    # 更新 JSON 報告
    json_output = (
        output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 顯示摘要
    print("\n" + "=" * 80)
    print("分析完成！摘要:")
    print("=" * 80)
    print("\n🐍 Python:")
    print(f"  總檔案數: {stats['total_files']}")
    print(f"  總行數: {stats['total_lines']:,}")
    print(f"  總函數數: {stats['total_functions']}")
    print(f"  總類別數: {stats['total_classes']}")
    print(
        f"  平均複雜度: {stats['total_complexity'] / max(stats['total_files'], 1):.2f}"
    )

    type_hint_pct = (
        stats["files_with_type_hints"] / max(stats["total_files"], 1)
    ) * 100
    docstring_pct = (
        stats["files_with_docstrings"] / max(stats["total_files"], 1)
    ) * 100
    print(f"  類型提示覆蓋率: {type_hint_pct:.1f}%")
    print(f"  文檔字串覆蓋率: {docstring_pct:.1f}%")

    # 顯示多語言統計
    total_multilang_lines = 0
    total_multilang_files = 0

    if multilang_stats["go"]["total_files"] > 0:
        print("\n🔷 Go:")
        print(f"  總檔案數: {multilang_stats['go']['total_files']}")
        print(f"  總行數: {multilang_stats['go']['total_lines']:,}")
        print(
            f"  平均行數: {multilang_stats['go']['total_lines'] / multilang_stats['go']['total_files']:.1f}"
        )
        total_multilang_lines += multilang_stats["go"]["total_lines"]
        total_multilang_files += multilang_stats["go"]["total_files"]

    if multilang_stats["rust"]["total_files"] > 0:
        print("\n🦀 Rust:")
        print(f"  總檔案數: {multilang_stats['rust']['total_files']}")
        print(f"  總行數: {multilang_stats['rust']['total_lines']:,}")
        print(
            f"  平均行數: {multilang_stats['rust']['total_lines'] / multilang_stats['rust']['total_files']:.1f}"
        )
        total_multilang_lines += multilang_stats["rust"]["total_lines"]
        total_multilang_files += multilang_stats["rust"]["total_files"]

    if multilang_stats["typescript"]["total_files"] > 0:
        print("\n📘 TypeScript:")
        print(f"  總檔案數: {multilang_stats['typescript']['total_files']}")
        print(f"  總行數: {multilang_stats['typescript']['total_lines']:,}")
        print(
            f"  平均行數: {multilang_stats['typescript']['total_lines'] / multilang_stats['typescript']['total_files']:.1f}"
        )
        total_multilang_lines += multilang_stats["typescript"]["total_lines"]
        total_multilang_files += multilang_stats["typescript"]["total_files"]

    if multilang_stats["javascript"]["total_files"] > 0:
        print("\n📜 JavaScript:")
        print(f"  總檔案數: {multilang_stats['javascript']['total_files']}")
        print(f"  總行數: {multilang_stats['javascript']['total_lines']:,}")
        print(
            f"  平均行數: {multilang_stats['javascript']['total_lines'] / multilang_stats['javascript']['total_files']:.1f}"
        )
        total_multilang_lines += multilang_stats["javascript"]["total_lines"]
        total_multilang_files += multilang_stats["javascript"]["total_files"]

    # 總計
    total_all_lines = stats["total_lines"] + total_multilang_lines
    total_all_files = stats["total_files"] + total_multilang_files

    print("\n" + "=" * 80)
    print("📊 全專案統計:")
    print("=" * 80)
    print(f"總檔案數: {total_all_files}")
    print(f"總行數: {total_all_lines:,}")
    print(
        f"  - Python: {stats['total_lines']:,} ({stats['total_lines']/total_all_lines*100:.1f}%)"
    )
    if total_multilang_lines > 0:
        print(
            f"  - 其他語言: {total_multilang_lines:,} ({total_multilang_lines/total_all_lines*100:.1f}%)"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
