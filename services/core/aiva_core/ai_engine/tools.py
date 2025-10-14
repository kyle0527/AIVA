"""
Tools - AI 代理工具系統
提供程式碼讀取、編輯、執行等操作能力
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class Tool(ABC):
    """工具基礎類別."""

    def __init__(self, name: str, description: str) -> None:
        """初始化工具.

        Args:
            name: 工具名稱
            description: 工具描述
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """執行工具.

        Args:
            **kwargs: 工具參數

        Returns:
            執行結果
        """
        ...


class CodeReader(Tool):
    """程式碼讀取工具."""

    def __init__(self, codebase_path: str) -> None:
        """初始化程式碼讀取器.

        Args:
            codebase_path: 程式碼庫根目錄
        """
        super().__init__(
            name="CodeReader",
            description="讀取程式碼檔案內容",
        )
        self.codebase_path = Path(codebase_path)

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """讀取檔案內容.

        Args:
            **kwargs: 工具參數
                path (str): 檔案路徑 (相對於程式碼庫根目錄)

        Returns:
            包含檔案內容的字典
        """
        path = kwargs.get("path", "")
        if not path:
            return {"status": "error", "error": "缺少必需參數: path"}

        try:
            full_path = self.codebase_path / path
            content = full_path.read_text(encoding="utf-8")
            return {
                "status": "success",
                "path": path,
                "content": content,
                "lines": len(content.splitlines()),
            }
        except Exception as e:
            return {"status": "error", "path": path, "error": str(e)}


class CodeWriter(Tool):
    """程式碼寫入工具."""

    def __init__(self, codebase_path: str) -> None:
        """初始化程式碼寫入器.

        Args:
            codebase_path: 程式碼庫根目錄
        """
        super().__init__(
            name="CodeWriter",
            description="寫入或修改程式碼檔案",
        )
        self.codebase_path = Path(codebase_path)

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """寫入檔案內容.

        Args:
            **kwargs: 工具參數
                path (str): 檔案路徑 (相對於程式碼庫根目錄)
                content (str): 要寫入的內容

        Returns:
            執行結果
        """
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")

        if not path:
            return {"status": "error", "error": "缺少必需參數: path"}
        if not content:
            return {"status": "error", "error": "缺少必需參數: content"}

        try:
            full_path = self.codebase_path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return {
                "status": "success",
                "path": path,
                "bytes_written": len(content.encode("utf-8")),
            }
        except Exception as e:
            return {"status": "error", "path": path, "error": str(e)}


class CodeAnalyzer(Tool):
    """程式碼分析工具."""

    def __init__(self, codebase_path: str) -> None:
        """初始化程式碼分析器.

        Args:
            codebase_path: 程式碼庫根目錄
        """
        super().__init__(
            name="CodeAnalyzer",
            description="分析程式碼結構和品質",
        )
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
                line for line in lines
                if line.strip().startswith("#") or line.strip().startswith('"""') or line.strip().startswith("'''")
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
                    import ast
                    tree = ast.parse(content)

                    # 統計各種節點
                    imports = []
                    functions = []
                    classes = []
                    async_functions = []

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

                    result.update({
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
                    })
                except SyntaxError as e:
                    result["syntax_error"] = str(e)
            else:
                # 簡單統計（向後兼容）
                import_count = sum(1 for line in lines if line.strip().startswith("import"))
                function_count = sum(1 for line in lines if line.strip().startswith("def "))
                class_count = sum(1 for line in lines if line.strip().startswith("class "))

                result.update({
                    "imports": import_count,
                    "functions": function_count,
                    "classes": class_count,
                })

            return result
        except Exception as e:
            return {"status": "error", "path": path, "error": str(e)}

    def _calculate_complexity(self, tree: Any) -> int:
        """計算循環複雜度.

        Args:
            tree: AST 樹

        Returns:
            複雜度分數
        """
        import ast
        complexity = 1  # 基礎複雜度

        for node in ast.walk(tree):
            # 每個決策點增加複雜度
            if isinstance(node, ast.If | ast.While | ast.For | ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _check_type_hints(self, tree: Any) -> bool:
        """檢查是否使用類型提示.

        Args:
            tree: AST 樹

        Returns:
            是否有類型提示
        """
        import ast

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # 檢查參數類型提示
                if node.args.args:
                    for arg in node.args.args:
                        if arg.annotation is not None:
                            return True
                # 檢查返回值類型提示
                if node.returns is not None:
                    return True
        return False

    def _check_docstrings(self, tree: Any) -> bool:
        """檢查是否有文檔字串.

        Args:
            tree: AST 樹

        Returns:
            是否有文檔字串
        """
        import ast

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module):
                docstring = ast.get_docstring(node)
                if docstring:
                    return True
        return False


class CommandExecutor(Tool):
    """命令執行工具."""

    def __init__(self, codebase_path: str) -> None:
        """初始化命令執行器.

        Args:
            codebase_path: 程式碼庫根目錄
        """
        super().__init__(
            name="CommandExecutor",
            description="執行系統命令",
        )
        self.codebase_path = Path(codebase_path)

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """執行命令.

        Args:
            **kwargs: 工具參數
                command (str): 要執行的命令

        Returns:
            執行結果
        """
        command = kwargs.get("command", "")
        if not command:
            return {"status": "error", "error": "缺少必需參數: command"}

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.codebase_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return {
                "status": "success",
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "command": command,
                "error": "Command timeout (30s)",
            }
        except Exception as e:
            return {"status": "error", "command": command, "error": str(e)}


class ScanTrigger(Tool):
    """掃描觸發工具."""

    def __init__(self) -> None:
        """初始化掃描觸發器."""
        super().__init__(
            name="ScanTrigger",
            description="觸發漏洞掃描任務",
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """觸發掃描.

        Args:
            **kwargs: 工具參數
                target_url (str): 目標 URL
                scan_type (str): 掃描類型 (full/quick/custom), 預設為 "full"

        Returns:
            掃描任務資訊
        """
        target_url = kwargs.get("target_url", "")
        scan_type = kwargs.get("scan_type", "full")

        if not target_url:
            return {"status": "error", "error": "缺少必需參數: target_url"}

        # TODO: 實際整合 AIVA 掃描系統
        return {
            "status": "success",
            "task_id": f"scan_{hash(target_url) % 10000}",
            "target": target_url,
            "scan_type": scan_type,
            "message": "掃描任務已建立 (待實作實際整合)",
        }


class VulnerabilityDetector(Tool):
    """漏洞檢測工具."""

    def __init__(self) -> None:
        """初始化漏洞檢測器."""
        super().__init__(
            name="VulnerabilityDetector",
            description="執行特定類型的漏洞檢測",
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """執行漏洞檢測.

        Args:
            **kwargs: 工具參數
                vuln_type (str): 漏洞類型 (xss/sqli/ssrf/idor)
                target (str): 目標

        Returns:
            檢測結果
        """
        vuln_type = kwargs.get("vuln_type", "")
        target = kwargs.get("target", "")

        if not vuln_type:
            return {"status": "error", "error": "缺少必需參數: vuln_type"}
        if not target:
            return {"status": "error", "error": "缺少必需參數: target"}

        # TODO: 實際整合 AIVA 漏洞檢測模組
        return {
            "status": "success",
            "vuln_type": vuln_type,
            "target": target,
            "findings": [],
            "message": f"{vuln_type.upper()} 檢測完成 (待實作實際整合)",
        }
