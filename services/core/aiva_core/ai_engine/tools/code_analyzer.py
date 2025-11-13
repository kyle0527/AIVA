"""程式碼分析工具 - 提供程式碼結構和品質分析

從原 tools.py 遷移的 CodeAnalyzer 功能，保留核心的 AST 分析能力
專注於有用的程式碼分析功能，去除不必要的複雜性
"""

import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tool(ABC):
    """工具基礎抽象類別"""
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_info(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "class": self.__class__.__name__}


class CodeAnalyzer(Tool):
    """程式碼分析工具 - 基於 AST 的程式碼分析
    
    提供功能：
    - Python 程式碼的 AST 解析和分析
    - 程式碼複雜度計算
    - 結構統計（函數、類別、導入等）
    - 程式碼品質指標（類型提示、文檔字串等）
    - 基本統計（行數、注釋等）
    """
    
    def __init__(self, codebase_path: str) -> None:
        """初始化程式碼分析器
        
        Args:
            codebase_path: 程式碼庫根目錄
        """
        super().__init__(
            name="CodeAnalyzer",
            description="分析 Python 程式碼結構、複雜度和品質指標"
        )
        self.codebase_path = Path(codebase_path).resolve()
    
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """分析程式碼
        
        Args:
            **kwargs: 工具參數
                path (str): 檔案路徑（相對於程式碼庫根目錄）
                detailed (bool): 是否進行詳細 AST 分析，預設 True
                include_metrics (bool): 是否包含程式碼複雜度等指標，預設 True
                
        Returns:
            分析結果字典：
            - status: 'success' | 'error'
            - path: 檔案路徑
            - file_stats: 基本檔案統計
            - code_structure: 程式碼結構分析（如果啟用詳細分析）
            - quality_metrics: 程式碼品質指標（如果啟用）
            - error: 錯誤信息（如果有）
        """
        path = kwargs.get("path", "")
        detailed = kwargs.get("detailed", True)
        include_metrics = kwargs.get("include_metrics", True)
        
        if not path:
            return {"status": "error", "error": "缺少必需參數: path"}
        
        try:
            # 安全路徑處理
            full_path = (self.codebase_path / path).resolve()
            try:
                full_path.relative_to(self.codebase_path)
            except ValueError:
                return {
                    "status": "error",
                    "path": path,
                    "error": "路徑超出程式碼庫範圍"
                }
            
            if not full_path.exists():
                return {
                    "status": "error",
                    "path": path,
                    "error": "檔案不存在"
                }
            
            # 讀取檔案內容
            content = full_path.read_text(encoding="utf-8")
            
            # 基本統計
            result = {
                "status": "success",
                "path": path,
                "file_stats": self._get_basic_stats(content)
            }
            
            # 詳細 AST 分析
            if detailed:
                ast_analysis = self._analyze_ast(content)
                if "error" in ast_analysis:
                    result["syntax_error"] = ast_analysis["error"]
                else:
                    result["code_structure"] = ast_analysis
            
            # 程式碼品質指標
            if include_metrics and "syntax_error" not in result:
                result["quality_metrics"] = self._calculate_quality_metrics(content)
            
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "path": path,
                "error": f"分析程式碼時發生錯誤: {str(e)}"
            }
    
    def _get_basic_stats(self, content: str) -> dict[str, Any]:
        """獲取基本統計信息
        
        Args:
            content: 檔案內容
            
        Returns:
            基本統計字典
        """
        lines = content.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [
            line for line in lines 
            if line.strip().startswith("#") 
            or '"""' in line.strip() 
            or "'''" in line.strip()
        ]
        
        return {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "comment_lines": len(comment_lines),
            "blank_lines": len(lines) - len(non_empty_lines),
            "file_size": len(content),
            "avg_line_length": sum(len(line) for line in lines) / len(lines) if lines else 0
        }
    
    def _analyze_ast(self, content: str) -> dict[str, Any]:
        """AST 分析
        
        Args:
            content: 檔案內容
            
        Returns:
            AST 分析結果或錯誤信息
        """
        try:
            tree = ast.parse(content)
            
            # 收集各種語法結構
            imports = set()
            functions = []
            async_functions = []
            classes = []
            variables = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if module:
                        imports.add(module)
                elif isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args_count": len(node.args.args),
                        "has_decorators": bool(node.decorator_list),
                        "is_private": node.name.startswith("_"),
                    })
                elif isinstance(node, ast.AsyncFunctionDef):
                    async_functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "args_count": len(node.args.args),
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "bases": [self._ast_name_to_string(base) for base in node.bases],
                        "methods_count": len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    })
                elif isinstance(node, ast.Assign):
                    # 收集變數賦值
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.add(target.id)
            
            return {
                "imports": sorted(list(imports)),
                "import_count": len(imports),
                "functions": functions,
                "function_count": len(functions),
                "async_functions": async_functions,
                "async_function_count": len(async_functions),
                "classes": classes,
                "class_count": len(classes),
                "variable_count": len(variables),
                "total_nodes": len(list(ast.walk(tree)))
            }
            
        except SyntaxError as e:
            return {"error": f"語法錯誤: {str(e)}"}
    
    def _calculate_quality_metrics(self, content: str) -> dict[str, Any]:
        """計算程式碼品質指標
        
        Args:
            content: 檔案內容
            
        Returns:
            品質指標字典
        """
        try:
            tree = ast.parse(content)
            
            complexity = self._calculate_cyclomatic_complexity(tree)
            has_type_hints = self._check_type_hints(tree)
            has_docstrings = self._check_docstrings(tree)
            
            return {
                "cyclomatic_complexity": complexity,
                "complexity_rating": self._rate_complexity(complexity),
                "has_type_hints": has_type_hints,
                "has_docstrings": has_docstrings,
                "code_quality_score": self._calculate_quality_score(
                    complexity, has_type_hints, has_docstrings
                )
            }
            
        except SyntaxError:
            return {"error": "無法計算品質指標，檔案包含語法錯誤"}
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """計算循環複雜度
        
        Args:
            tree: AST 樹
            
        Returns:
            複雜度分數
        """
        complexity = 1  # 基礎複雜度
        
        for node in ast.walk(tree):
            # 每個決策點增加複雜度
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # 邏輯運算符（and, or）增加複雜度
                complexity += len(node.values) - 1
        
        return complexity
    
    def _check_type_hints(self, tree: ast.AST) -> bool:
        """檢查是否使用類型提示
        
        Args:
            tree: AST 樹
            
        Returns:
            是否使用類型提示
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 檢查參數類型提示
                if node.args.args and any(arg.annotation for arg in node.args.args):
                    return True
                # 檢查返回值類型提示
                if node.returns is not None:
                    return True
        return False
    
    def _check_docstrings(self, tree: ast.AST) -> bool:
        """檢查是否有文檔字串
        
        Args:
            tree: AST 樹
            
        Returns:
            是否有文檔字串
        """
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                if ast.get_docstring(node):
                    return True
        return False
    
    def _ast_name_to_string(self, node: ast.AST) -> str:
        """將 AST 名稱節點轉為字串
        
        Args:
            node: AST 節點
            
        Returns:
            名稱字串
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._ast_name_to_string(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def _rate_complexity(self, complexity: int) -> str:
        """評估複雜度等級
        
        Args:
            complexity: 複雜度數值
            
        Returns:
            複雜度等級描述
        """
        if complexity <= 5:
            return "低"
        elif complexity <= 10:
            return "中"
        elif complexity <= 20:
            return "高"
        else:
            return "極高"
    
    def _calculate_quality_score(self, complexity: int, has_type_hints: bool, has_docstrings: bool) -> float:
        """計算程式碼品質分數（0-100）
        
        Args:
            complexity: 循環複雜度
            has_type_hints: 是否有類型提示
            has_docstrings: 是否有文檔字串
            
        Returns:
            品質分數
        """
        base_score = 100
        
        # 複雜度扣分
        if complexity > 20:
            base_score -= 40
        elif complexity > 10:
            base_score -= 20
        elif complexity > 5:
            base_score -= 10
        
        # 類型提示和文檔加分
        if has_type_hints:
            base_score += 10
        if has_docstrings:
            base_score += 10
        
        return max(0, min(100, base_score))


__all__ = ["CodeAnalyzer"]