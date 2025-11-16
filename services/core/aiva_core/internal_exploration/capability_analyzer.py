"""Capability Analyzer - 能力分析器

識別和分析 AIVA 系統的功能能力，通過 AST 解析識別 @register_capability 標記的函數

遵循 aiva_common 修復規範:
- 使用標準裝飾器模式
- 統一的能力元數據格式
"""

import ast
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CapabilityAnalyzer:
    """能力分析器
    
    職責：識別系統中所有註冊的能力函數，提取其元數據
    
    識別目標：
    - @register_capability 裝飾的函數
    - @capability 裝飾的函數
    - 包含 'capability' 關鍵字的裝飾器
    """
    
    def __init__(self):
        """初始化能力分析器"""
        self.capabilities_cache: dict[str, list[dict]] = {}
        logger.info("CapabilityAnalyzer initialized")
    
    async def analyze_capabilities(self, modules_info: dict) -> list[dict[str, Any]]:
        """分析模組中的能力函數
        
        Args:
            modules_info: ModuleExplorer 返回的模組資訊
            
        Returns:
            能力列表:
            [
                {
                    "name": str,
                    "module": str,
                    "description": str,
                    "parameters": list,
                    "file_path": str,
                    "return_type": str | None,
                    "is_async": bool,
                    "decorators": list
                }
            ]
        """
        logger.info(f"🔍 Starting capability analysis for {len(modules_info)} modules...")
        capabilities = []
        
        for module_name, module_data in modules_info.items():
            logger.info(f"  Analyzing module: {module_name}")
            module_path = Path(module_data["path"])
            
            for file_info in module_data["files"]:
                file_path = module_path / file_info["path"]
                
                # 跳過 __init__.py 和非能力相關文件
                if file_path.name == "__init__.py":
                    continue
                
                caps = await self._extract_capabilities_from_file(file_path, module_name)
                capabilities.extend(caps)
        
        logger.info(f"✅ Capability analysis completed: {len(capabilities)} capabilities found")
        return capabilities
    
    async def _extract_capabilities_from_file(self, file_path: Path, module: str) -> list[dict]:
        """從文件中提取能力
        
        Args:
            file_path: Python 文件路徑
            module: 所屬模組名稱
            
        Returns:
            能力列表
        """
        capabilities = []
        
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # 檢查是否有能力裝飾器
                    if self._has_capability_decorator(node):
                        cap = self._extract_capability_info(node, file_path, module)
                        capabilities.append(cap)
            
            if capabilities:
                logger.debug(f"  Found {len(capabilities)} capabilities in {file_path.name}")
            
        except SyntaxError as e:
            logger.warning(f"  Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"  Failed to parse {file_path}: {e}")
        
        return capabilities
    
    def _has_capability_decorator(self, node: ast.FunctionDef) -> bool:
        """檢查函數是否有能力裝飾器
        
        Args:
            node: AST 函數定義節點
            
        Returns:
            是否有能力裝飾器
        """
        for decorator in node.decorator_list:
            # 檢查 @capability 或 @register_capability
            if isinstance(decorator, ast.Name):
                if "capability" in decorator.id.lower():
                    return True
            
            # 檢查帶參數的裝飾器 @capability(...) 
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    if "capability" in decorator.func.id.lower():
                        return True
                # 檢查 @module.capability
                elif isinstance(decorator.func, ast.Attribute):
                    if "capability" in decorator.func.attr.lower():
                        return True
        
        return False
    
    def _extract_capability_info(
        self, 
        node: ast.FunctionDef, 
        file_path: Path,
        module: str
    ) -> dict[str, Any]:
        """提取能力詳細資訊
        
        Args:
            node: AST 函數定義節點
            file_path: 文件路徑
            module: 模組名稱
            
        Returns:
            能力資訊字典
        """
        # 提取參數
        parameters = []
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "annotation": ast.unparse(arg.annotation) if arg.annotation else None
            }
            parameters.append(param_info)
        
        # 提取返回類型
        return_type = None
        if node.returns:
            try:
                return_type = ast.unparse(node.returns)
            except Exception:
                return_type = "Unknown"
        
        # 提取裝飾器名稱
        decorators = []
        for decorator in node.decorator_list:
            try:
                decorators.append(ast.unparse(decorator))
            except Exception:
                decorators.append("Unknown")
        
        # 提取文檔字串
        docstring = ast.get_docstring(node) or ""
        description = docstring.split("\n")[0] if docstring else f"Function: {node.name}"
        
        return {
            "name": node.name,
            "module": module,
            "description": description,
            "parameters": parameters,
            "file_path": str(file_path),
            "return_type": return_type,
            "is_async": isinstance(node, ast.AsyncFunctionDef) or any(
                isinstance(n, ast.AsyncFunctionDef) for n in ast.walk(node)
            ),
            "decorators": decorators,
            "docstring": docstring,
            "line_number": node.lineno
        }
    
    def get_capabilities_by_module(self, capabilities: list[dict]) -> dict[str, list[dict]]:
        """按模組分組能力
        
        Args:
            capabilities: 能力列表
            
        Returns:
            按模組分組的字典
        """
        grouped = {}
        
        for cap in capabilities:
            module = cap["module"]
            if module not in grouped:
                grouped[module] = []
            grouped[module].append(cap)
        
        return grouped
    
    def generate_capability_summary(self, capabilities: list[dict]) -> str:
        """生成能力摘要報告
        
        Args:
            capabilities: 能力列表
            
        Returns:
            可讀的摘要字串
        """
        if not capabilities:
            return "No capabilities found"
        
        grouped = self.get_capabilities_by_module(capabilities)
        
        lines = [f"Total Capabilities: {len(capabilities)}\n"]
        
        for module, caps in grouped.items():
            lines.append(f"\nModule: {module} ({len(caps)} capabilities)")
            for cap in caps:
                params = ", ".join(p["name"] for p in cap["parameters"])
                lines.append(f"  - {cap['name']}({params})")
                if cap["description"]:
                    lines.append(f"    {cap['description'][:80]}")
        
        return "\n".join(lines)
