"""
AIVA 模組探索器 - 自動發現五大模組的能力與依賴

此模組負責掃描 AIVA 的五大服務模組 (Core, Common, Features, Integration, Scan),
自動發現所有能力、API、依賴關係,並建立架構知識圖譜。

Author: AIVA AI Engine Team
Version: 1.0.0
Created: 2025-11-13
"""

import ast
import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# 遵循 Core 模組開發規範 - 使用 aiva_common 標準枚舉
from aiva_common.enums import ModuleName, TaskStatus

logger = logging.getLogger(__name__)


@dataclass
class CapabilityInfo:
    """能力資訊數據結構"""
    
    capability_id: str
    name: str
    module: str
    file_path: str
    line_number: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    docstring: Optional[str] = None
    signature: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    source_code: Optional[str] = None
    language: str = "python"
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "module": self.module,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "docstring": self.docstring,
            "signature": self.signature,
            "decorators": self.decorators,
            "language": self.language,
        }


@dataclass
class DependencyInfo:
    """依賴關係資訊"""
    
    source_module: str
    target_module: str
    dependency_type: str  # "internal" or "external"
    import_statement: str
    file_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "source_module": self.source_module,
            "target_module": self.target_module,
            "dependency_type": self.dependency_type,
            "import_statement": self.import_statement,
            "file_path": self.file_path,
        }


@dataclass
class ModuleStructure:
    """模組結構資訊"""
    
    module_name: str
    root_path: Path
    total_files: int = 0
    python_files: int = 0
    typescript_files: int = 0
    rust_files: int = 0
    go_files: int = 0
    total_lines: int = 0
    directories: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            "module_name": self.module_name,
            "root_path": str(self.root_path),
            "total_files": self.total_files,
            "python_files": self.python_files,
            "typescript_files": self.typescript_files,
            "rust_files": self.rust_files,
            "go_files": self.go_files,
            "total_lines": self.total_lines,
            "directories": self.directories,
        }


class ModuleExplorer:
    """
    模組探索器 - 掃描並分析 AIVA 五大模組
    
    功能:
    1. 自動模組掃描 - 遍歷目錄結構,識別代碼文件
    2. 能力發現與註冊 - AST 分析找裝飾器、CLI 入口、API endpoints
    3. 依賴關係映射 - 導入分析、調用鏈追蹤
    4. 架構圖譜生成 - 模組依賴樹、能力調用圖
    
    遵循規範:
    - 使用 aiva_common.enums (ModuleName, TaskStatus)
    - 類型標註完整
    - 異步支持
    """
    
    # 五大模組定義 (使用 aiva_common.enums.ModuleName)
    MODULES = {
        "CoreModule": "core",
        "CommonModule": "aiva_common", 
        "FeaturesModule": "features",
        "IntegrationModule": "integration",
        "ScanModule": "scan",
    }
    
    # 能力裝飾器識別模式
    CAPABILITY_DECORATORS = {
        "register_capability",
        "cli_command",
        "api_endpoint",
        "tool_function",
    }
    
    def __init__(self, services_root: Optional[Path] = None):
        """
        初始化模組探索器
        
        Args:
            services_root: services/ 目錄的絕對路徑,默認為當前工作目錄的 services/
        """
        if services_root is None:
            # 從當前文件位置推斷 services 目錄
            current_file = Path(__file__).resolve()
            # aiva_core/ai_engine/module_explorer.py -> services/core/aiva_core/ai_engine/module_explorer.py
            # 向上 4 層到 services/
            services_root = current_file.parent.parent.parent.parent
        
        self.services_root = Path(services_root)
        logger.info(f"初始化 ModuleExplorer,services 根目錄: {self.services_root}")
        
        # 模組路徑映射
        self.module_paths: Dict[str, Path] = {}
        for module_name, folder_name in self.MODULES.items():
            module_path = self.services_root / folder_name
            if module_path.exists():
                self.module_paths[module_name] = module_path
            else:
                logger.warning(f"模組路徑不存在: {module_path}")
        
        # 探索結果緩存
        self._capabilities_cache: Dict[str, List[CapabilityInfo]] = {}
        self._dependencies_cache: Dict[str, List[DependencyInfo]] = {}
        self._structure_cache: Dict[str, ModuleStructure] = {}
    
    async def explore_all_modules(self) -> Dict[str, Any]:
        """
        探索所有五大模組
        
        Returns:
            Dict 包含所有模組的探索結果:
            {
                "core": {
                    "structure": ModuleStructure,
                    "capabilities": List[CapabilityInfo],
                    "dependencies": DependencyInfo,
                    "stats": {...}
                },
                ...
            }
        """
        logger.info("🔍 開始探索所有模組...")
        results = {}
        
        for module_name, module_path in self.module_paths.items():
            logger.info(f"📂 探索模組: {module_name}")
            
            try:
                # 1. 掃描目錄結構
                structure = await self._scan_directory_structure(module_name, module_path)
                
                # 2. 分析代碼找能力
                capabilities = await self._discover_capabilities(module_name, module_path)
                
                # 3. 分析依賴關係
                dependencies = await self._analyze_dependencies(module_name, module_path)
                
                # 4. 生成統計資料
                stats = self._generate_stats(structure, capabilities, dependencies)
                
                results[module_name] = {
                    "structure": structure.to_dict(),
                    "capabilities": [cap.to_dict() for cap in capabilities],
                    "dependencies": {
                        "internal": [dep.to_dict() for dep in dependencies["internal"]],
                        "external": [dep.to_dict() for dep in dependencies["external"]],
                    },
                    "stats": stats,
                }
                
                logger.info(f"✅ {module_name} 探索完成: {len(capabilities)} 個能力, {len(dependencies['internal'])} 個內部依賴")
            
            except Exception as e:
                logger.error(f"❌ 探索 {module_name} 時發生錯誤: {e}", exc_info=True)
                results[module_name] = {
                    "error": str(e),
                    "status": "failed",
                }
        
        logger.info(f"🎉 模組探索完成,共處理 {len(results)} 個模組")
        return results
    
    async def _scan_directory_structure(
        self, 
        module_name: str, 
        module_path: Path
    ) -> ModuleStructure:
        """
        掃描模組目錄結構
        
        Args:
            module_name: 模組名稱
            module_path: 模組根目錄路徑
        
        Returns:
            ModuleStructure 包含目錄結構資訊
        """
        structure = ModuleStructure(
            module_name=module_name,
            root_path=module_path,
        )
        
        # 遞歸掃描所有文件
        for item in module_path.rglob("*"):
            if item.is_file():
                structure.total_files += 1
                
                # 統計不同語言文件
                if item.suffix == ".py":
                    structure.python_files += 1
                    # 計算行數 - 處理編碼問題
                    try:
                        try:
                            lines = item.read_text(encoding="utf-8").count("\n")
                        except UnicodeDecodeError:
                            try:
                                lines = item.read_text(encoding="gbk").count("\n")
                            except UnicodeDecodeError:
                                lines = item.read_text(encoding="latin1").count("\n")
                        structure.total_lines += lines
                    except Exception:
                        pass
                
                elif item.suffix in [".ts", ".tsx"]:
                    structure.typescript_files += 1
                
                elif item.suffix == ".rs":
                    structure.rust_files += 1
                
                elif item.suffix == ".go":
                    structure.go_files += 1
            
            elif item.is_dir():
                # 記錄目錄結構
                relative_path = item.relative_to(module_path)
                structure.directories.append(str(relative_path))
        
        self._structure_cache[module_name] = structure
        return structure
    
    async def _discover_capabilities(
        self, 
        module_name: str, 
        module_path: Path
    ) -> List[CapabilityInfo]:
        """
        發現模組中的所有能力
        
        通過以下方式識別能力:
        1. @register_capability 裝飾器
        2. @cli_command 裝飾器
        3. @api_endpoint 裝飾器
        4. CLI 工具類 (繼承自 BaseTool)
        5. FastAPI 路由函數
        
        Args:
            module_name: 模組名稱
            module_path: 模組根目錄路徑
        
        Returns:
            List[CapabilityInfo] 能力列表
        """
        capabilities: List[CapabilityInfo] = []
        
        # 只掃描 Python 文件
        for py_file in module_path.rglob("*.py"):
            try:
                # 讀取並解析文件 - 處理編碼問題
                try:
                    code = py_file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    # 嘗試其他編碼
                    try:
                        code = py_file.read_text(encoding="gbk")
                    except UnicodeDecodeError:
                        code = py_file.read_text(encoding="latin1")
                
                tree = ast.parse(code)
                
                # 查找裝飾器函數
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # 檢查是否有能力裝飾器
                        for decorator in node.decorator_list:
                            if self._is_capability_decorator(decorator):
                                cap = self._extract_capability_from_function(
                                    node, py_file, code, module_name
                                )
                                if cap:
                                    capabilities.append(cap)
                                break
                    
                    elif isinstance(node, ast.ClassDef):
                        # CLI 工具類或 API 類
                        if self._is_tool_class(node):
                            cap = self._extract_capability_from_class(
                                node, py_file, code, module_name
                            )
                            if cap:
                                capabilities.append(cap)
            
            except SyntaxError:
                logger.warning(f"無法解析 {py_file}: 語法錯誤")
            except Exception as e:
                logger.warning(f"分析 {py_file} 時發生錯誤: {e}")
        
        self._capabilities_cache[module_name] = capabilities
        return capabilities
    
    def _is_capability_decorator(self, decorator: ast.AST) -> bool:
        """
        判斷是否為能力裝飾器
        
        Args:
            decorator: AST 裝飾器節點
        
        Returns:
            bool 是否為能力裝飾器
        """
        if isinstance(decorator, ast.Name):
            return decorator.id in self.CAPABILITY_DECORATORS
        
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id in self.CAPABILITY_DECORATORS
        
        return False
    
    def _is_tool_class(self, node: ast.ClassDef) -> bool:
        """
        判斷是否為工具類 (CLI 工具或 API 類)
        
        Args:
            node: AST 類定義節點
        
        Returns:
            bool 是否為工具類
        """
        # 檢查基類
        for base in node.bases:
            if isinstance(base, ast.Name):
                if base.id in ["BaseTool", "BaseAPI", "APIRouter"]:
                    return True
        
        return False
    
    def _extract_capability_from_function(
        self,
        node: ast.FunctionDef,
        file_path: Path,
        code: str,
        module_name: str,
    ) -> Optional[CapabilityInfo]:
        """
        從函數定義提取能力資訊
        
        Args:
            node: AST 函數定義節點
            file_path: 文件路徑
            code: 源代碼
            module_name: 模組名稱
        
        Returns:
            CapabilityInfo 或 None
        """
        try:
            # 提取裝飾器名稱
            decorators = []
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    decorators.append(decorator.id)
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                    decorators.append(decorator.func.id)
            
            # 提取 docstring
            docstring = ast.get_docstring(node)
            
            # 生成能力 ID
            capability_id = f"{module_name}.{file_path.stem}.{node.name}"
            
            # 提取函數簽名
            signature = self._extract_function_signature(node)
            
            return CapabilityInfo(
                capability_id=capability_id,
                name=node.name,
                module=module_name,
                file_path=str(file_path),
                line_number=node.lineno,
                function_name=node.name,
                docstring=docstring,
                signature=signature,
                decorators=decorators,
                language="python",
            )
        
        except Exception as e:
            logger.warning(f"提取函數能力時發生錯誤: {e}")
            return None
    
    def _extract_capability_from_class(
        self,
        node: ast.ClassDef,
        file_path: Path,
        code: str,
        module_name: str,
    ) -> Optional[CapabilityInfo]:
        """從類定義提取能力資訊"""
        try:
            docstring = ast.get_docstring(node)
            capability_id = f"{module_name}.{file_path.stem}.{node.name}"
            
            return CapabilityInfo(
                capability_id=capability_id,
                name=node.name,
                module=module_name,
                file_path=str(file_path),
                line_number=node.lineno,
                class_name=node.name,
                docstring=docstring,
                language="python",
            )
        
        except Exception as e:
            logger.warning(f"提取類能力時發生錯誤: {e}")
            return None
    
    def _extract_function_signature(self, node: ast.FunctionDef) -> str:
        """提取函數簽名"""
        try:
            args = []
            for arg in node.args.args:
                arg_name = arg.arg
                # 嘗試獲取類型標註
                if arg.annotation:
                    arg_type = ast.unparse(arg.annotation)
                    args.append(f"{arg_name}: {arg_type}")
                else:
                    args.append(arg_name)
            
            # 返回值類型
            return_type = ""
            if node.returns:
                return_type = f" -> {ast.unparse(node.returns)}"
            
            return f"{node.name}({', '.join(args)}){return_type}"
        
        except Exception:
            return f"{node.name}(...)"
    
    async def _analyze_dependencies(
        self, 
        module_name: str, 
        module_path: Path
    ) -> Dict[str, List[DependencyInfo]]:
        """
        分析依賴關係
        
        Returns:
            Dict 包含內部和外部依賴:
            {
                "internal": [DependencyInfo, ...],  # 內部模組依賴
                "external": [DependencyInfo, ...]   # 外部庫依賴
            }
        """
        internal_deps: List[DependencyInfo] = []
        external_deps: List[DependencyInfo] = []
        
        for py_file in module_path.rglob("*.py"):
            try:
                # 讀取並解析文件 - 處理編碼問題
                try:
                    code = py_file.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    # 嘗試其他編碼
                    try:
                        code = py_file.read_text(encoding="gbk")
                    except UnicodeDecodeError:
                        code = py_file.read_text(encoding="latin1")
                
                tree = ast.parse(code)
                
                # 分析 import
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            dep = self._create_dependency_info(
                                source_module=module_name,
                                import_name=alias.name,
                                import_statement=f"import {alias.name}",
                                file_path=py_file,
                            )
                            
                            if dep.dependency_type == "internal":
                                internal_deps.append(dep)
                            else:
                                external_deps.append(dep)
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            dep = self._create_dependency_info(
                                source_module=module_name,
                                import_name=node.module,
                                import_statement=f"from {node.module} import ...",
                                file_path=py_file,
                            )
                            
                            if dep.dependency_type == "internal":
                                internal_deps.append(dep)
                            else:
                                external_deps.append(dep)
            
            except Exception as e:
                logger.warning(f"依賴分析失敗 {py_file}: {e}")
        
        return {
            "internal": internal_deps,
            "external": external_deps,
        }
    
    def _create_dependency_info(
        self,
        source_module: str,
        import_name: str,
        import_statement: str,
        file_path: Path,
    ) -> DependencyInfo:
        """創建依賴資訊"""
        # 判斷是內部還是外部依賴
        is_internal = any(
            import_name.startswith(f"services.{folder}") or import_name.startswith(folder)
            for folder in self.MODULES.values()
        )
        
        return DependencyInfo(
            source_module=source_module,
            target_module=import_name,
            dependency_type="internal" if is_internal else "external",
            import_statement=import_statement,
            file_path=str(file_path),
        )
    
    def _generate_stats(
        self,
        structure: ModuleStructure,
        capabilities: List[CapabilityInfo],
        dependencies: Dict[str, List[DependencyInfo]],
    ) -> Dict[str, Any]:
        """生成統計資料"""
        return {
            "total_files": structure.total_files,
            "python_files": structure.python_files,
            "total_lines": structure.total_lines,
            "total_capabilities": len(capabilities),
            "internal_dependencies": len(dependencies["internal"]),
            "external_dependencies": len(dependencies["external"]),
            "directories": len(structure.directories),
        }
    
    def export_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """
        導出探索結果到 JSON 文件
        
        Args:
            results: explore_all_modules() 的返回結果
            output_path: 輸出文件路徑
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 探索結果已導出到: {output_path}")
        
        except Exception as e:
            logger.error(f"❌ 導出結果失敗: {e}", exc_info=True)
