"""Capability Analyzer - 能力分析器 (增強版)

識別和分析 AIVA 系統的功能能力,支援多語言分析:
- Python: AST 解析 @register_capability 裝飾器
- Go/Rust/TypeScript: 使用 language_extractors 正則提取

增強功能:
- 完善的錯誤處理和追蹤
- 詳細的提取報告
- 文件大小驗證
- 編碼錯誤處理

遵循 aiva_common 修復規範:
- 使用標準裝飾器模式
- 統一的能力元數據格式
"""

import ast
import logging
from pathlib import Path
from typing import Any
from dataclasses import dataclass
from datetime import datetime, timezone

from .language_extractors import get_extractor

logger = logging.getLogger(__name__)


@dataclass
class ExtractionError:
    """提取錯誤記錄"""
    file_path: str
    language: str
    error_type: str
    error_message: str
    timestamp: str


class CapabilityAnalyzer:
    """能力分析器
    
    職責：識別系統中所有註冊的能力函數，提取其元數據
    
    識別目標：
    - @register_capability 裝飾的函數
    - @capability 裝飾的函數
    - 包含 'capability' 關鍵字的裝飾器
    """
    
    def __init__(self):
        """初始化能力分析器 (增強版)"""
        self.capabilities_cache: dict[str, list[dict]] = {}
        self.extraction_errors: list[ExtractionError] = []  # ✅ 錯誤追蹤
        self.stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "skipped_files": 0
        }
        logger.info("CapabilityAnalyzer initialized (Enhanced Mode)")
    
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
                
                # 跳過 __init__.py
                if file_path.name == "__init__.py":
                    continue
                
                caps = await self._extract_capabilities_from_file(file_path, module_name)
                capabilities.extend(caps)
        
        logger.info(f"✅ Capability analysis completed: {len(capabilities)} capabilities found")
        return capabilities
    
    async def _extract_capabilities_from_file(self, file_path: Path, module: str) -> list[dict]:
        """從文件中提取能力 (增強錯誤處理版)
        
        Args:
            file_path: 文件路徑 (.py/.go/.rs/.ts/.js)
            module: 所屬模組名稱
            
        Returns:
            能力列表
        """
        self.stats["total_files"] += 1
        language = "unknown"
        
        try:
            # ✅ 驗證文件存在
            if not file_path.exists():
                logger.error(f"  ❌ File not found: {file_path}")
                self._record_error(
                    file_path, "unknown", "FileNotFoundError",
                    f"File does not exist: {file_path}"
                )
                self.stats["failed_files"] += 1
                return []
            
            # ✅ 驗證文件大小 (跳過過大文件)
            file_size = file_path.stat().st_size
            if file_size > 5 * 1024 * 1024:  # 5MB
                logger.warning(f"  ⚠️  Skipping large file: {file_path.name} ({file_size / 1024 / 1024:.1f}MB)")
                self.stats["skipped_files"] += 1
                return []
            
            # 檢測語言
            language = self._detect_language(file_path)
            if language == "unknown":
                logger.warning(f"  ⚠️  Unknown file type: {file_path.suffix}")
                self.stats["skipped_files"] += 1
                return []
            
            # 提取能力
            if language == "python":
                capabilities = self._extract_python_capabilities(file_path, module)
            else:
                capabilities = self._extract_non_python_capabilities(file_path, module, language)
            
            self.stats["successful_files"] += 1
            return capabilities
            
        except PermissionError as e:
            logger.error(f"  ❌ Permission denied: {file_path.name}")
            self._record_error(file_path, language, "PermissionError", str(e))
            self.stats["failed_files"] += 1
            return []
        
        except UnicodeDecodeError as e:
            logger.error(f"  ❌ Encoding error: {file_path.name}")
            self._record_error(file_path, language, "UnicodeDecodeError", str(e))
            self.stats["failed_files"] += 1
            return []
        
        except Exception as e:
            logger.exception(f"  ❌ Unexpected error extracting from {file_path.name}: {e}")
            self._record_error(file_path, language, type(e).__name__, str(e))
            self.stats["failed_files"] += 1
            return []
    
    def _detect_language(self, file_path: Path) -> str:
        """檢測文件語言
        
        Args:
            file_path: 文件路徑
            
        Returns:
            語言名稱: python, go, rust, typescript, javascript
        """
        suffix = file_path.suffix.lower()
        language_map = {
            ".py": "python",
            ".go": "go",
            ".rs": "rust",
            ".ts": "typescript",
            ".js": "javascript"
        }
        return language_map.get(suffix, "unknown")
    
    def _extract_python_capabilities(self, file_path: Path, module: str) -> list[dict]:
        """從 Python 文件提取能力 (使用 AST)
        
        支持兩種模式:
        1. @capability 裝飾器標記的函數
        2. Manager 類中的 public async 方法 (function_* 模組)
        
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
            
            # 檢測是否為 Manager 文件
            is_manager_file = file_path.name == "manager.py" and "function_" in str(file_path.parent.name)
            
            for node in ast.walk(tree):
                # 模式 1: 裝飾器標記的函數
                if isinstance(node, ast.FunctionDef):
                    if self._has_capability_decorator(node):
                        cap = self._extract_capability_info(node, file_path, module)
                        capabilities.append(cap)
                
                # 模式 2: Manager 類方法 (新增)
                if is_manager_file and isinstance(node, ast.ClassDef):
                    if node.name.endswith("Manager"):
                        # 提取 Manager 類中的所有 public async 方法
                        for item in node.body:
                            if isinstance(item, ast.AsyncFunctionDef):
                                # 排除私有方法和特殊方法
                                if not item.name.startswith('_') and item.name not in ['__init__', '__aenter__', '__aexit__']:
                                    cap = self._extract_capability_info(item, file_path, module)
                                    cap["manager_class"] = node.name
                                    cap["is_manager_method"] = True
                                    capabilities.append(cap)
                                    logger.debug(f"  Found Manager method: {node.name}.{item.name}")
            
            if capabilities:
                logger.debug(f"  Found {len(capabilities)} Python capabilities in {file_path.name}")
            
        except SyntaxError as e:
            logger.warning(f"  Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"  Failed to parse {file_path}: {e}")
        
        return capabilities
    
    def _extract_non_python_capabilities(
        self, 
        file_path: Path, 
        module: str,
        language: str
    ) -> list[dict]:
        """從非 Python 文件提取能力 (使用 language_extractors)
        
        Args:
            file_path: 文件路徑
            module: 所屬模組名稱
            language: 語言名稱
            
        Returns:
            能力列表
        """
        try:
            # 獲取對應語言的提取器
            extractor = get_extractor(language)
            if not extractor:
                logger.warning(f"  No extractor available for language: {language}")
                return []
            
            # 讀取文件內容
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            
            # 使用提取器提取能力
            capabilities = extractor.extract_capabilities(content, str(file_path))
            
            # 添加 module 信息 (extractor 可能沒有設置)
            for cap in capabilities:
                if "module" not in cap or not cap["module"]:
                    cap["module"] = module
            
            if capabilities:
                logger.debug(f"  Found {len(capabilities)} {language} capabilities in {file_path.name}")
            
            return capabilities
            
        except Exception as e:
            logger.error(f"  Failed to extract from {file_path}: {e}")
            return []
    
    def _has_capability_decorator(self, node: ast.FunctionDef) -> bool:
        """檢查函數是否為能力函數
        
        策略：
        1. 有 @capability 裝飾器的函數（明確標記）
        2. async def 函數（異步能力）
        3. 公開函數（非 _ 開頭）且有文檔字串
        
        Args:
            node: AST 函數定義節點
            
        Returns:
            是否為能力函數
        """
        # 策略 1: 檢查裝飾器
        if self._check_decorator_for_capability(node):
            return True
        
        # 策略 2 & 3: 公開的異步函數或有文檔的函數
        if node.name.startswith('_'):
            return False
            
        # 異步函數
        if isinstance(node, ast.AsyncFunctionDef):
            return True
        
        # 有實質性文檔的函數
        docstring = ast.get_docstring(node)
        return bool(docstring and len(docstring) > 20)
    
    def _check_decorator_for_capability(self, node: ast.FunctionDef) -> bool:
        """檢查裝飾器是否包含 capability
        
        Args:
            node: AST 函數定義節點
            
        Returns:
            是否有 capability 相關裝飾器
        """
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if "capability" in decorator.id.lower():
                    return True
            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    if "capability" in decorator.func.id.lower():
                        return True
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
            "language": "python",  # ✅ 添加語言欄位
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
    
    def _record_error(
        self,
        file_path: Path,
        language: str,
        error_type: str,
        error_message: str
    ):
        """記錄提取錯誤
        
        Args:
            file_path: 文件路徑
            language: 語言名稱
            error_type: 錯誤類型
            error_message: 錯誤訊息
        """
        error = ExtractionError(
            file_path=str(file_path),
            language=language,
            error_type=error_type,
            error_message=error_message,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        self.extraction_errors.append(error)
    
    def get_extraction_report(self) -> dict:
        """獲取詳細的提取報告
        
        Returns:
            包含統計和錯誤詳情的報告
        """
        return {
            "statistics": self.stats,
            "success_rate": (
                self.stats["successful_files"] / self.stats["total_files"] * 100
                if self.stats["total_files"] > 0 else 0
            ),
            "total_errors": len(self.extraction_errors),
            "errors_by_type": self._group_errors_by_type(),
            "errors_by_language": self._group_errors_by_language(),
            "recent_errors": [
                {
                    "file": err.file_path,
                    "language": err.language,
                    "type": err.error_type,
                    "message": err.error_message[:100]  # 截斷長訊息
                }
                for err in self.extraction_errors[:10]  # 前 10 個錯誤
            ]
        }
    
    def _group_errors_by_type(self) -> dict[str, int]:
        """按錯誤類型分組
        
        Returns:
            錯誤類型統計字典
        """
        error_counts = {}
        for err in self.extraction_errors:
            error_counts[err.error_type] = error_counts.get(err.error_type, 0) + 1
        return error_counts
    
    def _group_errors_by_language(self) -> dict[str, int]:
        """按語言分組錯誤
        
        Returns:
            語言錯誤統計字典
        """
        lang_counts = {}
        for err in self.extraction_errors:
            lang_counts[err.language] = lang_counts.get(err.language, 0) + 1
        return lang_counts
    
    def print_extraction_report(self):
        """打印提取報告 (美化輸出)"""
        report = self.get_extraction_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 Capability Extraction Report")
        logger.info("=" * 60)
        
        stats = report["statistics"]
        logger.info("\n📁 Files Processed:")
        logger.info(f"  Total:      {stats['total_files']}")
        logger.info(f"  ✅ Success:  {stats['successful_files']}")
        logger.info(f"  ❌ Failed:   {stats['failed_files']}")
        logger.info(f"  ⚠️  Skipped:  {stats['skipped_files']}")
        logger.info(f"  Success Rate: {report['success_rate']:.1f}%")
        
        if report["total_errors"] > 0:
            logger.warning(f"\n⚠️  Errors: {report['total_errors']}")
            logger.warning("\nBy Type:")
            for err_type, count in report["errors_by_type"].items():
                logger.warning(f"  - {err_type}: {count}")
            
            if report["errors_by_language"]:
                logger.warning("\nBy Language:")
                for lang, count in report["errors_by_language"].items():
                    logger.warning(f"  - {lang}: {count}")
        
        logger.info("=" * 60 + "\n")
