"""Module Explorer - 模組探索器

掃描 AIVA 五大模組的文件結構，為能力分析提供基礎數據

遵循 aiva_common 修復規範:
- 使用 aiva_common.enums 的統一枚舉
- 使用 aiva_common.schemas 的統一 Schema
- 錯誤處理使用 aiva_common.exceptions
"""

import ast
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ModuleExplorer:
    """模組探索器
    
    職責：掃描 AIVA 五大模組的文件結構，為 AI 自我認知提供數據源
    
    掃描目標模組：
    - core/aiva_core: 核心智能系統
    - scan: 掃描模組
    - features: 功能模組  
    - integration: 整合模組
    """
    
    def __init__(self, root_path: Path | None = None):
        """初始化模組探索器
        
        Args:
            root_path: 專案根目錄，默認自動推斷
        """
        self.root_path = root_path or self._infer_root_path()
        self.target_modules = [
            "core/aiva_core",
            "scan",
            "features",
            "integration"
        ]
        logger.info(f"ModuleExplorer initialized with root: {self.root_path}")
    
    def _infer_root_path(self) -> Path:
        """推斷專案根目錄"""
        current = Path(__file__).resolve()
        # 向上尋找直到找到 services 目錄
        while current.name != "services" and current.parent != current:
            current = current.parent
        
        if current.name == "services":
            return current
        
        # 降級方案：使用當前文件的相對路徑
        return Path(__file__).parent.parent.parent.parent
    
    async def explore_all_modules(self) -> dict[str, Any]:
        """掃描所有目標模組
        
        Returns:
            {
                "module_name": {
                    "path": str,
                    "files": [{"path": str, "type": str, "size": int}],
                    "structure": dict,
                    "stats": dict
                }
            }
        """
        logger.info("🔍 Starting module exploration...")
        results = {}
        
        for module in self.target_modules:
            module_path = self.root_path / module
            
            if module_path.exists():
                logger.info(f"  Exploring: {module}")
                results[module] = await self._explore_module(module_path)
            else:
                logger.warning(f"  Module not found: {module_path}")
        
        logger.info(f"✅ Module exploration completed: {len(results)} modules scanned")
        return results
    
    async def _explore_module(self, path: Path) -> dict[str, Any]:
        """探索單一模組
        
        Args:
            path: 模組路徑
            
        Returns:
            模組資訊字典
        """
        files = []
        total_size = 0
        python_files = 0
        
        # 掃描所有 Python 文件
        for py_file in path.rglob("*.py"):
            # 跳過 __pycache__ 和測試文件
            if "__pycache__" in str(py_file) or py_file.name.startswith("test_"):
                continue
            
            file_size = py_file.stat().st_size
            files.append({
                "path": str(py_file.relative_to(path)),
                "type": "python",
                "size": file_size,
                "name": py_file.name
            })
            total_size += file_size
            python_files += 1
        
        # 分析模組結構
        structure = self._analyze_structure(path)
        
        return {
            "path": str(path),
            "files": files,
            "structure": structure,
            "stats": {
                "total_files": python_files,
                "total_size": total_size,
                "subdirectories": len(structure.get("subdirectories", []))
            }
        }
    
    def _analyze_structure(self, path: Path) -> dict:
        """分析模組結構
        
        Args:
            path: 模組路徑
            
        Returns:
            結構資訊
        """
        subdirs = []
        
        for item in path.iterdir():
            if item.is_dir() and not item.name.startswith(("_", ".")):
                subdirs.append({
                    "name": item.name,
                    "has_init": (item / "__init__.py").exists(),
                    "is_package": (item / "__init__.py").exists()
                })
        
        return {
            "subdirectories": subdirs,
            "is_package": (path / "__init__.py").exists(),
            "has_readme": (path / "README.md").exists()
        }
    
    def get_module_summary(self, module_name: str) -> str:
        """獲取模組摘要資訊
        
        Args:
            module_name: 模組名稱
            
        Returns:
            可讀的摘要字串
        """
        module_path = self.root_path / module_name
        
        if not module_path.exists():
            return f"Module '{module_name}' not found"
        
        python_files = list(module_path.rglob("*.py"))
        total_lines = 0
        
        for py_file in python_files:
            try:
                with open(py_file, encoding="utf-8") as f:
                    total_lines += len(f.readlines())
            except Exception:
                pass
        
        return (
            f"Module: {module_name}\n"
            f"  Files: {len(python_files)}\n"
            f"  Total Lines: {total_lines}\n"
            f"  Path: {module_path}"
        )
