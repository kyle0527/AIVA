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

# 常數定義
INIT_FILE = "__init__.py"

logger = logging.getLogger(__name__)


class ModuleExplorer:
    """模組探索器
    
    職責：掃描 AIVA 五大模組的文件結構，為 AI 自我認知提供數據源
    
    掃描目標模組：
    - core/aiva_core: 核心智能系統
    - scan: 掃描模組
    - features: 功能模組  
    - integration: 整合模組
    
    支援語言:
    - Python (.py)
    - Go (.go)
    - Rust (.rs)
    - TypeScript (.ts)
    - JavaScript (.js)
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
        # 支援的文件類型
        self.file_extensions = {
            "python": "*.py",
            "go": "*.go",
            "rust": "*.rs",
            "typescript": "*.ts",
            "javascript": "*.js"
        }
        logger.info(f"ModuleExplorer initialized with root: {self.root_path}")
        logger.info(f"Supported languages: {', '.join(self.file_extensions.keys())}")
    
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
    
    def explore_all_modules(self) -> dict[str, Any]:
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
                results[module] = self._explore_module(module_path)
            else:
                logger.warning(f"  Module not found: {module_path}")
        
        logger.info(f"✅ Module exploration completed: {len(results)} modules scanned")
        return results
    
    def _explore_module(self, path: Path) -> dict[str, Any]:
        """探索單一模組 (掃描多語言文件)
        
        Args:
            path: 模組路徑
            
        Returns:
            模組資訊字典
        """
        files = []
        total_size = 0
        file_counts = dict.fromkeys(self.file_extensions.keys(), 0)
        
        # 掃描所有支援的語言文件
        for lang, pattern in self.file_extensions.items():
            for file_path in path.rglob(pattern):
                # 跳過特殊目錄和測試文件
                if any(skip in str(file_path) for skip in ["__pycache__", "node_modules", "target", ".git"]):
                    continue
                if file_path.name.startswith("test_") or file_path.name.endswith("_test.go"):
                    continue
                
                file_size = file_path.stat().st_size
                files.append({
                    "path": str(file_path.relative_to(path)),
                    "type": lang,
                    "size": file_size,
                    "name": file_path.name,
                    "language": lang
                })
                total_size += file_size
                file_counts[lang] += 1
        
        # 分析模組結構
        structure = self._analyze_structure(path)
        
        return {
            "path": str(path),
            "files": files,
            "structure": structure,
            "stats": {
                "total_files": sum(file_counts.values()),
                "total_size": total_size,
                "subdirectories": len(structure.get("subdirectories", [])),
                "by_language": file_counts
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
                    "has_init": (item / INIT_FILE).exists(),
                    "is_package": (item / INIT_FILE).exists()
                })
        
        return {
            "subdirectories": subdirs,
            "is_package": (path / INIT_FILE).exists(),
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
