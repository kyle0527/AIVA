"""
測試 ModuleExplorer - 模組探索器單元測試

Author: AIVA AI Engine Team
Version: 1.0.0
Created: 2025-11-13
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from aiva_core.ai_engine.module_explorer import (
    ModuleExplorer,
    CapabilityInfo,
    DependencyInfo,
    ModuleStructure,
)
from aiva_common.enums import ModuleName


class TestModuleExplorer:
    """測試 ModuleExplorer 類"""
    
    @pytest.fixture
    def services_root(self, tmp_path):
        """創建臨時 services 目錄結構"""
        # 創建五大模組目錄
        (tmp_path / "core").mkdir()
        (tmp_path / "aiva_common").mkdir()
        (tmp_path / "features").mkdir()
        (tmp_path / "integration").mkdir()
        (tmp_path / "scan").mkdir()
        
        return tmp_path
    
    @pytest.fixture
    def explorer(self, services_root):
        """創建 ModuleExplorer 實例"""
        return ModuleExplorer(services_root=services_root)
    
    def test_initialization(self, explorer, services_root):
        """測試初始化"""
        assert explorer.services_root == services_root
        assert len(explorer.module_paths) == 5
        assert ModuleName.CORE.value in explorer.module_paths
        assert ModuleName.COMMON.value in explorer.module_paths
    
    @pytest.mark.asyncio
    async def test_scan_directory_structure(self, explorer, services_root):
        """測試目錄掃描"""
        # 創建測試文件
        core_path = services_root / "core"
        (core_path / "test.py").write_text("# test file\nprint('hello')\n")
        (core_path / "subdir").mkdir()
        (core_path / "subdir" / "test2.py").write_text("# test2\n")
        
        structure = await explorer._scan_directory_structure(
            ModuleName.CORE.value,
            core_path
        )
        
        assert structure.module_name == ModuleName.CORE.value
        assert structure.total_files == 2
        assert structure.python_files == 2
        assert structure.total_lines > 0
    
    def test_is_capability_decorator(self, explorer):
        """測試能力裝飾器識別"""
        import ast
        
        # 測試簡單裝飾器
        code1 = "@register_capability\ndef test(): pass"
        tree1 = ast.parse(code1)
        func1 = tree1.body[0]
        assert explorer._is_capability_decorator(func1.decorator_list[0]) is True
        
        # 測試帶參數的裝飾器
        code2 = "@register_capability(name='test')\ndef test(): pass"
        tree2 = ast.parse(code2)
        func2 = tree2.body[0]
        assert explorer._is_capability_decorator(func2.decorator_list[0]) is True
        
        # 測試非能力裝飾器
        code3 = "@property\ndef test(): pass"
        tree3 = ast.parse(code3)
        func3 = tree3.body[0]
        assert explorer._is_capability_decorator(func3.decorator_list[0]) is False
    
    def test_extract_function_signature(self, explorer):
        """測試函數簽名提取"""
        import ast
        
        code = """
def test_func(a: int, b: str) -> bool:
    return True
"""
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        signature = explorer._extract_function_signature(func_node)
        
        assert "test_func" in signature
        assert "a: int" in signature
        assert "b: str" in signature
        assert "-> bool" in signature
    
    @pytest.mark.asyncio
    async def test_discover_capabilities(self, explorer, services_root):
        """測試能力發現"""
        # 創建包含能力的 Python 文件
        core_path = services_root / "core"
        test_file = core_path / "test_capabilities.py"
        
        test_code = '''
"""測試模組"""

from typing import Optional

def register_capability(func):
    """能力裝飾器"""
    return func

@register_capability
def scan_target(target: str) -> dict:
    """掃描目標"""
    return {"status": "ok"}

class BaseTool:
    """基礎工具類"""
    pass

class SQLMapTool(BaseTool):
    """SQLMap 工具"""
    
    def run(self):
        pass
'''
        test_file.write_text(test_code)
        
        capabilities = await explorer._discover_capabilities(
            ModuleName.CORE.value,
            core_path
        )
        
        # 應該找到 2 個能力 (函數 + 類)
        assert len(capabilities) >= 1
        
        # 檢查函數能力
        func_caps = [c for c in capabilities if c.function_name == "scan_target"]
        assert len(func_caps) == 1
        assert func_caps[0].module == ModuleName.CORE.value
        assert "register_capability" in func_caps[0].decorators
    
    @pytest.mark.asyncio
    async def test_analyze_dependencies(self, explorer, services_root):
        """測試依賴分析"""
        # 創建包含導入的 Python 文件
        core_path = services_root / "core"
        test_file = core_path / "test_deps.py"
        
        test_code = '''
import os
import sys
from typing import Dict
from services.aiva_common import Severity
from aiva_common.enums import TaskStatus
import requests
'''
        test_file.write_text(test_code)
        
        dependencies = await explorer._analyze_dependencies(
            ModuleName.CORE.value,
            core_path
        )
        
        # 應該有內部和外部依賴
        assert "internal" in dependencies
        assert "external" in dependencies
        
        # 檢查內部依賴
        internal_targets = [d.target_module for d in dependencies["internal"]]
        assert any("aiva_common" in t for t in internal_targets)
        
        # 檢查外部依賴
        external_targets = [d.target_module for d in dependencies["external"]]
        assert any(t in ["os", "sys", "typing", "requests"] for t in external_targets)
    
    @pytest.mark.asyncio
    async def test_explore_all_modules(self, explorer, services_root):
        """測試完整探索流程"""
        # 為每個模組創建測試文件
        for module_name in [ModuleName.CORE.value, ModuleName.COMMON.value]:
            module_path = services_root / explorer.MODULES[module_name]
            test_file = module_path / "test.py"
            test_file.write_text("""
def test_function():
    pass
""")
        
        results = await explorer.explore_all_modules()
        
        # 應該返回所有模組的結果
        assert len(results) == 5
        
        # 檢查 Core 模組結果
        core_result = results.get(ModuleName.CORE.value)
        assert core_result is not None
        assert "structure" in core_result
        assert "capabilities" in core_result
        assert "dependencies" in core_result
        assert "stats" in core_result
    
    def test_export_results(self, explorer, tmp_path):
        """測試結果導出"""
        results = {
            ModuleName.CORE.value: {
                "structure": {"total_files": 10},
                "capabilities": [],
                "dependencies": {"internal": [], "external": []},
                "stats": {},
            }
        }
        
        output_path = tmp_path / "results.json"
        explorer.export_results(results, output_path)
        
        assert output_path.exists()
        
        # 讀取並驗證
        import json
        with open(output_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        assert ModuleName.CORE.value in loaded


class TestCapabilityInfo:
    """測試 CapabilityInfo 數據類"""
    
    def test_to_dict(self):
        """測試轉換為字典"""
        cap = CapabilityInfo(
            capability_id="core.test.func",
            name="test_func",
            module=ModuleName.CORE.value,
            file_path="/path/to/file.py",
            line_number=10,
            function_name="test_func",
            docstring="Test function",
            signature="test_func() -> None",
            decorators=["register_capability"],
        )
        
        result = cap.to_dict()
        
        assert result["capability_id"] == "core.test.func"
        assert result["name"] == "test_func"
        assert result["module"] == ModuleName.CORE.value
        assert result["decorators"] == ["register_capability"]


class TestDependencyInfo:
    """測試 DependencyInfo 數據類"""
    
    def test_to_dict(self):
        """測試轉換為字典"""
        dep = DependencyInfo(
            source_module=ModuleName.CORE.value,
            target_module="aiva_common",
            dependency_type="internal",
            import_statement="from aiva_common import Severity",
            file_path="/path/to/file.py",
        )
        
        result = dep.to_dict()
        
        assert result["source_module"] == ModuleName.CORE.value
        assert result["target_module"] == "aiva_common"
        assert result["dependency_type"] == "internal"


class TestModuleStructure:
    """測試 ModuleStructure 數據類"""
    
    def test_to_dict(self):
        """測試轉換為字典"""
        structure = ModuleStructure(
            module_name=ModuleName.CORE.value,
            root_path=Path("/services/core"),
            total_files=100,
            python_files=80,
            total_lines=5000,
        )
        
        result = structure.to_dict()
        
        assert result["module_name"] == ModuleName.CORE.value
        assert result["total_files"] == 100
        assert result["python_files"] == 80
        assert result["total_lines"] == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
