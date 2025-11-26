#!/usr/bin/env python3
"""
智能功能模組恢復工具
- 分析 V1 備份和 V2 capability 的功能重疊
- 智能合併，保留最佳實現
- 建立 V1/V2 轉換合約
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Set
import json

class SmartFeatureRestorer:
    """智能功能恢復器"""
    
    def __init__(self):
        self.v1_backup_path = Path("C:/Users/User/Downloads/新增資料夾 (3)/AIVA_V1_features")
        self.v2_capability_path = Path("services/integration/capability")
        self.features_path = Path("services/features")
        
        self.analysis_result = {
            "v1_only": [],      # 只在 V1 中存在
            "v2_only": [],      # 只在 V2 中存在  
            "overlap": [],      # 兩邊都有，需要合併
            "conversion_map": {}  # 轉換映射
        }

    def analyze_functionality_overlap(self):
        """分析功能重疊情況"""
        print("🔍 分析功能重疊情況...")
        
        # V1 功能列表
        v1_functions = set()
        if self.v1_backup_path.exists():
            for item in self.v1_backup_path.iterdir():
                if item.is_dir() and item.name.startswith('function_'):
                    func_name = item.name.replace('function_', '')
                    v1_functions.add(func_name)
        
        # V2 功能列表
        v2_functions = set()
        if self.v2_capability_path.exists():
            for item in self.v2_capability_path.glob('*.py'):
                if 'sql' in item.name or 'xss' in item.name or 'injection' in item.name:
                    # 提取功能名
                    func_name = item.stem.replace('_tools', '').replace('_attack', '')
                    v2_functions.add(func_name)
        
        print(f"V1 功能: {v1_functions}")
        print(f"V2 功能: {v2_functions}")
        
        # 分類
        self.analysis_result["v1_only"] = list(v1_functions - v2_functions)
        self.analysis_result["v2_only"] = list(v2_functions - v1_functions)  
        self.analysis_result["overlap"] = list(v1_functions & v2_functions)
        
        return self.analysis_result

    def create_conversion_contracts(self):
        """建立轉換合約"""
        print("📋 建立 V1/V2 轉換合約...")
        
        conversion_template = """
# V1/V2 功能轉換合約
class {function_name}Converter:
    '''V1 到 V2 的功能轉換器'''
    
    @staticmethod
    def v1_to_v2_request(v1_request: dict) -> 'V2Request':
        '''V1 請求轉換為 V2 格式'''
        return V2Request(
            target=v1_request.get('url'),
            parameters=v1_request.get('params', {{}}),
            method=v1_request.get('method', 'GET')
        )
    
    @staticmethod  
    def v2_to_v1_response(v2_response: 'V2Response') -> dict:
        '''V2 響應轉換為 V1 格式'''
        return {{
            'vulnerable': v2_response.findings_count > 0,
            'findings': [f.to_dict() for f in v2_response.findings],
            'confidence': v2_response.confidence_score
        }}
"""
        
        # 為每個重疊功能創建轉換器
        converters_dir = self.features_path / "converters"
        converters_dir.mkdir(exist_ok=True)
        
        for func in self.analysis_result["overlap"]:
            converter_file = converters_dir / f"{func}_converter.py"
            with open(converter_file, 'w', encoding='utf-8') as f:
                f.write(conversion_template.format(function_name=func.title()))
        
        print(f"✅ 創建了 {len(self.analysis_result['overlap'])} 個轉換合約")

    def restore_v1_exclusive_features(self):
        """恢復 V1 獨有功能"""
        print("🔄 恢復 V1 獨有功能...")
        
        for func in self.analysis_result["v1_only"]:
            source_dir = self.v1_backup_path / f"function_{func}"
            target_dir = self.features_path / f"function_{func}"
            
            if source_dir.exists():
                shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                print(f"  ✅ 恢復: function_{func}")
        
        print(f"✅ 恢復了 {len(self.analysis_result['v1_only'])} 個 V1 獨有功能")

    def create_unified_interface(self):
        """創建統一接口"""
        print("🔗 創建統一功能接口...")
        
        interface_code = '''
"""
AIVA 統一功能接口 - 支援 V1/V2 雙架構
"""

from typing import Dict, Any, Union
from dataclasses import dataclass

@dataclass
class UnifiedRequest:
    """統一請求格式"""
    target: str
    method: str = "GET"
    parameters: Dict[str, Any] = None
    headers: Dict[str, str] = None

@dataclass  
class UnifiedResponse:
    """統一響應格式"""
    success: bool
    findings: List[Dict[str, Any]]
    confidence: float
    execution_time: float

class UnifiedFunctionInterface:
    """統一功能接口 - 自動路由到 V1 或 V2"""
    
    def __init__(self):
        self.v1_functions = {}  # V1 功能註冊
        self.v2_capabilities = {}  # V2 功能註冊
        self.converters = {}  # 轉換器註冊
    
    async def execute_function(self, 
                             function_name: str, 
                             request: UnifiedRequest) -> UnifiedResponse:
        """執行功能 - 自動選擇最佳版本"""
        
        # 優先使用 V2（如果可用）
        if function_name in self.v2_capabilities:
            return await self._execute_v2(function_name, request)
        
        # 回退到 V1
        elif function_name in self.v1_functions:
            return await self._execute_v1(function_name, request)
        
        else:
            raise ValueError(f"功能不存在: {function_name}")
    
    async def _execute_v1(self, func_name: str, request: UnifiedRequest):
        """執行 V1 功能"""
        v1_worker = self.v1_functions[func_name]
        
        # 轉換請求格式
        v1_request = {
            'url': request.target,
            'method': request.method,
            'params': request.parameters or {}
        }
        
        result = await v1_worker.execute(v1_request)
        
        # 標準化響應
        return UnifiedResponse(
            success=result.get('success', False),
            findings=result.get('findings', []),
            confidence=result.get('confidence', 0.0),
            execution_time=result.get('execution_time', 0.0)
        )
    
    async def _execute_v2(self, func_name: str, request: UnifiedRequest):
        """執行 V2 功能"""
        v2_capability = self.v2_capabilities[func_name]
        
        # 使用 V2 原生格式
        result = await v2_capability.execute(request)
        return result
'''

        interface_file = self.features_path / "unified_interface.py"
        with open(interface_file, 'w', encoding='utf-8') as f:
            f.write(interface_code)
        
        print("✅ 創建統一功能接口完成")

    def generate_feature_registry(self):
        """生成功能註冊表"""
        print("📋 生成功能註冊表...")
        
        registry = {
            "version": "1.0",
            "features": {
                "v1_functions": {},
                "v2_capabilities": {},
                "conversion_available": self.analysis_result["overlap"]
            }
        }
        
        # V1 功能註冊
        for func in self.analysis_result["v1_only"] + self.analysis_result["overlap"]:
            registry["features"]["v1_functions"][func] = {
                "module": f"services.features.function_{func}",
                "entry_point": "worker.py",
                "status": "active"
            }
        
        # V2 功能註冊 
        for func in self.analysis_result["v2_only"] + self.analysis_result["overlap"]:
            registry["features"]["v2_capabilities"][func] = {
                "module": f"services.integration.capability.{func}_tools",
                "entry_point": f"{func.title()}Capability",
                "status": "active" 
            }
        
        registry_file = self.features_path / "feature_registry.json"
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)
        
        print("✅ 功能註冊表生成完成")

    def run_full_restoration(self):
        """執行完整恢復流程"""
        print("🚀 開始智能功能模組恢復...")
        print("="*50)
        
        # 創建 Features 目錄結構
        self.features_path.mkdir(exist_ok=True)
        
        # 執行分析和恢復
        self.analyze_functionality_overlap()
        print()
        
        self.create_conversion_contracts()
        print()
        
        self.restore_v1_exclusive_features() 
        print()
        
        self.create_unified_interface()
        print()
        
        self.generate_feature_registry()
        print()
        
        print("🎉 智能功能模組恢復完成！")
        print("="*50)
        
        # 輸出結果摘要
        print("📊 恢復摘要:")
        print(f"  V1 獨有功能: {len(self.analysis_result['v1_only'])} 個")
        print(f"  V2 獨有功能: {len(self.analysis_result['v2_only'])} 個") 
        print(f"  重疊功能: {len(self.analysis_result['overlap'])} 個")
        print(f"  轉換合約: {len(self.analysis_result['overlap'])} 個")
        print()
        print("✅ Features 模組現在擁有完整功能 + 智能轉換能力!")

if __name__ == "__main__":
    restorer = SmartFeatureRestorer()
    restorer.run_full_restoration()