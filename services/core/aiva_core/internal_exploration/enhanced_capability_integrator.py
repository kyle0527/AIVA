"""
AIVA 增強能力集成器
Enhanced Capability Integrator

功能：
1. 讀取增強分類處理器的輸出
2. 為 aiva_internal_executor 提供AI友善的能力描述
3. 支持按模組、複雜度、風險等級篩選能力
4. 提供實際的CLI命令建議
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加路徑以便導入
sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class CapabilityReference:
    """能力引用數據結構"""
    flow_id: int
    capability_name: str
    display_name: str
    module_name: str
    complexity: str
    scope: str
    ai_level: str
    risk_level: str
    purpose: str
    example_command: str
    estimated_duration: str


class EnhancedCapabilityIntegrator:
    """增強能力集成器"""
    
    def __init__(self, capabilities_data_path: str = None, verbose: bool = True):
        """
        初始化集成器
        
        Args:
            capabilities_data_path: 增強分類數據路徑
            verbose: 詳細輸出模式
        """
        self.verbose = verbose
        self.capabilities_data = None
        self.capability_index = {}  # 快速查找索引
        self.module_index = {}      # 模組索引
        
        if capabilities_data_path:
            self.load_capabilities_data(capabilities_data_path)
    
    def load_capabilities_data(self, data_path: str) -> bool:
        """載入增強分類數據"""
        try:
            data_file = Path(data_path)
            if not data_file.exists():
                if self.verbose:
                    print(f"[Error] 找不到能力數據文件: {data_path}")
                return False
            
            with open(data_file, 'r', encoding='utf-8') as f:
                self.capabilities_data = json.load(f)
            
            # 建立索引
            self._build_indexes()
            
            if self.verbose:
                total_caps = self.capabilities_data['metadata']['total_capabilities']
                total_modules = self.capabilities_data['metadata']['total_modules']
                print(f"[OK] 載入能力數據：{total_caps} 個能力，{total_modules} 個模組")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[Error] 載入能力數據失敗: {e}")
            return False
    
    def _build_indexes(self):
        """建立快速查找索引"""
        if not self.capabilities_data:
            return
        
        self.capability_index = {}
        self.module_index = {}
        
        module_capabilities = self.capabilities_data.get('module_capabilities', {})
        
        for module_name, module_data in module_capabilities.items():
            module_id = module_data['module_info']['module_id']
            self.module_index[module_id] = module_name
            
            for group_key, group_data in module_data['capability_groups'].items():
                for capability in group_data['capabilities']:
                    flow_id = capability['flow_id']
                    
                    cap_ref = CapabilityReference(
                        flow_id=flow_id,
                        capability_name=capability['capability_name'],
                        display_name=capability['display_name'],
                        module_name=module_name,
                        complexity=capability['ai_description']['complexity'],
                        scope=capability['ai_description']['scope'],
                        ai_level=capability['ai_description']['ai_capability_level'],
                        risk_level=capability['usage_guide']['risk_level'],
                        purpose=capability['ai_description']['purpose'],
                        example_command=capability['usage_guide']['example_commands'][0] if capability['usage_guide']['example_commands'] else "",
                        estimated_duration=capability['technical_details']['estimated_duration']
                    )
                    
                    self.capability_index[flow_id] = cap_ref
        
        if self.verbose:
            print(f"[Index] 建立索引完成：{len(self.capability_index)} 個能力索引")
    
    def get_capability_by_flow_id(self, flow_id: int) -> Optional[CapabilityReference]:
        """根據Flow ID獲取能力引用"""
        return self.capability_index.get(flow_id)
    
    def get_capabilities_by_module(self, module_id: str) -> List[CapabilityReference]:
        """根據模組ID獲取所有能力"""
        results = []
        for cap_ref in self.capability_index.values():
            if cap_ref.module_name == self.module_index.get(module_id, ''):
                results.append(cap_ref)
        return results
    
    def search_capabilities(self, 
                          keyword: str = None,
                          module: str = None,
                          complexity: str = None,
                          scope: str = None,
                          risk_level: str = None,
                          ai_level: str = None,
                          limit: int = 10) -> List[CapabilityReference]:
        """搜索能力"""
        results = []
        
        for cap_ref in self.capability_index.values():
            # 應用篩選條件
            if module and cap_ref.module_name != self.module_index.get(module, ''):
                continue
            if complexity and cap_ref.complexity != complexity:
                continue
            if scope and cap_ref.scope != scope:
                continue
            if risk_level and cap_ref.risk_level != risk_level:
                continue
            if ai_level and cap_ref.ai_level != ai_level:
                continue
            if keyword:
                if (keyword.lower() not in cap_ref.display_name.lower() and
                    keyword.lower() not in cap_ref.purpose.lower()):
                    continue
            
            results.append(cap_ref)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_recommended_capabilities(self, 
                                   category: str = "ai_core",
                                   risk_preference: str = "low") -> List[CapabilityReference]:
        """獲取推薦能力列表"""
        recommendations = []
        
        if category == "ai_core":
            # AI核心能力推薦
            for cap_ref in self.capability_index.values():
                if (cap_ref.ai_level in ['core', 'advanced'] and
                    cap_ref.risk_level == risk_preference and
                    cap_ref.complexity in ['medium', 'complex']):
                    recommendations.append(cap_ref)
        
        elif category == "quick_start":
            # 快速開始推薦
            for cap_ref in self.capability_index.values():
                if (cap_ref.complexity == 'simple' and
                    cap_ref.risk_level == 'low'):
                    recommendations.append(cap_ref)
        
        elif category == "analysis":
            # 分析功能推薦
            for cap_ref in self.capability_index.values():
                if ('分析' in cap_ref.display_name or
                    'analyze' in cap_ref.purpose.lower()):
                    recommendations.append(cap_ref)
        
        return recommendations[:10]  # 限制10個推薦
    
    def generate_usage_report(self, flow_ids: List[int]) -> Dict[str, Any]:
        """生成使用報告"""
        if not flow_ids:
            return {"error": "未提供Flow ID"}
        
        report = {
            "requested_capabilities": len(flow_ids),
            "found_capabilities": 0,
            "missing_flow_ids": [],
            "capabilities_summary": [],
            "module_distribution": {},
            "risk_assessment": {"low": 0, "medium": 0, "high": 0},
            "complexity_distribution": {"simple": 0, "medium": 0, "complex": 0},
            "recommended_execution_order": [],
            "total_estimated_duration": "未知",
            "generated_at": datetime.now().isoformat()
        }
        
        found_capabilities = []
        
        for flow_id in flow_ids:
            cap_ref = self.get_capability_by_flow_id(flow_id)
            if cap_ref:
                found_capabilities.append(cap_ref)
                report["capabilities_summary"].append({
                    "flow_id": cap_ref.flow_id,
                    "name": cap_ref.display_name,
                    "module": cap_ref.module_name,
                    "complexity": cap_ref.complexity,
                    "risk": cap_ref.risk_level,
                    "purpose": cap_ref.purpose,
                    "command": cap_ref.example_command
                })
                
                # 統計信息
                report["module_distribution"][cap_ref.module_name] = \
                    report["module_distribution"].get(cap_ref.module_name, 0) + 1
                report["risk_assessment"][cap_ref.risk_level] += 1
                report["complexity_distribution"][cap_ref.complexity] += 1
            else:
                report["missing_flow_ids"].append(flow_id)
        
        report["found_capabilities"] = len(found_capabilities)
        
        # 生成執行順序建議（低風險先執行）
        sorted_caps = sorted(found_capabilities, 
                           key=lambda x: (
                               {'low': 0, 'medium': 1, 'high': 2}[x.risk_level],
                               {'simple': 0, 'medium': 1, 'complex': 2}[x.complexity]
                           ))
        
        report["recommended_execution_order"] = [cap.flow_id for cap in sorted_caps]
        
        return report
    
    def export_for_cli_integration(self, output_path: str = None) -> str:
        """導出CLI整合格式"""
        if not output_path:
            output_path = "enhanced_capabilities_cli_export.json"
        
        cli_export = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "source": "enhanced_classifier_processor",
                "total_capabilities": len(self.capability_index),
                "cli_integration_version": "1.0"
            },
            "capability_lookup": {},
            "module_capabilities": {},
            "quick_reference": {
                "by_complexity": {"simple": [], "medium": [], "complex": []},
                "by_risk": {"low": [], "medium": [], "high": []},
                "by_ai_level": {"none": [], "basic": [], "advanced": [], "core": []},
                "recommended_flows": {
                    "beginners": [],
                    "ai_users": [],
                    "advanced_users": []
                }
            }
        }
        
        # 構建能力查找表
        for flow_id, cap_ref in self.capability_index.items():
            cli_export["capability_lookup"][str(flow_id)] = {
                "name": cap_ref.display_name,
                "command": cap_ref.example_command,
                "purpose": cap_ref.purpose,
                "module": cap_ref.module_name,
                "complexity": cap_ref.complexity,
                "risk": cap_ref.risk_level,
                "ai_level": cap_ref.ai_level,
                "duration": cap_ref.estimated_duration
            }
            
            # 分類到快速參考
            cli_export["quick_reference"]["by_complexity"][cap_ref.complexity].append(flow_id)
            cli_export["quick_reference"]["by_risk"][cap_ref.risk_level].append(flow_id)
            cli_export["quick_reference"]["by_ai_level"][cap_ref.ai_level].append(flow_id)
        
        # 構建模組能力映射
        for module_id, module_name in self.module_index.items():
            module_caps = self.get_capabilities_by_module(module_id)
            cli_export["module_capabilities"][module_id] = {
                "name": module_name,
                "capability_count": len(module_caps),
                "flow_ids": [cap.flow_id for cap in module_caps]
            }
        
        # 生成推薦列表
        cli_export["quick_reference"]["recommended_flows"]["beginners"] = [
            cap.flow_id for cap in self.get_recommended_capabilities("quick_start")
        ]
        cli_export["quick_reference"]["recommended_flows"]["ai_users"] = [
            cap.flow_id for cap in self.get_recommended_capabilities("ai_core")
        ]
        cli_export["quick_reference"]["recommended_flows"]["advanced_users"] = [
            cap.flow_id for cap in self.search_capabilities(complexity="complex", limit=5)
        ]
        
        # 保存文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cli_export, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"📤 CLI整合數據已導出: {output_path}")
        
        return output_path
    
    def create_enhanced_executor_integration(self) -> str:
        """創建增強執行器的整合代碼"""
        integration_code = '''"""
AIVA 執行器增強功能集成
使用增強分類處理器的結果為用戶提供智能能力選擇
"""

import json
from pathlib import Path

class EnhancedExecutorIntegration:
    """執行器增強集成"""
    
    def __init__(self, cli_export_path: str):
        self.cli_data = self._load_cli_data(cli_export_path)
    
    def _load_cli_data(self, path: str):
        """載入CLI導出數據"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def get_capability_description(self, flow_id: int) -> str:
        """獲取能力的AI友善描述"""
        flow_str = str(flow_id)
        if flow_str in self.cli_data.get('capability_lookup', {}):
            cap_data = self.cli_data['capability_lookup'][flow_str]
            return f"""
能力名稱: {cap_data['name']}
模組: {cap_data['module']}
用途: {cap_data['purpose']}
複雜度: {cap_data['complexity']}
風險等級: {cap_data['risk']}
預估時間: {cap_data['duration']}
建議命令: {cap_data['command']}
"""
        return f"Flow {flow_id}: 未找到詳細描述"
    
    def suggest_related_flows(self, current_flow_id: int, limit: int = 3) -> List[int]:
        """建議相關的Flow"""
        current_cap = self.cli_data.get('capability_lookup', {}).get(str(current_flow_id))
        if not current_cap:
            return []
        
        current_module = current_cap['module']
        suggestions = []
        
        for flow_str, cap_data in self.cli_data.get('capability_lookup', {}).items():
            flow_id = int(flow_str)
            if (flow_id != current_flow_id and 
                cap_data['module'] == current_module and
                len(suggestions) < limit):
                suggestions.append(flow_id)
        
        return suggestions
    
    def get_recommendations_for_user_level(self, level: str = "beginners") -> List[int]:
        """根據用戶水平獲取推薦"""
        return self.cli_data.get('quick_reference', {}).get('recommended_flows', {}).get(level, [])
    
    def format_flow_list_with_descriptions(self, flow_ids: List[int]) -> str:
        """格式化Flow列表並添加描述"""
        if not flow_ids:
            return "沒有找到相關能力"
        
        formatted = "\\n推薦的能力:\\n"
        for i, flow_id in enumerate(flow_ids, 1):
            cap_data = self.cli_data.get('capability_lookup', {}).get(str(flow_id), {})
            name = cap_data.get('name', f'Flow {flow_id}')
            purpose = cap_data.get('purpose', '未知用途')
            risk = cap_data.get('risk', 'unknown')
            
            risk_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(risk, '⚪')
            
            formatted += f"  {i}. {risk_emoji} Flow {flow_id}: {name}\\n"
            formatted += f"     用途: {purpose}\\n"
        
        return formatted

# 使用範例:
# integrator = EnhancedExecutorIntegration("enhanced_capabilities_cli_export.json")
# description = integrator.get_capability_description(123)
# print(description)
'''
        return integration_code


def main():
    """主函數 - 示範集成器功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AIVA 增強能力集成器')
    parser.add_argument('--capabilities-data', type=str, 
                       default="test_enhanced_output/4_standardization/module_grouped_capabilities.json",
                       help='增強分類數據路徑')
    parser.add_argument('--flow-id', type=int, help='查詢特定Flow ID')
    parser.add_argument('--module', type=str, help='查詢特定模組')
    parser.add_argument('--search', type=str, help='搜索關鍵字')
    parser.add_argument('--export', action='store_true', help='導出CLI整合格式')
    parser.add_argument('--report', type=str, help='為指定Flow IDs生成報告（逗號分隔）')
    
    args = parser.parse_args()
    
    # 創建集成器
    integrator = EnhancedCapabilityIntegrator(args.capabilities_data)
    
    if not integrator.capabilities_data:
        print("[Error] 無法載入能力數據")
        return 1
    
    # 查詢特定Flow ID
    if args.flow_id:
        cap_ref = integrator.get_capability_by_flow_id(args.flow_id)
        if cap_ref:
            print(f"[Flow] Flow {args.flow_id}:")
            print(f"  名稱: {cap_ref.display_name}")
            print(f"  模組: {cap_ref.module_name}")
            print(f"  用途: {cap_ref.purpose}")
            print(f"  複雜度: {cap_ref.complexity}")
            print(f"  風險: {cap_ref.risk_level}")
            print(f"  命令: {cap_ref.example_command}")
        else:
            print(f"[Error] 未找到 Flow {args.flow_id}")
    
    # 模組查詢
    if args.module:
        caps = integrator.get_capabilities_by_module(args.module)
        print(f"[Module] 模組 {args.module} 包含 {len(caps)} 個能力:")
        for cap in caps[:5]:  # 只顯示前5個
            print(f"  Flow {cap.flow_id}: {cap.display_name}")
    
    # 搜索
    if args.search:
        results = integrator.search_capabilities(keyword=args.search)
        print(f"[Search] 搜索 '{args.search}' 找到 {len(results)} 個結果:")
        for cap in results:
            print(f"  Flow {cap.flow_id}: {cap.display_name} ({cap.module_name})")
    
    # 生成報告
    if args.report:
        flow_ids = [int(x.strip()) for x in args.report.split(',')]
        report = integrator.generate_usage_report(flow_ids)
        print("[Stats] 使用報告:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # 導出CLI格式
    if args.export:
        export_path = integrator.export_for_cli_integration()
        print(f"[Export] 已導出CLI整合格式: {export_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())