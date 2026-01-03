#!/usr/bin/env python3
"""
AIVA Core 數據流分類分析器 (完整版)
===================================
基於 AIVA Core 六大模組架構進行數據流分類和路徑差異分析

架構說明:
---------
1. 認知核心模組 (cognitive_core) - AI認知、神經網路、RAG、決策
2. 內探模組 (internal_exploration) - 自我認知、能力分析、內部監控
3. 任務規劃模組 (task_planning) - 規劃器、執行器、指揮官
4. 外學模組 (external_learning) - 分析、追蹤、訓練、模型
5. 核心能力模組 (core_capabilities) - 攻擊鏈、業務邏輯、對話、插件
6. 服務骨幹模組 (service_backbone) - API、協調、消息、存儲、狀態

功能:
-----
- 自動分類數據流到對應模組
- 標記組件類型 (AI組件/程式組件/混合組件)
- 完整列出所有數據流路徑及腳本順序
- 分析多路徑到相同終點的使用差異
- 生成詳細的分類報告
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from datetime import datetime

# 常量定義
SECTION_SEPARATOR = "---\n\n"


class AIVAFlowClassifier:
    """AIVA 數據流分類分析器"""
    
    # AIVA Core 六大模組定義
    MODULES = {
        "cognitive_core": "認知核心模組",
        "internal_exploration": "內探模組", 
        "task_planning": "任務規劃模組",
        "external_learning": "外學模組",
        "core_capabilities": "核心能力模組",
        "service_backbone": "服務骨幹模組"
    }
    
    # 腳本詳細說明 (擴展版 - 涵蓋所有關鍵腳本)
    SCRIPT_DESCRIPTIONS = {
        # 認知核心相關
        "ai_capability_query": "AI能力查詢器 - 預設指令查詢",
        "enhanced_decision_agent": "增強決策代理 - 程式邏輯決策",
        "external_loop_connector": "外部循環連接器 - 系統接口整合",
        "internal_loop_connector": "內部循環連接器 - 內部API協調",
        "neural_network": "神經網路核心 - AI計算引擎",
        "rag_system": "RAG系統 - 結構化資料檢索",
        
        # 內探模組相關
        "capability_cli": "能力命令行工具 - 能力管理界面",
        "self_aware_analyzer": "自我感知分析器 - 自我認知評估",
        "capability_analyzer": "能力分析器 - 能力評估分析",
        
        # 任務規劃相關
        "plan_executor": "計劃執行器 - 任務執行管理",
        "plan_comparator": "計劃比較器 - 方案對比評估",
        "task_commander": "任務指揮官 - 任務調度控制",
        "planner": "規劃器 - 智能任務規劃",
        
        # 外學模組相關
        "scalable_bio_trainer": "可擴展生物訓練器 - 大規模AI訓練",
        "bio_analysis": "生物分析 - 生物數據分析",
        "resource_tracker": "資源追蹤器 - 資源使用監控",
        "rl_models": "強化學習模型 - 智能決策學習",
        "model_manager": "模型管理器 - AI模型管理",
        "training_pipeline": "訓練管道 - AI模型訓練流程",
        
        # 核心能力相關
        "capability_registry": "能力註冊表 - 能力登記管理",
        "capability_orchestrator": "能力編排器 - 功能協調管理",
        "attack_chain": "攻擊鏈 - 預設攻擊序列執行",
        "business_logic": "業務邏輯 - 核心功能處理",
        "conversation_handler": "指令處理器 - 預設指令管理",
        "plugin_manager": "插件管理器 - 模組化功能管理",
        
        # 服務骨幹相關
        "backends": "後端存儲 - 數據持久化",
        "storage_manager": "存儲管理器 - 存儲資源管理",
        "api_gateway": "API網關 - 服務接口管理",
        "message_bus": "消息總線 - 異步消息傳遞",
        "state_manager": "狀態管理器 - 系統狀態追蹤",
        "orchestrator": "協調器 - 服務協調管理",
        
        # 通用組件
        "utils": "工具集 - 通用輔助函數",
        "config": "配置模組 - 系統配置管理",
        "logger": "日誌器 - 日誌記錄",
        "validator": "驗證器 - 數據驗證",
        "cache": "緩存 - 數據緩存",
        "monitor": "監控器 - 系統監控"
    }
    
    # 組件類型分類規則
    AI_COMPONENTS = {
        "neural_network", "scalable_bio_trainer", "rl_models", "model_manager",
        "training_pipeline", "bio_analysis", "enhanced_decision_agent"
    }
    
    PROGRAM_COMPONENTS = {
        "backends", "storage_manager", "api_gateway", "message_bus",
        "state_manager", "utils", "config", "logger", "validator", "cache"
    }
    
    # 混合組件 (同時包含AI和程式邏輯)
    MIXED_COMPONENTS = {
        "capability_orchestrator", "plan_executor", "task_commander",
        "planner", "orchestrator", "resource_tracker", "monitor",
        "capability_registry", "plugin_manager", "capability_cli",
        "self_aware_analyzer", "capability_analyzer", "ai_capability_query",
        "rag_system", "conversation_handler"
    }
    
    def __init__(self, input_dir: str, output_dir: str, verbose: bool = False):
        """初始化分類器"""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.flows = []
        
        # 新增：函數詳細信息
        self.function_details = {}
        self.function_map = {}
        self.script_functions = {}
        
        self.stats = {
            "total_flows": 0,
            "module_distribution": defaultdict(int),
            "component_type_distribution": defaultdict(int),
            "multi_path_endpoints": defaultdict(list)
        }
        
    def load_flow_data(self):
        """載入數據流數據"""
        # Check if input_dir is a file or directory
        if self.input_dir.suffix == '.json':
            analysis_file = self.input_dir
        else:
            analysis_file = self.input_dir / "analysis_results.json"
        
        if not analysis_file.exists():
            raise FileNotFoundError(f"找不到分析結果文件: {analysis_file}")
        
        with open(analysis_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 處理 flow_chains 格式
        flow_chains = data.get('flow_chains', [])
        
        # 載入函數詳細信息（新增）
        self.function_details = data.get('function_details', {})
        self.function_map = self.function_details.get('function_map', {})
        self.script_functions = self.function_details.get('script_functions', {})
        
        if self.verbose:
            print(f"載入了 {len(flow_chains)} 條數據流鏈")
            print(f"載入了 {len(self.function_map)} 個函數詳細信息")
            print(f"載入了 {len(self.script_functions)} 個腳本的函數映射")
        
        # 轉換為標準格式並提取腳本名稱
        for idx, chain in enumerate(flow_chains, 1):
            if not chain:
                continue
                
            # 提取路徑中的腳本名稱
            scripts = [self._extract_script_name(filepath) for filepath in chain]
            
            flow_data = {
                'id': idx,
                'path': scripts,
                'full_path': chain,
                'length': len(scripts),
                'start': scripts[0] if scripts else None,
                'end': scripts[-1] if scripts else None
            }
            
            self.flows.append(flow_data)
        
        self.stats['total_flows'] = len(self.flows)
        
        if self.verbose:
            print(f"成功處理 {len(self.flows)} 條數據流")
    
    def _extract_script_name(self, filepath: str) -> str:
        """從完整路徑提取腳本名稱"""
        return Path(filepath).stem
    
    def _get_script_description(self, script_name: str) -> str:
        """根據腳本名稱生成描述"""
        # 根據腳本名稱進行特殊處理
        script_desc = self._classify_script_by_name(script_name)
        
        return script_desc
        
    def _classify_script_by_name(self, script_name: str) -> str:
        """根據腳本名稱生成描述"""
        # 使用預定義描述或生成默認描述
        return self.SCRIPT_DESCRIPTIONS.get(script_name, f"{script_name} - 功能組件")
    
    def _classify_module(self, script_name: str) -> str:
        """根據腳本名稱分類到對應模組"""
        # 根據腳本名稱特徵進行模組分類
        if any(keyword in script_name for keyword in ['ai_', 'neural', 'rag', 'decision']):
            return 'cognitive_core'
        elif any(keyword in script_name for keyword in ['capability_cli', 'self_aware', 'capability_analyzer']):
            return 'internal_exploration'
        elif any(keyword in script_name for keyword in ['plan_', 'task_', 'planner', 'executor', 'commander']):
            return 'task_planning'
        elif any(keyword in script_name for keyword in ['trainer', 'training', 'model', 'bio_analysis', 'rl_', 'resource_tracker']):
            return 'external_learning'
        elif any(keyword in script_name for keyword in ['capability_registry', 'capability_orchestrator', 'attack_chain', 'business', 'conversation', 'plugin']):
            return 'core_capabilities'
        elif any(keyword in script_name for keyword in ['backend', 'storage', 'api_', 'message', 'state_', 'orchestrator']):
            return 'service_backbone'
        else:
            # 默認歸類為服務骨幹 (基礎設施)
            return 'service_backbone'
    
    def _classify_component_type(self, script_name: str) -> str:
        """分類組件類型"""
        if script_name in self.AI_COMPONENTS:
            return "AI組件"
        elif script_name in self.PROGRAM_COMPONENTS:
            return "程式組件"
        elif script_name in self.MIXED_COMPONENTS:
            return "混合組件"
        else:
            # 根據名稱推斷
            if any(keyword in script_name for keyword in ['ai_', 'neural', 'rag', 'model', 'trainer', 'rl_']):
                return "AI組件"
            else:
                return "程式組件"
    
    def classify_flows(self):
        """對所有數據流進行分類
        
        使用終點腳本分類法：
        - primary_module: 基於數據流終點腳本的模組
        - majority_module: 基於多數決的模組（保留作對比）
        """
        if self.verbose:
            print("\n開始分類數據流...")
            print("🎯 使用終點腳本分類法")
        
        for flow in self.flows:
            # 分類每個腳本
            flow['classifications'] = []
            flow['modules'] = []
            flow['component_types'] = []
            
            for script in flow['path']:
                module = self._classify_module(script)
                comp_type = self._classify_component_type(script)
                description = self._get_script_description(script)
                
                flow['classifications'].append({
                    'script': script,
                    'module': module,
                    'component_type': comp_type,
                    'description': description
                })
                
                flow['modules'].append(module)
                flow['component_types'].append(comp_type)
            
            # 統計主要模組（基於終點腳本分類）
            if flow['modules']:
                # 使用終點腳本的模組作為primary_module
                endpoint_module = flow['modules'][-1] if flow['modules'] else 'unknown'
                flow['primary_module'] = endpoint_module
                flow['endpoint_module'] = endpoint_module  # 新增終點模組欄位
                
                # 同時保留多數決結果做為對比
                majority_module = max(set(flow['modules']), key=flow['modules'].count)
                flow['majority_module'] = majority_module
                
                self.stats['module_distribution'][endpoint_module] += 1
            
            # 統計主要組件類型
            if flow['component_types']:
                primary_type = max(set(flow['component_types']), key=flow['component_types'].count)
                flow['primary_component_type'] = primary_type
                self.stats['component_type_distribution'][primary_type] += 1
            
            # 記錄終點用於多路徑分析
            if flow['end']:
                self.stats['multi_path_endpoints'][flow['end']].append(flow['id'])
        
        if self.verbose:
            print(f"分類完成: {len(self.flows)} 條數據流")
    
    def analyze_multi_path_endpoints(self) -> List[Dict]:
        """分析有多條路徑到達的終點"""
        multi_path_analysis = []
        
        for endpoint, flow_ids in self.stats['multi_path_endpoints'].items():
            if len(flow_ids) > 1:
                # 獲取所有到達此終點的流
                endpoint_flows = [f for f in self.flows if f['id'] in flow_ids]
                
                # 分析路徑差異
                lengths = [f['length'] for f in endpoint_flows]
                modules_used = [set(f['modules']) for f in endpoint_flows]
                
                analysis = {
                    'endpoint': endpoint,
                    'endpoint_description': self._get_script_description(endpoint),
                    'total_paths': len(flow_ids),
                    'flow_ids': flow_ids,
                    'path_length_range': (min(lengths), max(lengths)),
                    'average_length': sum(lengths) / len(lengths),
                    'all_modules_involved': list(set.union(*modules_used)) if modules_used else [],
                    'paths': []
                }
                
                # 詳細記錄每條路徑
                for flow in endpoint_flows:
                    path_info = {
                        'flow_id': flow['id'],
                        'length': flow['length'],
                        'scripts': flow['path'],
                        'modules': flow['modules'],
                        'primary_module': flow.get('primary_module', 'unknown'),
                        'description': ' → '.join([
                            f"{c['script']}[{c['component_type']}]" 
                            for c in flow['classifications']
                        ])
                    }
                    analysis['paths'].append(path_info)
                
                multi_path_analysis.append(analysis)
        
        # 按路徑數量排序
        multi_path_analysis.sort(key=lambda x: x['total_paths'], reverse=True)
        
        if self.verbose:
            print(f"\n找到 {len(multi_path_analysis)} 個多路徑終點")
        
        return multi_path_analysis
    
    def generate_reports(self):
        """生成完整報告"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 生成分類統計報告
        self._generate_classification_report()
        
        # 2. 生成完整數據流詳細報告
        self._generate_complete_flow_details()
        
        # 3. 生成多路徑分析報告
        multi_path_analysis = self.analyze_multi_path_endpoints()
        self._generate_multi_path_report(multi_path_analysis)
        
        # 4. 生成 JSON 格式數據
        self._generate_json_export(multi_path_analysis)
        
        if self.verbose:
            print(f"\n所有報告已生成到: {self.output_dir}")
    
    def _generate_classification_report(self):
        """生成分類統計報告"""
        report_file = self.output_dir / "classification_summary.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# AIVA Core 數據流分類統計報告\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本統計
            f.write("## 基本統計\n\n")
            f.write(f"- **總數據流數量**: {self.stats['total_flows']}\n")
            
            avg_length = sum(flow['length'] for flow in self.flows) / len(self.flows) if self.flows else 0
            f.write(f"- **平均流程長度**: {avg_length:.2f}\n")
            
            max_flow = max(self.flows, key=lambda x: x['length']) if self.flows else None
            if max_flow:
                f.write(f"- **最長流程**: {max_flow['length']} 步 (Flow {max_flow['id']})\n")
            
            min_flow = min(self.flows, key=lambda x: x['length']) if self.flows else None
            if min_flow:
                f.write(f"- **最短流程**: {min_flow['length']} 步 (Flow {min_flow['id']})\n\n")
            
            # 模組分布
            f.write("## 模組分布\n\n")
            f.write("| 模組 | 中文名稱 | 數據流數量 | 占比 |\n")
            f.write("|------|----------|------------|------|\n")
            
            for module_id, count in sorted(
                self.stats['module_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                module_name = self.MODULES.get(module_id, module_id)
                percentage = (count / self.stats['total_flows'] * 100) if self.stats['total_flows'] > 0 else 0
                f.write(f"| {module_id} | {module_name} | {count} | {percentage:.1f}% |\n")
            
            # 組件類型分布
            f.write("\n## 組件類型分布\n\n")
            f.write("| 組件類型 | 數量 | 占比 |\n")
            f.write("|----------|------|------|\n")
            
            for comp_type, count in sorted(
                self.stats['component_type_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (count / self.stats['total_flows'] * 100) if self.stats['total_flows'] > 0 else 0
                f.write(f"| {comp_type} | {count} | {percentage:.1f}% |\n")
            
            f.write("\n---\n")
        
        if self.verbose:
            print(f"生成分類統計報告: {report_file}")
    
    def _generate_complete_flow_details(self):
        """生成完整數據流詳細列表"""
        report_file = self.output_dir / "complete_flow_details.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 完整數據流詳細列表\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"總數據流數量: {len(self.flows)}\n\n")
            f.write(SECTION_SEPARATOR)
            
            # 按模組分組輸出
            for module_id in self.MODULES.keys():
                module_flows = [f for f in self.flows if f.get('primary_module') == module_id]
                
                if not module_flows:
                    continue
                
                module_name = self.MODULES[module_id]
                f.write(f"## {module_name} ({module_id})\n\n")
                f.write(f"包含 {len(module_flows)} 條數據流\n\n")
                
                for flow in module_flows:
                    f.write(f"### Flow {flow['id']}\n\n")
                    f.write(f"- **長度**: {flow['length']} 步\n")
                    f.write(f"- **起點**: {flow['start']}\n")
                    f.write(f"- **終點**: {flow['end']}\n")
                    f.write(f"- **主要模組**: {self.MODULES.get(flow['primary_module'], flow['primary_module'])}\n")
                    f.write(f"- **主要組件類型**: {flow['primary_component_type']}\n\n")
                    
                    f.write("**執行路徑**:\n\n")
                    for i, classification in enumerate(flow['classifications'], 1):
                        script_name = classification['script']
                        # 獲取該腳本的函數信息
                        script_info = self.script_functions.get(script_name, {})
                        functions = script_info.get('functions', {})
                        file_path = script_info.get('file_path', '')
                        file_name = file_path.split('\\')[-1].replace('.py', '') if file_path else script_name
                        
                        f.write(f"{i}. **{classification['component_type']}**\n")
                        
                        # 顯示函數名稱和檔案名稱
                        if functions:
                            func_names = list(functions.keys())
                            for func_name in func_names:
                                f.write(f"   {func_name}\n")
                                f.write(f"   {file_name}\n")
                                if func_name != func_names[-1]:  # 不是最後一個函數時
                                    f.write("   \n")
                        else:
                            f.write(f"   {script_name}\n")
                            f.write(f"   {file_name}\n")
                        
                        f.write(f"   - 模組: {self.MODULES.get(classification['module'], classification['module'])}\n\n")
                    
                    f.write("---\n\n")
        
        if self.verbose:
            print(f"生成完整流程詳細列表: {report_file}")
    
    def _generate_multi_path_report(self, multi_path_analysis: List[Dict]):
        """生成多路徑分析報告"""
        report_file = self.output_dir / "multi_path_analysis.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 多路徑終點分析報告\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"找到 {len(multi_path_analysis)} 個有多條路徑到達的終點\n\n")
            f.write(SECTION_SEPARATOR)
            
            for analysis in multi_path_analysis:
                f.write(f"## 終點: {analysis['endpoint']}\n\n")
                f.write(f"**說明**: {analysis['endpoint_description']}\n\n")
                f.write(f"- **路徑總數**: {analysis['total_paths']}\n")
                f.write(f"- **路徑長度範圍**: {analysis['path_length_range'][0]} - {analysis['path_length_range'][1]} 步\n")
                f.write(f"- **平均路徑長度**: {analysis['average_length']:.2f} 步\n")
                modules_str = ', '.join([m for m in [self.MODULES.get(m, m) for m in analysis['all_modules_involved']] if m is not None])
                f.write(f"- **涉及模組**: {modules_str}\n\n")
                
                f.write("### 路徑詳細對比\n\n")
                
                for i, path_info in enumerate(analysis['paths'], 1):
                    f.write(f"#### 路徑 {i} (Flow {path_info['flow_id']})\n\n")
                    f.write(f"- **長度**: {path_info['length']} 步\n")
                    f.write(f"- **主要模組**: {self.MODULES.get(path_info['primary_module'], path_info['primary_module'])}\n")
                    f.write(f"- **執行順序**: {path_info['description']}\n\n")
                    
                    f.write("**完整腳本列表**:\n")
                    for j, script in enumerate(path_info['scripts'], 1):
                        f.write(f"{j}. {script}\n")
                    f.write("\n")
                
                f.write("### 路徑差異分析\n\n")
                
                # 分析路徑間的差異
                if len(analysis['paths']) >= 2:
                    path1 = analysis['paths'][0]
                    path2 = analysis['paths'][1]
                    
                    set1 = set(path1['scripts'])
                    set2 = set(path2['scripts'])
                    
                    unique_to_path1 = set1 - set2
                    unique_to_path2 = set2 - set1
                    common = set1 & set2
                    
                    f.write("**路徑 1 vs 路徑 2 對比**:\n\n")
                    f.write(f"- 共同腳本數: {len(common)}\n")
                    f.write(f"- 路徑 1 獨有: {len(unique_to_path1)} 個腳本\n")
                    if unique_to_path1:
                        f.write(f"  - {', '.join(unique_to_path1)}\n")
                    f.write(f"- 路徑 2 獨有: {len(unique_to_path2)} 個腳本\n")
                    if unique_to_path2:
                        f.write(f"  - {', '.join(unique_to_path2)}\n")
                    f.write("\n")
                    
                    # 使用場景差異分析
                    f.write("**使用場景差異推測**:\n\n")
                    
                    module_diff = set(path1['modules']) != set(path2['modules'])
                    if module_diff:
                        modules1_str = ', '.join([m for m in [self.MODULES.get(m, m) for m in set(path1['modules'])] if m is not None])
                        modules2_str = ', '.join([m for m in [self.MODULES.get(m, m) for m in set(path2['modules'])] if m is not None])
                        f.write(f"- 路徑 1 主要涉及: {modules1_str}\n")
                        f.write(f"- 路徑 2 主要涉及: {modules2_str}\n")
                        f.write("- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景\n\n")
                    
                    length_diff = abs(path1['length'] - path2['length'])
                    if length_diff > 2:
                        f.write(f"- 路徑長度差異顯著 ({length_diff} 步)\n")
                        if path1['length'] < path2['length']:
                            f.write("- **推測**: 路徑 1 可能是快速路徑或直接調用,路徑 2 可能包含更多處理邏輯\n\n")
                        else:
                            f.write("- **推測**: 路徑 2 可能是快速路徑或直接調用,路徑 1 可能包含更多處理邏輯\n\n")
                
                f.write("---\n\n")
        
        if self.verbose:
            print(f"生成多路徑分析報告: {report_file}")
    
    def _generate_json_export(self, multi_path_analysis: List[Dict]):
        """生成 JSON 格式完整數據"""
        json_file = self.output_dir / "classification_data.json"
        
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_flows': self.stats['total_flows'],
                'module_distribution': dict(self.stats['module_distribution']),
                'component_type_distribution': dict(self.stats['component_type_distribution'])
            },
            'flows': self.flows,
            'multi_path_analysis': multi_path_analysis
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"生成 JSON 數據導出: {json_file}")
    
    def run(self):
        """執行完整分析流程"""
        print("="*60)
        print("AIVA Core 數據流分類分析器")
        print("="*60)
        print()
        
        try:
            # 1. 載入數據
            print("步驟 1/4: 載入數據流數據...")
            self.load_flow_data()
            
            # 2. 分類
            print("步驟 2/4: 分類數據流...")
            self.classify_flows()
            
            # 3. 多路徑分析
            print("步驟 3/4: 分析多路徑終點...")
            multi_path_count = len([k for k, v in self.stats['multi_path_endpoints'].items() if len(v) > 1])
            print(f"發現 {multi_path_count} 個多路徑終點")
            
            # 4. 生成報告
            print("步驟 4/4: 生成報告...")
            self.generate_reports()
            
            print("\n" + "="*60)
            print("分析完成!")
            print("="*60)
            print(f"\n總數據流: {self.stats['total_flows']}")
            print(f"多路徑終點: {multi_path_count}")
            print(f"\n報告輸出目錄: {self.output_dir}")
            print("\n生成的報告文件:")
            print("  1. classification_summary.md - 分類統計報告")
            print("  2. complete_flow_details.md - 完整數據流詳細列表")
            print("  3. multi_path_analysis.md - 多路徑分析報告")
            print("  4. classification_data.json - JSON 格式完整數據")
            
        except Exception as e:
            print(f"\n錯誤: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='AIVA Core 數據流分類分析器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python aiva_flow_classifier_final.py --input ./aiva_core_analysis_v4 --output ./classification_results --verbose
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='輸入目錄 (包含 analysis_results.json 的目錄)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./classification_results',
        help='輸出目錄 (默認: ./classification_results)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='顯示詳細處理信息'
    )
    
    args = parser.parse_args()
    
    # 創建分類器並執行
    classifier = AIVAFlowClassifier(
        input_dir=args.input,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    return classifier.run()


if __name__ == '__main__':
    exit(main())
