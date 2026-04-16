#!/usr/bin/env python3
"""
AIVA Core 數據流分類分析器 (完整版 v3.3)
========================================
基於 AIVA Core 五大模組架構進行數據流分類和路徑差異分析

版本歷史:
---------
v3.3 (2026-01-21) - 🔧 統一路徑配置
  - 整合 paths_config.py 作為路徑的單一數據源 (SOT)
  - 內部分類輸出至 internal_classification.json
  - 修復舊路徑引用問題
  
v3.2 (2026-01-01) - 🔧 修復模組分類算法
  - 使用文件路徑而非腳本名稱進行模組分類
  - 添加 _classify_module_from_path() 方法
  - 分類準確度從 46% 提升至 91.2%
  
v3.1 - 初始版本（存在分類錯誤）

架構說明:
---------
⚠️ 重要：此分類器僅用於 AI Core 內部模組（純 Python）
外部模組請使用 aiva_external_classifier.py

1. 認知核心模組 (cognitive_core) - AI認知、神經網路、RAG、決策、學習系統 (整合自 external_learning)
2. 內探模組 (internal_exploration) - 自我認知、能力分析、內部監控
3. 任務規劃模組 (task_planning) - 規劃器、執行器、指揮官
4. 核心能力模組 (core_capabilities) - 攻擊鏈、業務邏輯、對話、插件
5. 服務骨幹模組 (service_backbone) - API、協調、消息、存儲、狀態

數據存放位置:
-------------
輸出目錄: services/integration/data/internal_exploration/
分類文件: internal_classification.json
摘要文件: internal_classification_summary.md

功能:
-----
- ✅ 自動分類數據流到對應模組（使用文件路徑，準確度 91.2%）
- 標記組件類型 (AI組件/程式組件/混合組件)
- 完整列出所有數據流路徑及腳本順序
- 分析多路徑到相同終點的使用差異
- 生成詳細的分類報告
"""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

# ==========================================
# 路徑配置導入
# ==========================================
# 嘗試導入統一路徑配置，失敗則使用舊邏輯

try:
    # 添加路徑以便導入
    PATHS_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "integration" / "data" / "internal_exploration"
    
    from services.integration.data.internal_exploration.paths_config import (
        DATA_ROOT,
        CombinedPaths,
        InternalPaths,
        ensure_all_dirs,
    )
    USING_NEW_PATHS = True
except ImportError:
    USING_NEW_PATHS = False
    # 定義舊路徑作為後備
    DATA_ROOT = Path(__file__).resolve().parent / "classification_results"

# 常量定義
SECTION_SEPARATOR = "---\n\n"


class AIVAFlowClassifier:
    """AIVA Core 數據流分類分析器
    
    基於 AIVA Core 模組架構進行數據流分類和路徑差異分析
    """
    
    # 類常量：模組定義（五大模組架構 - 2026-01-03）
    # 注意：external_learning 已整合至 cognitive_core.learning_system
    MODULES = {
        "cognitive_core": "認知核心模組",
        "internal_exploration": "內探模組",
        "task_planning": "任務規劃模組",
        "core_capabilities": "核心能力模組",
        "service_backbone": "服務骨幹模組",
        # 向後相容：external_learning 映射到 cognitive_core
        "external_learning": "認知核心模組(學習子系統)",
        "learning_system": "認知核心模組(學習子系統)"
    }
    
    # AI能力類型分類：內部能力 vs 對外能力（2026-01-09）
    AI_INTERNAL_CAPABILITIES = [
        "neural_network", "rag_system", "model_manager", "training_pipeline",
        "rl_models", "scalable_bio_trainer", "bio_analysis", "learning_system",
        "self_aware_analyzer", "capability_analyzer", "internal_loop_connector"
    ]
    
    AI_EXTERNAL_CAPABILITIES = [
        "ai_capability_query", "enhanced_decision_agent", "external_loop_connector",
        "attack_chain", "business_logic", "conversation_handler", "plugin_manager",
        "capability_orchestrator", "task_commander", "planner", "plan_executor"
    ]

    # AI 能力類型字串常量
    AI_INTERNAL_TYPE = "AI內部能力"
    AI_EXTERNAL_TYPE = "AI對外能力"
    AI_COMPONENT_TYPE = "AI組件"

    # 腳本詳細說明：只有經過實際執行驗證成功的能力才應該有詳細描述
    SCRIPT_DESCRIPTIONS = {
        # AI 內部能力候選
        "internal_loop_connector": "internal_loop_connector - 功能組件",
        "capability_orchestrator": "capability_orchestrator - 功能組件",
        "capability_registry": "capability_registry - 功能組件",
        "ai_model_manager": "ai_model_manager - 功能組件",
        "scalable_bio_trainer": "scalable_bio_trainer - 功能組件",
        "rl_trainers": "rl_trainers - 功能組件",
        "rl_models": "rl_models - 功能組件",
        
        # AI 對外能力候選
        "ai_capability_query": "ai_capability_query - 功能組件",
        "enhanced_decision_agent": "enhanced_decision_agent - 功能組件",
        "external_loop_connector": "external_loop_connector - 功能組件",
        "event_listener": "event_listener - 功能組件",
        "attack_coordinator": "attack_coordinator - 功能組件",
        "two_phase_scan_orchestrator": "two_phase_scan_orchestrator - 功能組件",
        "unified_attack_executor": "unified_attack_executor - 功能組件",
        "unified_executor": "unified_executor - 功能組件",
        
        # AI 組件
        "neural_network": "neural_network - 功能組件",
        "rag_system": "rag_system - 功能組件",
        "rag_engine": "rag_engine - 功能組件",
        "real_bio_net_adapter": "real_bio_net_adapter - 功能組件",
        "real_bio_neuron_rag_agent": "real_bio_neuron_rag_agent - 功能組件",
        "skill_graph": "skill_graph - 功能組件",
        "vector_store": "vector_store - 功能組件",
        "unified_vector_store": "unified_vector_store - 功能組件",
        "capability_encoder": "capability_encoder - 功能組件",
        "weight_manager": "weight_manager - 功能組件",
        "unified_tracer": "unified_tracer - 功能組件",
        "experience_manager": "experience_manager - 功能組件",
        "continuous_learning_engine": "continuous_learning_engine - 功能組件",
        
        # 其他組件
        "capability_cli": "capability_cli - 功能組件",
        "self_aware_analyzer": "self_aware_analyzer - 功能組件",
        "capability_analyzer": "capability_analyzer - 功能組件",
        "plan_executor": "plan_executor - 功能組件",
        "plan_comparator": "plan_comparator - 功能組件",
        "task_commander": "task_commander - 功能組件",
        "planner": "planner - 功能組件",
        "bio_analysis": "bio_analysis - 功能組件",
        "resource_tracker": "resource_tracker - 功能組件",
        "model_manager": "model_manager - 功能組件",
        "training_pipeline": "training_pipeline - 功能組件",
        "attack_chain": "attack_chain - 功能組件",
        "business_logic": "business_logic - 功能組件",
        "conversation_handler": "conversation_handler - 功能組件",
        "plugin_manager": "plugin_manager - 功能組件",
        "backends": "backends - 功能組件",
        "sqlite_backend": "sqlite_backend - 功能組件",
        "storage_manager": "storage_manager - 功能組件",
        "api_gateway": "api_gateway - 功能組件",
        "message_bus": "message_bus - 功能組件",
        "state_manager": "state_manager - 功能組件",
        "orchestrator": "orchestrator - 功能組件",
        "command_repository": "command_repository - 功能組件",
        "trace_repository": "trace_repository - 功能組件",
        "utils": "utils - 功能組件",
        "config": "config - 功能組件",
        "logger": "logger - 功能組件",
        "validator": "validator - 功能組件",
        "cache": "cache - 功能組件",
        "monitor": "monitor - 功能組件"
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
    
    def __init__(self, input_dir: str | None = None, output_dir: str | None = None, verbose: bool = False, module_config_path: str = "modules_config.json", flows: list | None = None, merge_duplicate_flows: bool = True):
        """初始化分類器
        
        ⚠️ 修復版本（2025-12-16）：
        - 支持直接傳入 flows 列表進行重新分類
        - 符合 aiva_common 規範：保留未使用函數，維持向前兼容性
        - 保持原有文件讀取功能不變
        
        Args:
            input_dir: 輸入目錄或JSON文件路徑（可選，如果提供 flows 則不需要）
            output_dir: 輸出目錄路徑（可選，如果只需重新分類則不需要）
            verbose: 是否顯示詳細信息
            module_config_path: 模組配置文件路徑
            flows: 直接傳入的 flows 列表（可選，用於重新分類現有數據）
            merge_duplicate_flows: 是否合併相同起終點的數據流（預設False，因為中間路徑不同會產生不同結果）
        """
        # 基本參數
        self.input_dir = Path(input_dir) if input_dir else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.verbose = verbose
        self.merge_duplicate_flows = merge_duplicate_flows if merge_duplicate_flows is not None else False
        
        # 動態配置載入
        self.module_config_path = module_config_path
        self.config = self.load_module_config()
        self.dynamic_modules = self.config.get("modules", {})
        self.component_types_config = self.config.get("component_types", {})
        self.dynamic_script_descriptions = self.config.get("script_descriptions", {})
        
        # 從配置中建立所有模組的扁平化對應
        self.all_modules = {}
        for category, modules in self.dynamic_modules.items():
            self.all_modules.update(modules)

        # 數據流處理相關屬性
        # ⚠️ 修復（2025-12-16）：支持直接傳入 flows
        self.flows = flows if flows is not None else []
        self.function_details: dict[str, Any] = {}
        self.function_map: dict[str, Any] = {}
        self.script_functions: dict[str, Any] = {}
        
        # 統計信息
        self.stats = {
            "total_flows": len(self.flows) if flows else 0,
            "module_distribution": defaultdict(int),
            "component_type_distribution": defaultdict(int),
            "loop_type_distribution": defaultdict(int),  # ✅ 新增：循環類型統計
            "ai_notification_required": defaultdict(int),  # ✅ 新增：AI通知需求統計
            "multi_path_endpoints": defaultdict(list)
        }
    
    def load_module_config(self) -> dict:
        """載入模組配置文件"""
        try:
            if Path(self.module_config_path).exists():
                with open(self.module_config_path, encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 使用預設配置
                return self.get_default_config()
        except Exception as e:
            if self.verbose:
                print(f"載入配置失敗，使用預設配置: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """獲取預設配置（向後相容）"""
        return {
            "modules": {
                "aiva_core": {
                    "cognitive_core": "認知核心模組",
                    "internal_exploration": "內探模組", 
                    "task_planning": "任務規劃模組",
                    "external_learning": "外學模組",
                    "core_capabilities": "核心能力模組",
                    "service_backbone": "服務骨幹模組"
                }
            },
            "component_types": {
                "AI組件": {
                    "keywords": ["ai", "neural", "machine_learning", "model", "train"],
                    "patterns": [".*ai.*", ".*neural.*", ".*model.*"]
                },
                "程式組件": {
                    "keywords": ["config", "utils", "helper", "parser"],
                    "patterns": [".*config.*", ".*util.*"]
                },
                "混合組件": {
                    "keywords": ["analyzer", "processor", "engine"],
                    "patterns": [".*analyzer.*", ".*processor.*"]
                }
            }
        }
        
    def load_flow_data(self):
        """載入數據流數據
        
        ⚠️ 修復（2025-12-16）：如果已有 flows，則跳過載入
        符合 aiva_common 規範：保留未使用函數，維持向前兼容性
        """
        # 如果已經有 flows（通過 __init__ 傳入），則跳過載入
        if self.flows:
            if self.verbose:
                print(f"使用已傳入的 {len(self.flows)} 個 flows")
            self.stats['total_flows'] = len(self.flows)
            return
        
        # 原有的文件載入邏輯（保持向前兼容性）
        if not self.input_dir:
            raise ValueError("必須提供 input_dir 或 flows 參數")
        
        # Check if input_dir is a file or directory
        if self.input_dir.suffix == '.json':
            analysis_file = self.input_dir
        else:
            analysis_file = self.input_dir / "analysis_results.json"
        
        if not analysis_file.exists():
            raise FileNotFoundError(f"找不到分析結果文件: {analysis_file}")
        
        with open(analysis_file, encoding='utf-8') as f:
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
        
        # 轉換為標準格式
        # ✅ 修復 (2026-01-09): flow_chains 包含函數名稱，需從 function_map 查找路徑
        for idx, chain in enumerate(flow_chains, 1):
            if not chain:
                continue
            
            # chain 裡是函數名稱，從 function_map 查找對應的檔案路徑
            full_paths = []
            scripts = []
            for func_name in chain:
                func_info = self.function_map.get(func_name, {})
                file_path = func_info.get('file_path', '')
                full_paths.append(file_path)
                # 從路徑提取腳本名稱
                scripts.append(Path(file_path).stem if file_path else func_name)
            
            flow_data = {
                'id': idx,
                'name': f'{idx}',          # 內部直接用數字 ID
                'path': scripts,           # 腳本名稱列表
                'full_path': full_paths,   # 完整檔案路徑列表
                'func_names': chain,       # 原始函數名稱列表
                'length': len(chain),
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
    
    def _classify_module_from_path(self, filepath: str) -> str:
        """從完整文件路徑提取模組名稱
        
        ⚠️ 修復版本（2026-01-09）：
        - 直接從文件路徑中提取模組目錄名稱
        - 這是最準確的方法，不依賴關鍵字匹配
        - 只分析 aiva_core 下的五大模組
        """
        # 標準化路徑分隔符
        filepath_normalized = filepath.replace('\\', '/')
        
        # 確保是 aiva_core 下的檔案
        if '/aiva_core/' not in filepath_normalized:
            # 不在 aiva_core 下的檔案（如 ui/, ai_models.py）標記為 unknown
            return 'unknown'
        
        # 按順序檢查五大模組
        for module in [
            'cognitive_core',
            'internal_exploration',
            'task_planning',
            'core_capabilities',
            'service_backbone'
        ]:
            if f'/aiva_core/{module}/' in filepath_normalized:
                # 檢查是否為 learning_system 子目錄
                if module == 'cognitive_core' and '/learning_system/' in filepath_normalized:
                    return 'learning_system'  # 標記為學習子系統
                return module
        
        # aiva_core 根目錄下的檔案（如 __init__.py）
        # 根據檔名推斷模組
        filename = filepath_normalized.split('/')[-1].replace('.py', '')
        return self._classify_module(filename)
    
    def _merge_duplicate_flows(self):
        """合併相同起終點的數據流（2026-01-09）
        
        將相同起點和終點但路徑不同的數據流合併到第一個出現的位置，
        原位置標記為 merged（保留 ID 連續性），對AI分析友好。
        
        JSON結構：
        - 主流程：完整保留所有欄位 + path_variants + variant_count
        - 合併流程：僅保留 {id, merged_into, original_path, start, end}
        """
        if not self.merge_duplicate_flows:
            return
        
        if self.verbose:
            print("\n🔗 合併相同起終點的數據流...")
        
        # 建立起終點索引
        endpoint_map = {}  # {(start, end): [flow_indices]}
        for i, flow in enumerate(self.flows):
            if 'merged_into' in flow:  # 跳過已合併的
                continue
            key = (flow['start'], flow['end'])
            if key not in endpoint_map:
                endpoint_map[key] = []
            endpoint_map[key].append(i)
        
        # 找出需要合併的組
        merged_count = 0
        for key, indices in endpoint_map.items():
            if len(indices) > 1:
                # 保留第一個，其他標記為 merged
                primary_idx = indices[0]
                primary_flow = self.flows[primary_idx]
                
                # 收集所有路徑變體
                if 'path_variants' not in primary_flow:
                    primary_flow['path_variants'] = [primary_flow['path']]
                
                for idx in indices[1:]:
                    duplicate_flow = self.flows[idx]
                    # 添加路徑變體
                    if duplicate_flow['path'] not in primary_flow['path_variants']:
                        primary_flow['path_variants'].append(duplicate_flow['path'])
                    
                    # 標記為已合併，保留 ID（AI可追蹤）
                    self.flows[idx] = {
                        'id': duplicate_flow['id'],
                        'merged_into': primary_flow['id'],
                        'original_path': duplicate_flow['path'],
                        'start': duplicate_flow['start'],
                        'end': duplicate_flow['end']
                    }
                    merged_count += 1
                
                # 更新主流程的統計
                primary_flow['variant_count'] = len(primary_flow['path_variants'])
        
        if self.verbose:
            print(f"  ✅ 合併了 {merged_count} 條重複數據流")
            print(f"  ✅ 保留了 {len(endpoint_map)} 個唯一的起終點組合")
    
    def _classify_ai_capability_type(self, script_name: str) -> str:
        """分類AI能力類型：內部 vs 對外（2026-01-09）
        
        Returns:
            'AI內部能力': 神經網路、訓練、自我認知等內部處理
            'AI對外能力': 查詢、決策、執行等對外服務
            '非AI能力': 純程式邏輯組件
        """
        if script_name in self.AI_INTERNAL_CAPABILITIES:
            return self.AI_INTERNAL_TYPE
        elif script_name in self.AI_EXTERNAL_CAPABILITIES:
            return self.AI_EXTERNAL_TYPE
        else:
            return "非AI能力"
    
    def _classify_module(self, script_name: str) -> str:
        """根據腳本名稱分類到對應模組（降級方案）
        
        ⚠️ 注意：此方法已廢棄，優先使用 _classify_module_from_path()
        僅在沒有 full_path 時使用
        """
        script_lower = script_name.lower()
        
        # 1. Cognitive Core - AI 認知核心
        if any(keyword in script_lower for keyword in [
            'neural', 'rag', 'decision', 'ai_capability', 'ai_model',
            'orchestrator', 'skill_graph', 'anti_hallucination'
        ]):
            return 'cognitive_core'
        
        # 2. Internal Exploration - 內部探索
        elif any(keyword in script_lower for keyword in [
            'aiva_flow', 'aiva_cli', 'aiva_exploration', 'capability_cli',
            'self_aware', 'capability_analyzer', 'internal_loop',
            'analyze_dataflow', 'self_healing'
        ]):
            return 'internal_exploration'
        
        # 3. Task Planning - 任務規劃
        elif any(keyword in script_lower for keyword in [
            'plan_', 'planner', 'task_', 'commander', 'ai_commander',
            'executor', 'command_router'
        ]):
            return 'task_planning'
        
        # 4. External Learning - 外部學習
        elif any(keyword in script_lower for keyword in [
            'trainer', 'training', 'model_', 'bio_analysis', 'rl_',
            'resource_tracker', 'scalable_bio', 'external_loop',
            'risk_assessment', 'experience_manager'
        ]):
            return 'external_learning'
        
        # 5. Core Capabilities - 核心能力
        elif any(keyword in script_lower for keyword in [
            'capability_registry', 'attack_chain', 'business_logic',
            'conversation', 'plugin', 'assistant', 'initial_surface',
            'skill_graph', 'ingestion', 'output_formatter'
        ]):
            return 'core_capabilities'
        
        # 6. Service Backbone - 服務骨幹
        elif any(keyword in script_lower for keyword in [
            'backend', 'storage', 'api_gateway', 'message_bus',
            'state_manager', 'coordination', 'session_state',
            'monitoring', 'optimized_core', 'performance'
        ]):
            return 'service_backbone'
        
        # 7. 未匹配：使用更智能的推斷
        else:
            # 基於常見模式推斷
            if any(word in script_lower for word in ['core', 'engine', 'system']):
                return 'cognitive_core'
            elif any(word in script_lower for word in ['manager', 'coordinator', 'service']):
                return 'service_backbone'
            elif any(word in script_lower for word in ['analyze', 'explore', 'inspect']):
                return 'internal_exploration'
            else:
                # 最後才使用 service_backbone 作為兜底
                return 'service_backbone'
    
    def _classify_component_type(self, script_name: str) -> str:
        """分類組件類型（增強版 - 2026-01-09）"""
        # 優先檢查AI能力類型
        ai_type = self._classify_ai_capability_type(script_name)
        if ai_type != "非AI能力":
            return ai_type
        
        # 原有分類邏輯
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
    
    def _classify_loop_type(self, flow) -> str:
        """分類數據流的循環類型（用於雙循環AI優化系統）
        
        Args:
            flow: 包含路徑和模組資訊的數據流
            
        Returns:
            loop_type: 'internal_loop' | 'external_loop' | 'hybrid_loop' | 'no_loop'
            
        分類規則:
        - internal_loop: 內部循環，用於自我認知和內部分析
        - external_loop: 外部循環，用於實戰學習和外部反饋  
        - hybrid_loop: 混合循環，同時涉及內外部
        - no_loop: 非循環流程
        """
        # 獲取流程路徑和模組
        path_str = ' → '.join(flow.get('path', []))
        modules = flow.get('modules', [])
        endpoint_module = flow.get('primary_module', '')
        
        # 內部循環關鍵詞
        internal_keywords = [
            'system_self_explorer', 'internal_exploration', 'cognitive_core',
            'self_analysis', 'internal_monitor', 'system_health', 'capability_check',
            'neural', 'rag', 'learning_system', 'experience_manager',
            'ai_analysis', 'self_optimization', 'internal_feedback'
        ]
        
        # 外部循環關鍵詞  
        external_keywords = [
            'scan_engine', 'attack_engine', 'target_', 'external_',
            'penetration', 'vulnerability', 'exploit', 'battle_',
            'ssrf', 'sqli', 'xss', 'rce', 'auth_bypass',
            'external_feedback', 'target_analysis', 'war_feedback'
        ]
        
        # 檢查路徑字串中的關鍵詞
        path_lower = path_str.lower()
        internal_count = sum(1 for keyword in internal_keywords if keyword in path_lower)
        external_count = sum(1 for keyword in external_keywords if keyword in path_lower)
        
        # 檢查模組類型
        has_cognitive = 'cognitive_core' in modules
        has_internal_exploration = 'internal_exploration' in modules  
        has_scan_or_attack = any(module in ['scan_engine', 'attack_engine'] for module in modules)
        
        # 分類邏輯
        if internal_count > 0 and external_count > 0:
            return 'hybrid_loop'
        elif internal_count > 0 or has_cognitive or has_internal_exploration:
            return 'internal_loop'
        elif external_count > 0 or has_scan_or_attack:
            return 'external_loop'
        else:
            # 基於終點模組判斷
            if endpoint_module in ['cognitive_core', 'internal_exploration']:
                return 'internal_loop'
            elif endpoint_module in ['scan_engine', 'attack_engine']:
                return 'external_loop'
            else:
                return 'no_loop'
    
    def _get_connector_type(self, loop_type: str) -> str | None:
        """根據循環類型返回對應的連接器
        
        Args:
            loop_type: 循環類型 (internal_loop/external_loop/no_loop等)
            
        Returns:
            對應的連接器類名或 None
        """
        if loop_type == 'internal_loop':
            return 'InternalLoopConnector'
        elif loop_type == 'external_loop':
            return 'ExternalLoopConnector'
        return None
    
    def _get_notification_priority(self, loop_type: str) -> str:
        """根據循環類型返回通知優先級
        
        Args:
            loop_type: 循環類型
            
        Returns:
            優先級字符串 (high/medium/low)
        """
        if loop_type == 'internal_loop':
            return 'high'
        elif loop_type == 'external_loop':
            return 'medium'
        return 'low'
    
    def classify_flows(self):
        """對所有數據流進行分類
        
        ⚠️ 增強版本（2026-01-09）：
        - 使用終點腳本進行模組分類
        - 區分AI內部能力 vs 對外能力
        - 支持相同起終點數據流合併
        
        分類策略：
        - primary_module: 基於數據流終點腳本的模組（權威分類）
        - endpoint_component_type: 終點腳本的AI能力類型
        - majority_module: 基於多數決的模組（參考）
        """
        if self.verbose:
            print("\n開始分類數據流...")
            print("🎯 使用終點腳本進行精確分類")
        
        # 先合併重複數據流
        self._merge_duplicate_flows()
        
        for flow in self.flows:
            # 跳過已合併的流程（僅保留標記）
            if 'merged_into' in flow:
                continue
            
            # 分類每個腳本
            flow['classifications'] = []
            flow['modules'] = []
            flow['component_types'] = []
            
            # ✅ 修復：使用 full_path 而不是 path
            if 'full_path' in flow and flow['full_path']:
                for script, full_path in zip(flow['path'], flow['full_path'], strict=False):
                    module = self._classify_module_from_path(full_path)  # ✅ 使用路徑
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
            else:
                # 降級方案：沒有 full_path 時使用舊方法
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
                # ✅ 使用終點腳本的模組作為primary_module（AI模組分類）
                endpoint_module = flow['modules'][-1] if flow['modules'] else 'unknown'
                flow['primary_module'] = endpoint_module
                flow['endpoint_module'] = endpoint_module  # 新增終點模組欄位
                
                # 同時保留多數決結果做為對比
                majority_module = max(set(flow['modules']), key=flow['modules'].count)
                flow['majority_module'] = majority_module
                
                self.stats['module_distribution'][endpoint_module] += 1
            
            # 統計主要組件類型（基於終點腳本）
            if flow['component_types']:
                # ✅ 使用終點腳本的組件類型
                endpoint_component = flow['component_types'][-1] if flow['component_types'] else '未分類'
                flow['primary_component_type'] = endpoint_component
                flow['endpoint_component_type'] = endpoint_component
                
                # 標記是否為AI能力
                flow['is_ai_capability'] = endpoint_component in [
                    self.AI_INTERNAL_TYPE, 
                    self.AI_EXTERNAL_TYPE, 
                    self.AI_COMPONENT_TYPE
                ]
                
                self.stats['component_type_distribution'][endpoint_component] += 1
            
            # 記錄終點用於多路徑分析
            if flow['end']:
                self.stats['multi_path_endpoints'][flow['end']].append(flow['id'])
            
            # ✅ 新增：分類循環類型（用於雙循環AI優化系統）
            flow['loop_type'] = self._classify_loop_type(flow)
            
            # ✅ 新增：AI通知配置（基於循環類型）
            connector = self._get_connector_type(flow['loop_type'])
            priority = self._get_notification_priority(flow['loop_type'])
            
            flow['ai_notification'] = {
                'required': flow['loop_type'] in ['internal_loop', 'external_loop'],
                'connector': connector,
                'priority': priority
            }
            
            # ✅ 更新統計信息
            self.stats['loop_type_distribution'][flow['loop_type']] += 1
            self.stats['ai_notification_required'][flow['ai_notification']['required']] += 1
        
        if self.verbose:
            print(f"分類完成: {len(self.flows)} 條數據流")
    
    def analyze_multi_path_endpoints(self) -> list[dict]:
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
        if self.output_dir:
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
        if not self.output_dir:
            return
        report_file = self.output_dir / "classification_summary.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# AIVA Core 數據流分類統計報告\n\n")
            f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本統計
            f.write("## 基本統計\n\n")
            f.write(f"- **總數據流數量**: {self.stats['total_flows']}\n")
            
            # 過濾出活躍流程
            active_flows = [flow for flow in self.flows if 'merged_into' not in flow]
            merged_count = len(self.flows) - len(active_flows)
            if merged_count > 0:
                f.write(f"- **活躍數據流**: {len(active_flows)} 條（去重後）\n")
                f.write(f"- **已合併數據流**: {merged_count} 條\n")
            
            if active_flows:
                avg_length = sum(flow['length'] for flow in active_flows) / len(active_flows)
                f.write(f"- **平均流程長度**: {avg_length:.2f}\n")
                
                max_flow = max(active_flows, key=lambda x: x['length'])
                f.write(f"- **最長流程**: {max_flow['length']} 步 (Flow {max_flow['id']})\n")
                
                min_flow = min(active_flows, key=lambda x: x['length'])
                f.write(f"- **最短流程**: {min_flow['length']} 步 (Flow {min_flow['id']})\n\n")
                
                # AI能力分類統計（新增）
                f.write("## AI能力分類統計\n\n")
                ai_internal = len([f for f in active_flows if f.get('is_ai_capability') and f.get('endpoint_component_type') == self.AI_INTERNAL_TYPE])
                ai_external = len([f for f in active_flows if f.get('is_ai_capability') and f.get('endpoint_component_type') == self.AI_EXTERNAL_TYPE])
                non_ai = len(active_flows) - ai_internal - ai_external
                
                total = len(active_flows)
                f.write("| 能力類型 | 數量 | 占比 | 說明 |\n")
                f.write("|---------|------|------|------|\n")
                f.write(f"| {self.AI_INTERNAL_TYPE} | {ai_internal} | {ai_internal/total*100:.1f}% | 神經網路、訓練、RAG等內部處理 |\n")
                f.write(f"| {self.AI_EXTERNAL_TYPE} | {ai_external} | {ai_external/total*100:.1f}% | 查詢、決策、執行等對外服務 |\n")
                f.write(f"| 非AI能力 | {non_ai} | {non_ai/total*100:.1f}% | 純程式邏輯組件 |\n\n")
            
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
            
            # 各模組詳細統計（新增 - 2026-01-09）
            f.write("\n## 各模組詳細能力統計\n\n")
            f.write("| 模組 | 合併前 | 合併後 | AI內部 | AI對外 | 非AI | 能力種類 | 有路徑變體 |\n")
            f.write("|------|--------|--------|--------|--------|------|----------|------------|\n")
            
            for module_id, _ in sorted(
                self.stats['module_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                # 合併前的數量（包含所有流程）
                all_module_flows = [f for f in self.flows if f.get('primary_module') == module_id]
                original_count = len(all_module_flows)
                
                # 合併後的數量（僅活躍流程）
                module_flows = [f for f in active_flows if f.get('primary_module') == module_id]
                if not module_flows:
                    continue
                
                # 統計AI類型
                ai_internal = len([f for f in module_flows if f.get('endpoint_component_type') == self.AI_INTERNAL_TYPE])
                ai_external = len([f for f in module_flows if f.get('endpoint_component_type') == self.AI_EXTERNAL_TYPE])
                non_ai = len(module_flows) - ai_internal - ai_external
                
                # 統計能力種類（唯一的起終點組合）
                unique_capabilities = len({(f['start'], f['end']) for f in module_flows})
                
                # 統計有路徑變體的能力數量
                with_variants = len([f for f in module_flows if f.get('variant_count', 1) > 1])
                
                module_name = self.MODULES.get(module_id, module_id)
                f.write(f"| {module_name} | {original_count} | {len(module_flows)} | {ai_internal} | {ai_external} | {non_ai} | {unique_capabilities} | {with_variants} |\n")
            
            # 路徑變體詳細統計（新增 - 2026-01-09）
            f.write("\n## 路徑變體詳細統計\n\n")
            flows_with_variants = [f for f in active_flows if f.get('variant_count', 1) > 1]
            if flows_with_variants:
                f.write(f"共有 {len(flows_with_variants)} 個能力存在多條路徑變體：\n\n")
                
                # 按模組分組
                from collections import defaultdict
                by_module = defaultdict(list)
                for flow in flows_with_variants:
                    by_module[flow.get('primary_module', 'unknown')].append(flow)
                
                for module_id, module_flows in sorted(by_module.items(), key=lambda x: len(x[1]), reverse=True):
                    module_name = self.MODULES.get(module_id, module_id)
                    f.write(f"\n### {module_name}\n\n")
                    f.write(f"本模組有 {len(module_flows)} 種能力存在路徑變體\n\n")
                    f.write("| 能力 (起點→終點) | 合併前流程數 | 路徑變體數 | Flow ID |\n")
                    f.write("|------------------|-------------|-----------|----------|\n")
                    
                    for flow in sorted(module_flows, key=lambda x: x.get('variant_count', 1), reverse=True):
                        capability_name = f"{flow['start']} → {flow['end']}"
                        variant_count = flow.get('variant_count', 1)
                        # 計算合併前的流程數（1個主流程 + 合併的數量）
                        merged_flows = [f for f in self.flows 
                                      if f.get('merged_into') == flow['id']]
                        original_count = 1 + len(merged_flows)
                        f.write(f"| {capability_name} | {original_count} | {variant_count} | Flow {flow['id']} |\n")
            else:
                f.write("所有能力均為單一路徑，無路徑變體。\n")
            
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
        if not self.output_dir:
            return
        report_file = self.output_dir / "complete_flow_details.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            self._write_flow_details_header(f)
            self._write_flows_by_module(f)
        
        if self.verbose:
            print(f"生成完整流程詳細列表: {report_file}")
    
    def _write_flow_details_header(self, f):
        """寫入流程詳細列表標題"""
        f.write("# 完整數據流詳細列表\n\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"總數據流數量: {len(self.flows)}\n\n")
        f.write(SECTION_SEPARATOR)
    
    def _write_flows_by_module(self, f):
        """按模組寫入數據流詳細信息"""
        for module_id in self.MODULES.keys():
            module_flows = [flow for flow in self.flows if flow.get('primary_module') == module_id]
            if module_flows:
                self._write_module_section(f, module_id, module_flows)
    
    def _write_module_section(self, f, module_id: str, module_flows: list[dict]):
        """寫入單個模組的數據流信息"""
        module_name = self.MODULES[module_id]
        f.write(f"## {module_name} ({module_id})\n\n")
        f.write(f"包含 {len(module_flows)} 條數據流\n\n")
        
        for flow in module_flows:
            self._write_single_flow_details(f, flow)
    
    def _write_single_flow_details(self, f, flow: dict):
        """寫入單個數據流的詳細信息"""
        f.write(f"### Flow {flow['id']}\n\n")
        f.write(f"- **長度**: {flow['length']} 步\n")
        f.write(f"- **起點**: {flow['start']}\n")
        f.write(f"- **終點**: {flow['end']}\n")
        f.write(f"- **主要模組**: {self.MODULES.get(flow['primary_module'], flow['primary_module'])}\n")
        f.write(f"- **主要組件類型**: {flow['primary_component_type']}\n\n")
        
        f.write("**執行路徑**:\n\n")
        for i, classification in enumerate(flow['classifications'], 1):
            self._write_classification_step(f, i, classification)
        
        f.write(SECTION_SEPARATOR)
    
    def _write_classification_step(self, f, step_num: int, classification: dict):
        """寫入單個分類步驟的信息"""
        script_name = classification['script']
        script_info = self.script_functions.get(script_name, {})
        functions = script_info.get('functions', {})
        file_path = script_info.get('file_path', '')
        file_name = file_path.split('\\')[-1].replace('.py', '') if file_path else script_name
        
        f.write(f"{step_num}. **{classification['component_type']}**\n")
        
        if functions:
            self._write_function_details(f, functions, file_name)
        else:
            f.write(f"   {script_name}\n")
            f.write(f"   {file_name}\n")
        
        f.write(f"   - 模組: {self.MODULES.get(classification['module'], classification['module'])}\n\n")
    
    def _write_function_details(self, f, functions: dict, file_name: str):
        """寫入函數詳細信息"""
        func_names = list(functions.keys())
        for func_name in func_names:
            f.write(f"   {func_name}\n")
            f.write(f"   {file_name}\n")
            if func_name != func_names[-1]:
                f.write("   \n")
    
    def _generate_multi_path_report(self, multi_path_analysis: list[dict]):
        """生成多路徑分析報告"""
        if not self.output_dir:
            return
        report_file = self.output_dir / "multi_path_analysis.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            self._write_multi_path_header(f, multi_path_analysis)
            for analysis in multi_path_analysis:
                self._write_single_endpoint_analysis(f, analysis)
        
        if self.verbose:
            print(f"生成多路徑分析報告: {report_file}")
    
    def _write_multi_path_header(self, f, multi_path_analysis: list[dict]):
        """寫入多路徑報告標題"""
        f.write("# 多路徑終點分析報告\n\n")
        f.write(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"找到 {len(multi_path_analysis)} 個有多條路徑到達的終點\n\n")
        f.write(SECTION_SEPARATOR)
    
    def _write_single_endpoint_analysis(self, f, analysis: dict):
        """寫入單個終點的分析結果"""
        f.write(f"## 終點: {analysis['endpoint']}\n\n")
        f.write(f"**說明**: {analysis['endpoint_description']}\n\n")
        
        self._write_endpoint_summary(f, analysis)
        self._write_path_details(f, analysis['paths'])
        self._write_path_difference_analysis(f, analysis['paths'])
        
        f.write("---\n\n")
    
    def _write_endpoint_summary(self, f, analysis: dict):
        """寫入終點摘要信息"""
        f.write(f"- **路徑總數**: {analysis['total_paths']}\n")
        f.write(f"- **路徑長度範圍**: {analysis['path_length_range'][0]} - {analysis['path_length_range'][1]} 步\n")
        f.write(f"- **平均路徑長度**: {analysis['average_length']:.2f} 步\n")
        
        modules_str = ', '.join([m for m in [self.MODULES.get(m, m) for m in analysis['all_modules_involved']] if m is not None])
        f.write(f"- **涉及模組**: {modules_str}\n\n")
    
    def _write_path_details(self, f, paths: list[dict]):
        """寫入路徑詳細信息"""
        f.write("### 路徑詳細對比\n\n")
        
        for i, path_info in enumerate(paths, 1):
            f.write(f"#### 路徑 {i} (Flow {path_info['flow_id']})\n\n")
            f.write(f"- **長度**: {path_info['length']} 步\n")
            f.write(f"- **主要模組**: {self.MODULES.get(path_info['primary_module'], path_info['primary_module'])}\n")
            f.write(f"- **執行順序**: {path_info['description']}\n\n")
            
            f.write("**完整腳本列表**:\n")
            for j, script in enumerate(path_info['scripts'], 1):
                f.write(f"{j}. {script}\n")
            f.write("\n")
    
    def _write_path_difference_analysis(self, f, paths: list[dict]):
        """寫入路徑差異分析"""
        if len(paths) < 2:
            return
            
        f.write("### 路徑差異分析\n\n")
        
        path1, path2 = paths[0], paths[1]
        set1, set2 = set(path1['scripts']), set(path2['scripts'])
        
        unique_to_path1 = set1 - set2
        unique_to_path2 = set2 - set1
        common = set1 & set2
        
        self._write_script_comparison(f, unique_to_path1, unique_to_path2, common)
        self._write_usage_scenario_analysis(f, path1, path2)
    
    def _write_script_comparison(self, f, unique_to_path1: set[str], unique_to_path2: set[str], common: set[str]):
        """寫入腳本對比信息"""
        f.write("**路徑 1 vs 路徑 2 對比**:\n\n")
        f.write(f"- 共同腳本數: {len(common)}\n")
        f.write(f"- 路徑 1 獨有: {len(unique_to_path1)} 個腳本\n")
        if unique_to_path1:
            f.write(f"  - {', '.join(unique_to_path1)}\n")
        f.write(f"- 路徑 2 獨有: {len(unique_to_path2)} 個腳本\n")
        if unique_to_path2:
            f.write(f"  - {', '.join(unique_to_path2)}\n")
        f.write("\n")
    
    def _write_usage_scenario_analysis(self, f, path1: dict, path2: dict):
        """寫入使用場景差異分析"""
        f.write("**使用場景差異推測**:\n\n")
        
        # 模組差異分析
        module_diff = set(path1['modules']) != set(path2['modules'])
        if module_diff:
            modules1_str = ', '.join([m for m in [self.MODULES.get(m, m) for m in set(path1['modules'])] if m is not None])
            modules2_str = ', '.join([m for m in [self.MODULES.get(m, m) for m in set(path2['modules'])] if m is not None])
            f.write(f"- 路徑 1 主要涉及: {modules1_str}\n")
            f.write(f"- 路徑 2 主要涉及: {modules2_str}\n")
            f.write("- **差異**: 兩條路徑經過不同的架構模組,可能用於不同的功能場景\n\n")
        
        # 長度差異分析
        length_diff = abs(path1['length'] - path2['length'])
        if length_diff > 2:
            f.write(f"- 路徑長度差異顯著 ({length_diff} 步)\n")
            if path1['length'] < path2['length']:
                f.write("- **推測**: 路徑 1 可能是快速路徑或直接調用,路徑 2 可能包含更多處理邏輯\n\n")
            else:
                f.write("- **推測**: 路徑 2 可能是快速路徑或直接調用,路徑 1 可能包含更多處理邏輯\n\n")
    
    def _generate_json_export(self, multi_path_analysis: list[dict]):
        """生成 JSON 格式完整數據
        
        v3.3 (2026-01-21): 統一路徑配置
        - 輸出至 internal_classification.json (SOT)
        - 同時輸出至 output_dir/classification_data.json (向後相容)
        
        v3.3 (2026-01-04): 新增 5M AI 專用欄位
        - parameters: 從函數定義提取
        - return_type: 從 type hints 提取
        - cli_command: 根據模組自動生成
        """
        if not self.output_dir:
            return
        
        # v3.3: 為每個 flow 添加 AI 專用欄位
        enhanced_flows = []
        for flow in self.flows:
            enhanced_flow = flow.copy()
            
            # 添加 cli_command（基於終點腳本和模組）
            enhanced_flow['cli_command'] = self._generate_cli_command(flow)
            
            # 添加終點函數的 parameters 和 return_type
            endpoint_info = self._get_endpoint_function_info(flow)
            enhanced_flow['parameters'] = endpoint_info.get('parameters', [])
            enhanced_flow['return_type'] = endpoint_info.get('return_type', 'unknown')
            
            # 添加結構化標籤（用於 5M AI 向量編碼）
            enhanced_flow['structured_tags'] = self._generate_structured_tags(flow)
            
            enhanced_flows.append(enhanced_flow)
        
        export_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_flows': self.stats['total_flows'],
                'module_distribution': dict(self.stats['module_distribution']),
                'component_type_distribution': dict(self.stats['component_type_distribution']),
                # v3.3: 版本標記
                'schema_version': '3.3',
                'ai_compatible': True,  # 標記此格式支援 5M AI
                'classification_type': 'internal',  # 標記為內部分類
                'target_modules': ['cognitive_core', 'internal_exploration', 'task_planning', 
                                   'core_capabilities', 'service_backbone']
            },
            'flows': enhanced_flows,
            'multi_path_analysis': multi_path_analysis
        }
        
        # 輸出到 output_dir (向後相容)
        json_file = self.output_dir / "classification_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        # v3.3: 同時輸出到統一路徑 (SOT)
        if USING_NEW_PATHS:
            sot_file = InternalPaths.CLASSIFICATION_FILE
            sot_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sot_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            if self.verbose:
                print(f"生成 JSON 數據 (SOT): {sot_file}")
        
        if self.verbose:
            print(f"生成 JSON 數據導出: {json_file}")
    
    def _generate_cli_command(self, flow: dict) -> str:
        """根據 flow 生成 CLI 命令
        
        v3.3 (2026-01-04): 新增方法
        格式: python -m <module_path> <action> [--options]
        """
        if not flow.get('full_path'):
            return ""
        
        endpoint_path = flow['full_path'][-1] if flow['full_path'] else ""
        if not endpoint_path:
            return ""
        
        # 從路徑提取模組名稱
        # 例如: services/core/aiva_core/cognitive_core/rag/vector_store.py
        # -> services.core.aiva_core.cognitive_core.rag.vector_store
        normalized = endpoint_path.replace('\\', '/').replace('.py', '')
        
        # 找到 services 開始的位置
        if 'services/' in normalized:
            module_path = normalized.split('services/')[-1]
            module_path = 'services.' + module_path.replace('/', '.')
        else:
            module_path = normalized.split('/')[-1]
        
        # 根據模組類型生成不同的命令格式
        primary_module = flow.get('primary_module', 'unknown')
        
        if primary_module == 'internal_exploration':
            return f"python -m {module_path} --flow-id {flow.get('id', 0)}"
        elif primary_module == 'cognitive_core':
            return f"python -m {module_path} query"
        elif primary_module == 'task_planning':
            return f"python -m {module_path} execute"
        else:
            return f"python -m {module_path}"
    
    def _get_endpoint_function_info(self, flow: dict) -> dict:
        """獲取終點函數的詳細信息
        
        v3.3 (2026-01-04): 新增方法
        """
        if not flow.get('end'):
            return {'parameters': [], 'return_type': 'unknown'}
        
        endpoint_script = flow['end']
        
        # 從 script_functions 獲取函數信息
        script_info = self.script_functions.get(endpoint_script, {})
        functions = script_info.get('functions', {})
        
        # 優先使用入口點函數
        entry_points = script_info.get('entry_points', [])
        target_func = None
        
        for ep in entry_points:
            if ep in functions:
                target_func = functions[ep]
                break
        
        # 如果沒有入口點，使用第一個函數
        if not target_func and functions:
            target_func = list(functions.values())[0]
        
        if target_func:
            return {
                'parameters': target_func.get('parameters', []),
                'return_type': target_func.get('return_type', 'unknown')
            }
        
        return {'parameters': [], 'return_type': 'unknown'}
    
    def _generate_structured_tags(self, flow: dict) -> list[str]:
        """生成結構化標籤（用於 5M AI 向量編碼）
        
        v3.3 (2026-01-04): 新增方法
        
        標籤格式：
        - module:<module_name>
        - type:<component_type>
        - length:<short|medium|long>
        - async:<true|false>
        """
        tags = []
        
        # 模組標籤
        if flow.get('primary_module'):
            tags.append(f"module:{flow['primary_module']}")
        
        # 組件類型標籤
        if flow.get('primary_component_type'):
            comp_type = flow['primary_component_type'].replace('組件', '')
            tags.append(f"type:{comp_type}")
        
        # 長度標籤
        length = flow.get('length', 0)
        if length <= 3:
            tags.append("length:short")
        elif length <= 7:
            tags.append("length:medium")
        else:
            tags.append("length:long")
        
        # 檢查是否包含異步組件
        has_async = False
        for classification in flow.get('classifications', []):
            script_name = classification.get('script', '')
            script_info = self.script_functions.get(script_name, {})
            for func_info in script_info.get('functions', {}).values():
                if func_info.get('is_async', False):
                    has_async = True
                    break
            if has_async:
                break
        
        tags.append(f"async:{str(has_async).lower()}")
        
        return tags
    
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
        description='AIVA Core 數據流分類分析器 (內部模組專用)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️ 注意: 此分類器僅用於 AI Core 內部模組 (純 Python)
        外部模組請使用 aiva_external_classifier.py

數據存放位置:
  services/integration/data/internal_exploration/
  ├── internal_classification.json     # 分類數據
  ├── internal_classification_summary.md  # 摘要報告
  └── analysis_results/internal/       # 詳細分析

示例:
  # 使用預設路徑 (推薦)
  python aiva_internal_classifier.py --input ./aiva_core_analysis
  
  # 指定輸出目錄
  python aiva_internal_classifier.py --input ./analysis --output ./custom_output --verbose
        """
    )
    
    # 設定預設路徑
    if USING_NEW_PATHS:
        default_output = str(InternalPaths.OUTPUT_DIR)
    else:
        default_output = './classification_results'
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='輸入目錄 (包含 analysis_results.json 的目錄)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=default_output,
        help=f'輸出目錄 (默認: {default_output})'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='顯示詳細處理信息'
    )
    
    args = parser.parse_args()
    
    # 確保目錄存在
    if USING_NEW_PATHS:
        ensure_all_dirs()
    
    # 創建分類器並執行
    classifier = AIVAFlowClassifier(
        input_dir=args.input,
        output_dir=args.output,
        verbose=args.verbose,
        merge_duplicate_flows=False  # 不合併，因為中間路徑不同會產生不同結果
    )
    
    return classifier.run()


if __name__ == '__main__':
    exit(main())
