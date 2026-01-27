"""
AIVA 增強分類器處理器 v1.0
Enhanced Classifier Processor with Layer Recovery Architecture

功能描述：
1. 分層可恢復處理架構 - 支持斷點恢復和增量處理
2. LLM 友善的能力描述生成系統
3. 統一標準化格式輸出
4. 動態能力描述補充機制

技術特點：
- 基於現有 aiva_internal_classifier 擴展
- 支持多種輸入源（classification_data.json, flows 列表等）
- 分層處理：原始分析 -> 語義增強 -> LLM 補充 -> 標準化輸出
- 可恢復機制：每層處理保存檢查點
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import hashlib

# 導入現有分類器
sys.path.insert(0, str(Path(__file__).parent))
from aiva_internal_classifier import AIVAFlowClassifier


class ProcessingStage(Enum):
    """處理階段枚舉"""
    RAW_ANALYSIS = "raw_analysis"
    SEMANTIC_ENHANCEMENT = "semantic_enhancement"
    LLM_AUGMENTATION = "llm_augmentation"
    STANDARDIZATION = "standardization"
    COMPLETION = "completion"


@dataclass
class ProcessingCheckpoint:
    """處理檢查點數據結構"""
    stage: ProcessingStage
    timestamp: str
    data_hash: str
    metadata: Dict[str, Any]
    progress: int  # 0-100
    output_path: str


@dataclass
class EnhancedCapabilityDescription:
    """增強能力描述數據結構 - LLM 友善格式"""
    
    # 基本識別信息
    flow_id: int
    capability_name: str
    display_name: str
    
    # 功能性描述
    purpose: str                    # 用途說明
    input_description: str          # 輸入描述
    output_description: str         # 輸出描述
    side_effects: List[str]         # 副作用
    
    # 技術規格
    execution_complexity: str       # simple/medium/complex
    estimated_duration: str         # 預估執行時間
    dependencies: List[str]         # 依賴項目
    
    # AI 相關屬性
    ai_capability_level: str        # none/basic/advanced/core
    requires_human_input: bool      # 是否需要人工輸入
    risk_level: str                # low/medium/high
    
    # 使用場景
    usage_scenarios: List[str]      # 使用場景列表
    example_commands: List[str]     # 範例命令
    
    # 內部技術細節
    primary_module: str
    component_type: str
    path_variants: List[List[str]]  # 路徑變體
    
    # 元數據
    last_updated: str
    source_confidence: float        # 0.0-1.0


class EnhancedClassifierProcessor:
    """增強分類器處理器 - 分層可恢復架構"""
    
    def __init__(self, 
                 work_dir: str = None,
                 checkpoint_dir: str = None,
                 enable_recovery: bool = True,
                 verbose: bool = True):
        """
        初始化處理器
        
        Args:
            work_dir: 工作目錄
            checkpoint_dir: 檢查點目錄
            enable_recovery: 是否啟用斷點恢復
            verbose: 詳細輸出模式
        """
        self.work_dir = Path(work_dir) if work_dir else Path.cwd() / "enhanced_classification"
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.work_dir / "checkpoints"
        self.enable_recovery = enable_recovery
        self.verbose = verbose
        
        # 創建目錄結構
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化分層輸出目錄
        self.layer_dirs = {
            ProcessingStage.RAW_ANALYSIS: self.work_dir / "1_raw_analysis",
            ProcessingStage.SEMANTIC_ENHANCEMENT: self.work_dir / "2_semantic_enhancement", 
            ProcessingStage.LLM_AUGMENTATION: self.work_dir / "3_llm_augmentation",
            ProcessingStage.STANDARDIZATION: self.work_dir / "4_standardization",
            ProcessingStage.COMPLETION: self.work_dir / "5_completion"
        }
        
        for layer_dir in self.layer_dirs.values():
            layer_dir.mkdir(parents=True, exist_ok=True)
            
        # 初始化狀態
        self.current_stage = ProcessingStage.RAW_ANALYSIS
        self.checkpoints: List[ProcessingCheckpoint] = []
        self.raw_classifier_data = None
        self.enhanced_capabilities: Dict[int, EnhancedCapabilityDescription] = {}
        
        # 預設模板
        self.llm_templates = self._init_llm_templates()
        
    def _init_llm_templates(self) -> Dict[str, str]:
        """初始化 LLM 補充描述用的模板"""
        return {
            "purpose_template": "基於執行路徑 {path} 和模組 {module}，這個能力的主要用途是：",
            "scenario_template": "這個能力 {capability_name} 適用於以下場景：",
            "risk_template": "執行 {capability_name} 的風險等級評估：",
            "example_template": "使用 {capability_name} 的範例命令：",
        }
    
    def load_from_classifier_data(self, 
                                 classification_data_path: str = None,
                                 flows_data: List[Dict] = None) -> bool:
        """
        從分類器數據載入
        
        Args:
            classification_data_path: 分類數據文件路徑
            flows_data: 直接傳入的流程數據
            
        Returns:
            bool: 載入成功與否
        """
        try:
            if flows_data:
                # 直接使用傳入的數據
                self.raw_classifier_data = {'flows': flows_data}
                if self.verbose:
                    print(f"[OK] 從內存載入了 {len(flows_data)} 條流程")
            else:
                # 從文件載入
                if not classification_data_path:
                    classification_data_path = "features_classification/classification_data.json"
                
                data_path = Path(classification_data_path)
                if not data_path.exists():
                    if self.verbose:
                        print(f"[Error] 找不到分類數據文件: {data_path}")
                    return False
                
                with open(data_path, 'r', encoding='utf-8') as f:
                    self.raw_classifier_data = json.load(f)
                    
                if self.verbose:
                    flows_count = len(self.raw_classifier_data.get('flows', []))
                    print(f"[OK] 從文件載入了 {flows_count} 條流程")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[Error] 載入分類數據失敗: {e}")
            return False
    
    def check_recovery_point(self) -> Optional[ProcessingStage]:
        """檢查是否有可恢復的檢查點"""
        if not self.enable_recovery:
            return None
        
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoint_files:
            return None
        
        # 找到最新的檢查點
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_checkpoint, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            checkpoint = ProcessingCheckpoint(**checkpoint_data)
            self.checkpoints.append(checkpoint)
            
            if self.verbose:
                print(f"[Resume] 發現可恢復檢查點: {checkpoint.stage.value} ({checkpoint.progress}%)")
            
            return checkpoint.stage
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  檢查點文件損壞: {e}")
            return None
    
    def save_checkpoint(self, 
                       stage: ProcessingStage, 
                       progress: int, 
                       output_path: str,
                       metadata: Dict = None) -> bool:
        """保存處理檢查點"""
        if not self.enable_recovery:
            return True
            
        try:
            # 計算數據哈希
            data_hash = self._calculate_data_hash()
            
            checkpoint = ProcessingCheckpoint(
                stage=stage,
                timestamp=datetime.now().isoformat(),
                data_hash=data_hash,
                metadata=metadata or {},
                progress=progress,
                output_path=output_path
            )
            
            checkpoint_file = self.checkpoint_dir / f"checkpoint_{stage.value}_{int(datetime.now().timestamp())}.json"
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(checkpoint), f, indent=2, ensure_ascii=False, default=str)
            
            self.checkpoints.append(checkpoint)
            
            if self.verbose:
                print(f"[Save] 保存檢查點: {stage.value} ({progress}%)")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[Error] 保存檢查點失敗: {e}")
            return False
    
    def _calculate_data_hash(self) -> str:
        """計算當前數據的哈希值"""
        if not self.raw_classifier_data:
            return ""
        
        data_str = json.dumps(self.raw_classifier_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    def stage_1_raw_analysis(self) -> bool:
        """階段1: 原始分析 - 基於現有分類器進行基礎分析"""
        if self.verbose:
            print("[Stage1] 階段1: 執行原始分析...")
        
        if not self.raw_classifier_data:
            if self.verbose:
                print("[Error] 缺少原始分類數據")
            return False
        
        try:
            flows = self.raw_classifier_data.get('flows', [])
            
            # 使用現有分類器重新分類（確保數據一致性）
            classifier = AIVAFlowClassifier(flows=flows, verbose=False)
            classifier.classify_flows()
            
            # 提取原始分析結果
            raw_analysis = {
                'timestamp': datetime.now().isoformat(),
                'total_flows': len(flows),
                'flows_analyzed': [],
                'module_stats': dict(classifier.stats['module_distribution']),
                'component_stats': dict(classifier.stats['component_type_distribution'])
            }
            
            # 分析每個流程的基礎信息
            for flow in classifier.flows:
                if 'merged_into' in flow:  # 跳過合併的流程
                    continue
                
                # 確保模組分類正確執行
                if 'primary_module' not in flow or flow.get('primary_module') == 'unknown':
                    # 手動分類模組
                    flow['primary_module'] = self._extract_module_from_flow(flow)
                    flow['primary_component_type'] = self._classify_component_from_flow(flow)
                
                flow_analysis = {
                    'flow_id': flow['id'],
                    'basic_name': f"Flow_{flow['id']}",
                    'path': flow['path'],
                    'length': flow['length'],
                    'start_point': flow['start'],
                    'end_point': flow['end'],
                    'primary_module': flow.get('primary_module', 'service_backbone'),
                    'component_type': flow.get('primary_component_type', '程式組件'),
                    'has_variants': flow.get('variant_count', 1) > 1,
                    'variant_paths': flow.get('path_variants', []),
                    'ai_capability_indicators': self._detect_ai_indicators(flow),
                    'full_path': flow.get('full_path', [])
                }
                
                raw_analysis['flows_analyzed'].append(flow_analysis)
            
            # 保存原始分析結果
            output_path = self.layer_dirs[ProcessingStage.RAW_ANALYSIS] / "raw_analysis_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(raw_analysis, f, indent=2, ensure_ascii=False)
            
            # 保存檢查點
            self.save_checkpoint(
                ProcessingStage.RAW_ANALYSIS, 
                100, 
                str(output_path),
                {'flows_count': len(raw_analysis['flows_analyzed'])}
            )
            
            if self.verbose:
                print(f"[OK] 階段1完成 - 分析了 {len(raw_analysis['flows_analyzed'])} 個流程")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[Error] 階段1失敗: {e}")
            return False
    
    def _extract_module_from_flow(self, flow: Dict) -> str:
        """從flow數據中提取模組信息"""
        # 檢查full_path中的模組信息
        if 'full_path' in flow and flow['full_path']:
            for path in flow['full_path']:
                if isinstance(path, str) and 'aiva_core' in path:
                    # 提取模組名稱
                    parts = path.replace('\\', '/').split('/')
                    try:
                        aiva_core_idx = parts.index('aiva_core')
                        if aiva_core_idx + 1 < len(parts):
                            module_name = parts[aiva_core_idx + 1]
                            if module_name in ['cognitive_core', 'internal_exploration', 'task_planning', 'core_capabilities', 'service_backbone']:
                                return module_name
                    except ValueError:
                        continue
        
        # 基於路徑內容推斷
        path_text = ' '.join(flow.get('path', [])).lower()
        
        if any(keyword in path_text for keyword in ['neural', 'rag', 'decision', 'ai_']):
            return 'cognitive_core'
        elif any(keyword in path_text for keyword in ['aiva_', 'analyze', 'classifier', 'internal']):
            return 'internal_exploration'
        elif any(keyword in path_text for keyword in ['plan', 'task', 'execute']):
            return 'task_planning'
        elif any(keyword in path_text for keyword in ['capability', 'skill', 'ability']):
            return 'core_capabilities'
        else:
            return 'service_backbone'
    
    def _classify_component_from_flow(self, flow: Dict) -> str:
        """從flow數據中分類組件類型"""
        path_text = ' '.join(flow.get('path', [])).lower()
        
        if any(keyword in path_text for keyword in ['ai_', 'neural', 'rag', 'model']):
            return 'AI組件'
        elif any(keyword in path_text for keyword in ['config', 'setting', 'manage']):
            return '程式組件'
        else:
            return '混合組件'
    
    def _detect_ai_indicators(self, flow: Dict) -> Dict[str, Any]:
        """檢測流程中的AI能力指標"""
        indicators = {
            'has_neural_components': False,
            'has_rag_components': False,
            'has_training_components': False,
            'has_decision_components': False,
            'ai_intensity_score': 0.0  # 0.0-1.0
        }
        
        # 檢查路徑中的AI關鍵詞
        path_text = ' '.join(flow.get('path', [])).lower()
        
        ai_keywords = {
            'neural': ['neural', 'network', 'nn'],
            'rag': ['rag', 'retrieval', 'vector', 'embedding'],
            'training': ['train', 'trainer', 'learning', 'model'],
            'decision': ['decision', 'orchestrator', 'planner', 'ai_capability']
        }
        
        score = 0
        for category, keywords in ai_keywords.items():
            if any(keyword in path_text for keyword in keywords):
                indicators[f'has_{category}_components'] = True
                score += 0.25
        
        indicators['ai_intensity_score'] = min(score, 1.0)
        
        return indicators
    
    def stage_2_semantic_enhancement(self) -> bool:
        """階段2: 語義增強 - 添加語義信息和上下文"""
        if self.verbose:
            print("[Stage2] 階段2: 執行語義增強...")
        
        try:
            # 載入階段1的結果
            raw_results_path = self.layer_dirs[ProcessingStage.RAW_ANALYSIS] / "raw_analysis_results.json"
            with open(raw_results_path, 'r', encoding='utf-8') as f:
                raw_analysis = json.load(f)
            
            enhanced_flows = []
            
            for flow_data in raw_analysis['flows_analyzed']:
                # 進行語義增強
                enhanced_flow = flow_data.copy()
                
                # 生成更具描述性的名稱
                enhanced_flow['semantic_name'] = self._generate_semantic_name(flow_data)
                enhanced_flow['capability_category'] = self._categorize_capability(flow_data)
                enhanced_flow['complexity_estimate'] = self._estimate_complexity(flow_data)
                enhanced_flow['interaction_pattern'] = self._detect_interaction_pattern(flow_data)
                
                # 添加關聯性分析
                enhanced_flow['related_capabilities'] = self._find_related_capabilities(
                    flow_data, raw_analysis['flows_analyzed']
                )
                
                enhanced_flows.append(enhanced_flow)
            
            enhanced_analysis = {
                'timestamp': datetime.now().isoformat(),
                'base_analysis': raw_analysis,
                'enhanced_flows': enhanced_flows,
                'semantic_categories': self._build_semantic_taxonomy(enhanced_flows),
                'capability_clusters': self._cluster_capabilities(enhanced_flows)
            }
            
            # 保存語義增強結果
            output_path = self.layer_dirs[ProcessingStage.SEMANTIC_ENHANCEMENT] / "semantic_enhancement_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_analysis, f, indent=2, ensure_ascii=False)
            
            self.save_checkpoint(
                ProcessingStage.SEMANTIC_ENHANCEMENT,
                100,
                str(output_path),
                {'enhanced_flows_count': len(enhanced_flows)}
            )
            
            if self.verbose:
                print(f"[OK] 階段2完成 - 增強了 {len(enhanced_flows)} 個流程")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[Error] 階段2失敗: {e}")
            return False
    
    def _generate_semantic_name(self, flow_data: Dict) -> str:
        """生成語義化的能力名稱"""
        start = flow_data['start_point']
        end = flow_data['end_point']
        module = flow_data['primary_module']
        
        # 基於模組和終點生成有意義的名稱
        module_mapping = {
            'cognitive_core': '認知',
            'internal_exploration': '內探',
            'task_planning': '規劃',
            'core_capabilities': '核心',
            'service_backbone': '服務'
        }
        
        module_name = module_mapping.get(module, module)
        
        # 清理終點名稱
        clean_end = end.replace('_', ' ').title()
        
        return f"{module_name}：{clean_end}"
    
    def _categorize_capability(self, flow_data: Dict) -> str:
        """將能力分類到高層次類別"""
        ai_indicators = flow_data.get('ai_capability_indicators', {})
        
        if ai_indicators.get('ai_intensity_score', 0) > 0.5:
            return "AI核心能力"
        elif ai_indicators.get('ai_intensity_score', 0) > 0.25:
            return "AI輔助能力"
        elif 'manage' in flow_data['end_point'].lower():
            return "管理能力"
        elif 'analyze' in flow_data['end_point'].lower():
            return "分析能力"
        else:
            return "基礎能力"
    
    def _estimate_complexity(self, flow_data: Dict) -> str:
        """估算能力複雜度"""
        length = flow_data['length']
        has_variants = flow_data['has_variants']
        ai_score = flow_data.get('ai_capability_indicators', {}).get('ai_intensity_score', 0)
        
        score = 0
        score += min(length / 5, 1.0) * 0.4  # 路徑長度影響
        score += (1.0 if has_variants else 0.0) * 0.3  # 變體數量影響
        score += ai_score * 0.3  # AI複雜度影響
        
        if score < 0.3:
            return "simple"
        elif score < 0.7:
            return "medium"
        else:
            return "complex"
    
    def _detect_interaction_pattern(self, flow_data: Dict) -> str:
        """檢測交互模式"""
        path = flow_data.get('path', [])
        path_text = ' '.join(path).lower()
        
        if 'user' in path_text or 'input' in path_text:
            return "interactive"
        elif 'batch' in path_text or 'background' in path_text:
            return "batch"
        elif 'api' in path_text or 'service' in path_text:
            return "service"
        else:
            return "automatic"
    
    def _find_related_capabilities(self, target_flow: Dict, all_flows: List[Dict]) -> List[str]:
        """找到相關聯的能力"""
        related = []
        target_module = target_flow['primary_module']
        target_end = target_flow['end_point']
        
        for flow in all_flows:
            if flow['flow_id'] == target_flow['flow_id']:
                continue
            
            # 同模組或相似終點的能力
            if (flow['primary_module'] == target_module or 
                flow['end_point'] == target_end):
                related.append(f"Flow_{flow['flow_id']}")
        
        return related[:5]  # 限制最多5個相關項
    
    def _build_semantic_taxonomy(self, enhanced_flows: List[Dict]) -> Dict[str, List[str]]:
        """建立語義分類法"""
        taxonomy = {}
        
        for flow in enhanced_flows:
            category = flow['capability_category']
            if category not in taxonomy:
                taxonomy[category] = []
            
            taxonomy[category].append(flow['semantic_name'])
        
        return taxonomy
    
    def _cluster_capabilities(self, enhanced_flows: List[Dict]) -> List[Dict]:
        """聚類相似能力"""
        clusters = []
        processed = set()
        
        for flow in enhanced_flows:
            if flow['flow_id'] in processed:
                continue
            
            cluster = {
                'cluster_name': flow['capability_category'],
                'primary_flow': flow['flow_id'],
                'members': [flow['flow_id']],
                'shared_characteristics': []
            }
            
            # 查找相似的能力
            for other_flow in enhanced_flows:
                if (other_flow['flow_id'] != flow['flow_id'] and 
                    other_flow['capability_category'] == flow['capability_category'] and
                    other_flow['primary_module'] == flow['primary_module']):
                    
                    cluster['members'].append(other_flow['flow_id'])
                    processed.add(other_flow['flow_id'])
            
            processed.add(flow['flow_id'])
            clusters.append(cluster)
        
        return clusters
    
    def stage_3_llm_augmentation(self) -> bool:
        """階段3: LLM 增強 - 使用大語言模型補充能力描述"""
        if self.verbose:
            print("🤖 階段3: 執行LLM增強 (模擬模式)...")
        
        try:
            # 載入階段2的結果
            semantic_results_path = self.layer_dirs[ProcessingStage.SEMANTIC_ENHANCEMENT] / "semantic_enhancement_results.json"
            with open(semantic_results_path, 'r', encoding='utf-8') as f:
                semantic_analysis = json.load(f)
            
            llm_enhanced_flows = []
            
            for flow_data in semantic_analysis['enhanced_flows']:
                # 模擬LLM增強 (實際實現時可以接入真正的LLM API)
                llm_enhanced = flow_data.copy()
                
                # 生成LLM風格的描述
                llm_enhanced['llm_generated_purpose'] = self._simulate_llm_purpose(flow_data)
                llm_enhanced['llm_usage_scenarios'] = self._simulate_llm_scenarios(flow_data)
                llm_enhanced['llm_risk_assessment'] = self._simulate_llm_risk_assessment(flow_data)
                llm_enhanced['llm_example_commands'] = self._simulate_llm_examples(flow_data)
                
                llm_enhanced_flows.append(llm_enhanced)
            
            llm_analysis = {
                'timestamp': datetime.now().isoformat(),
                'llm_model_info': 'Simulated LLM v1.0 (實際可替換為真實LLM)',
                'base_analysis': semantic_analysis,
                'llm_enhanced_flows': llm_enhanced_flows,
                'llm_insights': self._generate_llm_insights(llm_enhanced_flows)
            }
            
            # 保存LLM增強結果
            output_path = self.layer_dirs[ProcessingStage.LLM_AUGMENTATION] / "llm_augmentation_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(llm_analysis, f, indent=2, ensure_ascii=False)
            
            self.save_checkpoint(
                ProcessingStage.LLM_AUGMENTATION,
                100,
                str(output_path),
                {'llm_enhanced_flows_count': len(llm_enhanced_flows)}
            )
            
            if self.verbose:
                print(f"[OK] 階段3完成 - LLM增強了 {len(llm_enhanced_flows)} 個流程")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[Error] 階段3失敗: {e}")
            return False
    
    def _simulate_llm_purpose(self, flow_data: Dict) -> str:
        """模擬LLM生成的用途描述"""
        semantic_name = flow_data['semantic_name']
        module = flow_data['primary_module']
        complexity = flow_data['complexity_estimate']
        
        purpose_templates = {
            'simple': f"提供{semantic_name}的基礎功能，適合快速執行簡單任務",
            'medium': f"執行{semantic_name}相關的中等複雜度操作，包含多步驟處理流程", 
            'complex': f"進行{semantic_name}的高級處理，涉及複雜的邏輯判斷和多系統協調"
        }
        
        return purpose_templates.get(complexity, f"提供{semantic_name}功能")
    
    def _simulate_llm_scenarios(self, flow_data: Dict) -> List[str]:
        """模擬LLM生成的使用場景"""
        category = flow_data['capability_category']
        interaction = flow_data['interaction_pattern']
        
        scenario_map = {
            "AI核心能力": [
                "需要智能決策的場景",
                "機器學習模型訓練階段",
                "複雜問題的自動化解決"
            ],
            "AI輔助能力": [
                "人工智能輔助分析",
                "半自動化處理流程",
                "智能推薦系統"
            ],
            "管理能力": [
                "系統資源管理",
                "配置和監控任務",
                "維護和優化作業"
            ],
            "分析能力": [
                "數據分析和報告",
                "性能監控和診斷",
                "趨勢預測和洞察"
            ],
            "基礎能力": [
                "日常操作任務",
                "基礎數據處理",
                "標準化流程執行"
            ]
        }
        
        base_scenarios = scenario_map.get(category, ["一般用途場景"])
        
        # 根據交互模式調整
        if interaction == "interactive":
            base_scenarios.append("需要用戶交互的場景")
        elif interaction == "batch":
            base_scenarios.append("批量處理任務")
        
        return base_scenarios
    
    def _simulate_llm_risk_assessment(self, flow_data: Dict) -> Dict[str, str]:
        """模擬LLM生成的風險評估"""
        complexity = flow_data['complexity_estimate']
        ai_score = flow_data.get('ai_capability_indicators', {}).get('ai_intensity_score', 0)
        
        risk_level = "low"
        if complexity == "complex" or ai_score > 0.7:
            risk_level = "high"
        elif complexity == "medium" or ai_score > 0.3:
            risk_level = "medium"
        
        risk_factors = []
        if ai_score > 0.5:
            risk_factors.append("AI決策不確定性")
        if flow_data['has_variants']:
            risk_factors.append("多路徑執行風險")
        if complexity == "complex":
            risk_factors.append("高複雜度操作")
        
        return {
            "level": risk_level,
            "factors": risk_factors,
            "mitigation": f"建議在{risk_level}風險等級下謹慎執行，並準備回滾機制"
        }
    
    def _simulate_llm_examples(self, flow_data: Dict) -> List[str]:
        """模擬LLM生成的範例命令"""
        flow_id = flow_data['flow_id']
        semantic_name = flow_data['semantic_name']
        
        examples = [
            f"aiva_internal_executor.py --flow {flow_id}",
            f"aiva_internal_executor.py --flow {flow_id} --dry-run",
        ]
        
        if flow_data['interaction_pattern'] == "interactive":
            examples.append(f"aiva_internal_executor.py --flow {flow_id} --interactive")
        
        return examples
    
    def _generate_llm_insights(self, llm_enhanced_flows: List[Dict]) -> Dict[str, Any]:
        """生成LLM整體洞察"""
        total_flows = len(llm_enhanced_flows)
        high_risk_count = sum(1 for flow in llm_enhanced_flows 
                             if flow['llm_risk_assessment']['level'] == 'high')
        
        insights = {
            "summary": f"分析了{total_flows}個能力，其中{high_risk_count}個為高風險",
            "recommendations": [
                "建議對高風險能力添加額外的監控機制",
                "可考慮為複雜能力增加中間檢查點",
                "建議定期回顧和更新能力描述"
            ],
            "patterns_detected": [
                f"識別出{len(set(flow['capability_category'] for flow in llm_enhanced_flows))}種能力類別",
                f"發現{len(set(flow['interaction_pattern'] for flow in llm_enhanced_flows))}種不同的交互模式"
            ]
        }
        
        return insights
    
    def stage_4_standardization(self) -> bool:
        """階段4: 標準化 - 按用戶需求的分類優先級生成格式"""
        if self.verbose:
            print("📋 階段4: 執行標準化（按模組優先級分組）...")
        
        try:
            # 載入階段3的結果
            llm_results_path = self.layer_dirs[ProcessingStage.LLM_AUGMENTATION] / "llm_augmentation_results.json"
            with open(llm_results_path, 'r', encoding='utf-8') as f:
                llm_analysis = json.load(f)
            
            # 按用戶需求的優先級進行分組
            module_grouped_capabilities = self._group_by_module_priority(llm_analysis['llm_enhanced_flows'])
            
            # 生成AI友善的描述格式
            ai_readable_capabilities = {}
            
            for module_name, module_data in module_grouped_capabilities.items():
                ai_readable_capabilities[module_name] = {
                    'module_info': module_data['module_info'],
                    'capability_groups': {}
                }
                
                for group_key, group_flows in module_data['capability_groups'].items():
                    # 創建能力組的AI描述
                    group_description = self._create_ai_readable_group_description(group_key, group_flows)
                    
                    ai_readable_capabilities[module_name]['capability_groups'][group_key] = {
                        'ai_description': group_description,
                        'capabilities': []
                    }
                    
                    # 為每個流程創建標準化描述
                    for flow_data in group_flows:
                        capability = self._create_standardized_capability(flow_data)
                        ai_readable_capabilities[module_name]['capability_groups'][group_key]['capabilities'].append(capability)
            
            # 創建完整的標準化輸出
            standardized_output = {
                'metadata': {
                    'schema_version': '3.0.0',
                    'generated_at': datetime.now().isoformat(),
                    'processing_pipeline': 'enhanced_classifier_processor_v1.0',
                    'classification_priority': ['module', 'endpoint_similarity', 'internal_external', 'ai_capability'],
                    'total_modules': len(ai_readable_capabilities),
                    'total_capabilities': sum(
                        len(group['capabilities']) 
                        for module in ai_readable_capabilities.values()
                        for group in module['capability_groups'].values()
                    )
                },
                'module_capabilities': ai_readable_capabilities,
                'ai_usage_guide': self._generate_ai_usage_guide(ai_readable_capabilities),
                'cli_integration': self._generate_enhanced_cli_integration(ai_readable_capabilities)
            }
            
            # 保存標準化結果
            output_path = self.layer_dirs[ProcessingStage.STANDARDIZATION] / "module_grouped_capabilities.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(standardized_output, f, indent=2, ensure_ascii=False)
            
            # 生成AI友善的Markdown文檔
            self._generate_ai_friendly_documentation(standardized_output)
            
            self.save_checkpoint(
                ProcessingStage.STANDARDIZATION,
                100,
                str(output_path),
                {'modules_processed': len(ai_readable_capabilities)}
            )
            
            if self.verbose:
                print(f"[OK] 階段4完成 - 按模組分組了 {len(ai_readable_capabilities)} 個模組的能力")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[Error] 階段4失敗: {e}")
            return False
    
    def _group_by_module_priority(self, enhanced_flows: List[Dict]) -> Dict[str, Dict]:
        """按用戶需求的優先級分組：1.模組 2.相同起終點 3.內外部"""
        module_groups = {}
        
        if self.verbose:
            print(f"[Group] 開始模組分組，共有 {len(enhanced_flows)} 個流程")
        
        # 模組映射
        module_names = {
            'cognitive_core': '認知核心模組',
            'internal_exploration': '內探模組',
            'task_planning': '任務規劃模組',
            'core_capabilities': '核心能力模組',
            'service_backbone': '服務骨幹模組',
            'external_learning': '外學模組'
        }
        
        for i, flow in enumerate(enhanced_flows):
            if self.verbose and i < 5:  # 只顯示前5個用於調試
                print(f"  Flow {i}: {flow.get('primary_module', 'MISSING_MODULE')} | {flow.get('start_point', 'MISSING_START')} → {flow.get('end_point', 'MISSING_END')}")
            
            module_id = flow.get('primary_module', 'unknown')
            module_name = module_names.get(module_id, module_id)
            
            if module_name not in module_groups:
                module_groups[module_name] = {
                    'module_info': {
                        'module_id': module_id,
                        'module_name': module_name,
                        'total_capabilities': 0,
                        'internal_capabilities': 0,
                        'external_capabilities': 0
                    },
                    'capability_groups': {}
                }
            
            # 生成起終點組合鍵
            start_point = flow.get('start_point', 'unknown_start')
            end_point = flow.get('end_point', 'unknown_end')
            endpoint_key = f"{start_point} → {end_point}"
            
            if endpoint_key not in module_groups[module_name]['capability_groups']:
                module_groups[module_name]['capability_groups'][endpoint_key] = []
            
            # 標記內部/外部能力
            flow['capability_scope'] = self._classify_capability_scope(flow)
            
            module_groups[module_name]['capability_groups'][endpoint_key].append(flow)
            
            # 更新統計
            module_groups[module_name]['module_info']['total_capabilities'] += 1
            if flow['capability_scope'] == 'internal':
                module_groups[module_name]['module_info']['internal_capabilities'] += 1
            else:
                module_groups[module_name]['module_info']['external_capabilities'] += 1
        
        if self.verbose:
            print(f"  [OK] 分組完成，共 {len(module_groups)} 個模組")
            for module_name, data in module_groups.items():
                print(f"    {module_name}: {data['module_info']['total_capabilities']} 個能力")
        
        # 排序：按模組重要性，然後按能力數量
        sorted_modules = {}
        module_priority = ['認知核心模組', '內探模組', '任務規劃模組', '核心能力模組', '服務骨幹模組', '外學模組']
        
        for module_name in module_priority:
            if module_name in module_groups:
                # 對每個模組內的能力組進行排序（相同起終點的放在一起）
                sorted_groups = dict(sorted(
                    module_groups[module_name]['capability_groups'].items(),
                    key=lambda x: (len(x[1]), x[0])  # 先按能力數量，再按名稱
                ))
                module_groups[module_name]['capability_groups'] = sorted_groups
                sorted_modules[module_name] = module_groups[module_name]
        
        return sorted_modules
    
    def _classify_capability_scope(self, flow: Dict) -> str:
        """分類能力的內外部屬性"""
        # 基於AI能力指標判斷
        ai_indicators = flow.get('ai_capability_indicators', {})
        ai_score = ai_indicators.get('ai_intensity_score', 0)
        
        # 基於模組判斷
        module = flow['primary_module']
        internal_modules = ['cognitive_core', 'internal_exploration']
        
        # 基於路徑內容判斷
        path_text = ' '.join(flow.get('path', [])).lower()
        external_keywords = ['api', 'service', 'external', 'client', 'user']
        internal_keywords = ['internal', 'core', 'self', 'analyze']
        
        if module in internal_modules or any(kw in path_text for kw in internal_keywords):
            return 'internal'
        elif any(kw in path_text for kw in external_keywords):
            return 'external'
        else:
            # 根據AI強度判斷
            return 'internal' if ai_score > 0.3 else 'external'
    
    def _create_ai_readable_group_description(self, group_key: str, group_flows: List[Dict]) -> Dict[str, Any]:
        """為能力組創建AI可讀的描述"""
        start_point, end_point = group_key.split(' → ')
        
        # 分析組內的路徑變體
        path_variants = []
        complexity_levels = set()
        scopes = set()
        
        for flow in group_flows:
            path_variants.append(flow.get('path', []))
            complexity_levels.add(flow['complexity_estimate'])
            scopes.add(flow['capability_scope'])
        
        # 生成AI友善的描述
        description = {
            'group_summary': {
                'capability_name': f"{start_point}到{end_point}的處理能力",
                'total_variants': len(group_flows),
                'path_variants_count': len(path_variants),
                'complexity_range': list(complexity_levels),
                'scope_types': list(scopes)
            },
            'ai_usage_context': {
                'when_to_use': self._generate_usage_context(start_point, end_point, group_flows),
                'expected_outcome': self._generate_expected_outcome(end_point, group_flows),
                'selection_criteria': self._generate_selection_criteria(group_flows)
            },
            'path_analysis': {
                'common_steps': self._find_common_path_steps(path_variants),
                'variant_differences': self._analyze_path_differences(path_variants),
                'recommended_path': self._recommend_optimal_path(group_flows)
            }
        }
        
        return description
    
    def _generate_usage_context(self, start_point: str, end_point: str, flows: List[Dict]) -> str:
        """生成使用情境描述"""
        primary_scope = flows[0]['capability_scope']
        complexity = flows[0]['complexity_estimate']
        
        context_templates = {
            ('internal', 'simple'): f"當需要快速從{start_point}獲取{end_point}的基礎信息時",
            ('internal', 'medium'): f"當需要對{start_point}進行深度分析並生成{end_point}結果時",
            ('internal', 'complex'): f"當需要從{start_point}執行複雜的多階段處理達到{end_point}時",
            ('external', 'simple'): f"當外部系統需要簡單的{start_point}到{end_point}轉換時",
            ('external', 'medium'): f"當需要為外部用戶提供{start_point}的{end_point}服務時",
            ('external', 'complex'): f"當需要為外部系統提供完整的{start_point}到{end_point}解決方案時"
        }
        
        return context_templates.get((primary_scope, complexity), 
                                   f"用於處理{start_point}到{end_point}的轉換需求")
    
    def _generate_expected_outcome(self, end_point: str, flows: List[Dict]) -> str:
        """生成預期結果描述"""
        ai_scores = [flow.get('ai_capability_indicators', {}).get('ai_intensity_score', 0) for flow in flows]
        avg_ai_score = sum(ai_scores) / len(ai_scores) if ai_scores else 0
        
        if avg_ai_score > 0.5:
            return f"獲得經過AI處理和優化的{end_point}結果，包含智能分析和建議"
        elif avg_ai_score > 0.2:
            return f"獲得部分AI輔助的{end_point}結果，結合程式邏輯和智能處理"
        else:
            return f"獲得基於程式邏輯的{end_point}結果，確保準確性和一致性"
    
    def _generate_selection_criteria(self, flows: List[Dict]) -> List[str]:
        """生成路徑選擇建議"""
        criteria = []
        
        if len(flows) == 1:
            criteria.append("單一路徑，直接使用")
        else:
            # 多路徑選擇建議
            criteria.append("根據以下條件選擇合適的路徑：")
            
            complexities = [flow['complexity_estimate'] for flow in flows]
            if 'simple' in complexities:
                criteria.append("- 快速處理需求：選擇simple複雜度路徑")
            if 'complex' in complexities:
                criteria.append("- 完整功能需求：選擇complex複雜度路徑")
            
            scopes = [flow['capability_scope'] for flow in flows]
            if 'internal' in scopes and 'external' in scopes:
                criteria.append("- 內部處理：選擇internal範圍路徑")
                criteria.append("- 對外服務：選擇external範圍路徑")
        
        return criteria
    
    def _find_common_path_steps(self, path_variants: List[List[str]]) -> List[str]:
        """找出路徑變體的共同步驟"""
        if not path_variants:
            return []
        
        # 找出所有路徑都包含的步驟
        common_steps = set(path_variants[0])
        for path in path_variants[1:]:
            common_steps &= set(path)
        
        return list(common_steps)
    
    def _analyze_path_differences(self, path_variants: List[List[str]]) -> Dict[str, List[str]]:
        """分析路徑變體的差異"""
        if len(path_variants) <= 1:
            return {}
        
        differences = {}
        for i, path in enumerate(path_variants):
            unique_steps = set(path)
            for other_path in path_variants[:i] + path_variants[i+1:]:
                unique_steps -= set(other_path)
            
            if unique_steps:
                differences[f"路徑變體{i+1}獨有"] = list(unique_steps)
        
        return differences
    
    def _recommend_optimal_path(self, flows: List[Dict]) -> Dict[str, Any]:
        """推薦最佳路徑"""
        if len(flows) == 1:
            return {
                'flow_id': flows[0]['flow_id'],
                'reason': '唯一路徑'
            }
        
        # 基於複雜度和AI分數推薦
        scored_flows = []
        for flow in flows:
            score = 0
            
            # 複雜度分數 (適中為最佳)
            complexity = flow['complexity_estimate']
            if complexity == 'medium':
                score += 3
            elif complexity == 'simple':
                score += 2
            else:  # complex
                score += 1
            
            # AI能力分數
            ai_score = flow.get('ai_capability_indicators', {}).get('ai_intensity_score', 0)
            score += ai_score * 2
            
            scored_flows.append((score, flow))
        
        best_flow = max(scored_flows, key=lambda x: x[0])[1]
        
        return {
            'flow_id': best_flow['flow_id'],
            'reason': f"最佳平衡的{best_flow['complexity_estimate']}複雜度路徑"
        }
    
    def _create_standardized_capability(self, flow_data: Dict) -> Dict[str, Any]:
        """為單個流程創建標準化能力描述"""
        return {
            'flow_id': flow_data['flow_id'],
            'capability_name': f"flow_{flow_data['flow_id']}",
            'display_name': flow_data['semantic_name'],
            'ai_description': {
                'purpose': flow_data['llm_generated_purpose'],
                'scope': flow_data['capability_scope'],
                'complexity': flow_data['complexity_estimate'],
                'interaction_pattern': flow_data['interaction_pattern'],
                'ai_capability_level': self._map_ai_capability_level(flow_data)
            },
            'technical_details': {
                'execution_path': flow_data.get('path', []),
                'start_point': flow_data['start_point'],
                'end_point': flow_data['end_point'],
                'estimated_duration': self._estimate_duration(flow_data),
                'dependencies': self._extract_dependencies(flow_data)
            },
            'usage_guide': {
                'scenarios': flow_data['llm_usage_scenarios'],
                'example_commands': flow_data['llm_example_commands'],
                'risk_level': flow_data['llm_risk_assessment']['level'],
                'precautions': flow_data['llm_risk_assessment'].get('factors', [])
            },
            'metadata': {
                'primary_module': flow_data['primary_module'],
                'component_type': flow_data['component_type'],
                'last_updated': datetime.now().isoformat(),
                'variant_paths': flow_data.get('variant_paths', [])
            }
        }
    
    def _generate_ai_usage_guide(self, module_capabilities: Dict) -> Dict[str, Any]:
        """生成AI使用指南"""
        return {
            'selection_strategy': {
                'step1': '首先確定需要的模組範圍',
                'step2': '識別起點和終點需求',
                'step3': '根據複雜度和範圍選擇具體能力',
                'step4': '參考AI描述欄位確認符合需求'
            },
            'module_priorities': [
                '認知核心模組：AI核心功能和決策',
                '內探模組：系統自我分析和監控',
                '任務規劃模組：規劃和執行管理',
                '核心能力模組：基礎功能和工具',
                '服務骨幹模組：基礎設施和支援'
            ],
            'ai_reading_tips': [
                '優先閱讀group_summary了解能力概況',
                '查看ai_usage_context確定使用時機',
                '根據selection_criteria選擇最適合的路徑變體',
                '參考expected_outcome確認是否符合預期'
            ]
        }
    
    def _generate_enhanced_cli_integration(self, module_capabilities: Dict) -> Dict[str, Any]:
        """生成增強的CLI整合指南"""
        cli_guide = {
            'module_commands': {},
            'capability_lookup': {},
            'batch_execution_examples': []
        }
        
        for module_name, module_data in module_capabilities.items():
            module_id = module_data['module_info']['module_id']
            cli_guide['module_commands'][module_name] = {
                'list_capabilities': f"aiva_internal_executor.py --module {module_id} --list",
                'module_stats': f"aiva_internal_executor.py --module {module_id} --stats"
            }
            
            for group_key, group_data in module_data['capability_groups'].items():
                for capability in group_data['capabilities']:
                    flow_id = capability['flow_id']
                    cli_guide['capability_lookup'][group_key] = {
                        'flow_id': flow_id,
                        'command': f"aiva_internal_executor.py --flow {flow_id}",
                        'dry_run': f"aiva_internal_executor.py --flow {flow_id} --dry-run"
                    }
        
        return cli_guide
    
    def _generate_ai_friendly_documentation(self, standardized_output: Dict):
        """生成AI友善的Markdown文檔"""
        doc_path = self.layer_dirs[ProcessingStage.STANDARDIZATION] / "ai_friendly_capabilities.md"
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write("# AIVA 模組化能力文檔（AI友善版本）\n\n")
            f.write(f"生成時間: {standardized_output['metadata']['generated_at']}\n\n")
            f.write("## 文檔說明\n\n")
            f.write("此文檔按照以下優先級組織：\n")
            f.write("1. **模組分類**（最高優先級）\n")
            f.write("2. **相同起終點能力集中**\n") 
            f.write("3. **內部/外部能力區分**\n")
            f.write("4. **AI可讀描述欄位**\n\n")
            
            # 按模組寫入
            for module_name, module_data in standardized_output['module_capabilities'].items():
                f.write(f"## {module_name}\n\n")
                
                module_info = module_data['module_info']
                f.write(f"**模組統計**:\n")
                f.write(f"- 總能力數量: {module_info['total_capabilities']}\n")
                f.write(f"- 內部能力: {module_info['internal_capabilities']}\n")
                f.write(f"- 外部能力: {module_info['external_capabilities']}\n\n")
                
                # 能力組
                for group_key, group_data in module_data['capability_groups'].items():
                    f.write(f"### {group_key}\n\n")
                    
                    ai_desc = group_data['ai_description']
                    
                    # AI可讀描述區塊
                    f.write("**AI描述欄位 📋**:\n")
                    f.write(f"- **能力概要**: {ai_desc['group_summary']['capability_name']}\n")
                    f.write(f"- **使用時機**: {ai_desc['ai_usage_context']['when_to_use']}\n")
                    f.write(f"- **預期結果**: {ai_desc['ai_usage_context']['expected_outcome']}\n")
                    f.write(f"- **路徑變體**: {ai_desc['group_summary']['total_variants']} 種\n\n")
                    
                    # 選擇建議
                    if ai_desc['ai_usage_context']['selection_criteria']:
                        f.write("**路徑選擇建議**:\n")
                        for criteria in ai_desc['ai_usage_context']['selection_criteria']:
                            f.write(f"- {criteria}\n")
                        f.write("\n")
                    
                    # 具體能力列表
                    f.write("**具體能力**:\n\n")
                    for capability in group_data['capabilities']:
                        f.write(f"#### Flow {capability['flow_id']}: {capability['display_name']}\n\n")
                        
                        ai_info = capability['ai_description']
                        f.write(f"- **範圍**: {ai_info['scope']}\n")
                        f.write(f"- **複雜度**: {ai_info['complexity']}\n")
                        f.write(f"- **AI等級**: {ai_info['ai_capability_level']}\n")
                        f.write(f"- **用途**: {ai_info['purpose']}\n")
                        
                        # 使用範例
                        usage = capability['usage_guide']
                        if usage['example_commands']:
                            f.write("\n**使用命令**:\n")
                            for cmd in usage['example_commands'][:2]:  # 限制2個範例
                                f.write(f"```bash\n{cmd}\n```\n")
                        
                        f.write("\n---\n\n")
            
            # 添加AI使用指南
            f.write("## AI使用指南\n\n")
            ai_guide = standardized_output['ai_usage_guide']
            
            f.write("### 選擇策略\n\n")
            for step, desc in ai_guide['selection_strategy'].items():
                f.write(f"**{step}**: {desc}\n\n")
            
            f.write("### 模組優先級\n\n")
            for priority in ai_guide['module_priorities']:
                f.write(f"- {priority}\n")
            
            f.write("\n### AI閱讀提示\n\n")
            for tip in ai_guide['ai_reading_tips']:
                f.write(f"- {tip}\n")
        
        if self.verbose:
            print(f"[Doc] 生成AI友善文檔: {doc_path}")
    
    
    def _generate_input_description(self, flow_data: Dict) -> str:
        """生成輸入描述"""
        interaction = flow_data['interaction_pattern']
        
        if interaction == 'interactive':
            return "需要用戶提供參數或確認操作"
        elif interaction == 'batch':
            return "接受批量數據或配置文件"
        else:
            return "使用系統預設參數或環境變量"
    
    def _generate_output_description(self, flow_data: Dict) -> str:
        """生成輸出描述"""
        category = flow_data['capability_category']
        
        output_map = {
            "AI核心能力": "返回AI處理結果、模型輸出或決策建議",
            "AI輔助能力": "提供分析報告、建議或處理狀態",
            "管理能力": "返回操作狀態、配置結果或系統信息",
            "分析能力": "生成分析報告、統計數據或可視化結果",
            "基礎能力": "執行結果狀態和基本輸出信息"
        }
        
        return output_map.get(category, "執行結果和狀態信息")
    
    def _identify_side_effects(self, flow_data: Dict) -> List[str]:
        """識別副作用"""
        side_effects = []
        
        if flow_data['complexity_estimate'] == 'complex':
            side_effects.append("可能影響系統性能")
        
        ai_score = flow_data.get('ai_capability_indicators', {}).get('ai_intensity_score', 0)
        if ai_score > 0.5:
            side_effects.append("可能更新AI模型狀態")
        
        if 'manage' in flow_data['end_point'].lower():
            side_effects.append("可能修改系統配置")
        
        return side_effects or ["無明顯副作用"]
    
    def _estimate_duration(self, flow_data: Dict) -> str:
        """估算執行時間"""
        complexity = flow_data['complexity_estimate']
        length = flow_data['length']
        
        if complexity == 'complex' or length > 7:
            return "30秒-2分鐘"
        elif complexity == 'medium' or length > 3:
            return "5-30秒"
        else:
            return "1-5秒"
    
    def _extract_dependencies(self, flow_data: Dict) -> List[str]:
        """提取依賴項"""
        dependencies = []
        
        # 基於模組添加依賴
        module_deps = {
            'cognitive_core': ['AI模型', '向量數據庫'],
            'task_planning': ['任務調度器', '資源管理器'],
            'service_backbone': ['數據庫連接', 'API服務']
        }
        
        module = flow_data['primary_module']
        if module in module_deps:
            dependencies.extend(module_deps[module])
        
        # 基於AI強度添加依賴
        ai_score = flow_data.get('ai_capability_indicators', {}).get('ai_intensity_score', 0)
        if ai_score > 0.3:
            dependencies.append('AI運行環境')
        
        return dependencies or ["基礎Python環境"]
    
    def _map_ai_capability_level(self, flow_data: Dict) -> str:
        """映射AI能力等級"""
        ai_score = flow_data.get('ai_capability_indicators', {}).get('ai_intensity_score', 0)
        
        if ai_score >= 0.8:
            return "core"
        elif ai_score >= 0.5:
            return "advanced"
        elif ai_score >= 0.2:
            return "basic"
        else:
            return "none"
    
    def _calculate_quality_score(self, capabilities: Dict) -> float:
        """計算整體質量分數"""
        total_score = 0
        count = 0
        
        for cap_data in capabilities.values():
            score = 0.8  # 基礎分數
            
            # 根據描述完整性調整
            if len(cap_data.get('purpose', '')) > 20:
                score += 0.1
            if len(cap_data.get('usage_scenarios', [])) >= 3:
                score += 0.1
            
            total_score += score
            count += 1
        
        return round(total_score / count if count > 0 else 0.0, 2)
    
    def _build_capability_taxonomy(self, capabilities: Dict) -> Dict[str, Dict]:
        """建立能力分類法"""
        taxonomy = {
            'by_module': {},
            'by_complexity': {},
            'by_ai_level': {},
            'by_risk': {}
        }
        
        for cap_data in capabilities.values():
            # 按模組分類
            module = cap_data['primary_module']
            if module not in taxonomy['by_module']:
                taxonomy['by_module'][module] = []
            taxonomy['by_module'][module].append(cap_data['capability_name'])
            
            # 按複雜度分類
            complexity = cap_data['execution_complexity']
            if complexity not in taxonomy['by_complexity']:
                taxonomy['by_complexity'][complexity] = []
            taxonomy['by_complexity'][complexity].append(cap_data['capability_name'])
            
            # 按AI等級分類
            ai_level = cap_data['ai_capability_level']
            if ai_level not in taxonomy['by_ai_level']:
                taxonomy['by_ai_level'][ai_level] = []
            taxonomy['by_ai_level'][ai_level].append(cap_data['capability_name'])
            
            # 按風險等級分類
            risk = cap_data['risk_level']
            if risk not in taxonomy['by_risk']:
                taxonomy['by_risk'][risk] = []
            taxonomy['by_risk'][risk].append(cap_data['capability_name'])
        
        return taxonomy
    
    def _generate_cli_integration_guide(self, capabilities: Dict) -> Dict[str, Any]:
        """生成CLI整合指南"""
        return {
            'execution_patterns': {
                'single_flow': "aiva_internal_executor.py --flow <flow_id>",
                'dry_run': "aiva_internal_executor.py --flow <flow_id> --dry-run",
                'interactive': "aiva_internal_executor.py --flow <flow_id> --interactive",
                'list_all': "aiva_internal_executor.py --list"
            },
            'risk_recommendations': {
                'high': "建議先使用 --dry-run 預覽執行流程",
                'medium': "建議在測試環境中先驗證",
                'low': "可直接執行，建議記錄執行日誌"
            },
            'flow_count_by_risk': {
                risk: len([cap for cap in capabilities.values() 
                          if cap['risk_level'] == risk])
                for risk in ['low', 'medium', 'high']
            }
        }
    
    def _generate_markdown_documentation(self, standardized_output: Dict):
        """生成Markdown格式文檔"""
        doc_path = self.layer_dirs[ProcessingStage.STANDARDIZATION] / "capabilities_documentation.md"
        
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write("# AIVA 核心能力文檔\n\n")
            f.write(f"生成時間: {standardized_output['metadata']['generated_at']}\n\n")
            f.write(f"總能力數量: {standardized_output['metadata']['total_capabilities']}\n\n")
            f.write(f"質量分數: {standardized_output['metadata']['quality_score']}/1.0\n\n")
            
            f.write("## 能力分類統計\n\n")
            
            taxonomy = standardized_output['taxonomy']
            
            # 按模組統計
            f.write("### 按模組分布\n\n")
            for module, cap_list in taxonomy['by_module'].items():
                f.write(f"- **{module}**: {len(cap_list)} 項能力\n")
            
            f.write("\n### 按複雜度分布\n\n")
            for complexity, cap_list in taxonomy['by_complexity'].items():
                f.write(f"- **{complexity}**: {len(cap_list)} 項能力\n")
            
            f.write("\n### 按風險等級分布\n\n")
            for risk, cap_list in taxonomy['by_risk'].items():
                f.write(f"- **{risk}**: {len(cap_list)} 項能力\n")
            
            f.write("\n## 詳細能力列表\n\n")
            
            # 詳細能力描述
            for cap_data in standardized_output['capabilities'].values():
                f.write(f"### {cap_data['display_name']}\n\n")
                f.write(f"**能力ID**: {cap_data['capability_name']}\n\n")
                f.write(f"**用途**: {cap_data['purpose']}\n\n")
                f.write(f"**複雜度**: {cap_data['execution_complexity']}\n\n")
                f.write(f"**預估執行時間**: {cap_data['estimated_duration']}\n\n")
                f.write(f"**風險等級**: {cap_data['risk_level']}\n\n")
                
                if cap_data['usage_scenarios']:
                    f.write("**使用場景**:\n")
                    for scenario in cap_data['usage_scenarios']:
                        f.write(f"- {scenario}\n")
                    f.write("\n")
                
                if cap_data['example_commands']:
                    f.write("**範例命令**:\n")
                    for cmd in cap_data['example_commands']:
                        f.write(f"```bash\n{cmd}\n```\n")
                    f.write("\n")
                
                f.write("---\n\n")
        
        if self.verbose:
            print(f"[Doc] 生成Markdown文檔: {doc_path}")
    
    def stage_5_completion(self) -> bool:
        """階段5: 完成 - 最終整合和清理"""
        if self.verbose:
            print("✨ 階段5: 執行最終完成...")
        
        try:
            # 載入標準化結果
            standard_results_path = self.layer_dirs[ProcessingStage.STANDARDIZATION] / "module_grouped_capabilities.json"
            with open(standard_results_path, 'r', encoding='utf-8') as f:
                standardized_output = json.load(f)
            
            # 計算質量分數
            total_capabilities = standardized_output['metadata'].get('total_capabilities', 0)
            
            # 生成最終報告
            final_report = {
                'processing_summary': {
                    'pipeline_version': 'enhanced_classifier_processor_v1.0',
                    'completion_time': datetime.now().isoformat(),
                    'total_capabilities_processed': total_capabilities,
                    'total_modules_processed': standardized_output['metadata'].get('total_modules', 0),
                    'processing_stages_completed': 5,
                    'classification_priority': standardized_output['metadata'].get('classification_priority', [])
                },
                'output_files': {
                    'primary_data': str(standard_results_path),
                    'documentation': str(self.layer_dirs[ProcessingStage.STANDARDIZATION] / "ai_friendly_capabilities.md"),
                    'checkpoints': [str(cp.name) for cp in self.checkpoint_dir.glob("checkpoint_*.json")]
                },
                'quality_metrics': {
                    'modules_with_capabilities': len(standardized_output.get('module_capabilities', {})),
                    'avg_capabilities_per_module': total_capabilities / max(len(standardized_output.get('module_capabilities', {})), 1),
                    'processing_success_rate': 100.0 if total_capabilities > 0 else 0.0
                },
                'recommendations': [
                    "建議定期重新運行分析以保持數據更新",
                    "可考慮將結果整合到現有CLI系統中", 
                    "建議根據使用情況調整風險評級",
                    f"已成功處理 {total_capabilities} 個能力，可開始實際應用"
                ]
            }
            
            # 保存最終報告
            final_report_path = self.layer_dirs[ProcessingStage.COMPLETION] / "processing_final_report.json"
            with open(final_report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            
            # 創建成功標記文件
            success_marker = self.work_dir / "PROCESSING_COMPLETE.txt"
            with open(success_marker, 'w', encoding='utf-8') as f:
                f.write(f"Enhanced Classifier Processing Completed Successfully\n")
                f.write(f"Completion Time: {datetime.now().isoformat()}\n")
                f.write(f"Total Capabilities: {total_capabilities}\n")
                f.write(f"Total Modules: {final_report['processing_summary']['total_modules_processed']}\n")
                f.write(f"Success Rate: {final_report['quality_metrics']['processing_success_rate']}%\n")
            
            self.save_checkpoint(
                ProcessingStage.COMPLETION,
                100,
                str(final_report_path),
                {'final_report_generated': True, 'capabilities_processed': total_capabilities}
            )
            
            if self.verbose:
                print(f"[OK] 階段5完成 - 處理流程全部完成!")
                print(f"[Stats] 最終統計: 處理了 {total_capabilities} 個能力")
                print(f"🏢 模組數量: {final_report['processing_summary']['total_modules_processed']} 個")
                print(f"📁 輸出目錄: {self.work_dir}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[Error] 階段5失敗: {e}")
            return False
    
    def run_full_pipeline(self, 
                         classification_data_path: str = None,
                         flows_data: List[Dict] = None,
                         resume_from_checkpoint: bool = True) -> bool:
        """運行完整的處理管道"""
        if self.verbose:
            print("[Start] 啟動增強分類器處理管道...")
        
        # 檢查恢復點
        resume_stage = None
        if resume_from_checkpoint and self.enable_recovery:
            resume_stage = self.check_recovery_point()
        
        # 載入數據
        if not self.load_from_classifier_data(classification_data_path, flows_data):
            return False
        
        # 決定起始階段
        stages_to_run = []
        if resume_stage:
            # 從檢查點恢復
            stage_order = list(ProcessingStage)
            start_index = stage_order.index(resume_stage) + 1
            stages_to_run = stage_order[start_index:]
            if self.verbose:
                print(f"[Resume] 從階段 {resume_stage.value} 後恢復執行")
        else:
            # 全新執行
            stages_to_run = list(ProcessingStage)
        
        # 執行各階段
        stage_methods = {
            ProcessingStage.RAW_ANALYSIS: self.stage_1_raw_analysis,
            ProcessingStage.SEMANTIC_ENHANCEMENT: self.stage_2_semantic_enhancement,
            ProcessingStage.LLM_AUGMENTATION: self.stage_3_llm_augmentation,
            ProcessingStage.STANDARDIZATION: self.stage_4_standardization,
            ProcessingStage.COMPLETION: self.stage_5_completion
        }
        
        success = True
        for stage in stages_to_run:
            if self.verbose:
                print(f"\n[Executing] 執行階段: {stage.value}")
            
            stage_method = stage_methods[stage]
            if not stage_method():
                if self.verbose:
                    print(f"[Error] 階段 {stage.value} 失敗，停止執行")
                success = False
                break
            
            self.current_stage = stage
        
        if success and self.verbose:
            print(f"\n[Complete] 增強分類器處理管道執行完成!")
            print(f"📁 所有結果保存在: {self.work_dir}")
        
        return success


def main():
    """主函數 - 示範如何使用增強分類器處理器"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AIVA 增強分類器處理器')
    parser.add_argument('--input', type=str, help='輸入分類數據文件路徑')
    parser.add_argument('--output', type=str, help='輸出目錄路徑')
    parser.add_argument('--checkpoint-dir', type=str, help='檢查點目錄路徑')
    parser.add_argument('--no-recovery', action='store_true', help='禁用斷點恢復')
    parser.add_argument('--verbose', action='store_true', default=True, help='詳細輸出模式')
    
    args = parser.parse_args()
    
    # 創建處理器實例
    processor = EnhancedClassifierProcessor(
        work_dir=args.output,
        checkpoint_dir=args.checkpoint_dir,
        enable_recovery=not args.no_recovery,
        verbose=args.verbose
    )
    
    # 運行完整管道
    success = processor.run_full_pipeline(
        classification_data_path=args.input,
        resume_from_checkpoint=not args.no_recovery
    )
    
    if success:
        print("[OK] 處理完成")
        return 0
    else:
        print("[Error] 處理失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())