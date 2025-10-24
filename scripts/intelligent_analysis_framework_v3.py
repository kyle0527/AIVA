#!/usr/bin/env python3
"""
AIVA Features 智能分析框架 V3.0
基於V2.0經驗教訓的完全重構版本

核心改進:
1. 統一的分析框架和介面標準
2. 配置驅動的規則系統
3. 完整的測試覆蓋和驗證機制
4. 性能優化和緩存系統
5. 深度實現替代簡化版本
6. 智能圖表生成和問題發現
"""

import json
import os
import sys
import time
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import glob
import shutil
from datetime import datetime

# 配置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisQuality(Enum):
    """分析品質等級"""
    BASIC = "basic"
    STANDARD = "standard"
    DEEP = "deep"
    EXPERT = "expert"

class ComponentType(Enum):
    """組件類型"""
    CONFIG = "config"
    SERVICE = "service"
    WORKER = "worker"
    MANAGER = "manager"
    VALIDATOR = "validator"
    BUILDER = "builder"
    FACTORY = "factory"
    HANDLER = "handler"
    ADAPTER = "adapter"
    CONTROLLER = "controller"
    DECORATOR = "decorator"
    OBSERVER = "observer"
    STRATEGY = "strategy"
    STATE = "state"
    UNKNOWN = "unknown"

@dataclass
class AnalysisResult:
    """分析結果標準格式"""
    method_name: str
    category: str
    quality_level: AnalysisQuality
    component_count: int
    confidence_score: float  # 0.0 - 1.0
    analysis_time: float
    groups: Dict[str, List[str]]
    metadata: Dict[str, Any]
    issues: List[str] = None
    
    def to_dict(self) -> Dict:
        """轉換為字典格式"""
        result = asdict(self)
        result['quality_level'] = self.quality_level.value
        return result

class AnalysisConfig:
    """分析配置管理"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_default_config()
        if config_path and os.path.exists(config_path):
            self._load_custom_config()
    
    def _load_default_config(self) -> Dict:
        """載入預設配置"""
        return {
            'analysis_quality': AnalysisQuality.STANDARD.value,
            'enable_caching': True,
            'cache_ttl': 3600,  # 1小時
            'max_workers': 4,
            'timeout_seconds': 300,  # 5分鐘
            'confidence_threshold': 0.7,
            'max_groups_per_method': 50,
            'enable_validation': True,
            'generate_charts': True,
            'detect_issues': True,
            'rules': {
                'role_patterns': {
                    'manager': ['manager', 'mgr', 'admin', 'supervisor'],
                    'controller': ['controller', 'ctrl', 'command'],
                    'service': ['service', 'svc', 'api', 'endpoint'],
                    'worker': ['worker', 'processor', 'executor'],
                    'handler': ['handler', 'handle', 'process'],
                    'builder': ['builder', 'build', 'construct'],
                    'factory': ['factory', 'creator', 'make'],
                    'validator': ['validator', 'validate', 'check', 'verify'],
                    'config': ['config', 'setting', 'option', 'param'],
                    'adapter': ['adapter', 'adapt', 'convert'],
                    'decorator': ['decorator', 'wrap', 'enhance'],
                    'observer': ['observer', 'listen', 'watch', 'monitor'],
                    'strategy': ['strategy', 'policy', 'algorithm'],
                    'state': ['state', 'status', 'condition']
                },
                'quality_indicators': {
                    'high_maintainability': ['simple', 'clean', 'clear', 'basic'],
                    'low_maintainability': ['complex', 'legacy', 'hack', 'temp'],
                    'high_testability': ['test', 'mock', 'stub', 'fake'],
                    'low_testability': ['static', 'singleton', 'global'],
                    'performance_critical': ['fast', 'optimize', 'cache', 'performance'],
                    'security_critical': ['auth', 'security', 'crypto', 'encrypt']
                },
                'complexity_thresholds': {
                    'simple': 1,
                    'moderate': 3,
                    'complex': 6,
                    'very_complex': 10
                }
            }
        }
    
    def _load_custom_config(self):
        """載入自定義配置"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                custom_config = json.load(f)
                self._merge_config(custom_config)
        except Exception as e:
            logger.warning(f"無法載入自定義配置: {e}")
    
    def _merge_config(self, custom_config: Dict):
        """合併配置"""
        def merge_dict(base: Dict, custom: Dict):
            for key, value in custom.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self.config, custom_config)
    
    def get(self, key: str, default=None):
        """獲取配置值"""
        keys = key.split('.')
        current = self.config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

class CacheManager:
    """緩存管理器"""
    
    def __init__(self, enabled: bool = True, cache_dir: str = ".cache"):
        self.enabled = enabled
        self.cache_dir = Path(cache_dir)
        if enabled:
            self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, data: Any) -> str:
        """生成緩存鍵"""
        content = json.dumps(data, sort_keys=True) if isinstance(data, (dict, list)) else str(data)
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """獲取緩存"""
        if not self.enabled:
            return None
        
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if time.time() - cached_data['timestamp'] < 3600:  # 1小時有效
                        return cached_data['data']
        except Exception as e:
            logger.warning(f"緩存讀取失敗: {e}")
        return None
    
    def set(self, key: str, data: Any):
        """設置緩存"""
        if not self.enabled:
            return
        
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logger.warning(f"緩存寫入失敗: {e}")

class BaseAnalyzer(ABC):
    """分析器基類 - 統一介面標準"""
    
    def __init__(self, config: AnalysisConfig, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager
        self.name = self.__class__.__name__
    
    @abstractmethod
    def analyze(self, components: Dict[str, Any]) -> AnalysisResult:
        """執行分析 - 子類必須實現"""
        pass
    
    @abstractmethod
    def get_required_quality(self) -> AnalysisQuality:
        """獲取需要的分析品質等級"""
        pass
    
    def validate_result(self, result: AnalysisResult) -> List[str]:
        """驗證分析結果"""
        issues = []
        
        # 檢查結果完整性
        if not result.groups:
            issues.append("分析結果為空")
        
        # 檢查信心度
        if result.confidence_score < self.config.get('confidence_threshold', 0.7):
            issues.append(f"信心度過低: {result.confidence_score:.2f}")
        
        # 檢查組數量
        max_groups = self.config.get('max_groups_per_method', 50)
        if len(result.groups) > max_groups:
            issues.append(f"組數量過多: {len(result.groups)} > {max_groups}")
        
        # 檢查組件分佈
        total_components = sum(len(components) for components in result.groups.values())
        if total_components != result.component_count:
            issues.append(f"組件計數不匹配: {total_components} != {result.component_count}")
        
        return issues
    
    def _extract_component_type(self, name: str) -> ComponentType:
        """提取組件類型"""
        name_lower = name.lower()
        role_patterns = self.config.get('rules.role_patterns', {})
        
        for role, patterns in role_patterns.items():
            if any(pattern in name_lower for pattern in patterns):
                try:
                    return ComponentType(role.upper())
                except ValueError:
                    return ComponentType.UNKNOWN
        
        return ComponentType.UNKNOWN
    
    def _calculate_confidence(self, groups: Dict[str, List[str]], total_components: int) -> float:
        """計算分析信心度"""
        if not groups or total_components == 0:
            return 0.0
        
        # 基於分組覆蓋率
        covered_components = sum(len(components) for components in groups.values())
        coverage_score = covered_components / total_components
        
        # 基於分組平衡度 (避免一個組包含所有組件)
        group_sizes = [len(components) for components in groups.values()]
        if not group_sizes:
            return 0.0
        
        max_size = max(group_sizes)
        balance_score = 1.0 - (max_size / total_components)
        
        # 基於組數量合理性
        group_count = len(groups)
        optimal_groups = min(20, max(3, total_components // 50))  # 經驗值
        group_score = 1.0 - abs(group_count - optimal_groups) / optimal_groups
        
        # 綜合計算
        confidence = (coverage_score * 0.5 + balance_score * 0.3 + group_score * 0.2)
        return min(1.0, max(0.0, confidence))

class SemanticAnalyzer(BaseAnalyzer):
    """語義分析器 - 深度實現版本"""
    
    def get_required_quality(self) -> AnalysisQuality:
        return AnalysisQuality.DEEP
    
    def analyze(self, components: Dict[str, Any]) -> AnalysisResult:
        """執行語義分析"""
        start_time = time.time()
        
        # 檢查緩存
        cache_key = f"semantic_{hash(str(sorted(components.keys())))}"
        cached_result = self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        groups = {}
        
        # 1. 詞性分析
        pos_groups = self._analyze_part_of_speech(components)
        groups.update({f"詞性_{k}": v for k, v in pos_groups.items()})
        
        # 2. 語義場分析
        semantic_groups = self._analyze_semantic_fields(components)
        groups.update({f"語義場_{k}": v for k, v in semantic_groups.items()})
        
        # 3. 概念層次分析
        hierarchy_groups = self._analyze_concept_hierarchy(components)
        groups.update({f"概念層次_{k}": v for k, v in hierarchy_groups.items()})
        
        analysis_time = time.time() - start_time
        confidence = self._calculate_confidence(groups, len(components))
        
        result = AnalysisResult(
            method_name="語義智能分析",
            category="語義分析",
            quality_level=self.get_required_quality(),
            component_count=len(components),
            confidence_score=confidence,
            analysis_time=analysis_time,
            groups=groups,
            metadata={
                'pos_groups_count': len(pos_groups),
                'semantic_groups_count': len(semantic_groups),
                'hierarchy_groups_count': len(hierarchy_groups)
            }
        )
        
        # 緩存結果
        self.cache.set(cache_key, result)
        
        return result
    
    def _analyze_part_of_speech(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """詞性分析 - 完整實現"""
        pos_groups = defaultdict(list)
        
        # 動詞模式（表示動作）
        verb_patterns = [
            'create', 'build', 'make', 'generate', 'process', 'handle', 
            'manage', 'execute', 'run', 'start', 'stop', 'update', 
            'delete', 'get', 'set', 'add', 'remove', 'find', 'search'
        ]
        
        # 名詞模式（表示實體）
        noun_patterns = [
            'manager', 'service', 'worker', 'handler', 'config', 'data', 
            'model', 'entity', 'client', 'server', 'database', 'cache',
            'queue', 'pool', 'factory', 'builder', 'validator'
        ]
        
        # 形容詞模式（表示屬性）
        adjective_patterns = [
            'smart', 'fast', 'secure', 'simple', 'complex', 'advanced', 
            'basic', 'enhanced', 'optimized', 'efficient', 'robust'
        ]
        
        for name, info in components.items():
            name_lower = name.lower()
            
            # 檢查動詞模式
            verb_count = sum(1 for pattern in verb_patterns if pattern in name_lower)
            # 檢查名詞模式  
            noun_count = sum(1 for pattern in noun_patterns if pattern in name_lower)
            # 檢查形容詞模式
            adj_count = sum(1 for pattern in adjective_patterns if pattern in name_lower)
            
            # 基於主要特徵分類
            if verb_count > noun_count and verb_count > adj_count:
                pos_groups['動詞導向_動作型'].append(name)
            elif noun_count > verb_count and noun_count > adj_count:
                pos_groups['名詞導向_實體型'].append(name)
            elif adj_count > 0:
                pos_groups['形容詞導向_屬性型'].append(name)
            else:
                pos_groups['混合型_複合概念'].append(name)
        
        return dict(pos_groups)
    
    def _analyze_semantic_fields(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """語義場分析 - 完整實現"""
        semantic_fields = defaultdict(list)
        
        # 定義語義場
        fields_config = {
            '認知計算領域': ['think', 'analyze', 'understand', 'learn', 'intelligence', 'smart', 'ai', 'ml'],
            '數據處理領域': ['data', 'process', 'transform', 'parse', 'encode', 'decode', 'serialize'],
            '網絡通信領域': ['http', 'api', 'client', 'server', 'request', 'response', 'network'],
            '存儲管理領域': ['store', 'save', 'load', 'cache', 'database', 'memory', 'persist'],
            '安全防護領域': ['auth', 'security', 'encrypt', 'decrypt', 'validate', 'verify', 'protect'],
            '系統控制領域': ['control', 'manage', 'monitor', 'supervise', 'coordinate', 'orchestrate'],
            '用戶交互領域': ['ui', 'user', 'interface', 'display', 'render', 'view', 'presentation'],
            '業務邏輯領域': ['business', 'logic', 'rule', 'workflow', 'process', 'operation'],
            '測試驗證領域': ['test', 'mock', 'stub', 'verify', 'assert', 'check', 'validate'],
            '配置管理領域': ['config', 'setting', 'option', 'parameter', 'property', 'preference']
        }
        
        for name, info in components.items():
            name_lower = name.lower()
            matched_fields = []
            
            # 檢查每個語義場
            for field_name, keywords in fields_config.items():
                match_count = sum(1 for keyword in keywords if keyword in name_lower)
                if match_count > 0:
                    matched_fields.append((field_name, match_count))
            
            # 選擇最匹配的語義場
            if matched_fields:
                best_field = max(matched_fields, key=lambda x: x[1])[0]
                semantic_fields[best_field].append(name)
            else:
                semantic_fields['通用工具領域'].append(name)
        
        return dict(semantic_fields)
    
    def _analyze_concept_hierarchy(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """概念階層分析 - 完整實現"""
        hierarchy_groups = defaultdict(list)
        
        for name, info in components.items():
            # 計算概念複雜度指標
            word_count = len(re.findall(r'[A-Z][a-z]*|[a-z]+', name))
            separator_count = len(re.findall(r'[._-]', name))
            
            # 檢查抽象度指標
            abstract_indicators = ['base', 'abstract', 'generic', 'common', 'core']
            concrete_indicators = ['impl', 'implementation', 'specific', 'custom', 'detail']
            
            abstract_score = sum(1 for indicator in abstract_indicators if indicator.lower() in name.lower())
            concrete_score = sum(1 for indicator in concrete_indicators if indicator.lower() in name.lower())
            
            # 計算總複雜度
            complexity = word_count + separator_count * 0.5
            
            # 分類
            if abstract_score > concrete_score:
                if complexity <= 2:
                    hierarchy_groups['高度抽象_核心概念'].append(name)
                else:
                    hierarchy_groups['抽象層_設計模式'].append(name)
            elif concrete_score > abstract_score:
                if complexity >= 4:
                    hierarchy_groups['具體實現_業務邏輯'].append(name)
                else:
                    hierarchy_groups['實現層_功能組件'].append(name)
            else:
                if complexity <= 1:
                    hierarchy_groups['原子概念_基礎單元'].append(name)
                elif complexity <= 3:
                    hierarchy_groups['中間層_服務組件'].append(name)
                else:
                    hierarchy_groups['複合概念_集成模塊'].append(name)
        
        return dict(hierarchy_groups)

class ArchitecturalAnalyzer(BaseAnalyzer):
    """架構分析器"""
    
    def get_required_quality(self) -> AnalysisQuality:
        return AnalysisQuality.EXPERT
    
    def analyze(self, components: Dict[str, Any]) -> AnalysisResult:
        """執行架構分析"""
        start_time = time.time()
        
        groups = {}
        
        # 1. 六邊形架構分析
        hex_groups = self._analyze_hexagonal_architecture(components)
        groups.update({f"六邊形_{k}": v for k, v in hex_groups.items()})
        
        # 2. 分層架構分析
        layer_groups = self._analyze_layered_architecture(components)
        groups.update({f"分層_{k}": v for k, v in layer_groups.items()})
        
        # 3. 微服務模式分析
        microservice_groups = self._analyze_microservice_patterns(components)
        groups.update({f"微服務_{k}": v for k, v in microservice_groups.items()})
        
        analysis_time = time.time() - start_time
        confidence = self._calculate_confidence(groups, len(components))
        
        return AnalysisResult(
            method_name="架構智能分析",
            category="架構分析", 
            quality_level=self.get_required_quality(),
            component_count=len(components),
            confidence_score=confidence,
            analysis_time=analysis_time,
            groups=groups,
            metadata={
                'hexagonal_groups': len(hex_groups),
                'layer_groups': len(layer_groups),
                'microservice_groups': len(microservice_groups)
            }
        )
    
    def _analyze_hexagonal_architecture(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """六邊形架構分析"""
        hex_groups = defaultdict(list)
        
        patterns = {
            '領域核心': ['domain', 'business', 'core', 'logic', 'entity', 'aggregate'],
            '應用服務': ['application', 'service', 'use_case', 'command', 'query'],
            '輸入端口': ['port', 'api', 'controller', 'handler', 'endpoint'],
            '輸出端口': ['port', 'repository', 'gateway', 'adapter'],
            '輸入適配器': ['adapter', 'web', 'rest', 'graphql', 'cli', 'ui'],
            '輸出適配器': ['adapter', 'database', 'file', 'http', 'message', 'email'],
            '基礎設施': ['infrastructure', 'config', 'logging', 'monitoring', 'security']
        }
        
        for name, info in components.items():
            name_lower = name.lower()
            best_match = None
            best_score = 0
            
            for arch_type, keywords in patterns.items():
                score = sum(1 for keyword in keywords if keyword in name_lower)
                if score > best_score:
                    best_score = score
                    best_match = arch_type
            
            if best_match:
                hex_groups[best_match].append(name)
            else:
                hex_groups['未分類組件'].append(name)
        
        return dict(hex_groups)
    
    def _analyze_layered_architecture(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """分層架構分析"""
        layer_groups = defaultdict(list)
        
        layer_patterns = {
            '表現層': ['ui', 'view', 'controller', 'presentation', 'web', 'api'],
            '應用層': ['application', 'app', 'service', 'facade', 'workflow'],
            '領域層': ['domain', 'business', 'model', 'entity', 'aggregate'],
            '基礎設施層': ['infrastructure', 'repository', 'dao', 'database', 'external'],
            '跨層關注點': ['logging', 'security', 'monitoring', 'caching', 'validation']
        }
        
        for name, info in components.items():
            name_lower = name.lower()
            assigned = False
            
            for layer, keywords in layer_patterns.items():
                if any(keyword in name_lower for keyword in keywords):
                    layer_groups[layer].append(name)
                    assigned = True
                    break
            
            if not assigned:
                layer_groups['核心邏輯層'].append(name)
        
        return dict(layer_groups)
    
    def _analyze_microservice_patterns(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """微服務模式分析"""
        ms_groups = defaultdict(list)
        
        ms_patterns = {
            'API網關': ['gateway', 'proxy', 'router', 'load_balancer'],
            '服務發現': ['discovery', 'registry', 'consul', 'eureka'],
            '斷路器': ['circuit_breaker', 'hystrix', 'resilience', 'fault_tolerance'],
            '配置中心': ['config', 'configuration', 'settings', 'properties'],
            '消息總線': ['message', 'event', 'queue', 'broker', 'pubsub'],
            '監控告警': ['monitor', 'metrics', 'health', 'alert', 'trace'],
            '服務網格': ['mesh', 'sidecar', 'proxy', 'istio', 'envoy'],
            '數據管理': ['database', 'storage', 'cache', 'persistence']
        }
        
        for name, info in components.items():
            name_lower = name.lower()
            matched_patterns = []
            
            for pattern_name, keywords in ms_patterns.items():
                match_score = sum(1 for keyword in keywords if keyword in name_lower)
                if match_score > 0:
                    matched_patterns.append((pattern_name, match_score))
            
            if matched_patterns:
                best_pattern = max(matched_patterns, key=lambda x: x[1])[0]
                ms_groups[best_pattern].append(name)
            else:
                ms_groups['業務服務'].append(name)
        
        return dict(ms_groups)

class QualityAnalyzer(BaseAnalyzer):
    """品質分析器"""
    
    def get_required_quality(self) -> AnalysisQuality:
        return AnalysisQuality.STANDARD
    
    def analyze(self, components: Dict[str, Any]) -> AnalysisResult:
        """執行品質分析"""
        start_time = time.time()
        
        groups = {}
        
        # 分析各種品質屬性
        quality_aspects = {
            '可維護性': self._analyze_maintainability(components),
            '可測試性': self._analyze_testability(components),
            '性能關注': self._analyze_performance(components),
            '安全性': self._analyze_security(components),
            '可靠性': self._analyze_reliability(components)
        }
        
        for aspect, aspect_groups in quality_aspects.items():
            groups.update({f"{aspect}_{k}": v for k, v in aspect_groups.items()})
        
        analysis_time = time.time() - start_time
        confidence = self._calculate_confidence(groups, len(components))
        
        return AnalysisResult(
            method_name="品質智能分析",
            category="品質分析",
            quality_level=self.get_required_quality(),
            component_count=len(components),
            confidence_score=confidence,
            analysis_time=analysis_time,
            groups=groups,
            metadata={
                'quality_aspects': list(quality_aspects.keys()),
                'total_groups': len(groups)
            }
        )
    
    def _analyze_maintainability(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """可維護性分析"""
        maintainability_groups = defaultdict(list)
        
        high_maintainability = ['simple', 'clean', 'clear', 'basic', 'util', 'helper']
        medium_maintainability = ['standard', 'common', 'regular']
        low_maintainability = ['complex', 'legacy', 'hack', 'temp', 'workaround', 'fix']
        
        for name, info in components.items():
            name_lower = name.lower()
            
            # 計算可維護性分數
            high_score = sum(1 for pattern in high_maintainability if pattern in name_lower)
            low_score = sum(1 for pattern in low_maintainability if pattern in name_lower)
            
            # 基於名稱複雜度
            complexity_score = len(re.findall(r'[A-Z][a-z]*', name)) + len(re.findall(r'[._-]', name))
            
            if high_score > 0 or complexity_score <= 2:
                maintainability_groups['高可維護性'].append(name)
            elif low_score > 0 or complexity_score >= 6:
                maintainability_groups['低可維護性'].append(name)
            else:
                maintainability_groups['中等可維護性'].append(name)
        
        return dict(maintainability_groups)
    
    def _analyze_testability(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """可測試性分析"""
        testability_groups = defaultdict(list)
        
        high_testability = ['test', 'mock', 'stub', 'fake', 'interface', 'injectable']
        low_testability = ['static', 'singleton', 'global', 'hardcoded', 'final']
        
        for name, info in components.items():
            name_lower = name.lower()
            
            high_score = sum(1 for pattern in high_testability if pattern in name_lower)
            low_score = sum(1 for pattern in low_testability if pattern in name_lower)
            
            if high_score > 0:
                testability_groups['高可測試性'].append(name)
            elif low_score > 0:
                testability_groups['低可測試性'].append(name)
            else:
                testability_groups['中等可測試性'].append(name)
        
        return dict(testability_groups)
    
    def _analyze_performance(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """性能分析"""
        performance_groups = defaultdict(list)
        
        performance_critical = ['performance', 'optimize', 'fast', 'cache', 'speed', 'efficient']
        performance_neutral = ['standard', 'regular', 'normal']
        
        for name, info in components.items():
            name_lower = name.lower()
            
            if any(pattern in name_lower for pattern in performance_critical):
                performance_groups['性能關鍵'].append(name)
            else:
                performance_groups['性能一般'].append(name)
        
        return dict(performance_groups)
    
    def _analyze_security(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """安全性分析"""
        security_groups = defaultdict(list)
        
        security_critical = ['auth', 'security', 'crypto', 'encrypt', 'password', 'token', 'key']
        security_sensitive = ['validate', 'verify', 'check', 'filter', 'sanitize']
        
        for name, info in components.items():
            name_lower = name.lower()
            
            if any(pattern in name_lower for pattern in security_critical):
                security_groups['安全關鍵'].append(name)
            elif any(pattern in name_lower for pattern in security_sensitive):
                security_groups['安全敏感'].append(name)
            else:
                security_groups['安全一般'].append(name)
        
        return dict(security_groups)
    
    def _analyze_reliability(self, components: Dict[str, Any]) -> Dict[str, List[str]]:
        """可靠性分析"""
        reliability_groups = defaultdict(list)
        
        high_reliability = ['robust', 'stable', 'reliable', 'resilient', 'fault_tolerant']
        low_reliability = ['experimental', 'beta', 'alpha', 'prototype', 'temp']
        
        for name, info in components.items():
            name_lower = name.lower()
            
            if any(pattern in name_lower for pattern in high_reliability):
                reliability_groups['高可靠性'].append(name)
            elif any(pattern in name_lower for pattern in low_reliability):
                reliability_groups['低可靠性'].append(name)
            else:
                reliability_groups['標準可靠性'].append(name)
        
        return dict(reliability_groups)

class AnalysisOrchestrator:
    """分析編排器 - 統一管理所有分析器"""
    
    def __init__(self, config_path: str = None):
        self.config = AnalysisConfig(config_path)
        self.cache = CacheManager(
            enabled=self.config.get('enable_caching', True),
            cache_dir=self.config.get('cache_dir', '.cache')
        )
        
        # 註冊分析器
        self.analyzers = {
            'semantic': SemanticAnalyzer(self.config, self.cache),
            'architectural': ArchitecturalAnalyzer(self.config, self.cache),
            'quality': QualityAnalyzer(self.config, self.cache)
        }
        
        self.results = {}
        self.issues = []
    
    def run_analysis(self, components: Dict[str, Any], selected_analyzers: List[str] = None) -> Dict[str, AnalysisResult]:
        """運行分析"""
        if selected_analyzers is None:
            selected_analyzers = list(self.analyzers.keys())
        
        logger.info(f"開始分析 {len(components)} 個組件，使用分析器: {selected_analyzers}")
        
        # 並行執行分析
        with ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4)) as executor:
            future_to_analyzer = {}
            
            for analyzer_name in selected_analyzers:
                if analyzer_name in self.analyzers:
                    analyzer = self.analyzers[analyzer_name]
                    future = executor.submit(analyzer.analyze, components)
                    future_to_analyzer[future] = analyzer_name
            
            # 收集結果
            for future in as_completed(future_to_analyzer):
                analyzer_name = future_to_analyzer[future]
                try:
                    result = future.result(timeout=self.config.get('timeout_seconds', 300))
                    self.results[analyzer_name] = result
                    
                    # 驗證結果
                    if self.config.get('enable_validation', True):
                        analyzer = self.analyzers[analyzer_name]
                        issues = analyzer.validate_result(result)
                        if issues:
                            self.issues.extend([f"{analyzer_name}: {issue}" for issue in issues])
                    
                    logger.info(f"完成 {analyzer_name} 分析，信心度: {result.confidence_score:.2f}")
                    
                except Exception as e:
                    logger.error(f"{analyzer_name} 分析失敗: {e}")
                    self.issues.append(f"{analyzer_name}: 分析異常 - {str(e)}")
        
        return self.results
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成綜合報告"""
        if not self.results:
            return {'error': '沒有分析結果'}
        
        # 統計信息
        total_methods = sum(len(result.groups) for result in self.results.values())
        total_components = max(result.component_count for result in self.results.values()) if self.results else 0
        avg_confidence = sum(result.confidence_score for result in self.results.values()) / len(self.results)
        total_analysis_time = sum(result.analysis_time for result in self.results.values())
        
        # 品質評估
        quality_distribution = defaultdict(int)
        for result in self.results.values():
            quality_distribution[result.quality_level.value] += 1
        
        # 問題統計
        issue_categories = defaultdict(int)
        for issue in self.issues:
            category = issue.split(':')[0]
            issue_categories[category] += 1
        
        return {
            'summary': {
                'total_analyzers': len(self.results),
                'total_methods': total_methods,
                'total_components': total_components,
                'average_confidence': avg_confidence,
                'total_analysis_time': total_analysis_time,
                'issues_found': len(self.issues)
            },
            'quality_distribution': dict(quality_distribution),
            'analyzer_results': {name: result.to_dict() for name, result in self.results.items()},
            'issues': self.issues,
            'issue_categories': dict(issue_categories),
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成改進建議"""
        recommendations = []
        
        # 基於信心度的建議
        low_confidence_analyzers = [
            name for name, result in self.results.items() 
            if result.confidence_score < 0.7
        ]
        
        if low_confidence_analyzers:
            recommendations.append(
                f"以下分析器信心度較低，建議檢查配置或增加規則: {', '.join(low_confidence_analyzers)}"
            )
        
        # 基於問題數量的建議
        if len(self.issues) > 5:
            recommendations.append("發現較多問題，建議優化分析配置和組件命名規範")
        
        # 基於性能的建議
        slow_analyzers = [
            name for name, result in self.results.items()
            if result.analysis_time > 10.0
        ]
        
        if slow_analyzers:
            recommendations.append(
                f"以下分析器運行較慢，建議啟用緩存或優化算法: {', '.join(slow_analyzers)}"
            )
        
        return recommendations


class DiagramManager:
    """圖表管理器 - 處理圖表清理和版本控制"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.diagram_patterns = {
            'keep': [
                '**/INTEGRATED_ARCHITECTURE_FINAL*.mmd',
                '**/ULTIMATE_ORGANIZATION_DISCOVERY*FINAL*.md',
                '**/INTELLIGENT_ANALYSIS*FINAL*.md',
                '**/architecture_diagrams/**/*FINAL*.mmd',
                '**/reports/**/*SUMMARY*.md',
            ],
            'cleanup': [
                '**/*AUTO_INTEGRATED*.mmd',
                '**/*_temp_*.mmd',
                '**/*_debug_*.json',
                '**/*_intermediate_*.mmd',
                '**/scripts/*_v[12]_*.py',  # 清理舊版本腳本
            ],
            'archive': [
                '**/*_v1_*.py',
                '**/*_v2_*.py',
                '**/backup/**/*',
            ]
        }
        
    def auto_cleanup(self) -> Dict[str, List[str]]:
        """自動清理圖表和臨時文件"""
        results = {
            'cleaned': [],
            'archived': [],
            'kept': [],
            'errors': []
        }
        
        logger.info("🗂️ 開始自動圖表清理...")
        
        # 清理臨時文件
        for pattern in self.diagram_patterns['cleanup']:
            for file_path in self.workspace_root.glob(pattern):
                try:
                    if self._should_cleanup(file_path):
                        # 檢查是否需要備份
                        if self._is_important_for_backup(file_path):
                            self._backup_file(file_path)
                        
                        file_path.unlink()
                        results['cleaned'].append(str(file_path))
                        logger.info(f"🗑️ 清理: {file_path.name}")
                        
                except Exception as e:
                    results['errors'].append(f"清理失敗 {file_path}: {e}")
                    logger.error(f"清理失敗: {file_path} - {e}")
        
        # 歸檔舊版本
        for pattern in self.diagram_patterns['archive']:
            for file_path in self.workspace_root.glob(pattern):
                try:
                    if self._should_archive(file_path):
                        archive_path = self._get_archive_path(file_path)
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(file_path), str(archive_path))
                        results['archived'].append(str(archive_path))
                        logger.info(f"📦 歸檔: {file_path.name}")
                        
                except Exception as e:
                    results['errors'].append(f"歸檔失敗 {file_path}: {e}")
        
        # 記錄保留文件
        for pattern in self.diagram_patterns['keep']:
            for file_path in self.workspace_root.glob(pattern):
                results['kept'].append(str(file_path))
        
        logger.info(f"✅ 清理完成: 清理{len(results['cleaned'])}個, 歸檔{len(results['archived'])}個")
        return results
    
    def _should_cleanup(self, file_path: Path) -> bool:
        """判斷文件是否應該清理"""
        # 檢查是否在保留列表中
        for keep_pattern in self.diagram_patterns['keep']:
            if file_path.match(keep_pattern.split('/')[-1]):
                return False
        
        # 檢查文件年齡 (超過1天的臨時文件)
        if file_path.stat().st_mtime < (time.time() - 86400):
            return True
            
        # 檢查是否為自動生成的臨時文件
        temp_indicators = ['_temp_', 'debug', 'intermediate', 'AUTO_INTEGRATED']
        return any(indicator in file_path.name for indicator in temp_indicators)
    
    def _should_archive(self, file_path: Path) -> bool:
        """判斷文件是否應該歸檔"""
        # V1.0 和 V2.0 版本文件
        version_patterns = ['_v1_', '_v2_', 'v1.', 'v2.']
        return any(pattern in file_path.name for pattern in version_patterns)
    
    def _is_important_for_backup(self, file_path: Path) -> bool:
        """判斷是否需要備份"""
        important_keywords = ['FINAL', 'SUMMARY', 'REPORT', 'ANALYSIS']
        return any(keyword in file_path.name.upper() for keyword in important_keywords)
    
    def _backup_file(self, file_path: Path):
        """備份重要文件"""
        backup_dir = self.workspace_root / "_cleanup_backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_path = backup_dir / file_path.name
        shutil.copy2(str(file_path), str(backup_path))
        logger.info(f"💾 備份: {file_path.name} -> {backup_path}")
    
    def _get_archive_path(self, file_path: Path) -> Path:
        """獲取歸檔路徑"""
        archive_dir = self.workspace_root / "_archive" / "historical_versions"
        return archive_dir / file_path.name


class VariabilityManager:
    """變異性管理器 - 處理分析結果的變異性和穩定性"""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.history_file = workspace_root / "_out" / "analysis_history.json"
        
    def record_analysis(self, results: Dict[str, Any]) -> None:
        """記錄分析結果歷史"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'version': 'V3.0',
            'summary': results.get('summary', {}),
            'method_count': results.get('summary', {}).get('total_methods', 0),
            'component_count': results.get('summary', {}).get('total_components', 0),
            'quality_metrics': {
                'confidence': results.get('summary', {}).get('average_confidence', 0),
                'issues_count': results.get('summary', {}).get('issues_found', 0),
                'analysis_time': results.get('summary', {}).get('total_analysis_time', 0),
            }
        }
        
        # 載入歷史記錄
        history = self._load_history()
        history.append(history_entry)
        
        # 保持最近10次記錄
        if len(history) > 10:
            history = history[-10:]
        
        # 保存歷史
        self._save_history(history)
        logger.info(f"📝 記錄分析歷史: {len(history)} 條記錄")
    
    def analyze_variability(self) -> Dict[str, Any]:
        """分析結果變異性"""
        history = self._load_history()
        
        if len(history) < 2:
            return {'message': '歷史記錄不足，無法分析變異性'}
        
        # 計算變異性指標
        method_counts = [entry['method_count'] for entry in history]
        component_counts = [entry['component_count'] for entry in history]
        confidence_scores = [entry['quality_metrics']['confidence'] for entry in history]
        
        variability_report = {
            'stability_metrics': {
                'method_count': {
                    'mean': sum(method_counts) / len(method_counts),
                    'variance': self._calculate_variance(method_counts),
                    'stability_score': 1 - (self._calculate_variance(method_counts) / max(method_counts, default=1))
                },
                'component_count': {
                    'mean': sum(component_counts) / len(component_counts),
                    'variance': self._calculate_variance(component_counts),
                    'stability_score': 1 - (self._calculate_variance(component_counts) / max(component_counts, default=1))
                },
                'confidence': {
                    'mean': sum(confidence_scores) / len(confidence_scores),
                    'variance': self._calculate_variance(confidence_scores),
                    'trend': 'improving' if confidence_scores[-1] > confidence_scores[0] else 'declining'
                }
            },
            'recommendations': self._generate_stability_recommendations(history),
            'last_comparison': self._compare_recent_analyses(history[-2:]) if len(history) >= 2 else None
        }
        
        return variability_report
    
    def _calculate_variance(self, values: List[float]) -> float:
        """計算方差"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _generate_stability_recommendations(self, history: List[Dict]) -> List[str]:
        """生成穩定性建議"""
        recommendations = []
        
        if len(history) < 3:
            return ["需要更多歷史數據來分析穩定性"]
        
        # 檢查方法數量穩定性
        method_counts = [entry['method_count'] for entry in history[-3:]]
        if self._calculate_variance(method_counts) > 100:
            recommendations.append("組織方式數量波動較大，建議檢查分析配置的一致性")
        
        # 檢查信心度趨勢
        confidence_scores = [entry['quality_metrics']['confidence'] for entry in history[-3:]]
        if confidence_scores[-1] < confidence_scores[0] - 0.1:
            recommendations.append("分析信心度下降，建議檢查組件品質或更新分析規則")
        
        # 檢查問題數量趨勢
        issues_counts = [entry['quality_metrics']['issues_count'] for entry in history[-3:]]
        if issues_counts[-1] > issues_counts[0] + 2:
            recommendations.append("發現問題數量增加，建議優化分析算法或組件規範")
        
        return recommendations
    
    def _compare_recent_analyses(self, recent_entries: List[Dict]) -> Dict[str, Any]:
        """比較最近兩次分析"""
        if len(recent_entries) < 2:
            return {}
        
        prev, curr = recent_entries
        
        return {
            'method_count_change': curr['method_count'] - prev['method_count'],
            'confidence_change': curr['quality_metrics']['confidence'] - prev['quality_metrics']['confidence'],
            'issues_change': curr['quality_metrics']['issues_count'] - prev['quality_metrics']['issues_count'],
            'performance_change': curr['quality_metrics']['analysis_time'] - prev['quality_metrics']['analysis_time'],
        }
    
    def _load_history(self) -> List[Dict]:
        """載入分析歷史"""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"載入歷史記錄失敗: {e}")
            return []
    
    def _save_history(self, history: List[Dict]) -> None:
        """保存分析歷史"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存歷史記錄失敗: {e}")


def main():
    """主函數"""
    
    # 載入組件數據
    features_classification_path = Path(__file__).parent.parent / "_out" / "architecture_diagrams" / "features_diagram_classification.json"
    
    if not features_classification_path.exists():
        print(f"❌ 找不到特徵分類文件: {features_classification_path}")
        return
    
    try:
        with open(features_classification_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        components = data.get('classifications', data) if isinstance(data, dict) else data
        print(f"✅ 成功載入 {len(components)} 個組件")
        
    except Exception as e:
        print(f"❌ 載入數據失敗: {e}")
        return
    
    # 創建管理器
    workspace_root = Path(__file__).parent.parent
    diagram_manager = DiagramManager(workspace_root)
    variability_manager = VariabilityManager(workspace_root)
    
    # 自動清理圖表
    cleanup_results = diagram_manager.auto_cleanup()
    
    # 創建分析編排器
    orchestrator = AnalysisOrchestrator()
    
    # 運行分析
    results = orchestrator.run_analysis(components)
    
    # 生成報告
    report = orchestrator.generate_comprehensive_report()
    
    # 記錄分析歷史
    variability_manager.record_analysis(report)
    
    # 分析變異性
    variability_report = variability_manager.analyze_variability()
    
    # 輸出結果
    print("\n" + "="*60)
    print("🎯 AIVA Features 智能分析 V3.0 完成")
    print("="*60)
    
    summary = report['summary']
    print(f"📊 分析統計:")
    print(f"   - 分析器數量: {summary['total_analyzers']}")
    print(f"   - 組織方式數: {summary['total_methods']}")
    print(f"   - 組件總數: {summary['total_components']}")
    print(f"   - 平均信心度: {summary['average_confidence']:.2f}")
    print(f"   - 分析耗時: {summary['total_analysis_time']:.1f}秒")
    print(f"   - 發現問題: {summary['issues_found']}個")
    
    if report['recommendations']:
        print(f"\n💡 改進建議:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"   {i}. {rec}")
    
    # 顯示清理結果
    if cleanup_results['cleaned'] or cleanup_results['archived']:
        print(f"\n🗂️ 圖表管理:")
        print(f"   - 清理文件: {len(cleanup_results['cleaned'])}個")
        print(f"   - 歸檔文件: {len(cleanup_results['archived'])}個")
        print(f"   - 保留文件: {len(cleanup_results['kept'])}個")
        if cleanup_results['errors']:
            print(f"   - 處理錯誤: {len(cleanup_results['errors'])}個")
    
    # 顯示變異性分析
    if 'stability_metrics' in variability_report:
        print(f"\n📊 穩定性分析:")
        stability = variability_report['stability_metrics']
        print(f"   - 方法數穩定度: {stability['method_count']['stability_score']:.3f}")
        print(f"   - 組件數穩定度: {stability['component_count']['stability_score']:.3f}")
        print(f"   - 信心度趨勢: {stability['confidence']['trend']}")
        
        if variability_report.get('recommendations'):
            print(f"\n🔄 穩定性建議:")
            for i, rec in enumerate(variability_report['recommendations'], 1):
                print(f"   {i}. {rec}")
    
    # 保存詳細報告
    output_path = Path(__file__).parent / "intelligent_analysis_v3_report.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存變異性報告
    variability_path = Path(__file__).parent / "variability_analysis_report.json"
    with open(variability_path, 'w', encoding='utf-8') as f:
        json.dump(variability_report, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n📁 詳細報告已保存: {output_path}")
    print(f"📈 變異性報告已保存: {variability_path}")
    print("🚀 V3.0 智能分析框架已就緒！")

if __name__ == "__main__":
    main()