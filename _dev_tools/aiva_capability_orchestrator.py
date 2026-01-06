#!/usr/bin/env python3
"""
🔗 AIVA 核心能力與5M神經網路串接器 (開發測試版本)

⚠️ 注意: 此為獨立測試版本，正式實現位於:
    services/core/aiva_core/cognitive_core/capability_orchestrator.py

此版本用途:
✅ 獨立測試和驗證能力編排邏輯
✅ 快速原型開發和實驗
✅ CLI 介面測試

生產環境請使用:
    from aiva_core.cognitive_core.capability_orchestrator import CapabilityOrchestrator

---
功能說明:
整合AIVA現有的所有核心能力模組：
- 靜態分析 (程式碼分析、漏洞檢測)
- 動態掃描 (網路掃描、漏洞掃描)
- 風險評估 (CVSS計算、威脅情報)
- 攻擊編排 (SQL注入、XSS、SSRF等)
- 情報收集 (目標探索、技術棧識別)

將所有能力的輸出統一格式化為512維特徵向量，
餵入5M神經網路進行智能決策。
"""

import asyncio
import json
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# 設定路徑
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'services' / 'core'))

logger = logging.getLogger(__name__)

class CapabilityType(str, Enum):
    """能力類型枚舉"""
    STATIC_ANALYSIS = "static_analysis"
    VULNERABILITY_SCANNING = "vulnerability_scanning" 
    NETWORK_RECONNAISSANCE = "network_reconnaissance"
    WEB_ATTACK = "web_attack"
    RISK_ASSESSMENT = "risk_assessment"
    INTELLIGENCE_GATHERING = "intelligence_gathering"
    CODE_EXPLORATION = "code_exploration"

@dataclass
class CapabilityResult:
    """能力執行結果"""
    capability_type: CapabilityType
    status: str
    confidence: float
    data: Dict[str, Any]
    features: Optional[np.ndarray] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None

class FeatureExtractor:
    """特徵提取器 - 將各種能力結果轉換為512維特徵向量"""
    
    def __init__(self):
        self.feature_dimensions = {
            'vulnerability_features': 100,    # 漏洞特徵
            'network_features': 80,           # 網路特徵
            'code_features': 90,              # 程式碼特徵
            'risk_features': 60,              # 風險特徵
            'attack_features': 70,            # 攻擊特徵
            'intel_features': 50,             # 情報特徵
            'meta_features': 62               # 元特徵（時間、信心度等）
        }
        total_dims = sum(self.feature_dimensions.values())
        assert total_dims == 512, f"特徵維度總和必須為512，當前為{total_dims}"
    
    def extract_vulnerability_features(self, vuln_data: Dict) -> np.ndarray:
        """提取漏洞特徵 (100維) - 重構後主函數，複雜度≤8"""
        features = np.zeros(100)
        
        if not vuln_data:
            return features
        
        # 按職責分離提取不同類型特徵
        self._extract_vulnerability_type_features(vuln_data, features)
        self._extract_severity_features(vuln_data, features)  
        self._extract_cvss_features(vuln_data, features)
        self._extract_exploit_difficulty_features(vuln_data, features)
        self._extract_impact_scope_features(vuln_data, features)
        self._extract_detection_confidence_features(vuln_data, features)
        
        return features
    
    def _extract_vulnerability_type_features(self, vuln_data: Dict, features: np.ndarray) -> None:
        """提取漏洞類型特徵 (20維) - 複雜度≤5"""
        vuln_types = ['sqli', 'xss', 'ssrf', 'idor', 'bola', 'info_leak', 
                     'weak_auth', 'rce', 'lfi', 'rfi', 'csrf', 'xxe',
                     'deserialization', 'injection', 'broken_auth', 
                     'sensitive_exposure', 'xml_injection', 'ldap_injection',
                     'command_injection', 'path_traversal']
        
        vuln_data_str = str(vuln_data).lower()
        for i, vuln_type in enumerate(vuln_types):
            if vuln_type in vuln_data_str:
                features[i] = 1.0
    
    def _extract_severity_features(self, vuln_data: Dict, features: np.ndarray) -> None:
        """提取嚴重度特徵 (10維) - 複雜度≤8"""
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'info': 0}
        
        vulnerabilities = vuln_data.get('vulnerabilities', [])
        for vuln in vulnerabilities:
            severity = vuln.get('severity', '').lower()
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # 歸一化嚴重度計數
        max_count = max(severity_counts.values()) if any(severity_counts.values()) else 1
        for i, (sev, count) in enumerate(severity_counts.items()):
            features[20 + i] = count / max_count
            features[25 + i] = min(count / 10.0, 1.0)  # 原始計數(歸一化)
    
    def _extract_cvss_features(self, vuln_data: Dict, features: np.ndarray) -> None:
        """提取CVSS特徵 (20維) - 複雜度≤6"""
        cvss_scores = []
        
        vulnerabilities = vuln_data.get('vulnerabilities', [])
        for vuln in vulnerabilities:
            score = vuln.get('cvss_score', 0.0)
            if score > 0:
                cvss_scores.append(score)
        
        if cvss_scores:
            features[30] = np.mean(cvss_scores) / 10.0  # 平均CVSS
            features[31] = np.max(cvss_scores) / 10.0   # 最高CVSS
            features[32] = np.min(cvss_scores) / 10.0   # 最低CVSS
            features[33] = np.std(cvss_scores) / 3.0    # CVSS標準差
            features[34] = len(cvss_scores) / 50.0      # 漏洞數量
    
    def _extract_exploit_difficulty_features(self, vuln_data: Dict, features: np.ndarray) -> None:
        """提取利用難度特徵 (15維) - 複雜度≤4"""
        exploit_difficulty = vuln_data.get('exploit_difficulty', {})
        difficulties = ['trivial', 'easy', 'medium', 'hard', 'expert']
        
        for i, difficulty in enumerate(difficulties):
            features[35 + i] = exploit_difficulty.get(difficulty, 0) / 10.0
    
    def _extract_impact_scope_features(self, vuln_data: Dict, features: np.ndarray) -> None:
        """提取影響範圍特徵 (15維) - 複雜度≤3"""
        impact_scope = vuln_data.get('impact_scope', {})
        scopes = ['confidentiality', 'integrity', 'availability']
        
        for i, scope in enumerate(scopes):
            features[40 + i] = impact_scope.get(scope, 0.0)
    
    def _extract_detection_confidence_features(self, vuln_data: Dict, features: np.ndarray) -> None:
        """提取檢測置信度特徵 (20維) - 複雜度≤4"""
        detection_confidence = vuln_data.get('detection_confidence', [])
        
        if detection_confidence:
            conf_array = np.array(detection_confidence[:20])
            features[50:70] = np.pad(conf_array, (0, max(0, 20 - len(conf_array))))
    
    def extract_network_features(self, network_data: Dict) -> np.ndarray:
        """提取網路特徵 (80維)"""
        features = np.zeros(80)
        
        if not network_data:
            return features
        
        # 開放端口特徵 (20維)
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 
                       1433, 3306, 3389, 5432, 5900, 6379, 27017, 8080, 8443]
        
        open_ports = network_data.get('open_ports', [])
        for i, port in enumerate(common_ports):
            if port in open_ports:
                features[i] = 1.0
        
        # 服務特徵 (20維)
        services = network_data.get('services', {})
        service_types = ['http', 'https', 'ftp', 'ssh', 'telnet', 'smtp', 'dns',
                        'mysql', 'postgresql', 'redis', 'mongodb', 'mssql',
                        'vnc', 'rdp', 'ldap', 'snmp', 'nfs', 'smb', 'nginx', 'apache']
        
        for i, service in enumerate(service_types):
            if service in str(services).lower():
                features[20 + i] = 1.0
        
        # 網路拓撲特徵 (20維)
        # topology = network_data.get("topology", {})  # 暫時移除未使用變數
        features[40] = len(open_ports) / 65535.0 if open_ports else 0
        features[41] = network_data.get('response_time', 0) / 1000.0  # 響應時間
        features[42] = network_data.get('bandwidth', 0) / 1000000.0   # 帶寬
        features[43] = len(services) / 50.0 if services else 0        # 服務數量
        
        # TTL和指紋特徵 (20維)
        fingerprint = network_data.get('os_fingerprint', {})
        features[60] = fingerprint.get('ttl', 64) / 255.0
        features[61] = fingerprint.get('window_size', 0) / 65535.0
        
        # WAF檢測
        waf_detected = network_data.get('waf_detected', False)
        features[70] = 1.0 if waf_detected else 0.0
        
        # CDN檢測  
        cdn_detected = network_data.get('cdn_detected', False)
        features[71] = 1.0 if cdn_detected else 0.0
        
        return features
    
    def extract_code_features(self, code_data: Dict) -> np.ndarray:
        """提取程式碼特徵 (90維)"""
        features = np.zeros(90)
        
        if not code_data:
            return features
        
        # 程式語言特徵 (20維)
        languages = ['python', 'javascript', 'java', 'csharp', 'cpp', 'c',
                    'go', 'rust', 'php', 'ruby', 'swift', 'kotlin',
                    'typescript', 'scala', 'shell', 'powershell',
                    'sql', 'html', 'css', 'xml']
        
        detected_languages = code_data.get('languages', [])
        for i, lang in enumerate(languages):
            if lang in detected_languages:
                features[i] = 1.0
        
        # 複雜度特徵 (20維)
        complexity = code_data.get('complexity', {})
        features[20] = min(complexity.get('cyclomatic_complexity', 0) / 50.0, 1.0)
        features[21] = min(complexity.get('halstead_difficulty', 0) / 100.0, 1.0)
        features[22] = min(complexity.get('lines_of_code', 0) / 10000.0, 1.0)
        features[23] = min(complexity.get('maintainability_index', 0) / 100.0, 1.0)
        
        # 安全模式特徵 (25維)
        security_patterns = code_data.get('security_patterns', {})
        dangerous_patterns = ['eval', 'exec', 'system', 'shell_exec', 'passthru',
                            'deserialize', 'unserialize', 'pickle', 'yaml_load',
                            'sql_query', 'raw_input', 'input', 'os.system',
                            'subprocess.call', '__import__', 'compile', 'globals',
                            'locals', 'vars', 'getattr', 'setattr', 'hasattr',
                            'delattr', 'callable', 'isinstance']
        
        for i, pattern in enumerate(dangerous_patterns):
            if pattern in security_patterns:
                features[24 + i] = security_patterns[pattern] / 10.0
        
        # 依賴項特徵 (25維)
        dependencies = code_data.get('dependencies', [])
        risky_deps = ['requests', 'urllib', 'subprocess', 'os', 'sys',
                     'socket', 'pickle', 'yaml', 'json', 'xml',
                     'sqlite3', 'mysql', 'psycopg2', 'redis', 'mongodb']
        
        for i, dep in enumerate(risky_deps):
            if dep in dependencies:
                features[49 + i] = 1.0
        
        return features
    
    def extract_risk_features(self, risk_data: Dict) -> np.ndarray:
        """提取風險特徵 (60維)"""
        features = np.zeros(60)
        
        if not risk_data:
            return features
        
        # CVSS v3 基礎特徵 (20維)
        cvss = risk_data.get('cvss_v3', {})
        features[0] = cvss.get('attack_vector', 0.0)      # 攻擊向量
        features[1] = cvss.get('attack_complexity', 0.0)  # 攻擊複雜度
        features[2] = cvss.get('privileges_required', 0.0) # 所需權限
        features[3] = cvss.get('user_interaction', 0.0)   # 用戶交互
        features[4] = cvss.get('confidentiality_impact', 0.0) # 機密性影響
        features[5] = cvss.get('integrity_impact', 0.0)   # 完整性影響
        features[6] = cvss.get('availability_impact', 0.0) # 可用性影響
        
        # 威脅情報特徵 (20維)
        threat_intel = risk_data.get('threat_intelligence', {})
        features[20] = threat_intel.get('actively_exploited', 0.0)
        features[21] = threat_intel.get('exploit_available', 0.0)
        features[22] = threat_intel.get('in_the_wild', 0.0)
        features[23] = threat_intel.get('threat_actor_interest', 0.0)
        
        # 資產價值特徵 (20維)
        asset_value = risk_data.get('asset_value', {})
        features[40] = asset_value.get('business_criticality', 0.0)
        features[41] = asset_value.get('data_sensitivity', 0.0)
        features[42] = asset_value.get('regulatory_impact', 0.0)
        features[43] = asset_value.get('financial_impact', 0.0)
        
        return features
    
    def extract_attack_features(self, attack_data: Dict) -> np.ndarray:
        """提取攻擊特徵 (70維)"""
        features = np.zeros(70)
        
        if not attack_data:
            return features
        
        # 攻擊類型特徵 (25維)
        attack_types = ['sql_injection', 'xss', 'csrf', 'ssrf', 'xxe',
                       'deserialization', 'command_injection', 'file_inclusion',
                       'directory_traversal', 'authentication_bypass',
                       'authorization_bypass', 'business_logic_bypass',
                       'race_condition', 'timing_attack', 'bruteforce',
                       'enumeration', 'information_disclosure', 'dos',
                       'buffer_overflow', 'format_string', 'integer_overflow',
                       'privilege_escalation', 'lateral_movement', 'persistence',
                       'exfiltration']
        
        for i, attack_type in enumerate(attack_types):
            if attack_type in attack_data.get('attack_vectors', []):
                features[i] = 1.0
        
        # 攻擊成功率特徵 (15維)
        success_rates = attack_data.get('success_rates', {})
        for i, attack_type in enumerate(['sqli', 'xss', 'ssrf', 'rce', 'lfi']):
            features[25 + i] = success_rates.get(attack_type, 0.0)
        
        # 攻擊複雜度特徵 (15維)
        complexity = attack_data.get('attack_complexity', {})
        features[40] = complexity.get('skill_level_required', 0.0)
        features[41] = complexity.get('time_required', 0.0)
        features[42] = complexity.get('tools_required', 0.0)
        
        # 防禦機制特徵 (15維)
        defenses = attack_data.get('defenses_detected', {})
        features[55] = 1.0 if defenses.get('waf') else 0.0
        features[56] = 1.0 if defenses.get('rate_limiting') else 0.0
        features[57] = 1.0 if defenses.get('input_validation') else 0.0
        features[58] = 1.0 if defenses.get('csrf_protection') else 0.0
        
        return features
    
    def extract_intelligence_features(self, intel_data: Dict) -> np.ndarray:
        """提取情報特徵 (50維)"""
        features = np.zeros(50)
        
        if not intel_data:
            return features
        
        # 技術棧特徵 (20維)
        tech_stack = intel_data.get('technology_stack', {})
        technologies = ['apache', 'nginx', 'iis', 'tomcat', 'nodejs',
                       'php', 'python', 'java', 'dotnet', 'ruby',
                       'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                       'docker', 'kubernetes', 'aws', 'azure', 'gcp']
        
        for i, tech in enumerate(technologies):
            if tech in tech_stack:
                features[i] = tech_stack[tech]
        
        # 組織特徵 (15維)
        org_info = intel_data.get('organization', {})
        features[20] = org_info.get('company_size', 0.0)
        features[21] = org_info.get('industry_risk_level', 0.0)
        features[22] = org_info.get('geographic_risk', 0.0)
        features[23] = org_info.get('regulatory_environment', 0.0)
        
        # 歷史攻擊特徵 (15維)
        history = intel_data.get('attack_history', {})
        features[35] = history.get('previous_breaches', 0.0)
        features[36] = history.get('vulnerability_disclosure_history', 0.0)
        features[37] = history.get('threat_actor_targeting', 0.0)
        
        return features
    
    def extract_meta_features(self, results: List[CapabilityResult]) -> np.ndarray:
        """提取元特徵 (62維)"""
        features = np.zeros(62)
        
        # 能力執行統計 (20維)
        total_capabilities = len(results)
        successful_capabilities = sum(1 for r in results if r.status == 'success')
        
        features[0] = successful_capabilities / max(total_capabilities, 1)  # 成功率
        features[1] = total_capabilities / 20.0  # 歸一化能力數量
        
        # 平均置信度和執行時間
        confidences = [r.confidence for r in results if r.confidence > 0]
        exec_times = [r.execution_time for r in results if r.execution_time > 0]
        
        features[2] = np.mean(confidences) if confidences else 0.0
        features[3] = np.mean(exec_times) / 10.0 if exec_times else 0.0  # 歸一化到0-1
        
        # 各類型能力的執行狀態 (7*6=42維)
        capability_types = list(CapabilityType)
        for i, cap_type in enumerate(capability_types):
            type_results = [r for r in results if r.capability_type == cap_type]
            if type_results:
                # 每種能力類型6個特徵
                features[4 + i*6] = len(type_results) / 10.0  # 數量
                features[5 + i*6] = sum(1 for r in type_results if r.status == 'success') / len(type_results)  # 成功率
                features[6 + i*6] = np.mean([r.confidence for r in type_results if r.confidence > 0] or [0])  # 平均置信度
                features[7 + i*6] = np.mean([r.execution_time for r in type_results if r.execution_time > 0] or [0]) / 5.0  # 平均時間
                features[8 + i*6] = len([r for r in type_results if r.error_message]) / len(type_results)  # 錯誤率
                features[9 + i*6] = 1.0  # 該類型已執行標記
        
        return features
    
    def combine_features(self, results: List[CapabilityResult]) -> np.ndarray:
        """組合所有特徵為512維向量"""
        # 初始化各類特徵
        vuln_features = np.zeros(100)
        network_features = np.zeros(80)
        code_features = np.zeros(90)
        risk_features = np.zeros(60)
        attack_features = np.zeros(70)
        intel_features = np.zeros(50)
        
        # 從結果中提取各類特徵
        for result in results:
            if result.capability_type == CapabilityType.VULNERABILITY_SCANNING:
                vuln_features = np.maximum(vuln_features, self.extract_vulnerability_features(result.data))
            elif result.capability_type == CapabilityType.NETWORK_RECONNAISSANCE:
                network_features = np.maximum(network_features, self.extract_network_features(result.data))
            elif result.capability_type in [CapabilityType.STATIC_ANALYSIS, CapabilityType.CODE_EXPLORATION]:
                code_features = np.maximum(code_features, self.extract_code_features(result.data))
            elif result.capability_type == CapabilityType.RISK_ASSESSMENT:
                risk_features = np.maximum(risk_features, self.extract_risk_features(result.data))
            elif result.capability_type == CapabilityType.WEB_ATTACK:
                attack_features = np.maximum(attack_features, self.extract_attack_features(result.data))
            elif result.capability_type == CapabilityType.INTELLIGENCE_GATHERING:
                intel_features = np.maximum(intel_features, self.extract_intelligence_features(result.data))
        
        # 提取元特徵
        meta_features = self.extract_meta_features(results)
        
        # 組合所有特徵
        combined_features = np.concatenate([
            vuln_features,      # 100維
            network_features,   # 80維
            code_features,      # 90維
            risk_features,      # 60維
            attack_features,    # 70維
            intel_features,     # 50維
            meta_features       # 62維
        ])
        
        assert len(combined_features) == 512, f"特徵向量長度必須為512，當前為{len(combined_features)}"
        return combined_features

class AIVACapabilityOrchestrator:
    """AIVA能力編排器 - 統一管理所有核心能力"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.capabilities = {}
        self.results_cache = {}
        
        # 載入5M神經網路
        self.ai_core = None
        self._load_5m_neural_network()
    
    def _load_5m_neural_network(self):
        """載入5M特化神經網路 - 統一匯入路徑修復"""
        try:
            # 嘗試相對路徑匯入 (指南推薦方式)
            from services.core.aiva_core.ai_engine.real_neural_core import RealAICore
            
            self.ai_core = RealAICore(
                use_5m_model=True,
                weights_path="services/core/aiva_core/ai_engine/aiva_5M_weights.pth"
            )
            self.ai_core.load_weights()
            logger.info("✅ 5M特化神經網路載入成功 (相對路徑)")
            
        except ImportError:
            try:
                # 備用：動態路徑匯入
                sys.path.insert(0, str(Path(__file__).parent / 'services/core/aiva_core/ai_engine'))
                from real_neural_core import RealAICore
                
                self.ai_core = RealAICore(
                    use_5m_model=True,
                    weights_path="services/core/aiva_core/ai_engine/aiva_5M_weights.pth"
                )
                self.ai_core.load_weights()
                logger.info("✅ 5M特化神經網路載入成功 (動態路徑)")
                
            except Exception as e:
                logger.error(f"❌ 5M神經網路載入失敗: {e}")
                self.ai_core = None
    
    def register_capability(self, name: str, capability_instance):
        """註冊能力模組"""
        self.capabilities[name] = capability_instance
        logger.info(f"📝 註冊能力模組: {name}")
    
    def execute_static_analysis(self, **kwargs) -> CapabilityResult:
        """執行靜態分析"""
        target_code = kwargs.get('target_code', kwargs.get('target', ''))
        
        try:
            # 模擬靜態分析結果
            analysis_data = {
                'languages': ['python', 'javascript'],
                'complexity': {
                    'cyclomatic_complexity': 15,
                    'halstead_difficulty': 25.5,
                    'lines_of_code': 1250,
                    'maintainability_index': 75.2
                },
                'security_patterns': {
                    'eval': 2, 'exec': 1, 'sql_query': 5,
                    'subprocess.call': 3, 'os.system': 1
                },
                'dependencies': ['requests', 'urllib', 'json', 'sqlite3']
            }
            
            return CapabilityResult(
                capability_type=CapabilityType.STATIC_ANALYSIS,
                status="success",
                confidence=0.85,
                data=analysis_data,
                execution_time=0.15
            )
            
        except Exception as e:
            return CapabilityResult(
                capability_type=CapabilityType.STATIC_ANALYSIS,
                status="error",
                confidence=0.0,
                data={},
                error_message=str(e)
            )
    
    def execute_vulnerability_scanning(self, **kwargs) -> CapabilityResult:
        """執行漏洞掃描"""
        target_url = kwargs.get('target_url', kwargs.get('target', ''))
        
        try:
            # 模擬漏洞掃描結果
            vuln_data = {
                'vulnerabilities': [
                    {
                        'type': 'sqli',
                        'severity': 'high',
                        'cvss_score': 8.5,
                        'location': '/login?id=1',
                        'payload': "1' OR '1'='1"
                    },
                    {
                        'type': 'xss',
                        'severity': 'medium', 
                        'cvss_score': 6.2,
                        'location': '/search?q=<script>',
                        'payload': '<script>alert("XSS")</script>'
                    }
                ],
                'exploit_difficulty': {'easy': 2, 'medium': 1},
                'detection_confidence': [0.95, 0.87, 0.76]
            }
            
            return CapabilityResult(
                capability_type=CapabilityType.VULNERABILITY_SCANNING,
                status="success", 
                confidence=0.92,
                data=vuln_data,
                execution_time=2.5
            )
            
        except Exception as e:
            return CapabilityResult(
                capability_type=CapabilityType.VULNERABILITY_SCANNING,
                status="error",
                confidence=0.0,
                data={},
                error_message=str(e)
            )
    
    def execute_network_reconnaissance(self, **kwargs) -> CapabilityResult:
        """執行網路偵察"""
        target_host = kwargs.get('target_host', kwargs.get('target', ''))
        
        try:
            # 模擬網路偵察結果
            network_data = {
                'open_ports': [22, 80, 443, 3306, 8080],
                'services': {
                    '22': 'OpenSSH 7.4',
                    '80': 'Apache httpd 2.4.29',
                    '443': 'Apache httpd 2.4.29 (SSL)',
                    '3306': 'MySQL 5.7.24',
                    '8080': 'Tomcat 9.0.14'
                },
                'os_fingerprint': {
                    'ttl': 64,
                    'window_size': 29200,
                    'os_class': 'Linux 3.X|4.X'
                },
                'waf_detected': True,
                'cdn_detected': False,
                'response_time': 120,
                'bandwidth': 1000000
            }
            
            return CapabilityResult(
                capability_type=CapabilityType.NETWORK_RECONNAISSANCE,
                status="success",
                confidence=0.88,
                data=network_data,
                execution_time=5.2
            )
            
        except Exception as e:
            return CapabilityResult(
                capability_type=CapabilityType.NETWORK_RECONNAISSANCE,
                status="error",
                confidence=0.0,
                data={},
                error_message=str(e)
            )
    
    def execute_risk_assessment(self, **kwargs) -> CapabilityResult:
        """執行風險評估"""
        target_info = kwargs.get('target_info', kwargs.get('target', {}))
        
        try:
            # 模擬風險評估結果
            risk_data = {
                'cvss_v3': {
                    'attack_vector': 0.85,      # Network
                    'attack_complexity': 0.77,  # Low
                    'privileges_required': 0.62, # None
                    'user_interaction': 0.85,   # None
                    'confidentiality_impact': 0.56,  # High
                    'integrity_impact': 0.56,   # High
                    'availability_impact': 0.56  # High
                },
                'threat_intelligence': {
                    'actively_exploited': 0.8,
                    'exploit_available': 1.0,
                    'in_the_wild': 0.6,
                    'threat_actor_interest': 0.7
                },
                'asset_value': {
                    'business_criticality': 0.9,
                    'data_sensitivity': 0.85,
                    'regulatory_impact': 0.7,
                    'financial_impact': 0.8
                }
            }
            
            return CapabilityResult(
                capability_type=CapabilityType.RISK_ASSESSMENT,
                status="success",
                confidence=0.91,
                data=risk_data,
                execution_time=0.8
            )
            
        except Exception as e:
            return CapabilityResult(
                capability_type=CapabilityType.RISK_ASSESSMENT,
                status="error",
                confidence=0.0,
                data={},
                error_message=str(e)
            )
    
    def execute_comprehensive_analysis(self, target: str) -> Tuple[List[CapabilityResult], np.ndarray, Optional[Dict[str, Any]]]:
        """執行綜合分析 - 整合所有能力模組"""
        logger.info(f"🔍 開始對目標 {target} 進行綜合分析...")
        
        results = []
        
        # 同步執行多種能力分析
        try:
            result1 = self.execute_static_analysis(target=target)
            results.append(result1)
        except Exception as e:
            logger.error(f"靜態分析失敗: {e}")
        
        try:
            result2 = self.execute_vulnerability_scanning(target=target)
            results.append(result2)
        except Exception as e:
            logger.error(f"漏洞掃描失敗: {e}")
            
        try:
            result3 = self.execute_network_reconnaissance(target=target)
            results.append(result3)
        except Exception as e:
            logger.error(f"網路偵察失敗: {e}")
            
        try:
            result4 = self.execute_risk_assessment(target=target)
            results.append(result4)
        except Exception as e:
            logger.error(f"風險評估失敗: {e}")
        
        # 提取特徵向量
        feature_vector = self.feature_extractor.combine_features(results)
        logger.info("✅ 提取512維特徵向量完成")
        
        # 使用5M神經網路進行決策
        ai_decision = None
        if self.ai_core:
            ai_decision = self.make_ai_decision(feature_vector)
        
        return results, feature_vector, ai_decision
    
    def make_ai_decision(self, feature_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        """使用5M神經網路做決策 - 修復async問題，添加None檢查"""
        if not self.ai_core:
            logger.warning("AI核心未初始化，無法進行決策")
            return None
            
        try:
            import torch
            
            # 轉換為PyTorch張量
            input_tensor = torch.FloatTensor(feature_vector).unsqueeze(0)  # [1, 512]
            
            with torch.no_grad():
                # 獲得主輸出和輔助輸出
                main_output, aux_output = self.ai_core.forward_with_aux(input_tensor)
                
                # 決策分析
                decision_class = torch.argmax(main_output, dim=1).item()
                confidence = torch.max(torch.softmax(main_output, dim=1)).item()
                
                # 獲得前5個推薦動作
                top5_actions = torch.topk(torch.softmax(main_output, dim=1), 5, dim=1)
                top5_classes = top5_actions.indices[0].tolist()
                top5_probs = top5_actions.values[0].tolist()
                
                # 分析輔助特徵
                aux_features = aux_output[0].tolist()  # 531維特徵
                
                return {
                    'primary_decision': {
                        'class': decision_class,
                        'confidence': confidence,
                        'description': f'推薦動作類別 {decision_class}'
                    },
                    'alternative_actions': [
                        {
                            'class': cls,
                            'probability': prob,
                            'description': f'備選動作 {cls}'
                        }
                        for cls, prob in zip(top5_classes, top5_probs)
                    ],
                    'auxiliary_analysis': {
                        'feature_count': len(aux_features),
                        'max_activation': max(aux_features),
                        'min_activation': min(aux_features),
                        'mean_activation': sum(aux_features) / len(aux_features),
                        'active_features': sum(1 for f in aux_features if abs(f) > 0.01)
                    },
                    'reasoning': f'基於512維特徵分析，5M神經網路推薦執行類別{decision_class}的動作，信心度{confidence:.3f}'
                }
                
        except Exception as e:
            logger.error(f"AI決策失敗: {e}")
            return {
                'error': str(e),
                'fallback_decision': 'manual_analysis_required'
            }

def main():
    """演示AIVA能力串接"""
    def run_demo():
        """AIVA演示函數 - 移除不必要的async"""
        print("🚀 AIVA 核心能力與5M神經網路串接演示")
        print("=" * 60)
        
        # 初始化編排器
        orchestrator = AIVACapabilityOrchestrator()
        
        # 目標測試
        test_target = "https://testphp.vulnweb.com"
        
        # 執行綜合分析 - 移除await，因為函數不是async
        results, features, decision = orchestrator.execute_comprehensive_analysis(test_target)
        
        print("\n📊 分析結果總覽:")
        print(f"   - 執行能力數量: {len(results)}")
        print(f"   - 成功執行: {sum(1 for r in results if r.status == 'success')}")
        print(f"   - 特徵向量維度: {len(features)}")
        
        print("\n🧠 AI決策結果:")
        if decision and 'primary_decision' in decision:
            primary = decision['primary_decision']
            print(f"   - 主要決策: 類別 {primary['class']}")
            print(f"   - 信心度: {primary['confidence']:.3f}")
            print(f"   - 推理: {decision.get('reasoning', 'N/A')}")
            
            print("\n🎯 備選動作:")
            for i, alt in enumerate(decision['alternative_actions'][:3]):
                print(f"   {i+1}. 類別 {alt['class']} (機率: {alt['probability']:.3f})")
        
        print("\n📈 詳細分析:")
        for result in results:
            print(f"   - {result.capability_type.value}: {result.status} (信心度: {result.confidence:.3f})")
        
        print("\n🎉 AIVA核心能力串接演示完成!")
    
    # 運行演示
    run_demo()  # 直接調用，不需要asyncio.run

if __name__ == "__main__":
    main()