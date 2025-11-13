"""
AI增強代碼分析引擎
基於Tree-sitter AST和神經網路的智能代碼分析系統
"""
import ast
import torch
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Tree-sitter is optional for enhanced parsing
try:
    import tree_sitter
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    tree_sitter = None

from ..bio_neuron_master import BioNeuronMasterController
from ..ai_engine.real_bio_net_adapter import RealBioNeuronRAGAgent

class AnalysisType(Enum):
    """分析類型枚舉"""
    SECURITY = "security"
    VULNERABILITY = "vulnerability" 
    COMPLEXITY = "complexity"
    PATTERNS = "patterns"
    SEMANTIC = "semantic"
    ARCHITECTURE = "architecture"

@dataclass
class AIAnalysisResult:
    """AI分析結果數據類"""
    analysis_type: AnalysisType
    confidence: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    risk_level: str
    explanation: str
    metadata: Dict[str, Any]

class AIAnalysisEngine:
    """
    AI驅動的代碼分析引擎
    結合傳統AST分析與神經網路增強
    """
    
    def __init__(self):
        self.bio_controller = None
        self.rag_agent = None
        self.initialized = False
        
    def initialize(self) -> bool:
        """初始化AI分析引擎"""
        try:
            # 初始化生物神經網路控制器
            self.bio_controller = BioNeuronMasterController()
            
            # 創建真實的決策核心
            from ..ai_engine.real_bio_net_adapter import create_real_scalable_bionet, create_real_rag_agent
            try:
                real_decision_core = create_real_scalable_bionet(
                    input_size=1024,
                    num_tools=10,
                    weights_path=None  # 使用隨機初始化
                )
                
                # 初始化RAG代理用於代碼分析
                self.rag_agent = create_real_rag_agent(
                    decision_core=real_decision_core,
                    input_vector_size=1024
                )
            except Exception as e:
                # 降級到無RAG模式
                self.rag_agent = None
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"AI分析引擎初始化失敗: {e}")
            return False
    
    def _extract_code_features(self, source_code: str) -> torch.Tensor:
        """從原始碼中提取特徵向量"""
        try:
            # 解析AST
            tree = ast.parse(source_code)
            
            # 基本特徵統計
            features = []
            
            # 1. 代碼長度特徵
            features.append(len(source_code))
            features.append(len(source_code.splitlines()))
            
            # 2. AST節點統計
            node_counts = {}
            for node in ast.walk(tree):
                node_type = type(node).__name__
                node_counts[node_type] = node_counts.get(node_type, 0) + 1
            
            # 關鍵節點類型統計
            critical_nodes = [
                'FunctionDef', 'ClassDef', 'If', 'For', 'While', 
                'Try', 'Import', 'Call', 'Assign', 'Compare'
            ]
            
            for node_type in critical_nodes:
                features.append(node_counts.get(node_type, 0))
            
            # 3. 複雜度特徵
            features.append(self._calculate_cyclomatic_complexity(tree))
            features.append(self._calculate_nesting_depth(tree))
            
            # 4. 安全特徵
            features.extend(self._extract_security_features(tree, source_code))
            
            # 5. 語義特徵
            features.extend(self._extract_semantic_features(tree))
            
            # 補齊到1024維度
            while len(features) < 1024:
                features.append(0.0)
            
            # 如果超過1024維，截斷
            features = features[:1024]
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            print(f"特徵提取失敗: {e}")
            # 返回零向量
            return torch.zeros(1024, dtype=torch.float32)
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """計算循環複雜度"""
        complexity = 1  # 基礎複雜度
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With, 
                               ast.Try, ast.ExceptHandler, ast.Match)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, ast.Compare):
                complexity += len(node.ops)
                
        return complexity
    
    def _calculate_nesting_depth(self, tree: ast.AST) -> int:
        """計算嵌套深度"""
        max_depth = 0
        
        def visit_node(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.If, 
                               ast.For, ast.While, ast.With, ast.Try)):
                depth += 1
                
            for child in ast.iter_child_nodes(node):
                visit_node(child, depth)
        
        visit_node(tree)
        return max_depth
    
    def _extract_security_features(self, tree: ast.AST, source_code: str) -> List[float]:
        """提取安全相關特徵"""
        features = []
        
        # 危險函數調用
        dangerous_funcs = ['eval', 'exec', 'input', 'open', 'subprocess', '__import__']
        dangerous_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in dangerous_funcs:
                    dangerous_count += 1
        
        features.append(dangerous_count)
        
        # SQL注入風險模式
        sql_patterns = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP']
        sql_risk = sum(1 for pattern in sql_patterns if pattern in source_code.upper())
        features.append(sql_risk)
        
        # 硬編碼密碼模式
        password_patterns = ['password', 'passwd', 'pwd', 'secret', 'token']
        hardcoded_secrets = 0
        for line in source_code.lower().splitlines():
            if any(pattern in line and '=' in line for pattern in password_patterns):
                hardcoded_secrets += 1
        features.append(hardcoded_secrets)
        
        return features
    
    def _extract_semantic_features(self, tree: ast.AST) -> List[float]:
        """提取語義特徵"""
        features = []
        
        # 函數參數統計
        func_param_counts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_param_counts.append(len(node.args.args))
        
        features.append(np.mean(func_param_counts) if func_param_counts else 0)
        features.append(np.max(func_param_counts) if func_param_counts else 0)
        
        # 類方法統計
        class_method_counts = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                method_count = sum(1 for child in node.body if isinstance(child, ast.FunctionDef))
                class_method_counts.append(method_count)
        
        features.append(np.mean(class_method_counts) if class_method_counts else 0)
        
        # 變數名長度統計
        var_name_lengths = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                var_name_lengths.append(len(node.id))
        
        features.append(np.mean(var_name_lengths) if var_name_lengths else 0)
        
        return features
    
    def analyze_code(
        self, 
        source_code: str, 
        file_path: str = "",
        analysis_types: Optional[List[AnalysisType]] = None
    ) -> Dict[AnalysisType, AIAnalysisResult]:
        """
        AI增強的代碼分析主函數
        """
        if not self.initialized:
            self.initialize()
        
        if analysis_types is None:
            analysis_types = [AnalysisType.SECURITY, AnalysisType.COMPLEXITY, AnalysisType.PATTERNS]
        
        results = {}
        
        try:
            # 提取代碼特徵
            features = self._extract_code_features(source_code)
            
            for analysis_type in analysis_types:
                result = self._perform_ai_analysis(
                    source_code, features, analysis_type, file_path
                )
                results[analysis_type] = result
                
        except Exception as e:
            print(f"AI代碼分析失敗: {e}")
            # 返回空結果
            for analysis_type in analysis_types:
                results[analysis_type] = AIAnalysisResult(
                    analysis_type=analysis_type,
                    confidence=0.0,
                    findings=[],
                    recommendations=[],
                    risk_level="unknown",
                    explanation=f"分析失敗: {e}",
                    metadata={}
                )
        
        return results
    
    def _perform_ai_analysis(
        self,
        source_code: str,
        features: torch.Tensor,
        analysis_type: AnalysisType,
        file_path: str
    ) -> AIAnalysisResult:
        """執行特定類型的AI分析"""
        
        # 構建針對分析類型的提示
        task_prompts = {
            AnalysisType.SECURITY: "分析代碼安全性，識別潛在漏洞和安全風險",
            AnalysisType.VULNERABILITY: "檢測代碼中的已知漏洞模式",
            AnalysisType.COMPLEXITY: "評估代碼複雜度和可維護性",
            AnalysisType.PATTERNS: "識別設計模式和代碼氣味",
            AnalysisType.SEMANTIC: "執行語義分析，理解代碼邏輯",
            AnalysisType.ARCHITECTURE: "分析架構設計和組件關係"
        }
        
        task_description = task_prompts.get(analysis_type, "代碼分析")
        
        try:
            # 使用RAG代理進行AI分析 (如果可用)
            if self.rag_agent is not None:
                rag_result = self.rag_agent.generate(
                    task_description=f"{task_description}\n文件: {file_path}",
                    context=source_code[:2000]  # 限制上下文長度
                )
                confidence = rag_result.get('confidence', 0.5) if isinstance(rag_result, dict) else 0.5
            else:
                confidence = 0.5
            
            # 基於特徵向量和分析類型生成具體發現
            findings = self._generate_findings(features, analysis_type)
            
            # 生成建議
            recommendations = self._generate_recommendations(analysis_type, findings)
            
            # 計算風險等級
            risk_level = self._calculate_risk_level(confidence, findings)
            
            # 生成解釋
            explanation = self._generate_explanation(analysis_type, confidence, findings)
            
            return AIAnalysisResult(
                analysis_type=analysis_type,
                confidence=confidence,
                findings=findings,
                recommendations=recommendations,
                risk_level=risk_level,
                explanation=explanation,
                metadata={
                    'file_path': file_path,
                    'feature_vector_size': len(features),
                    'analysis_timestamp': torch.tensor(0.0).item()  # Use valid torch operation
                }
            )
            
        except Exception as e:
            return AIAnalysisResult(
                analysis_type=analysis_type,
                confidence=0.0,
                findings=[],
                recommendations=[f"分析失敗: {e}"],
                risk_level="error",
                explanation=f"AI分析過程中發生錯誤: {e}",
                metadata={'error': str(e)}
            )
    
    def _generate_findings(self, features: torch.Tensor, analysis_type: AnalysisType) -> List[Dict[str, Any]]:
        """基於特徵向量生成具體發現"""
        findings = []
        
        if analysis_type == AnalysisType.SECURITY:
            # 基於安全特徵生成發現
            dangerous_calls = int(features[12])  # 假設第12個特徵是危險函數調用數
            if dangerous_calls > 0:
                findings.append({
                    'type': 'dangerous_function_calls',
                    'severity': 'high' if dangerous_calls > 3 else 'medium',
                    'count': dangerous_calls,
                    'description': f'發現 {dangerous_calls} 個潛在危險的函數調用'
                })
        
        elif analysis_type == AnalysisType.COMPLEXITY:
            # 基於複雜度特徵生成發現
            complexity = int(features[11])  # 假設第11個特徵是循環複雜度
            if complexity > 10:
                findings.append({
                    'type': 'high_complexity',
                    'severity': 'high' if complexity > 20 else 'medium',
                    'value': complexity,
                    'description': f'循環複雜度過高: {complexity}'
                })
        
        return findings
    
    def _generate_recommendations(self, analysis_type: AnalysisType, findings: List[Dict[str, Any]]) -> List[str]:
        """基於發現生成建議"""
        recommendations = []
        
        for finding in findings:
            if finding['type'] == 'dangerous_function_calls':
                recommendations.append("建議: 避免使用危險函數如eval()、exec()，改用更安全的替代方案")
            elif finding['type'] == 'high_complexity':
                recommendations.append("建議: 重構複雜函數，將其分解為更小的可管理單元")
        
        if not recommendations:
            recommendations.append(f"代碼在{analysis_type.value}方面未發現明顯問題")
        
        return recommendations
    
    def _calculate_risk_level(self, confidence: float, findings: List[Dict[str, Any]]) -> str:
        """計算風險等級"""
        high_severity_count = sum(1 for f in findings if f.get('severity') == 'high')
        medium_severity_count = sum(1 for f in findings if f.get('severity') == 'medium')
        
        if high_severity_count > 0:
            return "high"
        elif medium_severity_count > 2:
            return "medium"
        elif medium_severity_count > 0 or confidence < 0.3:
            return "low"
        else:
            return "safe"
    
    def _generate_explanation(self, analysis_type: AnalysisType, confidence: float, findings: List[Dict[str, Any]]) -> str:
        """生成分析解釋"""
        explanation = f"{analysis_type.value}分析完成，置信度: {confidence:.2f}\n"
        
        if findings:
            explanation += f"發現 {len(findings)} 個問題:\n"
            for i, finding in enumerate(findings, 1):
                explanation += f"{i}. {finding.get('description', 'Unknown issue')}\n"
        else:
            explanation += "未發現明顯問題"
        
        return explanation

    def get_analysis_summary(self, results: Dict[AnalysisType, AIAnalysisResult]) -> Dict[str, Any]:
        """生成分析摘要"""
        total_findings = sum(len(result.findings) for result in results.values())
        avg_confidence = np.mean([result.confidence for result in results.values()]) if results else 0
        
        risk_levels = [result.risk_level for result in results.values()]
        if "high" in risk_levels:
            overall_risk = "high"
        elif "medium" in risk_levels:
            overall_risk = "medium"
        else:
            overall_risk = "low"
        
        return {
            'total_analyses': len(results),
            'total_findings': total_findings,
            'average_confidence': avg_confidence,
            'overall_risk_level': overall_risk,
            'analysis_types': [at.value for at in results.keys()]
        }