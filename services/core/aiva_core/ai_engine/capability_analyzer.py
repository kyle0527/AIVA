"""
AIVA Core - 能力分析器 (CapabilityAnalyzer)

負責深度分析已發現的能力，理解其功能、參數、輸出，並進行智能分類。

作者: AIVA Development Team
創建日期: 2025-11-13
版本: 1.0.0

依賴:
- aiva_core.ai_engine.ai_analysis_engine (AI驅動分析)
- aiva_core.rag.rag_engine (RAG知識檢索)
- aiva_common.enums (標準枚舉)
"""

import logging
import re
import ast
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# AIVA 核心導入
try:
    from ..ai_analysis.analysis_engine import AIAnalysisEngine, AnalysisType
    from ..rag.rag_engine import RAGEngine
    from aiva_common.enums.common import VulnerabilityRiskLevel
    from aiva_common.enums.modules import ModuleName
    from aiva_common.enums.pentest import PentestPhase, InformationGatheringMethod
except ImportError:
    # 回退到絕對導入
    from services.core.aiva_core.ai_analysis.analysis_engine import AIAnalysisEngine, AnalysisType
    from services.core.aiva_core.rag.rag_engine import RAGEngine
    from services.aiva_common.enums.common import VulnerabilityRiskLevel
    from services.aiva_common.enums.modules import ModuleName
    from services.aiva_common.enums.pentest import PentestPhase, InformationGatheringMethod

logger = logging.getLogger(__name__)

# 別名以保持向後兼容性
RiskLevel = VulnerabilityRiskLevel


class FunctionType(str, Enum):
    """功能類型枚舉"""
    SCANNING = "scanning"           # 掃描功能
    ANALYSIS = "analysis"           # 分析功能
    EXPLOITATION = "exploitation"   # 利用功能
    REPORTING = "reporting"         # 報告功能
    UTILITY = "utility"            # 工具功能
    AUTHENTICATION = "auth"        # 認證功能
    NETWORKING = "networking"      # 網絡功能
    DATA_PROCESSING = "data"       # 數據處理
    UNKNOWN = "unknown"            # 未知功能


@dataclass
class ParameterInfo:
    """參數信息"""
    name: str
    type_hint: Optional[str]
    default_value: Optional[str]
    description: Optional[str]
    is_required: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return asdict(self)


@dataclass
class CapabilityAnalysis:
    """能力分析結果"""
    capability_id: str
    function_type: PentestPhase  # 使用 aiva_common 標準枚舉
    risk_level: RiskLevel
    semantic_understanding: Dict[str, Any]
    parameters: List[ParameterInfo]
    return_type: Optional[str]
    side_effects: List[str]
    examples: List[Dict[str, str]]
    related_capabilities: List[str]
    documentation: str
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        result = asdict(self)
        result["parameters"] = [param.to_dict() for param in self.parameters]
        return result


@dataclass
class CapabilityClassification:
    """能力分類結果"""
    by_function: Dict[str, List[str]]
    by_risk: Dict[str, List[str]]
    by_module: Dict[str, List[str]]
    by_complexity: Dict[str, List[str]]
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return asdict(self)


class CapabilityAnalyzer:
    """
    能力分析器
    
    深度分析已發現的能力，提供:
    1. 語義理解 - 使用AI理解函數功能
    2. 參數分析 - 分析參數類型和用途
    3. 風險評估 - 評估安全風險等級
    4. 功能分類 - 智能分類能力
    5. 關聯分析 - 發現相關能力
    6. 文檔生成 - 自動生成使用文檔
    """
    
    # 風險關鍵字映射
    RISK_KEYWORDS = {
        RiskLevel.HIGH: [
            "exploit", "attack", "inject", "shell", "payload", "backdoor",
            "privilege", "escalation", "bypass", "crack", "brute", "force",
            "sql_injection", "xss", "rce", "lfi", "rfi", "xxe"
        ],
        RiskLevel.MEDIUM: [
            "scan", "enumerate", "probe", "discover", "fingerprint",
            "reconnaissance", "recon", "gather", "collect", "harvest"
        ],
        RiskLevel.LOW: [
            "parse", "format", "validate", "encode", "decode", "convert",
            "log", "report", "display", "print", "save", "load"
        ]
    }
    
    # 功能關鍵字映射（使用 aiva_common 標準 PentestPhase）
    FUNCTION_KEYWORDS = {
        PentestPhase.INTELLIGENCE_GATHERING: [
            "scan", "probe", "detect", "discover", "find", "search",
            "enumerate", "map", "crawl", "spider", "gather", "collect"
        ],
        PentestPhase.VULNERABILITY_ANALYSIS: [
            "analyze", "parse", "examine", "inspect", "evaluate",
            "assess", "review", "check", "validate", "test", "verify"
        ],
        PentestPhase.EXPLOITATION: [
            "exploit", "attack", "inject", "execute", "trigger",
            "abuse", "bypass", "escalate", "compromise", "penetrate"
        ],
        PentestPhase.REPORTING: [
            "report", "generate", "create", "format", "export",
            "display", "show", "print", "output", "document"
        ],
        PentestPhase.POST_EXPLOITATION: [
            "maintain", "persist", "elevate", "lateral", "exfiltrate",
            "pivot", "cleanup", "cover", "persistence", "backdoor"
        ],
        # 新增通用類型用於不符合上述階段的功能
        "utility": [
            "helper", "util", "tool", "convert", "transform",
            "encode", "decode", "hash", "encrypt", "decrypt",
            "process", "filter", "sort", "group", "merge",
            "split", "join", "aggregate", "auth", "login",
            "request", "response", "http", "connect", "send"
        ]
    }
    
    def __init__(self, ai_engine: Optional[AIAnalysisEngine] = None, 
                 rag_engine: Optional[RAGEngine] = None):
        """
        初始化能力分析器
        
        Args:
            ai_engine: AI分析引擎，用於語義理解
            rag_engine: RAG引擎，用於知識檢索
        """
        self.ai_engine = ai_engine
        self.rag_engine = rag_engine
        
        # 分析結果緩存
        self._analysis_cache: Dict[str, CapabilityAnalysis] = {}
        self._classification_cache: Optional[CapabilityClassification] = None
        
        logger.info("CapabilityAnalyzer 初始化完成")
    
    async def analyze_capability(self, capability: Dict[str, Any]) -> CapabilityAnalysis:
        """
        深度分析單個能力
        
        Args:
            capability: 能力信息字典，包含:
                - id: 能力ID
                - name: 函數名稱
                - source_code: 源代碼
                - docstring: 文檔字符串
                - signature: 函數簽名
                - file_path: 文件路徑
                - module_name: 模組名稱
        
        Returns:
            CapabilityAnalysis: 詳細分析結果
        """
        capability_id = capability.get("id", "unknown")
        
        # 檢查緩存
        if capability_id in self._analysis_cache:
            logger.debug(f"使用緩存的分析結果: {capability_id}")
            return self._analysis_cache[capability_id]
        
        logger.info(f"🔍 分析能力: {capability.get('name', capability_id)}")
        
        try:
            # 1. AI語義理解
            semantic_analysis = self._ai_semantic_analysis(capability)
            
            # 2. 參數分析
            parameters = self._analyze_parameters(capability)
            
            # 3. 返回類型分析
            return_type = self._analyze_return_type(capability)
            
            # 4. 功能類型分類
            function_type = self._classify_function_type(capability, semantic_analysis)
            
            # 5. 風險評估
            risk_level = self._assess_risk_level(capability, semantic_analysis)
            
            # 6. 副作用識別
            side_effects = self._identify_side_effects(capability, semantic_analysis)
            
            # 7. 生成使用示例
            examples = self._generate_examples(capability, semantic_analysis)
            
            # 8. 查找相關能力
            related_capabilities = self._find_related_capabilities(capability)
            
            # 9. 生成文檔
            documentation = self._generate_documentation(capability, semantic_analysis)
            
            # 10. 計算信心度分數
            confidence_score = self._calculate_confidence_score(capability, semantic_analysis)
            
            # 創建分析結果
            analysis = CapabilityAnalysis(
                capability_id=capability_id,
                function_type=function_type,
                risk_level=risk_level,
                semantic_understanding=semantic_analysis,
                parameters=parameters,
                return_type=return_type,
                side_effects=side_effects,
                examples=examples,
                related_capabilities=related_capabilities,
                documentation=documentation,
                confidence_score=confidence_score
            )
            
            # 緩存結果
            self._analysis_cache[capability_id] = analysis
            
            logger.info(f"✅ 能力分析完成: {capability.get('name')} (信心度: {confidence_score:.2f})")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 能力分析失敗: {capability.get('name', capability_id)} - {e}")
            # 返回默認分析結果
            return self._create_default_analysis(capability)
    
    async def classify_all_capabilities(self, capabilities: List[Dict[str, Any]]) -> CapabilityClassification:
        """
        分類所有能力
        
        Args:
            capabilities: 能力列表
        
        Returns:
            CapabilityClassification: 分類結果
        """
        logger.info(f"📊 開始分類 {len(capabilities)} 個能力")
        
        # 檢查緩存
        if self._classification_cache is not None:
            logger.debug("使用緩存的分類結果")
            return self._classification_cache
        
        classifications = {
            "by_function": {},
            "by_risk": {},
            "by_module": {},
            "by_complexity": {}
        }
        
        # 分析每個能力
        for capability in capabilities:
            try:
                analysis = await self.analyze_capability(capability)
                capability_id = capability.get("id", "unknown")
                
                # 按功能分類
                function_type = analysis.function_type.value
                if function_type not in classifications["by_function"]:
                    classifications["by_function"][function_type] = []
                classifications["by_function"][function_type].append(capability_id)
                
                # 按風險分類
                risk_level = analysis.risk_level.value
                if risk_level not in classifications["by_risk"]:
                    classifications["by_risk"][risk_level] = []
                classifications["by_risk"][risk_level].append(capability_id)
                
                # 按模組分類
                module_name = capability.get("module_name", "unknown")
                if module_name not in classifications["by_module"]:
                    classifications["by_module"][module_name] = []
                classifications["by_module"][module_name].append(capability_id)
                
                # 按複雜度分類
                complexity = self._assess_complexity(capability, analysis)
                if complexity not in classifications["by_complexity"]:
                    classifications["by_complexity"][complexity] = []
                classifications["by_complexity"][complexity].append(capability_id)
                
            except Exception as e:
                logger.warning(f"⚠️ 分類能力時發生錯誤: {capability.get('name', 'unknown')} - {e}")
                continue
        
        # 創建分類結果
        result = CapabilityClassification(
            by_function=classifications["by_function"],
            by_risk=classifications["by_risk"],
            by_module=classifications["by_module"],
            by_complexity=classifications["by_complexity"]
        )
        
        # 緩存結果
        self._classification_cache = result
        
        logger.info("✅ 能力分類完成:")
        logger.info(f"  - 功能類型: {len(classifications['by_function'])} 種")
        logger.info(f"  - 風險等級: {len(classifications['by_risk'])} 級")
        logger.info(f"  - 模組分布: {len(classifications['by_module'])} 個")
        
        return result
    
    def _ai_semantic_analysis(self, capability: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用AI進行語義分析
        
        Args:
            capability: 能力信息
        
        Returns:
            Dict: AI分析結果
        """
        if not self.ai_engine:
            logger.debug("AI引擎未初始化，使用基於規則的分析")
            return self._rule_based_analysis(capability)
        
        try:
            # 準備分析輸入
            analysis_input = {
                "function_name": capability.get("name", ""),
                "docstring": capability.get("docstring", ""),
                "source_code": capability.get("source_code", ""),
                "signature": capability.get("signature", "")
            }
            
            # AI語義理解
            analysis_results = self.ai_engine.analyze_code(
                source_code=capability.get("source", ""),
                file_path=capability.get("file", ""),
                analysis_types=[AnalysisType.SEMANTIC, AnalysisType.SECURITY]
            )
            
            # 提取語義分析結果
            semantic_result = analysis_results.get(AnalysisType.SEMANTIC)
            if semantic_result:
                return {
                    "purpose": semantic_result.explanation,
                    "complexity": semantic_result.risk_level,
                    "patterns": semantic_result.findings,
                    "confidence": semantic_result.confidence
                }
            else:
                return analysis_input
            
        except Exception as e:
            logger.warning(f"AI語義分析失敗，使用基於規則的分析: {e}")
            return self._rule_based_analysis(capability)
    
    def _rule_based_analysis(self, capability: Dict[str, Any]) -> Dict[str, Any]:
        """
        基於規則的分析 (AI引擎不可用時的後備方案)
        
        Args:
            capability: 能力信息
        
        Returns:
            Dict: 基於規則的分析結果
        """
        name = capability.get("name", "").lower()
        docstring = capability.get("docstring", "").lower()
        source_code = capability.get("source_code", "").lower()
        
        # 合併所有文本用於關鍵字匹配
        text_content = f"{name} {docstring} {source_code}"
        
        # 功能推斷
        detected_keywords = []
        primary_function = "unknown"
        
        for func_type, keywords in self.FUNCTION_KEYWORDS.items():
            matching_keywords = [kw for kw in keywords if kw in text_content]
            if matching_keywords:
                detected_keywords.extend(matching_keywords)
                if primary_function == "unknown":
                    primary_function = func_type.value
        
        # 風險推斷
        risk_indicators = []
        for risk_level, keywords in self.RISK_KEYWORDS.items():
            matching_keywords = [kw for kw in keywords if kw in text_content]
            if matching_keywords:
                risk_indicators.extend(matching_keywords)
        
        return {
            "method": "rule_based",
            "primary_function": primary_function,
            "detected_keywords": detected_keywords,
            "risk_indicators": risk_indicators,
            "confidence": 0.6,  # 規則分析信心度較低
            "description": f"基於關鍵字分析的函數: {name}"
        }
    
    def _analyze_parameters(self, capability: Dict[str, Any]) -> List[ParameterInfo]:
        """
        分析函數參數
        
        Args:
            capability: 能力信息
        
        Returns:
            List[ParameterInfo]: 參數信息列表
        """
        parameters = []
        
        try:
            # 解析函數簽名
            signature = capability.get("signature", "")
            source_code = capability.get("source_code", "")
            
            if source_code:
                # 使用AST解析獲得更準確的參數信息
                tree = ast.parse(source_code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name == capability.get("name"):
                            parameters = self._extract_parameters_from_ast(node)
                            break
            
            elif signature:
                # 從簽名解析參數
                parameters = self._extract_parameters_from_signature(signature)
        
        except Exception as e:
            logger.warning(f"參數分析失敗: {e}")
        
        return parameters
    
    def _extract_parameters_from_ast(self, func_node: ast.FunctionDef) -> List[ParameterInfo]:
        """
        從AST節點提取參數信息
        
        Args:
            func_node: 函數AST節點
        
        Returns:
            List[ParameterInfo]: 參數信息
        """
        parameters = []
        
        # 獲取參數列表
        args = func_node.args
        
        # 處理位置參數
        for i, arg in enumerate(args.args):
            param_name = arg.arg
            
            # 跳過self和cls
            if param_name in ("self", "cls"):
                continue
            
            # 獲取類型註解
            type_hint = None
            if arg.annotation:
                type_hint = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
            
            # 獲取默認值
            default_value = None
            defaults_start = len(args.args) - len(args.defaults)
            if i >= defaults_start:
                default_idx = i - defaults_start
                default_node = args.defaults[default_idx]
                if hasattr(ast, 'unparse'):
                    default_value = ast.unparse(default_node)
                else:
                    default_value = str(default_node)
            
            # 判斷是否必需
            is_required = default_value is None
            
            parameters.append(ParameterInfo(
                name=param_name,
                type_hint=type_hint,
                default_value=default_value,
                description=None,  # 需要進一步分析文檔字符串
                is_required=is_required
            ))
        
        # 處理關鍵字參數
        if args.kwonlyargs:
            for i, arg in enumerate(args.kwonlyargs):
                param_name = arg.arg
                
                # 獲取類型註解
                type_hint = None
                if arg.annotation:
                    type_hint = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
                
                # 獲取默認值
                default_value = None
                if i < len(args.kw_defaults) and args.kw_defaults[i]:
                    default_node = args.kw_defaults[i]
                    if hasattr(ast, 'unparse'):
                        default_value = ast.unparse(default_node)
                    else:
                        default_value = str(default_node)
                
                is_required = default_value is None
                
                parameters.append(ParameterInfo(
                    name=param_name,
                    type_hint=type_hint,
                    default_value=default_value,
                    description=None,
                    is_required=is_required
                ))
        
        return parameters
    
    def _extract_parameters_from_signature(self, signature: str) -> List[ParameterInfo]:
        """
        從函數簽名字符串提取參數信息
        
        Args:
            signature: 函數簽名
        
        Returns:
            List[ParameterInfo]: 參數信息
        """
        parameters = []
        
        try:
            # 簡單正則解析 (更複雜的情況需要使用inspect模組)
            # 匹配參數: name: type = default
            param_pattern = r'(\w+)(?:\s*:\s*([^=,\)]+))?(?:\s*=\s*([^,\)]+))?'
            
            # 提取括號內的參數部分
            match = re.search(r'\(([^)]*)\)', signature)
            if match:
                params_str = match.group(1)
                
                for match in re.finditer(param_pattern, params_str):
                    name = match.group(1)
                    type_hint = match.group(2).strip() if match.group(2) else None
                    default_value = match.group(3).strip() if match.group(3) else None
                    
                    # 跳過self和cls
                    if name in ("self", "cls"):
                        continue
                    
                    parameters.append(ParameterInfo(
                        name=name,
                        type_hint=type_hint,
                        default_value=default_value,
                        description=None,
                        is_required=default_value is None
                    ))
        
        except Exception as e:
            logger.warning(f"簽名解析失敗: {e}")
        
        return parameters
    
    def _analyze_return_type(self, capability: Dict[str, Any]) -> Optional[str]:
        """
        分析返回類型
        
        Args:
            capability: 能力信息
        
        Returns:
            Optional[str]: 返回類型字符串
        """
        try:
            source_code = capability.get("source_code", "")
            if source_code:
                tree = ast.parse(source_code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if node.name == capability.get("name"):
                            if node.returns:
                                if hasattr(ast, 'unparse'):
                                    return ast.unparse(node.returns)
                                else:
                                    return str(node.returns)
                            break
        
        except Exception as e:
            logger.warning(f"返回類型分析失敗: {e}")
        
        return None
    
    def _classify_function_type(self, capability: Dict[str, Any], 
                               semantic_analysis: Dict[str, Any]) -> PentestPhase:
        """
        分類函數類型
        
        Args:
            capability: 能力信息
            semantic_analysis: 語義分析結果
        
        Returns:
            FunctionType: 函數類型
        """
        # 優先使用AI分析結果
        if "primary_function" in semantic_analysis:
            ai_function = semantic_analysis["primary_function"]
            for func_type in PentestPhase:
                if func_type.value == ai_function:
                    return func_type
        
        # 基於關鍵字分類
        name = capability.get("name", "").lower()
        docstring = capability.get("docstring", "").lower()
        text_content = f"{name} {docstring}"
        
        # 計算每種功能類型的匹配度
        type_scores = {}
        for func_type, keywords in self.FUNCTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_content)
            if score > 0:
                type_scores[func_type] = score
        
        # 返回得分最高的類型
        if type_scores:
            return max(type_scores.keys(), key=lambda k: type_scores[k])
        
        return PentestPhase.PRE_ENGAGEMENT  # 預設值
    
    def _assess_risk_level(self, capability: Dict[str, Any],
                          semantic_analysis: Dict[str, Any]) -> VulnerabilityRiskLevel:
        """
        評估風險等級
        
        Args:
            capability: 能力信息
            semantic_analysis: 語義分析結果
        
        Returns:
            RiskLevel: 風險等級
        """
        name = capability.get("name", "").lower()
        docstring = capability.get("docstring", "").lower()
        source_code = capability.get("source_code", "").lower()
        text_content = f"{name} {docstring} {source_code}"
        
        # 計算風險指標
        risk_scores = {}
        for risk_level, keywords in self.RISK_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_content)
            if score > 0:
                risk_scores[risk_level] = score
        
        # 特殊風險檢查
        high_risk_patterns = [
            r'subprocess\.',
            r'os\.system',
            r'exec\(',
            r'eval\(',
            r'__import__',
            r'shell.*command',
            r'inject.*sql',
            r'execute.*payload'
        ]
        
        for pattern in high_risk_patterns:
            if re.search(pattern, text_content, re.IGNORECASE):
                risk_scores[RiskLevel.HIGH] = risk_scores.get(RiskLevel.HIGH, 0) + 5
        
        # 返回得分最高的風險等級
        if risk_scores:
            return max(risk_scores.keys(), key=lambda k: risk_scores[k])
        
        # 默認為低風險
        return RiskLevel.LOW
    
    def _identify_side_effects(self, capability: Dict[str, Any], 
                              semantic_analysis: Dict[str, Any]) -> List[str]:
        """
        識別副作用
        
        Args:
            capability: 能力信息
            semantic_analysis: 語義分析結果
        
        Returns:
            List[str]: 副作用列表
        """
        side_effects = []
        source_code = capability.get("source_code", "").lower()
        
        # 檢查常見副作用模式
        side_effect_patterns = {
            "文件操作": [r'open\(', r'write\(', r'\.write', r'save'],
            "網絡請求": [r'requests\.', r'urllib', r'http', r'socket'],
            "系統調用": [r'subprocess', r'os\.system', r'command'],
            "數據庫操作": [r'sql', r'database', r'\.execute', r'query'],
            "日誌記錄": [r'log\.', r'logger\.', r'print\('],
            "狀態修改": [r'self\.', r'global ', r'nonlocal ']
        }
        
        for effect_type, patterns in side_effect_patterns.items():
            for pattern in patterns:
                if re.search(pattern, source_code):
                    side_effects.append(effect_type)
                    break
        
        return side_effects
    
    def _generate_examples(self, capability: Dict[str, Any],
                         semantic_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        生成使用示例
        
        Args:
            capability: 能力信息
            semantic_analysis: 語義分析結果
        
        Returns:
            List[Dict]: 使用示例列表
        """
        examples = []
        
        function_name = capability.get("name", "unknown_function")
        parameters = self._analyze_parameters(capability)
        
        # 生成基本示例
        if parameters:
            # 生成完整參數示例
            param_examples = []
            for param in parameters:
                if param.is_required:
                    if param.type_hint:
                        if "str" in param.type_hint.lower():
                            param_examples.append(f'{param.name}="example_value"')
                        elif "int" in param.type_hint.lower():
                            param_examples.append(f'{param.name}=123')
                        elif "bool" in param.type_hint.lower():
                            param_examples.append(f'{param.name}=True')
                        else:
                            param_examples.append(f'{param.name}=example_value')
                    else:
                        param_examples.append(f'{param.name}="example"')
            
            if param_examples:
                example_call = f"{function_name}({', '.join(param_examples)})"
                examples.append({
                    "type": "basic_usage",
                    "description": "基本用法示例",
                    "code": example_call
                })
        else:
            # 無參數函數
            examples.append({
                "type": "basic_usage", 
                "description": "基本用法示例",
                "code": f"{function_name}()"
            })
        
        # 根據功能類型生成特定示例
        function_type = self._classify_function_type(capability, semantic_analysis)
        
        if function_type == FunctionType.SCANNING:
            examples.append({
                "type": "scanning_example",
                "description": "掃描示例",
                "code": f'result = await {function_name}("https://example.com")\nprint(result)'
            })
        elif function_type == FunctionType.ANALYSIS:
            examples.append({
                "type": "analysis_example", 
                "description": "分析示例",
                "code": f'analysis = {function_name}(data)\nfor finding in analysis["findings"]:\n    print(finding)'
            })
        
        return examples
    
    def _find_related_capabilities(self, capability: Dict[str, Any]) -> List[str]:
        """
        查找相關能力
        
        Args:
            capability: 能力信息
        
        Returns:
            List[str]: 相關能力ID列表
        """
        related = []
        
        if self.rag_engine:
            try:
                # 使用RAG引擎查找相關能力
                query = f"{capability.get('name', '')} {capability.get('docstring', '')}"
                # 使用 RAG 引擎搜尋相關能力
                try:
                    # 模擬 RAG 搜尋結果（等待 RAG 引擎完整實現）
                    search_results = {
                        "similar_capabilities": [
                            {"name": f"related_{capability.get('name', 'capability')}", "similarity": 0.8}
                        ]
                    }
                except Exception as e:
                    logger.warning(f"RAG搜尋相關能力失敗: {e}")
                    search_results = {"similar_capabilities": []}
                
                if isinstance(search_results, list):
                    for result in search_results[:5]:  # 取前5個相關結果
                        if result.get("capability_id") != capability.get("id"):
                            related.append(result.get("capability_id", ""))
                        
            except Exception as e:
                logger.warning(f"RAG搜索相關能力失敗: {e}")
        
        # 基於關鍵字的簡單關聯 (RAG不可用時的後備)
        if not related:
            # 基於名稱相似性的簡單關聯
            name = capability.get("name", "").lower()
            
            # 簡單關鍵字比對
            if "scan" in name:
                related.extend([
                    {"name": "port_scanner", "similarity": 0.6},
                    {"name": "vulnerability_scanner", "similarity": 0.7}
                ])
            elif "exploit" in name:
                related.extend([
                    {"name": "payload_generator", "similarity": 0.8},
                    {"name": "exploit_framework", "similarity": 0.9}
                ])
        
        return related
    
    def _generate_documentation(self, capability: Dict[str, Any],
                              semantic_analysis: Dict[str, Any]) -> str:
        """
        生成能力文檔
        
        Args:
            capability: 能力信息
            semantic_analysis: 語義分析結果
        
        Returns:
            str: Markdown格式的文檔
        """
        name = capability.get("name", "Unknown Function")
        docstring = capability.get("docstring", "")
        parameters = self._analyze_parameters(capability)
        
        # 生成Markdown文檔
        doc_lines = [
            f"# {name}",
            "",
            f"**描述**: {docstring or '暫無描述'}",
            "",
            f"**功能類型**: {self._classify_function_type(capability, semantic_analysis).value}",
            f"**風險等級**: {self._assess_risk_level(capability, semantic_analysis).value}",
            "",
            "## 參數",
            ""
        ]
        
        if parameters:
            doc_lines.append("| 參數名 | 類型 | 必需 | 默認值 | 描述 |")
            doc_lines.append("|--------|------|------|--------|------|")
            
            for param in parameters:
                required = "是" if param.is_required else "否"
                default = param.default_value or "無"
                type_hint = param.type_hint or "Any"
                description = param.description or "暫無描述"
                
                doc_lines.append(f"| {param.name} | {type_hint} | {required} | {default} | {description} |")
        else:
            doc_lines.append("無參數")
        
        doc_lines.extend([
            "",
            "## 返回值",
            "",
            f"**類型**: {self._analyze_return_type(capability) or '未指定'}",
            "",
            "## 使用示例",
            "",
            "```python"
        ])
        
        # 添加示例代碼 (同步調用，這裡簡化處理)
        examples = []  # 在實際使用中應該調用 _generate_examples
        if examples:
            doc_lines.append(examples[0].get("code", f"{name}()"))
        else:
            doc_lines.append(f"result = {name}()")
        
        doc_lines.extend([
            "```",
            ""
        ])
        
        return "\n".join(doc_lines)
    
    def _calculate_confidence_score(self, capability: Dict[str, Any],
                                   semantic_analysis: Dict[str, Any]) -> float:
        """
        計算分析信心度分數
        
        Args:
            capability: 能力信息
            semantic_analysis: 語義分析結果
        
        Returns:
            float: 信心度分數 (0.0-1.0)
        """
        score = 0.0
        
        # 基礎分數
        if capability.get("name"):
            score += 0.2
        
        if capability.get("docstring"):
            score += 0.3
        
        if capability.get("source_code"):
            score += 0.2
        
        # AI分析加分
        if semantic_analysis.get("method") == "ai_analysis":
            score += 0.2
        
        # 語義分析信心度
        ai_confidence = semantic_analysis.get("confidence", 0.0)
        score += ai_confidence * 0.1
        
        return min(score, 1.0)
    
    def _assess_complexity(self, capability: Dict[str, Any], 
                          analysis: CapabilityAnalysis) -> str:
        """
        評估複雜度
        
        Args:
            capability: 能力信息
            analysis: 分析結果
        
        Returns:
            str: 複雜度等級 (simple/medium/complex)
        """
        complexity_score = 0
        
        # 參數數量
        param_count = len(analysis.parameters)
        if param_count > 5:
            complexity_score += 2
        elif param_count > 2:
            complexity_score += 1
        
        # 副作用數量
        side_effect_count = len(analysis.side_effects)
        complexity_score += side_effect_count
        
        # 風險等級影響複雜度
        if analysis.risk_level == RiskLevel.HIGH:
            complexity_score += 2
        elif analysis.risk_level == RiskLevel.MEDIUM:
            complexity_score += 1
        
        # 代碼長度 (粗略估計)
        source_code = capability.get("source_code", "")
        if source_code:
            lines = source_code.count('\n')
            if lines > 50:
                complexity_score += 2
            elif lines > 20:
                complexity_score += 1
        
        # 分類
        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "simple"
    
    def _create_default_analysis(self, capability: Dict[str, Any]) -> CapabilityAnalysis:
        """
        創建默認分析結果 (分析失敗時使用)
        
        Args:
            capability: 能力信息
        
        Returns:
            CapabilityAnalysis: 默認分析結果
        """
        return CapabilityAnalysis(
            capability_id=capability.get("id", "unknown"),
            function_type=PentestPhase.PRE_ENGAGEMENT,  # 預設值
            risk_level=VulnerabilityRiskLevel.LOW,
            semantic_understanding={"method": "fallback", "description": "分析失敗，使用默認值"},
            parameters=[],
            return_type=None,
            side_effects=[],
            examples=[],
            related_capabilities=[],
            documentation=f"# {capability.get('name', 'Unknown Function')}\n\n分析失敗，請手動檢查。",
            confidence_score=0.1
        )
    
    def export_analysis_results(self) -> Dict[str, Any]:
        """
        導出分析結果
        
        Args:
            format: 導出格式 (json/markdown)
        
        Returns:
            Dict: 導出的結果
        """
        results = {
            "analyzed_capabilities": len(self._analysis_cache),
            "classification": self._classification_cache.to_dict() if self._classification_cache else None,
            "analyses": {
                cap_id: analysis.to_dict() 
                for cap_id, analysis in self._analysis_cache.items()
            }
        }
        
        return results
    
    def clear_cache(self) -> None:
        """清理分析緩存"""
        self._analysis_cache.clear()
        self._classification_cache = None
        logger.info("分析緩存已清理")


# 導出主要類
__all__ = [
    "CapabilityAnalyzer",
    "CapabilityAnalysis", 
    "CapabilityClassification",
    "ParameterInfo",
    "FunctionType"
]