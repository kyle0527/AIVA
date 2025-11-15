"""
測試 CapabilityAnalyzer - 能力分析器

測試能力分析器的各項功能:
1. 能力分析
2. 功能分類  
3. 風險評估
4. 參數分析
5. 文檔生成

作者: AIVA Development Team
創建日期: 2025-11-13
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock
from typing import Dict, Any, List

# 導入測試目標
from aiva_core.ai_engine.capability_analyzer import (
    CapabilityAnalyzer,
    CapabilityAnalysis,
    CapabilityClassification,
    ParameterInfo,
    FunctionType  # 這是 PentestPhase 的別名
)
from aiva_common.enums.common import VulnerabilityRiskLevel
from aiva_common.enums.pentest import PentestPhase

# 別名以保持向後兼容性
RiskLevel = VulnerabilityRiskLevel


@pytest.fixture
def sample_capability():
    """示例能力數據"""
    return {
        "id": "test_scan_target",
        "name": "scan_target",
        "docstring": "掃描目標網站，檢測常見漏洞",
        "source_code": '''
def scan_target(url: str, timeout: int = 30) -> dict:
    """
    掃描目標網站，檢測常見漏洞
    
    Args:
        url: 目標URL
        timeout: 超時時間，默認30秒
    
    Returns:
        dict: 掃描結果
    """
    import requests
    
    result = {"vulnerabilities": []}
    try:
        response = requests.get(url, timeout=timeout)
        # 掃描邏輯...
        return result
    except Exception as e:
        logger.error(f"掃描失敗: {e}")
        return {"error": str(e)}
''',
        "signature": "scan_target(url: str, timeout: int = 30) -> dict",
        "file_path": "/test/scanner.py",
        "module_name": "CoreModule"
    }


@pytest.fixture
def high_risk_capability():
    """高風險能力示例"""
    return {
        "id": "test_sql_inject", 
        "name": "sql_inject_test",
        "docstring": "測試SQL注入漏洞",
        "source_code": '''
def sql_inject_test(target: str, payload: str) -> dict:
    """測試SQL注入漏洞"""
    import subprocess
    
    # 執行SQL注入測試
    command = f"sqlmap -u {target} --data '{payload}'"
    result = subprocess.run(command, shell=True, capture_output=True)
    
    return {"success": True, "output": result.stdout}
''',
        "signature": "sql_inject_test(target: str, payload: str) -> dict",
        "module_name": "FunctionModule"
    }


@pytest.fixture
def mock_ai_engine():
    """模擬AI引擎（簡化版本）"""
    # 返回None，讓系統使用基本語義分析
    return None


@pytest.fixture  
def mock_rag_engine():
    """模擬RAG引擎（簡化版本）"""
    # 返回None，讓系統使用基本相關能力查找
    return None


@pytest.fixture
def analyzer(mock_ai_engine, mock_rag_engine):
    """創建能力分析器實例"""
    return CapabilityAnalyzer(
        ai_engine=mock_ai_engine,
        rag_engine=mock_rag_engine
    )


@pytest.fixture
def analyzer_no_ai():
    """沒有AI引擎的分析器（測試規則分析）"""
    return CapabilityAnalyzer(ai_engine=None, rag_engine=None)


class TestCapabilityAnalyzer:
    """CapabilityAnalyzer 測試類"""
    
    def test_initialization(self, analyzer):
        """測試初始化"""
        assert analyzer.ai_engine is not None
        assert analyzer.rag_engine is not None
        assert len(analyzer._analysis_cache) == 0
    
    def test_initialization_without_engines(self):
        """測試無引擎初始化"""
        analyzer = CapabilityAnalyzer()
        assert analyzer.ai_engine is None
        assert analyzer.rag_engine is None
    
    @pytest.mark.asyncio
    async def test_analyze_capability_basic(self, analyzer, sample_capability):
        """測試基本能力分析"""
        analysis = await analyzer.analyze_capability(sample_capability)
        
        # 檢查基本屬性
        assert isinstance(analysis, CapabilityAnalysis)
        assert analysis.capability_id == "test_scan_target"
        assert analysis.function_type == PentestPhase.VULNERABILITY_ANALYSIS
        assert isinstance(analysis.risk_level, VulnerabilityRiskLevel)
        assert analysis.confidence_score > 0.0
        
        # 檢查參數分析
        assert len(analysis.parameters) == 2
        url_param = next(p for p in analysis.parameters if p.name == "url")
        assert url_param.is_required == True
        assert url_param.type_hint == "str"
        
        timeout_param = next(p for p in analysis.parameters if p.name == "timeout")
        assert timeout_param.is_required == False
        assert timeout_param.default_value == "30"
    
    @pytest.mark.asyncio
    async def test_analyze_high_risk_capability(self, analyzer, high_risk_capability):
        """測試高風險能力分析"""
        analysis = await analyzer.analyze_capability(high_risk_capability)
        
        # 高風險能力應該被正確識別
        assert analysis.risk_level == VulnerabilityRiskLevel.HIGH
        assert analysis.function_type == PentestPhase.EXPLOITATION
        assert "系統調用" in analysis.side_effects
    
    @pytest.mark.asyncio
    async def test_rule_based_analysis(self, analyzer_no_ai, sample_capability):
        """測試基於規則的分析（無AI引擎）"""
        analysis = await analyzer_no_ai.analyze_capability(sample_capability)
        
        # 應該回退到規則分析
        assert analysis.semantic_understanding["method"] == "rule_based"
        assert analysis.function_type == PentestPhase.INTELLIGENCE_GATHERING
        assert analysis.confidence_score < 0.8  # 規則分析信心度較低
    
    def test_classify_function_type_scanning(self, analyzer, sample_capability):
        """測試掃描功能分類"""
        semantic_analysis = {"primary_function": "scanning"}
        
        function_type = analyzer._classify_function_type(
            sample_capability, semantic_analysis
        )
        
        assert function_type == PentestPhase.INTELLIGENCE_GATHERING
    
    def test_classify_function_type_by_keywords(self, analyzer):
        """測試基於關鍵字的功能分類"""
        capability = {
            "name": "exploit_vulnerability",
            "docstring": "exploit detected vulnerability"
        }
        semantic_analysis = {}
        
        function_type = analyzer._classify_function_type(
            capability, semantic_analysis
        )
        
        assert function_type == PentestPhase.EXPLOITATION
    
    def test_assess_risk_level_high(self, analyzer, high_risk_capability):
        """測試高風險評估"""
        semantic_analysis = {}
        
        risk_level = analyzer._assess_risk_level(
            high_risk_capability, semantic_analysis
        )
        
        assert risk_level == RiskLevel.HIGH
    
    def test_assess_risk_level_low(self, analyzer):
        """測試低風險評估"""
        capability = {
            "name": "format_report",
            "docstring": "format scan report",
            "source_code": "def format_report(data): return json.dumps(data)"
        }
        semantic_analysis = {}
        
        risk_level = analyzer._assess_risk_level(
            capability, semantic_analysis
        )
        
        assert risk_level == VulnerabilityRiskLevel.LOW
    
    def test_analyze_parameters_from_ast(self, analyzer, sample_capability):
        """測試從AST分析參數"""
        parameters = analyzer._analyze_parameters(sample_capability)
        
        assert len(parameters) == 2
        
        # 檢查url參數
        url_param = next(p for p in parameters if p.name == "url")
        assert url_param.type_hint == "str"
        assert url_param.is_required == True
        assert url_param.default_value is None
        
        # 檢查timeout參數
        timeout_param = next(p for p in parameters if p.name == "timeout") 
        assert timeout_param.type_hint == "int"
        assert timeout_param.is_required == False
        assert timeout_param.default_value == "30"
    
    def test_analyze_parameters_from_signature(self, analyzer):
        """測試從簽名分析參數"""
        capability = {
            "signature": "test_func(name: str, age: int = 25, active: bool = True)",
            "name": "test_func"
        }
        
        parameters = analyzer._analyze_parameters(capability)
        
        assert len(parameters) == 3
        
        name_param = next(p for p in parameters if p.name == "name")
        assert name_param.type_hint == "str"
        assert name_param.is_required == True
        
        age_param = next(p for p in parameters if p.name == "age")  
        assert age_param.type_hint == "int"
        assert age_param.default_value == "25"
        
        active_param = next(p for p in parameters if p.name == "active")
        assert active_param.type_hint == "bool" 
        assert active_param.default_value == "True"
    
    def test_analyze_return_type(self, analyzer, sample_capability):
        """測試返回類型分析"""
        return_type = analyzer._analyze_return_type(sample_capability)
        assert return_type == "dict"
    
    def test_identify_side_effects(self, analyzer, sample_capability):
        """測試副作用識別"""
        semantic_analysis = {}
        side_effects = analyzer._identify_side_effects(
            sample_capability, semantic_analysis
        )
        
        # 應該檢測到網絡請求副作用
        assert "網絡請求" in side_effects
    
    def test_generate_examples(self, analyzer, sample_capability):
        """測試生成使用示例"""
        semantic_analysis = {"primary_function": "scanning"}
        
        examples = analyzer._generate_examples(
            sample_capability, semantic_analysis
        )
        
        assert len(examples) >= 1
        basic_example = next(e for e in examples if e["type"] == "basic_usage")
        assert "scan_target" in basic_example["code"]
    
    @pytest.mark.asyncio 
    def test_find_related_capabilities(self, analyzer, sample_capability):
        """測試查找相關能力"""
        related = analyzer._find_related_capabilities(sample_capability)
        
        # 應該找到模擬的相關能力
        assert len(related) == 2
        assert "related_scanner_1" in related
        assert "related_scanner_2" in related
    
    def test_generate_documentation(self, analyzer, sample_capability):
        """測試生成文檔"""
        semantic_analysis = {"primary_function": "scanning"}
        
        doc = analyzer._generate_documentation(
            sample_capability, semantic_analysis
        )
        
        # 檢查Markdown格式
        assert "# scan_target" in doc
        assert "## 參數" in doc
        assert "## 返回值" in doc
        assert "## 使用示例" in doc
    
    def test_calculate_confidence_score(self, analyzer, sample_capability):
        """測試信心度計算"""
        semantic_analysis = {
            "method": "ai_analysis",
            "confidence": 0.9
        }
        
        score = analyzer._calculate_confidence_score(
            sample_capability, semantic_analysis
        )
        
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # 完整的能力信息應該有高信心度
    
    def test_assess_complexity(self, analyzer, sample_capability):
        """測試複雜度評估"""
        analysis = CapabilityAnalysis(
            capability_id="test",
            function_type=PentestPhase.INTELLIGENCE_GATHERING,
            risk_level=VulnerabilityRiskLevel.MEDIUM,
            semantic_understanding={},
            parameters=[
                ParameterInfo("param1", "str", None, None, True),
                ParameterInfo("param2", "int", "30", None, False)
            ],
            return_type="dict",
            side_effects=["網絡請求"],
            examples=[],
            related_capabilities=[],
            documentation="",
            confidence_score=0.8
        )
        
        complexity = analyzer._assess_complexity(sample_capability, analysis)
        assert complexity in ["simple", "medium", "complex"]
    
    @pytest.mark.asyncio
    async def test_classify_all_capabilities(self, analyzer, sample_capability, high_risk_capability):
        """測試分類所有能力"""
        capabilities = [sample_capability, high_risk_capability]
        
        classification = await analyzer.classify_all_capabilities(capabilities)
        
        assert isinstance(classification, CapabilityClassification)
        assert "vulnerability_analysis" in classification.by_function
        assert "exploitation" in classification.by_function
        assert "high" in classification.by_risk
        assert "medium" in classification.by_risk or "low" in classification.by_risk
    
    @pytest.mark.asyncio
    async def test_export_analysis_results(self, analyzer, sample_capability):
        """測試導出分析結果"""
        # 先分析一個能力
        await analyzer.analyze_capability(sample_capability)
        
        # 導出結果
        results = analyzer.export_analysis_results()
        
        assert "analyzed_capabilities" in results
        assert results["analyzed_capabilities"] == 1
        assert "analyses" in results
        assert "test_scan_target" in results["analyses"]
    
    def test_clear_cache(self, analyzer, sample_capability):
        """測試清理緩存"""
        # 添加到緩存
        analyzer._analysis_cache["test"] = Mock()
        analyzer._classification_cache = Mock()
        
        # 清理緩存
        analyzer.clear_cache()
        
        assert len(analyzer._analysis_cache) == 0
        assert analyzer._classification_cache is None
    
    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer):
        """測試錯誤處理"""
        # 無效的能力數據
        invalid_capability = {
            "id": "invalid",
            "name": None,  # 無效名稱
            "source_code": "invalid python code {"  # 無效語法
        }
        
        analysis = await analyzer.analyze_capability(invalid_capability)
        
        # 應該返回預設分析結果
        assert analysis.capability_id == "invalid"
        assert analysis.function_type == PentestPhase.PRE_ENGAGEMENT  # 預設值
        assert analysis.confidence_score < 0.5


class TestParameterInfo:
    """ParameterInfo 測試類"""
    
    def test_parameter_info_creation(self):
        """測試參數信息創建"""
        param = ParameterInfo(
            name="test_param",
            type_hint="str", 
            default_value="default",
            description="Test parameter",
            is_required=False
        )
        
        assert param.name == "test_param"
        assert param.type_hint == "str"
        assert param.default_value == "default"
        assert param.is_required == False
    
    def test_parameter_info_to_dict(self):
        """測試參數信息轉換為字典"""
        param = ParameterInfo(
            name="test",
            type_hint="int",
            default_value=None,
            description=None,
            is_required=True
        )
        
        param_dict = param.to_dict()
        
        assert param_dict["name"] == "test"
        assert param_dict["type_hint"] == "int"
        assert param_dict["is_required"] == True


class TestCapabilityAnalysis:
    """CapabilityAnalysis 測試類"""
    
    def test_capability_analysis_to_dict(self):
        """測試能力分析結果轉換為字典"""
        analysis = CapabilityAnalysis(
            capability_id="test",
            function_type=PentestPhase.INTELLIGENCE_GATHERING,
            risk_level=VulnerabilityRiskLevel.LOW,
            semantic_understanding={"method": "test"},
            parameters=[
                ParameterInfo("param1", "str", None, None, True)
            ],
            return_type="dict",
            side_effects=["網絡請求"],
            examples=[{"type": "basic", "code": "test()"}],
            related_capabilities=["related1"],
            documentation="Test doc",
            confidence_score=0.8
        )
        
        analysis_dict = analysis.to_dict()
        
        assert analysis_dict["capability_id"] == "test"
        assert analysis_dict["function_type"] == "vulnerability_analysis"
        assert analysis_dict["risk_level"] == "low"
        assert len(analysis_dict["parameters"]) == 1
        assert abs(analysis_dict["confidence_score"] - 0.8) < 0.001


class TestCapabilityClassification:
    """CapabilityClassification 測試類"""
    
    def test_classification_to_dict(self):
        """測試分類結果轉換為字典"""
        classification = CapabilityClassification(
            by_function={"scanning": ["cap1"], "analysis": ["cap2"]},
            by_risk={"high": ["cap1"], "low": ["cap2"]},
            by_module={"core": ["cap1", "cap2"]},
            by_complexity={"simple": ["cap2"], "medium": ["cap1"]}
        )
        
        class_dict = classification.to_dict()
        
        assert "by_function" in class_dict
        assert "by_risk" in class_dict
        assert "by_module" in class_dict
        assert "by_complexity" in class_dict
        assert class_dict["by_function"]["scanning"] == ["cap1"]


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v"])