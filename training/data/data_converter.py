#!/usr/bin/env python3
"""數據轉換器 - 確保外部模組數據正確轉換為學習系統格式

這個模組負責：
1. 將 FindingPayload（外部模組產出）轉換為 TrainingDataSample（學習系統輸入）
2. 確保數據格式與 distillation_train.json 一致
3. 確保 Embedding/Tokenization 能正確處理轉換後的文本

數據流：
    外部模組（XSS/SQLi/SSRF）
        ↓ 產出 FindingPayload
    DataConverter
        ↓ 轉換為 TrainingDataSample
    AI 學習系統
        ↓ Tokenization + Embedding
    權重更新
"""

import logging
from typing import Dict, Any
from datetime import datetime

from services.aiva_common.schemas.findings import FindingPayload
from services.aiva_common.schemas.dual_loop import TrainingDataSample

logger = logging.getLogger(__name__)


class DataConverter:
    """數據轉換器
    
    確保外部模組數據正確轉換為學習系統格式
    """
    
    # ✅ 嚴重性映射（與 distillation_train.json 一致）
    SEVERITY_MAPPING = {
        "CRITICAL": 1.0,
        "HIGH": 0.8,
        "MEDIUM": 0.5,
        "LOW": 0.3,
        "INFO": 0.1,
    }
    
    # ✅ 置信度映射（與 distillation_train.json 一致）
    CONFIDENCE_MAPPING = {
        "CONFIRMED": 1.0,
        "HIGH": 0.85,
        "MEDIUM": 0.6,
        "LOW": 0.4,
    }
    
    # ✅ 漏洞類型映射（統一命名）
    VULNERABILITY_TYPE_MAPPING = {
        "XSS": "xss",
        "SQLI": "sql_injection",
        "SSRF": "ssrf",
        "IDOR": "idor",
        "RCE": "rce",
        "JWT_ATTACK": "jwt_attack",
        "GRAPHQL_INTROSPECTION": "graphql_introspection",
    }
    
    @staticmethod
    def finding_to_training_sample(finding: FindingPayload) -> TrainingDataSample:
        """將 FindingPayload 轉換為 TrainingDataSample
        
        確保格式與 distillation_train.json 一致：
        - scenario_text: 場景描述（簡潔，用於 Embedding）
        - raw_context: 原始上下文（HTTP 請求/響應）
        - teacher_vulnerability_type: 漏洞類型（小寫下劃線）
        - teacher_severity: 嚴重性（0.0-1.0）
        - teacher_confidence: 置信度（0.0-1.0）
        - teacher_reasoning: 推理過程
        - source_doc: 來源（實戰數據）
        - scenario_id: 唯一ID
        - difficulty_level: 難度等級
        
        Args:
            finding: 外部模組產出的漏洞發現
            
        Returns:
            TrainingDataSample: 學習系統格式的訓練樣本
        """
        
        # 1️⃣ 構建 scenario_text（簡潔描述，適合 Embedding）
        scenario_text = DataConverter._build_scenario_text(finding)
        
        # 2️⃣ 構建 raw_context（完整 HTTP 上下文）
        raw_context = DataConverter._build_raw_context(finding)
        
        # 3️⃣ 轉換漏洞類型（統一命名）
        vuln_type = DataConverter._convert_vulnerability_type(finding)
        
        # 4️⃣ 轉換嚴重性（0.0-1.0）
        severity = DataConverter._convert_severity(finding)
        
        # 5️⃣ 轉換置信度（0.0-1.0）
        confidence = DataConverter._convert_confidence(finding)
        
        # 6️⃣ 生成推理過程
        reasoning = DataConverter._generate_reasoning(finding, severity, confidence)
        
        # 7️⃣ 評估難度
        difficulty = DataConverter._assess_difficulty(finding)
        
        # ✅ 構建 TrainingDataSample（與 distillation_train.json 格式完全一致）
        sample = TrainingDataSample(
            scenario_text=scenario_text,
            raw_context=raw_context,
            teacher_vulnerability_type=vuln_type,
            teacher_severity=severity,
            teacher_confidence=confidence,
            teacher_reasoning=reasoning,
            source_doc="production_data",  # 實戰數據
            scenario_id=finding.finding_id,
            difficulty_level=difficulty
        )
        
        logger.info(f"✅ 轉換完成: {finding.finding_id}")
        logger.debug(f"   scenario_text: {scenario_text[:100]}...")
        logger.debug(f"   vulnerability: {vuln_type} (severity={severity:.2f}, confidence={confidence:.2f})")
        
        return sample
    
    @staticmethod
    def _build_scenario_text(finding: FindingPayload) -> str:
        """構建場景文本（簡潔描述，適合 Embedding）
        
        格式參考 distillation_train.json:
        - "在 HTTP 響應中發現 XSS 相關的 <script> 模式"
        - "通過分析請求參數與響應差異，識別出 SSRF 漏洞"
        
        Args:
            finding: 漏洞發現數據
            
        Returns:
            場景文本（100-200字符）
        """
        vuln_name = finding.vulnerability.name
        target_url = finding.target.url if hasattr(finding.target, 'url') else "unknown"
        parameter = getattr(finding.target, 'parameter', None) or "unknown"
        method = getattr(finding.target, 'method', None) or "GET"
        
        # 提取關鍵證據
        key_evidence = ""
        if finding.evidence and finding.evidence.payload:
            payload = finding.evidence.payload
            # 提取關鍵模式（前50字符）
            key_evidence = payload[:50] if len(payload) > 50 else payload
        
        # ✅ 構建簡潔描述（與 distillation_train.json 格式一致）
        if key_evidence:
            scenario = f"在 HTTP 響應中發現 {vuln_name} 相關的 {key_evidence} 模式"
        else:
            scenario = f"通過分析 {method} 請求參數 {parameter}，識別出 {vuln_name} 漏洞"
        
        return scenario
    
    @staticmethod
    def _build_raw_context(finding: FindingPayload) -> str:
        """構建原始上下文（完整 HTTP 請求/響應）
        
        格式參考 distillation_train.json:
        ```
        GET /api/user?id=<script>alert(1)</script> HTTP/1.1
        Host: target.com
        
        HTTP/1.1 200 OK
        Content-Type: text/html
        
        <html><body><script>alert(1)</script></body></html>
        ```
        
        Args:
            finding: 漏洞發現數據
            
        Returns:
            原始上下文字符串
        """
        lines = []
        
        # 1️⃣ HTTP 請求
        if finding.evidence and finding.evidence.request:
            lines.append("=== REQUEST ===")
            lines.append(finding.evidence.request)
        else:
            # 手動構建請求
            method = getattr(finding.target, 'method', 'GET')
            url = finding.target.url if hasattr(finding.target, 'url') else ""
            parameter = getattr(finding.target, 'parameter', None)
            payload = finding.evidence.payload if finding.evidence else None
            
            if parameter and payload:
                lines.append(f"{method} {url}?{parameter}={payload} HTTP/1.1")
            else:
                lines.append(f"{method} {url} HTTP/1.1")
            
            lines.append(f"Host: {finding.target.host if hasattr(finding.target, 'host') else 'unknown'}")
        
        # 2️⃣ HTTP 響應
        if finding.evidence and finding.evidence.response:
            lines.append("\n=== RESPONSE ===")
            lines.append(finding.evidence.response)
        else:
            # 基本響應信息
            lines.append("\n=== RESPONSE ===")
            if finding.evidence and hasattr(finding.evidence, 'response_time_delta'):
                lines.append(f"Response Time: {finding.evidence.response_time_delta}s")
        
        # 3️⃣ 證據
        if finding.evidence and finding.evidence.proof:
            lines.append("\n=== PROOF ===")
            lines.append(finding.evidence.proof)
        
        return "\n".join(lines)
    
    @staticmethod
    def _convert_vulnerability_type(finding: FindingPayload) -> str:
        """轉換漏洞類型（統一命名）
        
        distillation_train.json 使用的類型：
        - xss
        - sql_injection
        - ssrf
        - idor
        - rce
        - jwt_attack
        - graphql_introspection
        
        Args:
            finding: 漏洞發現數據
            
        Returns:
            標準化漏洞類型名稱
        """
        vuln_type = str(finding.vulnerability.name).upper()
        
        # 標準化命名
        return DataConverter.VULNERABILITY_TYPE_MAPPING.get(
            vuln_type,
            vuln_type.lower()  # 預設轉小寫
        )
    
    @staticmethod
    def _convert_severity(finding: FindingPayload) -> float:
        """轉換嚴重性（0.0-1.0）
        
        Args:
            finding: 漏洞發現數據
            
        Returns:
            嚴重性分數（0.0-1.0）
        """
        severity_str = str(finding.vulnerability.severity).upper()
        return DataConverter.SEVERITY_MAPPING.get(severity_str, 0.5)
    
    @staticmethod
    def _convert_confidence(finding: FindingPayload) -> float:
        """轉換置信度（0.0-1.0）
        
        Args:
            finding: 漏洞發現數據
            
        Returns:
            置信度分數（0.0-1.0）
        """
        confidence_str = str(finding.vulnerability.confidence).upper()
        return DataConverter.CONFIDENCE_MAPPING.get(confidence_str, 0.6)
    
    @staticmethod
    def _generate_reasoning(
        finding: FindingPayload,
        severity: float,
        confidence: float
    ) -> str:
        """生成推理過程（解釋為什麼判定為該漏洞）
        
        格式參考 distillation_train.json:
        "基於 反射型 XSS 特徵分析，發現 <script> 標籤未轉義，判定為 xss，嚴重性評估為 0.80"
        
        Args:
            finding: 漏洞發現數據
            severity: 嚴重性分數
            confidence: 置信度分數
            
        Returns:
            推理過程文本
        """
        vuln_name = finding.vulnerability.name
        vuln_type = DataConverter._convert_vulnerability_type(finding)
        
        # 提取關鍵證據
        evidence_text = "相關特徵"
        if finding.evidence and finding.evidence.payload:
            evidence_text = finding.evidence.payload[:30]
        
        reasoning = (
            f"基於 {vuln_name} 特徵分析，"
            f"發現 {evidence_text} 指標，"
            f"判定為 {vuln_type}，"
            f"嚴重性評估為 {severity:.2f}"
        )
        
        return reasoning
    
    @staticmethod
    def _assess_difficulty(finding: FindingPayload) -> str:
        """評估難度等級
        
        Based on:
        - 嚴重性（高嚴重性 = 高難度）
        - 置信度（低置信度 = 高難度）
        - 證據複雜度
        
        Args:
            finding: 漏洞發現數據
            
        Returns:
            難度等級: "easy", "medium", "hard"
        """
        severity = DataConverter._convert_severity(finding)
        confidence = DataConverter._convert_confidence(finding)
        
        # 高嚴重性 + 低置信度 = 高難度
        if severity >= 0.8 and confidence <= 0.6:
            return "hard"
        
        # 高嚴重性 或 中等置信度 = 中等難度
        if severity >= 0.7 or confidence >= 0.7:
            return "medium"
        
        # 其他 = 簡單
        return "easy"
    
    @staticmethod
    def validate_sample(sample: TrainingDataSample) -> bool:
        """驗證訓練樣本是否符合格式要求
        
        檢查項目：
        1. scenario_text 不為空（用於 Embedding）
        2. raw_context 不為空（提供完整上下文）
        3. teacher_vulnerability_type 在允許範圍內
        4. teacher_severity 在 [0.0, 1.0] 範圍
        5. teacher_confidence 在 [0.0, 1.0] 範圍
        
        Args:
            sample: 訓練樣本
            
        Returns:
            是否有效
        """
        # 1. 必填字段檢查
        if not sample.scenario_text or not sample.raw_context:
            logger.error("❌ scenario_text 或 raw_context 為空")
            return False
        
        # 2. 漏洞類型檢查
        valid_types = {"xss", "sql_injection", "ssrf", "idor", "rce", "jwt_attack", "graphql_introspection"}
        if sample.teacher_vulnerability_type not in valid_types:
            logger.warning(f"⚠️  未知漏洞類型: {sample.teacher_vulnerability_type}")
            # 不阻止，只警告
        
        # 3. 數值範圍檢查
        if not (0.0 <= sample.teacher_severity <= 1.0):
            logger.error(f"❌ severity 超出範圍: {sample.teacher_severity}")
            return False
        
        if not (0.0 <= sample.teacher_confidence <= 1.0):
            logger.error(f"❌ confidence 超出範圍: {sample.teacher_confidence}")
            return False
        
        logger.debug(f"✅ 訓練樣本驗證通過: {sample.scenario_id}")
        return True
    
    @staticmethod
    def batch_convert(findings: list[FindingPayload]) -> list[TrainingDataSample]:
        """批量轉換
        
        Args:
            findings: 漏洞發現列表
            
        Returns:
            訓練樣本列表
        """
        samples = []
        failed = 0
        
        for finding in findings:
            try:
                sample = DataConverter.finding_to_training_sample(finding)
                
                # 驗證
                if DataConverter.validate_sample(sample):
                    samples.append(sample)
                else:
                    failed += 1
                    logger.error(f"❌ 樣本驗證失敗: {finding.finding_id}")
                    
            except Exception as e:
                failed += 1
                logger.error(f"❌ 轉換失敗: {finding.finding_id} - {e}")
        
        logger.info(f"📊 批量轉換完成: {len(samples)} 成功, {failed} 失敗")
        return samples


# ==================== 快速測試 ====================

def test_data_conversion():
    """測試數據轉換"""
    from services.aiva_common.schemas.findings import (
        FindingPayload,
        Vulnerability,
        FindingEvidence,
    )
    from services.aiva_common.schemas.security.findings import Target
    from services.aiva_common.enums import VulnerabilityType, Severity, Confidence
    
    # 模擬外部模組產出
    finding = FindingPayload(
        finding_id="finding_test_001",
        task_id="task_test_001",
        scan_id="scan_test_001",
        status="confirmed",
        vulnerability=Vulnerability(
            name=VulnerabilityType.XSS,
            severity=Severity.HIGH,
            confidence=Confidence.HIGH,
            description="Reflected XSS in search parameter"
        ),
        target=Target(
            url="https://example.com/search?q=<script>alert(1)</script>",
            method="GET",
            parameter="q"
        ),
        evidence=FindingEvidence(
            payload="<script>alert(1)</script>",
            response='<html><body><script>alert(1)</script></body></html>',
            request='GET /search?q=<script>alert(1)</script> HTTP/1.1'
        )
    )
    
    # 轉換
    sample = DataConverter.finding_to_training_sample(finding)
    
    # 驗證
    assert DataConverter.validate_sample(sample)
    
    print("\n✅ 數據轉換測試通過:")
    print(f"   scenario_text: {sample.scenario_text}")
    print(f"   vulnerability: {sample.teacher_vulnerability_type}")
    print(f"   severity: {sample.teacher_severity}")
    print(f"   confidence: {sample.teacher_confidence}")
    print(f"   difficulty: {sample.difficulty_level}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_data_conversion()
