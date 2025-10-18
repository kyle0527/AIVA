# -*- coding: utf-8 -*-
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import time

@dataclass
class Finding:
    """
    單一安全發現的結構化表示
    
    設計用於直接匯出到 HackerOne/Bugcrowd 等平台的報告格式，
    包含所有必要的證據和重現步驟。
    """
    vuln_type: str                          # 漏洞類型，如 "IDOR", "Reflected XSS", "JWT confusion"
    severity: str                           # 嚴重度：low/medium/high/critical
    title: str                              # 漏洞標題，簡潔描述問題
    evidence: Dict[str, Any]               # 結構化證據（請求/回應片段、狀態碼差異等）
    reproduction: List[Dict[str, Any]]     # 可重現步驟（包含具體的 HTTP 請求）
    impact: Optional[str] = None           # 業務影響描述
    recommendation: Optional[str] = None   # 修復建議
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式，便於 JSON 序列化"""
        return asdict(self)
    
    def to_hackerone_format(self) -> Dict[str, Any]:
        """
        轉換為 HackerOne 報告格式
        
        Returns:
            適合直接貼進 HackerOne 表單的格式化字典
        """
        steps = []
        for i, step in enumerate(self.reproduction, 1):
            if "request" in step:
                req = step["request"]
                method = req.get("method", "GET")
                url = req.get("url", "")
                
                step_text = f"**Step {i}:** Send {method} request to `{url}`"
                
                if req.get("headers"):
                    step_text += f"\nHeaders: ```{req['headers']}```"
                if req.get("json"):
                    step_text += f"\nBody: ```json\n{req['json']}\n```"
                if req.get("data"):
                    step_text += f"\nData: ```{req['data']}```"
                if step.get("expect"):
                    step_text += f"\n\nExpected: {step['expect']}"
                    
                steps.append(step_text)
        
        return {
            "title": self.title,
            "vulnerability_type": self.vuln_type,
            "severity": self.severity.upper(),
            "description": f"## Summary\n{self.title}\n\n## Impact\n{self.impact or 'See evidence below'}\n\n## Steps to Reproduce\n" + "\n\n".join(steps),
            "impact": self.impact,
            "recommendation": self.recommendation,
            "evidence": self.evidence
        }

@dataclass
class FeatureResult:
    """
    功能模組執行結果的統一格式
    
    包含執行狀態、發現的漏洞、命令記錄和元數據，
    提供完整的執行上下文供後續處理使用。
    """
    ok: bool                               # 執行是否成功
    feature: str                           # 功能模組名稱
    command_record: Dict[str, Any]         # 面板可顯示的命令記錄
    findings: List[Finding]                # 發現的安全問題列表
    meta: Dict[str, Any]                   # 額外的元數據（追蹤資訊、統計等）
    
    def __post_init__(self):
        """自動填入時間戳"""
        if "timestamp" not in self.command_record:
            self.command_record["timestamp"] = int(time.time())
        if "execution_time" not in self.meta:
            self.meta["execution_time"] = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式，便於序列化和傳輸"""
        return {
            "ok": self.ok,
            "feature": self.feature,
            "command_record": self.command_record,
            "findings": [f.to_dict() for f in self.findings],
            "meta": self.meta
        }
    
    def has_critical_findings(self) -> bool:
        """檢查是否有 Critical 級別的發現"""
        return any(f.severity.lower() == "critical" for f in self.findings)
    
    def has_high_findings(self) -> bool:
        """檢查是否有 High 級別或以上的發現"""
        return any(f.severity.lower() in ("critical", "high") for f in self.findings)
    
    def get_findings_by_severity(self, severity: str) -> List[Finding]:
        """根據嚴重度篩選發現"""
        return [f for f in self.findings if f.severity.lower() == severity.lower()]
    
    def get_summary(self) -> Dict[str, Any]:
        """取得執行摘要"""
        severity_counts = {}
        for finding in self.findings:
            sev = finding.severity.lower()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        return {
            "feature": self.feature,
            "total_findings": len(self.findings),
            "severity_breakdown": severity_counts,
            "has_high_risk": self.has_high_findings(),
            "execution_ok": self.ok,
            "timestamp": self.command_record.get("timestamp")
        }