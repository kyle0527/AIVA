# ⚙️ Processing - 結果處理系統

**導航**: [← 返回 Core Capabilities](../README.md) | [← 返回 AIVA Core](../../README.md)

> **版本**: 3.0.0-alpha  
> **代碼量**: 1 個 Python 檔案，約 290 行代碼  
> **角色**: AIVA 的「數據處理器」- 掃描結果的智能分析和處理

---

## 📋 目錄

- [模組概述](#模組概述)
- [檔案列表](#檔案列表)
- [核心組件](#核心組件)
- [處理流程](#處理流程)
- [使用範例](#使用範例)

---

## 🎯 模組概述

**Processing** 子模組負責處理從 Ingestion 模組攝取的掃描結果，包括去重、優先級排序、風險評估、關聯分析和結果聚合等智能處理功能。

### 核心能力
1. **結果去重** - 識別和合併重複的發現
2. **優先級排序** - 根據嚴重程度和影響範圍排序
3. **風險評估** - 計算綜合風險評分
4. **關聯分析** - 關聯相關的漏洞發現
5. **結果聚合** - 生成統計報告和摘要

---

## 📂 檔案列表

| 檔案名 | 行數 | 核心功能 | 狀態 |
|--------|------|----------|------|
| **scan_result_processor.py** | 290 | 掃描結果處理器 - 智能分析和處理 | ✅ 生產 |
| **__init__.py** | - | 模組初始化 | - |

---

## 🔧 核心組件

### ScanResultProcessor - 掃描結果處理器

**檔案**: `scan_result_processor.py` (290 行)

提供掃描結果的智能處理能力，包括去重、排序、評估和聚合。

#### 核心類別

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ProcessingAction(Enum):
    """處理動作"""
    DEDUPLICATE = "deduplicate"     # 去重
    PRIORITIZE = "prioritize"       # 優先級排序
    ASSESS_RISK = "assess_risk"     # 風險評估
    CORRELATE = "correlate"         # 關聯分析
    AGGREGATE = "aggregate"         # 聚合統計

@dataclass
class ProcessedResult:
    """處理後的結果"""
    original_findings: List[Dict]
    deduplicated_findings: List[Dict]
    prioritized_findings: List[Dict]
    risk_scores: Dict[str, float]
    correlations: List[Dict]
    summary: Dict[str, Any]

class ScanResultProcessor:
    """掃描結果處理器
    
    功能:
    - 去重處理
    - 優先級排序
    - 風險評估
    - 漏洞關聯
    - 結果聚合
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化處理器"""
        self.config = config or {}
        self.dedup_threshold = self.config.get("dedup_similarity", 0.9)
        self.risk_weights = self.config.get("risk_weights", {
            "severity": 0.4,
            "exploitability": 0.3,
            "impact": 0.2,
            "confidence": 0.1
        })
    
    async def process(
        self,
        findings: List[Dict],
        actions: List[ProcessingAction] = None
    ) -> ProcessedResult:
        """處理掃描結果"""
        
    def deduplicate(self, findings: List[Dict]) -> List[Dict]:
        """去重處理"""
        
    def prioritize(self, findings: List[Dict]) -> List[Dict]:
        """優先級排序"""
        
    def assess_risk(self, finding: Dict) -> float:
        """評估單個發現的風險評分"""
        
    def correlate(self, findings: List[Dict]) -> List[Dict]:
        """關聯分析"""
        
    def aggregate(self, findings: List[Dict]) -> Dict[str, Any]:
        """聚合統計"""
```

---

## 🔄 處理流程

```
原始發現
    ↓
┌──────────────────┐
│   1. 去重處理     │  相似度檢測 → 合併重複項
└────────┬─────────┘
         ↓
┌──────────────────┐
│ 2. 優先級排序     │  嚴重程度 + 可利用性 → 排序
└────────┬─────────┘
         ↓
┌──────────────────┐
│  3. 風險評估      │  多維度評分 → 風險等級
└────────┬─────────┘
         ↓
┌──────────────────┐
│  4. 關聯分析      │  漏洞鏈識別 → 攻擊路徑
└────────┬─────────┘
         ↓
┌──────────────────┐
│  5. 結果聚合      │  統計摘要 → 可視化數據
└────────┬─────────┘
         ↓
    處理完成
```

---

## 🚀 使用範例

### 完整處理流程

```python
from core_capabilities.processing import (
    ScanResultProcessor,
    ProcessingAction,
    ProcessedResult
)

# 1. 初始化處理器
processor = ScanResultProcessor(config={
    "dedup_similarity": 0.85,  # 相似度閾值
    "risk_weights": {
        "severity": 0.4,
        "exploitability": 0.3,
        "impact": 0.2,
        "confidence": 0.1
    }
})

# 2. 準備原始發現數據
findings = [
    {
        "id": "finding-001",
        "name": "SQL Injection",
        "severity": "high",
        "url": "https://example.com/login",
        "parameter": "username"
    },
    {
        "id": "finding-002",
        "name": "SQL Injection",  # 重複
        "severity": "high",
        "url": "https://example.com/login",
        "parameter": "username"
    },
    {
        "id": "finding-003",
        "name": "XSS",
        "severity": "medium",
        "url": "https://example.com/search",
        "parameter": "query"
    }
]

# 3. 執行處理
result = await processor.process(
    findings=findings,
    actions=[
        ProcessingAction.DEDUPLICATE,
        ProcessingAction.PRIORITIZE,
        ProcessingAction.ASSESS_RISK,
        ProcessingAction.CORRELATE,
        ProcessingAction.AGGREGATE
    ]
)

# 4. 查看處理結果
print(f"原始發現數: {len(result.original_findings)}")
print(f"去重後: {len(result.deduplicated_findings)}")
print(f"\n=== 優先級排序 ===")
for i, finding in enumerate(result.prioritized_findings[:5], 1):
    risk_score = result.risk_scores.get(finding['id'], 0)
    print(f"{i}. [{finding['severity'].upper()}] {finding['name']}")
    print(f"   風險評分: {risk_score:.2f}")
    print(f"   URL: {finding['url']}")

print(f"\n=== 統計摘要 ===")
summary = result.summary
print(f"總發現數: {summary['total']}")
print(f"Critical: {summary['by_severity']['critical']}")
print(f"High: {summary['by_severity']['high']}")
print(f"Medium: {summary['by_severity']['medium']}")
print(f"Low: {summary['by_severity']['low']}")
```

### 去重處理

```python
# 去重算法
def deduplicate(findings: List[Dict]) -> List[Dict]:
    """基於相似度的去重
    
    比較維度:
    - 漏洞類型 (name)
    - URL 路徑
    - 參數名稱
    - HTTP 方法
    """
    
    deduplicated = []
    seen_signatures = set()
    
    for finding in findings:
        # 生成特徵簽名
        signature = (
            finding.get('name', '').lower(),
            finding.get('url', '').split('?')[0],  # 忽略查詢參數
            finding.get('parameter', ''),
            finding.get('method', 'GET')
        )
        
        if signature not in seen_signatures:
            deduplicated.append(finding)
            seen_signatures.add(signature)
        else:
            # 合併重複發現的證據
            existing = next(f for f in deduplicated 
                          if self._match_signature(f, signature))
            if 'evidence' in finding:
                existing.setdefault('evidence', []).extend(finding['evidence'])
    
    return deduplicated

# 使用
original_count = len(findings)
deduplicated = processor.deduplicate(findings)
print(f"去重: {original_count} → {len(deduplicated)} (-{original_count - len(deduplicated)})")
```

### 風險評估

```python
def assess_risk(finding: Dict) -> float:
    """多維度風險評分 (0-10)
    
    評分因子:
    - 嚴重程度 (40%)
    - 可利用性 (30%)
    - 影響範圍 (20%)
    - 置信度 (10%)
    """
    
    # 1. 嚴重程度評分
    severity_scores = {
        "critical": 10,
        "high": 8,
        "medium": 5,
        "low": 3,
        "info": 1
    }
    severity_score = severity_scores.get(
        finding.get("severity", "info").lower(), 
        1
    )
    
    # 2. 可利用性評分
    exploitability = finding.get("exploitability", "medium")
    exploit_scores = {
        "high": 10,
        "medium": 6,
        "low": 3
    }
    exploit_score = exploit_scores.get(exploitability, 6)
    
    # 3. 影響範圍評分
    impact = finding.get("impact", "limited")
    impact_scores = {
        "complete": 10,
        "high": 8,
        "partial": 5,
        "limited": 3
    }
    impact_score = impact_scores.get(impact, 5)
    
    # 4. 置信度評分
    confidence = finding.get("confidence", "certain")
    confidence_scores = {
        "certain": 10,
        "firm": 8,
        "tentative": 5
    }
    confidence_score = confidence_scores.get(confidence, 8)
    
    # 加權計算
    weights = self.risk_weights
    final_score = (
        severity_score * weights["severity"] +
        exploit_score * weights["exploitability"] +
        impact_score * weights["impact"] +
        confidence_score * weights["confidence"]
    )
    
    return round(final_score, 2)

# 批量評估
for finding in findings:
    risk_score = processor.assess_risk(finding)
    finding["risk_score"] = risk_score
    print(f"{finding['name']}: {risk_score}/10")
```

### 關聯分析

```python
def correlate(findings: List[Dict]) -> List[Dict]:
    """識別漏洞之間的關聯關係
    
    關聯類型:
    - 攻擊鏈 (一個漏洞可利用另一個)
    - 同源漏洞 (相同根本原因)
    - 組合攻擊 (多個漏洞組合利用)
    """
    
    correlations = []
    
    # 1. 識別攻擊鏈
    for i, finding_a in enumerate(findings):
        for finding_b in findings[i+1:]:
            if self._is_attack_chain(finding_a, finding_b):
                correlations.append({
                    "type": "attack_chain",
                    "findings": [finding_a["id"], finding_b["id"]],
                    "description": f"{finding_a['name']} 可用於利用 {finding_b['name']}",
                    "severity": "high"
                })
    
    # 2. 識別同源漏洞
    by_root_cause = {}
    for finding in findings:
        root = self._identify_root_cause(finding)
        by_root_cause.setdefault(root, []).append(finding)
    
    for root, related in by_root_cause.items():
        if len(related) > 1:
            correlations.append({
                "type": "common_root_cause",
                "findings": [f["id"] for f in related],
                "root_cause": root,
                "count": len(related)
            })
    
    # 3. 識別組合攻擊
    # 例如: XSS + CSRF = 完整攻擊鏈
    xss_findings = [f for f in findings if "xss" in f.get("name", "").lower()]
    csrf_findings = [f for f in findings if "csrf" in f.get("name", "").lower()]
    
    if xss_findings and csrf_findings:
        correlations.append({
            "type": "combined_attack",
            "findings": [xss_findings[0]["id"], csrf_findings[0]["id"]],
            "description": "XSS + CSRF 可實現完整的跨站請求偽造攻擊",
            "severity": "critical"
        })
    
    return correlations

# 使用
correlations = processor.correlate(findings)
print(f"\n=== 發現 {len(correlations)} 個關聯 ===")
for corr in correlations:
    print(f"[{corr['type'].upper()}] {corr.get('description', '')}")
    print(f"  涉及發現: {', '.join(corr['findings'])}")
```

### 結果聚合

```python
def aggregate(findings: List[Dict]) -> Dict[str, Any]:
    """生成統計摘要"""
    
    from collections import Counter
    
    # 按嚴重程度統計
    by_severity = Counter(f.get("severity", "info") for f in findings)
    
    # 按類型統計
    by_type = Counter(f.get("name", "Unknown") for f in findings)
    
    # 按 URL 統計
    by_url = Counter(f.get("url", "Unknown") for f in findings)
    
    # 高風險發現
    high_risk = [
        f for f in findings 
        if f.get("risk_score", 0) >= 8.0
    ]
    
    # 計算平均風險評分
    avg_risk = sum(f.get("risk_score", 0) for f in findings) / len(findings) if findings else 0
    
    return {
        "total": len(findings),
        "by_severity": dict(by_severity),
        "by_type": dict(by_type.most_common(10)),
        "by_url": dict(by_url.most_common(10)),
        "high_risk_count": len(high_risk),
        "average_risk_score": round(avg_risk, 2),
        "top_vulnerabilities": [
            {
                "name": f["name"],
                "url": f["url"],
                "risk": f.get("risk_score", 0)
            }
            for f in sorted(findings, key=lambda x: x.get("risk_score", 0), reverse=True)[:5]
        ]
    }

# 使用
summary = processor.aggregate(findings)
print(json.dumps(summary, indent=2, ensure_ascii=False))
```

---

## 📊 處理統計示例

```json
{
  "total": 127,
  "deduplicated": 85,
  "by_severity": {
    "critical": 3,
    "high": 15,
    "medium": 42,
    "low": 20,
    "info": 5
  },
  "by_type": {
    "SQL Injection": 12,
    "XSS": 18,
    "CSRF": 8,
    "Authentication Bypass": 5,
    "Information Disclosure": 22
  },
  "high_risk_count": 18,
  "average_risk_score": 6.3,
  "correlations": [
    {
      "type": "attack_chain",
      "findings": ["finding-001", "finding-045"],
      "description": "XSS 可用於繞過 CSRF 保護"
    }
  ],
  "top_vulnerabilities": [
    {
      "name": "SQL Injection",
      "url": "https://api.example.com/login",
      "risk": 9.2
    },
    {
      "name": "Authentication Bypass",
      "url": "https://api.example.com/admin",
      "risk": 8.8
    }
  ]
}
```

---

## 📚 相關文檔

- [Core Capabilities 主文檔](../README.md)
- [Ingestion 子模組](../ingestion/README.md) - 數據攝取
- [Output 子模組](../output/README.md) - 輸出轉換
- [Plugins 子模組](../plugins/README.md) - AI 摘要插件

---

**版權所有** © 2024 AIVA Project. 保留所有權利。
