# 三大決策方法實作說明

## 概述

本文檔詳細說明 AIVA 系統中三個核心 AI 決策方法的實作，這些方法基於實際 HackerOne/Bugcrowd Bug Bounty 測試場景設計。

**實作位置**: `services/core/aiva_core/cognitive_core/decision/enhanced_decision_agent.py`

---

## 1. decide_phase1_strategy() - Phase1 深度掃描決策

### 步驟對應
- **13 步流程**: Step 6 (AI 決策是否需要 Phase1 深度掃描)
- **OWASP WSTG**: 4.1 Information Gathering → 4.2 Configuration Testing

### 核心功能
根據 Phase0 偵察結果，決定是否值得投入時間進行 Phase1 深度掃描。

### 決策因素

| 因素 | 權重 | 說明 |
|------|------|------|
| 高價值目標評分 | 核心 | API、文件上傳、Admin 面板、支付流程 |
| 技術棧風險 | 1.0-1.5x | PHP/WordPress/Struts = 高風險 |
| WAF 檢測 | -30~50% | 降低成功率但可嘗試繞過 |
| ROI 計算 | $75/hr 門檻 | 專業 Hunter 的參考標準 |
| AI 信心度 | 0-1 | 5M 神經網路評估 |

### 高價值目標識別

```python
high_value_indicators = {
    "api_endpoints": apis_found,           # API 通常有 IDOR/Auth 問題
    "file_upload_forms": ...,              # 文件上傳是高價值目標
    "auth_endpoints": ...,                 # 認證相關端點
    "admin_panels": ...,                   # Admin 面板通常獎金高
    "payment_flows": ...,                  # 支付相關最高價值
    "graphql_endpoints": ...,              # GraphQL 特殊處理
}
```

### 返回結構

```python
{
    "need_phase1": bool,                   # 是否執行 Phase1
    "reasoning": str,                      # 決策理由
    "roi": float,                          # 投資回報率 ($/hr)
    "high_value_score": float,             # 高價值評分 (0-1)
    "priority_targets": [                  # 優先測試目標
        {"type": "payment", "priority": 1, "vuln_focus": ["idor", "logic_bypass"]},
        ...
    ],
    "phase1_config": {                     # Phase1 執行配置
        "scan_depth": "intensive"|"standard",
        "focus_areas": [...],
        "parallel_workers": int,
        "stealth_mode": bool
    },
    "waf_bypass_plan": {...}               # WAF 繞過策略 (如檢測到)
}
```

---

## 2. decide_phase2_targets() - Phase2 攻擊目標排序

### 步驟對應
- **13 步流程**: Step 9 (AI 決策 Phase2 攻擊目標)
- **OWASP WSTG**: 4.6 Session Management, 4.7 Input Validation, 4.10 Business Logic

### 核心功能
根據 Phase1 掃描結果，智慧排序攻擊目標優先級。

### 漏洞優先級體系 (基於 HackerOne 獎金)

#### Tier 1 - Critical ($10k+)
| 漏洞類型 | 預設獎金 | 攻擊鏈潛力 |
|---------|---------|-----------|
| RCE/Command Injection | $15,000 | Shell Access → Lateral Movement |
| SSRF | $8,000 | Cloud Metadata → AWS Keys → RCE |
| Account Takeover | $12,000 | 完整帳戶控制 |
| Payment Bypass | $10,000+ | 財務損失 |

#### Tier 2 - High ($3k-$10k)
| 漏洞類型 | 預設獎金 | 攻擊鏈潛力 |
|---------|---------|-----------|
| SQL Injection | $5,000 | Credential Dump → ATO |
| IDOR | $3,000 | Bulk Data Access → PII Leak |
| Auth Bypass | $4,000 | Admin Access → Full Control |
| XXE | $3,000 | Internal File Read |
| SSTI | $4,000 | Template → RCE |

#### Tier 3 - Medium ($500-$3k)
| 漏洞類型 | 預設獎金 | 說明 |
|---------|---------|------|
| Stored XSS | $1,500 | 可串聯成 ATO |
| DOM XSS | $1,000 | JS 執行 |
| Reflected XSS | $500 | 需社工 |
| CSRF | $1,000 | 敏感操作 |
| API Key Disclosure | $1,500 | 認證資料洩露 |

### 綜合評分算法

```python
final_score = (
    ai_score * 0.35 +              # AI 神經網路評估
    bounty_value_score * 0.25 +    # 獎金價值
    (1.0 - waf_interference) * 0.15 +  # WAF 抗性
    historical_success * 0.10 +    # 歷史成功率
    (1.0 - duplicate_risk) * 0.15  # 重複風險
)

# Tier 加權
if tier == 1: final_score *= 1.3
elif tier == 2: final_score *= 1.1
```

### 推薦工具映射

| 漏洞類型 | 推薦工具 |
|---------|---------|
| SQL Injection | sqlmap, burp_intruder, ghauri |
| XSS | xsstrike, dalfox, domdig |
| SSRF | ssrfmap, gopherus, burp_collaborator |
| IDOR | autorize, burp_match_replace |
| XXE | xxeinjector |
| SSTI | tplmap, sstimap |

### 返回結構

```python
[
    {
        "asset": {...},                    # 原始資產
        "score": 0.85,                     # 綜合評分
        "tier": 1,                         # 優先級等級
        "cvss_estimate": 9.0,              # CVSS 3.1 估計
        "attack_vector": "ssrf",           # 攻擊向量
        "recommended_tools": ["ssrfmap"], # 推薦工具
        "estimated_bounty": 8000,          # 預估獎金
        "duplicate_risk": 0.2,             # 重複風險
        "reasoning": "Tier1|AI:0.8|獎金:0.8|WAF干擾:0.1"
    },
    ...
]
```

---

## 3. evaluate_phase2_results() - Phase2 結果評估

### 步驟對應
- **13 步流程**: Step 11 (AI 評估 Phase2 結果)
- **HackerOne 報告流程**: 品質評估 → 報告撰寫 → 提交

### 核心功能
評估 Phase2 攻擊結果，決定後續行動。

### 決策選項

| 行動 | 觸發條件 | 優先級 |
|-----|---------|--------|
| **SUBMIT_REPORT** | Critical/High + POC Ready + Confidence > 0.85 | HIGH/URGENT |
| **CHAIN_VULNERABILITIES** | 可串聯漏洞 + Chain Score > 0.7 | HIGH |
| **CONTINUE_DEEP_DIVE** | ROI > $50/hr + 時間充裕 | NORMAL |
| **SWITCH_STRATEGY** | Confidence < 0.4 + Findings < 3 | NORMAL |
| **ABANDON_TARGET** | Duplicate Rate > 60% 或 無有效發現 | LOW |

### 漏洞串聯分析

系統自動檢測可組合的攻擊鏈：

| 組合 | 結果 | 嚴重性提升 |
|-----|------|-----------|
| XSS + CSRF | Account Takeover | → CRITICAL |
| SSRF + Internal | RCE Chain | → CRITICAL |
| IDOR + Info Disclosure | Mass Data Exposure | → HIGH |
| SQLi + Auth Bypass | Full DB Access | → CRITICAL |
| Open Redirect + OAuth | Token Theft | → HIGH |

### HackerOne 報告指南生成

當決策為 `SUBMIT_REPORT` 時，系統自動生成報告指南：

```python
{
    "title_template": "[Critical] SSRF in api.example.com/internal",
    "sections": [
        "Summary (1-2 sentences)",
        "Steps to Reproduce (numbered list)",
        "Impact (business perspective)",
        "Proof of Concept (code/screenshots)",
        "Suggested Fix",
        "References (CVE, OWASP)"
    ],
    "cvss_estimate": {
        "base_score": 9.0,
        "vector_string": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N",
        "severity_rating": "CRITICAL"
    },
    "tips": [
        "使用清晰的重現步驟",
        "強調商業影響",
        "提供修復建議",
        "附上視頻 POC 可提升報告品質"
    ]
}
```

### 返回結構

```python
{
    "action": "SUBMIT_REPORT",
    "priority": "HIGH",
    "reasoning": "發現 2 個高危漏洞，POC 已準備",
    "findings_summary": {
        "total": 5,
        "by_severity": {"CRITICAL": 1, "HIGH": 1, "MEDIUM": 2, "LOW": 1},
        "poc_ready": 3,
        "avg_confidence": 0.87,
        "estimated_bounty": 12500
    },
    "chain_analysis": {
        "can_chain": true,
        "score": 0.8,
        "chain_description": "XSS + CSRF → 帳戶劫持",
        "severity_boost": "CRITICAL"
    },
    "report_guidance": {...},
    "next_steps": ["撰寫詳細報告", "準備 POC 視頻", "計算 CVSS"],
    "time_metrics": {
        "remaining_minutes": 45,
        "recommended_action_time": 30
    }
}
```

---

## 整合流程

```
Phase0 (偵察)
    ↓
┌─────────────────────────────────────────────┐
│ decide_phase1_strategy()                    │
│ - 評估攻擊面                                 │
│ - 計算 ROI                                   │
│ - WAF 檢測                                   │
│ - 返回: need_phase1, phase1_config          │
└─────────────────────────────────────────────┘
    ↓ (如果 need_phase1 = true)
Phase1 (深度掃描)
    ↓
┌─────────────────────────────────────────────┐
│ decide_phase2_targets()                     │
│ - Tier 排序 (1/2/3)                         │
│ - 獎金評估                                   │
│ - 重複風險分析                               │
│ - 返回: 排序後的攻擊目標列表                  │
└─────────────────────────────────────────────┘
    ↓
Phase2 (攻擊執行)
    ↓
┌─────────────────────────────────────────────┐
│ evaluate_phase2_results()                   │
│ - 結果統計                                   │
│ - 攻擊鏈分析                                 │
│ - 決策: SUBMIT/CONTINUE/CHAIN/SWITCH/ABANDON│
│ - 報告指南生成                               │
└─────────────────────────────────────────────┘
    ↓
Phase3 (報告/繼續)
```

---

## 與 5M 神經網路整合

三個決策方法都與 `RealDecisionEngine` (5M 參數) 整合：

```python
# 在 decide_phase1_strategy 中
ai_result = self.neural_engine.generate_decision(
    task_description="decide_phase1_strategy",
    context=neural_context
)
ai_confidence = ai_result.get("confidence", 0.5)
ai_attack_vector = ai_result.get("attack_vector", "reconnaissance")

# 在 decide_phase2_targets 中
ai_result = self.neural_engine.generate_decision(
    task_description="target_prioritization",
    context=f"Target: {url} | VulnType: {vuln_type} | Tier: {tier}"
)

# 在 evaluate_phase2_results 中
ai_result = self.neural_engine.generate_decision(
    task_description="evaluate_phase2_results",
    context=context
)
```

---

## 實際使用範例

```python
from aiva_core.cognitive_core.decision import EnhancedDecisionAgent

agent = EnhancedDecisionAgent(use_neural_decision=True)

# Step 6: Phase1 決策
phase0_result = {
    "summary": {"urls_found": 50, "forms_found": 10, "apis_found": 15},
    "fingerprints": {"waf_detected": True, "waf_vendor": "cloudflare"},
    "endpoints": [...]
}
phase1_decision = agent.decide_phase1_strategy(phase0_result, target_value=5000)

# Step 9: Phase2 目標排序
if phase1_decision["need_phase1"]:
    phase1_result = run_phase1_scan(...)
    targets = agent.decide_phase2_targets(phase1_result, max_targets=10)

# Step 11: 結果評估
phase2_results = run_phase2_attack(targets)
evaluation = agent.evaluate_phase2_results(
    phase2_results,
    time_budget_remaining=3600  # 1 小時
)

if evaluation["action"] == "SUBMIT_REPORT":
    print(evaluation["report_guidance"])
```

---

## 參考資料

- [OWASP Web Security Testing Guide (WSTG)](https://owasp.org/www-project-web-security-testing-guide/)
- [HackerOne Severity Ratings](https://docs.hackerone.com/hackers/severity.html)
- [CVSS 3.1 Calculator](https://www.first.org/cvss/calculator/3.1)
- [Bug Bounty Hunter Methodology](https://github.com/jhaddix/tbhm)

---

**文件版本**: 1.0.0  
**最後更新**: 2025-01-14  
**作者**: AIVA Development Team
