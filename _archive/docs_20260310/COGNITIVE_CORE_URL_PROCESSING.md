# 認知核心網址處理流程分析

## 📑 目錄

- [📋 總覽](#-總覽)
- [🔍 詳細流程分析](#-詳細流程分析)
  - [階段 1: 內部能力查詢（InternalLoopConnector）](#階段-1-內部能力查詢internalloopconnector)
  - [階段 2: embedded_knowledge 深度分析](#階段-2-embedded_knowledge-深度分析)
    - [2.1 漏洞檢測（VulnerabilityDetector）](#21-漏洞檢測vulnerabilitydetector)
    - [2.2 架構分析（WebArchitectureAnalyzer）](#22-架構分析webarchitectureanalyzer)
    - [2.3 CVE 識別（CVEIdentifier）](#23-cve-識別cveidentifier)
    - [2.4 WAF 繞過分析（WAFBypassEngine）](#24-waf-繞過分析wafbypassengine)
  - [階段 3: 多源融合決策](#階段-3-多源融合決策)
  - [階段 4: 增強結果輸出](#階段-4-增強結果輸出)
- [🎯 實際案例：davincisnotebook.blog](#-實際案例davincisnotebookblog)
  - [階段 1: 內部能力查詢](#階段-1-內部能力查詢)
  - [階段 2: embedded_knowledge 分析](#階段-2-embedded_knowledge-分析)
  - [階段 3: 多源融合決策](#階段-3-多源融合決策)
  - [階段 4: 增強輸出](#階段-4-增強輸出)
- [📊 知識源權重](#-知識源權重)
- [🔧 關鍵方法總結](#-關鍵方法總結)
- [💡 設計優勢](#-設計優勢)
- [🚀 後續優化方向](#-後續優化方向)

---


**研究日期**: 2026-01-19  
**研究對象**: EnhancedDecisionAgent 如何處理網址  
**文件**: `enhanced_decision_agent.py`

---

## 📋 總覽

當用戶提供網址（如 `https://davincisnotebook.blog/`）時，認知核心通過 4 個階段進行處理：

```
網址輸入
    ↓
【階段 1】內部能力查詢
    ↓
【階段 2】embedded_knowledge 分析
    ↓
【階段 3】多源融合決策
    ↓
【階段 4】增強結果輸出
```

---

## 🔍 詳細流程分析

### 階段 1: 內部能力查詢（InternalLoopConnector）

**入口**: `make_enhanced_decision()` line 934

```python
if self.internal_connector:
    target_type = context.target_info.get("type", "web")
    capabilities = self.query_internal_capabilities(
        query=f"{target_type} vulnerability scan",
        top_k=3
    )
```

**處理邏輯**:
1. 提取目標類型（web/api/network）
2. 查詢內部能力庫（RAG 語義搜索）
3. 返回最相關的 3 個能力
4. 將能力名稱添加到 `context.available_tools`

**知識來源**: 
- ChromaDB 向量數據庫
- 210 個 flows 的能力描述
- 功能模組的使用經驗

---

### 階段 2: embedded_knowledge 深度分析

**入口**: `make_enhanced_decision()` line 944

#### 2.1 漏洞檢測（VulnerabilityDetector）

```python
vuln_result = self.analyze_target_vulnerabilities(
    target_url=target_url,
    response_body="",
    response_time=0.0
)
```

**分析內容**:
- SQL 注入特徵
- XSS 漏洞模式
- 命令注入風險
- 路徑遍歷可能性
- 認證繞過機會

**輸出**: 
```python
{
    "vulnerabilities": [
        {"type": "sql_injection", "confidence": 0.85, "location": "/api/search"},
        {"type": "xss", "confidence": 0.72, "location": "/comments"}
    ],
    "risk_score": 8.5
}
```

#### 2.2 架構分析（WebArchitectureAnalyzer）

```python
arch_result = self.analyze_web_architecture(
    response_headers={},
    response_body=""
)
```

**識別內容**:
- 架構類型: REST/GraphQL/gRPC/Microservices
- 技術棧: Node.js/Django/Spring Boot
- 安全機制: WAF/CDN/Rate Limiting
- API 版本和端點結構

**輸出**:
```python
{
    "architecture_type": "graphql",
    "framework": "apollo-server",
    "security_issues": ["no_rate_limit", "introspection_enabled"],
    "recommendations": ["Enable rate limiting", "Disable introspection in production"]
}
```

#### 2.3 CVE 識別（CVEIdentifier）

**觸發條件**: 檢測到特定軟件版本時

```python
cve_result = self.identify_cves(
    software="express",
    version="4.17.1"
)
```

**輸出**:
```python
{
    "cves": [
        {
            "id": "CVE-2022-24999",
            "severity": "HIGH",
            "description": "qs prototype pollution",
            "cvss_score": 7.5
        }
    ]
}
```

#### 2.4 WAF 繞過分析（WAFBypassEngine）

**觸發條件**: 檢測到 WAF 時

```python
waf_payloads = self.generate_waf_bypass_payloads(
    vuln_type="sqli",
    waf_type="cloudflare"
)
```

**生成技術**:
- 編碼變換（URL/Base64/Unicode）
- 大小寫混淆
- 註釋插入
- 協議層繞過

---

### 階段 3: 多源融合決策

**入口**: `make_enhanced_decision()` line 973

```python
decision = await self.make_decision(context)
```

**決策流程**:

```
make_decision()
    ↓
├─ 風險評估決策 (_assess_risk_decision)
│   ├─ CRITICAL → STOP_OPERATION
│   ├─ HIGH → SWITCH_MODE (需要用戶確認)
│   └─ MEDIUM/LOW → 繼續
│
├─ 經驗驅動決策 (_make_experience_driven_decision)
│   ├─ 查詢相似成功案例
│   ├─ 計算相似度 > 0.6
│   └─ 返回成功率 > 0.8 的策略
│
├─ 神經網路決策 (neural_engine)
│   ├─ 5M 參數模型
│   ├─ 輸入: 目標特徵 + 環境上下文
│   └─ 輸出: 行動建議 + 信心度
│
└─ 規則引擎決策 (_make_rule_based_decision)
    └─ 基於預定義規則返回默認策略
```

**融合策略**:
1. 優先級: 風險評估 > 經驗 > 神經網路 > 規則
2. 信心度加權平均
3. 多源驗證（交叉確認）

---

### 階段 4: 增強結果輸出

**入口**: `make_enhanced_decision()` line 976

```python
decision.params["enhanced_mode"] = True
decision.params["knowledge_sources"] = [
    "neural_network",
    "experience_db",
    "rule_engine",
    "internal_capabilities",
    "embedded_knowledge"
]
```

**Decision 對象結構**:
```python
{
    "action": "EXECUTE_SCAN",           # 決策行動
    "confidence": 0.87,                 # 信心度
    "reasoning": "High confidence...",  # 推理過程
    "params": {
        "enhanced_mode": True,
        "knowledge_sources": [...],
        "selected_tools": ["sqlmap", "nikto"],
        "scan_depth": "deep",
        "time_limit": 1800
    },
    "alternatives": [                   # 備選方案
        {"action": "QUICK_SCAN", "confidence": 0.65}
    ],
    "risk_assessment": {                # 風險評估
        "level": "MEDIUM",
        "factors": ["unknown_tech_stack", "no_rate_limit"]
    }
}
```

---

## 🎯 實際案例：davincisnotebook.blog

假設輸入: `https://davincisnotebook.blog/`

### 階段 1: 內部能力查詢
```
Query: "web vulnerability scan"
Results:
  1. flow_8: unified_executor → web_scan (score: 0.92)
  2. flow_15: recon → web_fingerprint (score: 0.88)
  3. flow_23: xss_scanner → dom_analysis (score: 0.81)
```

### 階段 2: embedded_knowledge 分析

**漏洞檢測**:
```json
{
  "vulnerabilities": [
    {"type": "clickjacking", "confidence": 0.65, "reason": "No X-Frame-Options"},
    {"type": "information_disclosure", "confidence": 0.58, "reason": "Server header exposed"}
  ],
  "risk_score": 5.2
}
```

**架構分析**:
```json
{
  "architecture_type": "wordpress",
  "framework": "wordpress_5.8",
  "security_issues": ["outdated_version", "weak_security_headers"],
  "recommendations": ["Update WordPress", "Enable security headers"]
}
```

**CVE 識別**:
```json
{
  "cves": [
    {
      "id": "CVE-2021-29447",
      "severity": "CRITICAL",
      "description": "WordPress XXE vulnerability",
      "cvss_score": 9.8
    }
  ]
}
```

### 階段 3: 多源融合決策

**風險評估**: MEDIUM（有 CVE 但不是攻擊性掃描，可繼續）

**經驗驅動**: 查到 5 個類似 WordPress 掃描案例，成功率 0.82

**神經網路**: 建議 "EXECUTE_COMPREHENSIVE_SCAN"，信心度 0.85

**最終決策**:
```json
{
  "action": "EXECUTE_SCAN",
  "confidence": 0.84,
  "reasoning": "WordPress site with known CVEs, comprehensive scan recommended",
  "params": {
    "scan_type": "wordpress_full",
    "tools": ["wpscan", "nikto", "sqlmap"],
    "priority": "CVE-2021-29447"
  }
}
```

### 階段 4: 增強輸出

返回完整 Decision 對象，包含：
- ✅ 行動建議: EXECUTE_SCAN
- ✅ 工具選擇: wpscan, nikto, sqlmap
- ✅ 優先級: CVE 驗證優先
- ✅ 知識來源: 5 個（神經網路、經驗、規則、內部能力、embedded_knowledge）
- ✅ 風險評估: MEDIUM
- ✅ 信心度: 84%

---

## 📊 知識源權重

| 知識源 | 權重 | 觸發條件 | 主要用途 |
|--------|------|----------|----------|
| 神經網路 (5M) | 35% | 始終啟用 | 複雜模式識別 |
| 經驗數據庫 | 30% | 有相似案例 | 成功策略複用 |
| embedded_knowledge | 25% | 啟用時 | 專業領域知識 |
| 內部能力庫 | 7% | 有連接器 | 能力匹配 |
| 規則引擎 | 3% | 兜底方案 | 默認策略 |

---

## 🔧 關鍵方法總結

| 方法 | 行號 | 用途 |
|------|------|------|
| `make_enhanced_decision()` | 912 | 主入口，協調所有分析 |
| `query_internal_capabilities()` | - | 查詢內部能力 |
| `analyze_target_vulnerabilities()` | - | 漏洞檢測 |
| `analyze_web_architecture()` | - | 架構識別 |
| `identify_cves()` | - | CVE 匹配 |
| `generate_waf_bypass_payloads()` | - | WAF 繞過 |
| `make_decision()` | - | 多源融合決策 |
| `_assess_risk_decision()` | 1007 | 風險評估 |
| `_make_experience_driven_decision()` | 1027 | 經驗查詢 |
| `_calculate_similarity()` | 1127 | 相似度計算 |

---

## 💡 設計優勢

1. **多源驗證**: 5 個知識源交叉確認，降低誤判
2. **增量學習**: 每次決策結果會被記錄，持續優化
3. **風險優先**: 風險評估具有最高優先級，確保安全
4. **經驗複用**: 成功案例自動應用到相似場景
5. **專業知識**: embedded_knowledge 提供領域專家級分析

---

## 🚀 後續優化方向

1. **實時響應分析**: 目前 response_body/headers 為空，應傳入實際數據
2. **動態權重調整**: 根據歷史準確率動態調整知識源權重
3. **並行分析**: 漏洞檢測、架構分析可並行執行以提升速度
4. **缺失數據處理**: 當某些知識源不可用時的降級策略
5. **決策解釋**: 增強 reasoning 的可解釋性，方便審計

---

**結論**: 認知核心通過 4 階段處理網址，整合 5 個知識源，最終產生高信心度的智能決策，為後續的 Phase0 掃描提供精準指導。
