# AIVA Embedded Security Knowledge

> 內嵌式安全知識庫 - AIVA 攻擊決策系統的專業安全知識模組

## 📋 概述

`embedded_knowledge` 是 AIVA 認知核心的關鍵組件，提供了無需外部查詢即可直接調用的專業安全知識。基於以下外部知識文檔設計：

1. **AI 掃描器漏洞判斷邏輯資料庫.md** → `vulnerability_detection.py`
2. **AI 識別高危險 CVE 模組.md** → `cve_identification.py`
3. **WAF 繞過技術字典生成.md** → `waf_bypass.py`
4. **Web 架構安全漏洞檢測指南.md** → `web_architecture.py`

## 🎯 設計理念

### 為什麼選擇 Embedded 而非 RAG?

- **零延遲訪問**: 所有知識直接編碼為 Python 代碼，無需搜索或查詢
- **AI 可讀性**: 結構化數據類型 (dataclass)，便於 AI 決策系統理解
- **確定性輸出**: 每次調用返回一致的結果，避免 RAG 的不確定性
- **離線可用**: 無需網絡連接或外部服務

### 架構原則

1. **結構化知識**: 使用 Enum, dataclass, TypeAlias 等強類型
2. **置信度驅動**: 每個檢測結果包含置信度評分 (0.0-1.0)
3. **可擴展設計**: 預留 `register_*` 方法支援動態擴展
4. **AI 友好**: 所有結果提供 `to_dict()` 方法便於 JSON 序列化

## 📦 模組結構

```
embedded_knowledge/
├── __init__.py              # 統一導出接口
├── base.py                  # 基礎類型 (DetectionResult, ConfidenceLevel, etc.)
├── vulnerability_detection.py  # SQLi/XSS/SSRF/IDOR 檢測
├── cve_identification.py    # 高危 CVE 識別 (Log4Shell, Spring4Shell, etc.)
├── waf_bypass.py           # WAF 繞過技術 (Cloudflare, AWS, Imperva, etc.)
├── web_architecture.py     # 現代架構安全 (GraphQL, JWT, WebSocket, BOLA)
├── USAGE.md                # 詳細使用指南
└── README.md               # 本文件
```

## 🚀 快速開始

### 安裝

模組已內建在 AIVA 中，無需額外安裝。

### 基本使用

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    VulnerabilityDetector,
    CVEIdentifier,
    WAFBypassEngine,
    WebArchitectureAnalyzer,
)

# SQLi 檢測
result = VulnerabilityDetector.check_sqli(
    response_body="You have an error in your SQL syntax",
    response_time=0.15,
)

if result.should_exploit():
    print(f"發現 SQLi 漏洞 (置信度: {result.confidence_score:.0%})")

# CVE 識別
matches = CVEIdentifier.identify(target_fingerprint)
for match in matches:
    if match.is_exploitable():
        print(f"高危 CVE: {match.cve_id} (CVSS: {match.cvss_score})")

# WAF 檢測
is_waf, vendor, _ = WAFBypassEngine.detect_waf(
    response_body="Cloudflare Ray ID: 123",
    response_headers={"cf-ray": "123"},
    status_code=403,
)

# GraphQL 安全
result = WebArchitectureAnalyzer.detect_graphql_introspection(
    endpoint="https://api.example.com/graphql",
    response_data=introspection_response,
)
```

詳細使用方式請參考 [USAGE.md](./USAGE.md)。

## 📊 功能矩陣

### 1. vulnerability_detection.py

| 漏洞類型 | 檢測方法 | 支援數據庫 | 特性 |
|---------|---------|-----------|------|
| SQLi (Error-Based) | `check_sqli()` | MySQL, PostgreSQL, MSSQL, Oracle | 400+ 錯誤指紋 |
| SQLi (Time-Based) | `check_sqli()` | 全部 | 響應時間分析 |
| XSS (Reflected) | `check_xss()` | N/A | 反射檢測 + CSP 檢測 |
| SSRF | `check_ssrf()` | N/A | AWS/GCP/Azure 元數據檢測 |
| IDOR | `check_idor()` | N/A | 成對測試 + 相似度分析 |

**亮點**: 
- 自動 WAF 檢測 (18 種簽名)
- 數據庫指紋識別
- 誤報風險評估

### 2. cve_identification.py

內建 **8 個高危 CVE** (CVSS ≥ 9.0):

| CVE ID | 名稱 | CVSS | 影響 |
|--------|------|------|------|
| CVE-2021-44228 | Log4Shell | 10.0 | Log4j RCE |
| CVE-2022-22965 | Spring4Shell | 9.8 | Spring RCE |
| CVE-2022-26134 | Confluence OGNL | 9.8 | Confluence RCE |
| CVE-2021-34473 | ProxyShell | 9.8 | Exchange RCE |
| CVE-2022-1388 | F5 BIG-IP | 9.8 | F5 Auth Bypass |
| CVE-2023-4966 | Citrix Bleed | 9.4 | Citrix 信息洩露 |
| CVE-2024-23897 | Jenkins CLI | 9.8 | Jenkins RCE |
| CVE-2023-46805 | Ivanti Chain | 9.1 | Ivanti Auth Bypass |

**三層信號架構**:
- **Tier 3** (概率): 技術棧觸發 (如 "java", "log4j")
- **Tier 2** (確定性): Payload 響應驗證
- **Tier 1** (絕對): 漏洞利用成功證據

### 3. waf_bypass.py

支援 **6 大 WAF 廠商**:
- Cloudflare
- AWS WAF
- Imperva (Incapsula)
- Akamai
- ModSecurity
- F5 BIG-IP ASM

**20+ 繞過技術**:
- 編碼混淆 (IBM037, Double URL, Unicode)
- HTTP 協議層 (Chunked Transfer, Header Spoofing)
- 特定廠商 (AWS 8KB 限制, Cloudflare 屬性超載)
- Payload 變形 (SQL 註釋注入, XSS 實體編碼)

**功能**:
- `detect_waf()`: 自動識別 WAF 類型
- `get_bypass_techniques()`: 獲取針對性繞過方法
- `mutate_payload()`: 自動 payload 變形
- `generate_chunked_body()`: 生成分塊編碼

### 4. web_architecture.py

現代 Web 架構安全分析:

| 架構類型 | 檢測能力 | 方法 |
|---------|---------|------|
| GraphQL | Introspection 暴露 | `detect_graphql_introspection()` |
| JWT | None Algorithm, 弱算法 | `analyze_jwt()` |
| REST API | BOLA/IDOR | `check_bola()` |
| WebSocket | 劫持, Origin 繞過 | `check_websocket_security()` |
| 通用 | 架構指紋識別 | `identify_architecture()` |

**JWT 攻擊支援**:
- None algorithm bypass
- Algorithm confusion (RS256 → HS256)
- kid header injection
- jku/jwk injection

## 🧠 與 AI 決策系統整合

### 在 EnhancedDecisionAgent 中使用

```python
class EnhancedDecisionAgent:
    def decide_next_action(self, attack_result: dict) -> dict:
        # 1. 檢測漏洞
        detection = VulnerabilityDetector.check_sqli(
            response_body=attack_result["response"],
            response_time=attack_result["time"],
        )
        
        # 2. AI 可讀的結構化數據
        detection_dict = detection.to_dict()
        
        # 3. 置信度驅動決策
        if detection.should_exploit(risk_threshold=0.8):
            # 4. 檢測 WAF
            is_waf, vendor, _ = WAFBypassEngine.detect_waf(...)
            
            if is_waf:
                # 5. 獲取繞過策略
                bypass_techniques = WAFBypassEngine.get_bypass_techniques(
                    waf_vendor=vendor,
                    attack_type="sqli",
                )
                return {"action": "bypass_waf", "techniques": bypass_techniques}
            
            return {"action": "exploit", "confidence": detection.confidence_score}
        
        return {"action": "try_different_payload"}
```

## 📈 性能指標

| 指標 | 值 |
|-----|-----|
| 響應延遲 | < 1ms (無網絡請求) |
| 內存佔用 | ~8MB (指紋庫 + CVE 數據) |
| 並發安全 | 是 (無狀態 classmethod) |
| SQLi 指紋數 | 400+ |
| WAF 簽名數 | 18 |
| CVE 數量 | 8 (可擴展) |

## 🔧 擴展知識庫

### 動態註冊新知識

```python
from services.core.aiva_core.cognitive_core.embedded_knowledge import (
    KnowledgeRegistry,
    CVEIdentifier,
    WAFBypassEngine,
)

# 註冊 SQLi 指紋
KnowledgeRegistry.register_sqli_fingerprint(
    database=DatabaseType.MYSQL,
    pattern=r"custom error pattern",
)

# 註冊 CVE
CVEIdentifier.register_cve(custom_cve_signature)

# 註冊 WAF 繞過技術
WAFBypassEngine.register_technique(custom_bypass_technique)
```

## 📝 數據流

```
攻擊響應 (HTTP Response)
    ↓
VulnerabilityDetector.check_sqli()
    ↓
DetectionResult (detected=True, confidence=0.95)
    ↓
decision_agent.should_exploit()  ← AI 決策點
    ↓
WAFBypassEngine.detect_waf()
    ↓
bypass_techniques = get_bypass_techniques()
    ↓
mutated_payloads = mutate_payload()
    ↓
執行新一輪攻擊
```

## 🎓 最佳實踐

1. **始終使用 `should_exploit()`**: 避免低置信度誤報
2. **檢查 `false_positive_risk`**: 特別是有 WAF 時
3. **交叉驗證**: 結合多個檢測器提高準確性
4. **序列化結果**: 使用 `to_dict()` 進行日誌記錄
5. **動態擴展**: 通過 `KnowledgeRegistry` 添加新知識

## 🗺️ 未來計劃

### 短期 (v1.1)
- [ ] NoSQL 注入檢測 (MongoDB, Redis)
- [ ] OAuth 2.0 漏洞檢測
- [ ] 更多 CVE 簽名 (目標: 20+)

### 中期 (v1.5)
- [ ] 機器學習模型集成
- [ ] 動態指紋更新機制
- [ ] 性能優化 (緩存機制)

### 長期 (v2.0)
- [ ] 自動化知識學習
- [ ] 多語言支持 (不只是 Python)
- [ ] 分布式知識庫

## 📄 文檔

- [USAGE.md](./USAGE.md) - 詳細使用指南
- [外部知識文檔](../external_knowledge/) - 原始知識來源

## 🤝 貢獻

如需添加新的漏洞檢測邏輯、CVE 或繞過技術，請:

1. 在對應模組中實現檢測邏輯
2. 添加單元測試
3. 更新 USAGE.md 文檔
4. 提交 Pull Request

## 📜 版本歷史

- **v1.0.0** (2026-01-19) - 初始發布
  - 4 個核心模組
  - 8 個高危 CVE
  - 20+ WAF 繞過技術
  - 完整的 Web 架構安全分析

## 📞 支持

如有問題或建議，請聯繫 AIVA 開發團隊。

---

**License**: Internal Use Only  
**Maintainer**: AIVA Security Team  
**Last Updated**: 2026-01-19
