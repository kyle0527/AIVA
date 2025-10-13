# P0 模組 Schema 命名問題分析報告

**生成時間**: 2025-01-13  
**分析範圍**: 所有 P0 模組腳本

---

## 📋 schemas.py 正確定義 (官方規範)

### 1. **Authentication** (認證模型)

```python
class Authentication(BaseModel):
    method: str = "none"
    credentials: dict[str, str] | None = None
```

**❌ 錯誤使用**:

- `auth.bearer_token` → 不存在
- `auth.username` → 不存在
- `auth.password` → 不存在
- `auth.cookies` → 不存在
- `auth.custom_headers` → 不存在

**✅ 正確使用**:

```python
# 使用 credentials 字典
auth.credentials.get("bearer_token")  # 如果 method="bearer"
auth.credentials.get("username")      # 如果 method="basic"
auth.credentials.get("password")      # 如果 method="basic"
```

---

### 2. **Vulnerability** (漏洞模型)

```python
class Vulnerability(BaseModel):
    name: VulnerabilityType    # 枚舉類型
    cwe: str | None = None
    severity: Severity         # 枚舉類型
    confidence: Confidence     # 枚舉類型
```

**❌ 錯誤使用**:

- `vuln.type` → 應為 `vuln.name`
- `vuln.description` → 不存在
- `vuln.cve_id` → 應為 `vuln.cwe`
- `vuln.owasp_category` → 不存在

**✅ 正確使用**:

```python
vuln.name        # VulnerabilityType 枚舉 (SQLI, XSS, IDOR, etc.)
vuln.cwe         # CWE 編號字串 (例如 "CWE-89")
vuln.severity    # Severity 枚舉 (CRITICAL, HIGH, MEDIUM, LOW, INFORMATIONAL)
vuln.confidence  # Confidence 枚舉 (CERTAIN, FIRM, TENTATIVE)
```

---

### 3. **FindingPayload** (發現結果模型)

```python
class FindingPayload(BaseModel):
    finding_id: str
    task_id: str
    scan_id: str
    status: str
    vulnerability: Vulnerability      # 嵌套對象
    target: FindingTarget            # 嵌套對象
    strategy: str | None = None
    evidence: FindingEvidence | None = None
    impact: FindingImpact | None = None
    recommendation: FindingRecommendation | None = None
```

**❌ 錯誤使用**:

- `finding.severity` → 應為 `finding.vulnerability.severity`
- `finding.confidence` → 應為 `finding.vulnerability.confidence`
- `finding.type` → 應為 `finding.vulnerability.name`
- `finding.tags` → 不存在
- `finding.remediation` → 應為 `finding.recommendation.fix`
- `finding.references` → 不存在
- `finding.proof_of_concept` → 應為 `finding.evidence.proof`
- `finding.confidentiality` → 不存在
- `finding.integrity` → 不存在
- `finding.availability` → 不存在

**✅ 正確使用**:

```python
# 漏洞基本信息
finding.vulnerability.name         # VulnerabilityType
finding.vulnerability.severity     # Severity
finding.vulnerability.confidence   # Confidence
finding.vulnerability.cwe          # CWE 編號

# 目標信息
finding.target.url                 # 目標 URL
finding.target.parameter           # 參數名稱
finding.target.method              # HTTP 方法

# 證據信息
finding.evidence.payload           # 測試 payload
finding.evidence.request           # 請求內容
finding.evidence.response          # 響應內容
finding.evidence.proof             # 證明

# 影響與建議
finding.impact.description         # 影響描述
finding.recommendation.fix         # 修復建議
```

---

### 4. **Asset** (資產模型)

```python
class Asset(BaseModel):
    asset_id: str
    type: str
    value: str
    parameters: list[str] | None = None
    has_form: bool = False
```

**❌ 錯誤使用**:

- `asset.url` → 應為 `asset.value`
- `asset.category` → 不存在
- `asset.name` → 不存在

**✅ 正確使用**:

```python
asset.asset_id    # 資產 ID
asset.type        # 資產類型 (例如 "url", "api", "form")
asset.value       # 資產值 (例如 URL 字串)
asset.parameters  # 參數列表
asset.has_form    # 是否包含表單
```

---

## 🔍 各模組錯誤清單

### **Module-APISec (Python)**

#### `bfla_tester.py` - 33 處錯誤

**Authentication 錯誤** (10 處):

```python
# ❌ 行 103-117
if auth.bearer_token:  # 應為 auth.credentials.get("bearer_token")
if auth.username and auth.password:  # 應為 auth.credentials
if auth.cookies:  # 應為從 target 或其他地方獲取
if auth.custom_headers:  # 應為從 target 獲取

# ❌ 行 339-347 (兩處)
Authentication(username="...", password="...", bearer_token="...")
# 應為:
Authentication(method="basic", credentials={"username": "...", "password": "..."})
Authentication(method="bearer", credentials={"bearer_token": "..."})
```

**Vulnerability 創建錯誤** (5 處):

```python
# ❌ 行 245-253
Vulnerability(
    type=VulnerabilityType.BOLA,        # 應為 name=
    description="...",                   # 不存在此參數
    cwe_id="CWE-285",                   # 應為 cwe=
    owasp_category="API1:2023",         # 不存在此參數
    proof_of_concept="...",             # 不存在此參數 (應在 evidence 中)
)

# ✅ 正確寫法:
Vulnerability(
    name=VulnerabilityType.BOLA,
    cwe="CWE-285",
    severity=Severity.HIGH,
    confidence=Confidence.FIRM,
)
```

**FindingPayload 創建錯誤** (15 處):

```python
# ❌ 行 305-316
FindingPayload(
    finding_id="...",
    task_id="...",
    # ❌ 缺少必要參數 scan_id, status
    severity=Severity.HIGH,              # 不存在 (應在 vulnerability 中)
    confidence=Confidence.FIRM,          # 不存在 (應在 vulnerability 中)
    tags=["BFLA"],                       # 不存在此參數
    confidentiality="HIGH",              # 不存在 (應在 impact 中)
    integrity="HIGH",                    # 不存在 (應在 impact 中)
    availability="NONE",                 # 不存在 (應在 impact 中)
    remediation="...",                   # 不存在 (應在 recommendation 中)
    references=[...],                    # 不存在此參數
)

# ✅ 正確寫法:
FindingPayload(
    finding_id="finding_...",
    task_id="task_...",
    scan_id="scan_...",
    status="detected",
    vulnerability=Vulnerability(
        name=VulnerabilityType.BOLA,
        cwe="CWE-285",
        severity=Severity.HIGH,
        confidence=Confidence.FIRM,
    ),
    target=FindingTarget(url=target_url, method=method),
    evidence=FindingEvidence(
        payload=payload,
        request=request_str,
        response=response_str,
        proof=proof_text,
    ),
    impact=FindingImpact(
        description="影響描述",
        business_impact="業務影響",
    ),
    recommendation=FindingRecommendation(
        fix="修復建議",
        priority="HIGH",
    ),
)
```

**其他錯誤** (3 處):

```python
# ❌ 行 206-209
if success:
    return True
else:
    return False
# 建議: return success

# ❌ 行 370
logger.info(f"Severity: {finding.severity}")
# 應為: finding.vulnerability.severity
```

---

#### `mass_assignment_tester.py` - 預估 25+ 處類似錯誤

類似 `bfla_tester.py` 的錯誤模式:

- Authentication 屬性錯誤
- Vulnerability 參數錯誤
- FindingPayload 結構錯誤

---

### **Module-AttackPath (Python)**

#### `engine.py` - 已修正 ✅

所有命名問題已在最新版本中修正:

- ✅ 使用 `asset.value` 而非 `asset.url`
- ✅ 使用 `finding.vulnerability.name` 而非 `finding.vulnerability.type`
- ✅ 使用 `finding.vulnerability.severity` 而非 `finding.severity`
- ✅ 使用 `Severity.INFORMATIONAL` 而非 `Severity.INFO`

---

## 📊 錯誤統計

| 模組 | 檔案 | Authentication | Vulnerability | FindingPayload | Asset | 總計 |
|------|------|---------------|--------------|---------------|-------|------|
| APISec | bfla_tester.py | 10 | 5 | 15 | 0 | 30+ |
| APISec | mass_assignment_tester.py | ~8 | ~4 | ~12 | 0 | ~24 |
| AttackPath | engine.py | 0 | 0 | 0 | 0 | ✅ 已修正 |
| **總計** | | **~18** | **~9** | **~27** | **0** | **~54** |

---

## 🎯 修正優先級

### P0 (立即修正 - 影響功能)

1. **FindingPayload 必要參數缺失**
   - 缺少 `scan_id`, `status` 參數
   - 影響: 無法正常創建 Finding 對象

2. **Vulnerability 創建錯誤**
   - 使用不存在的參數 `type`, `description`, `cwe_id`
   - 影響: Pydantic 驗證失敗,拋出異常

3. **FindingPayload 嵌套結構錯誤**
   - 直接訪問 `finding.severity` 而非 `finding.vulnerability.severity`
   - 影響: AttributeError 運行時錯誤

### P1 (強烈建議 - 影響可維護性)

1. **Authentication 屬性訪問錯誤**
   - 使用不存在的 `auth.bearer_token`, `auth.username` 等
   - 影響: AttributeError 或邏輯錯誤

2. **Asset 屬性錯誤**
   - 使用 `asset.url` 而非 `asset.value`
   - 影響: AttributeError

### P2 (建議改進 - 代碼品質)

1. **Import 排序問題** (Ruff I001)
2. **未使用的 Import** (Ruff F401: `defaultdict`)
3. **簡化布爾返回** (Ruff SIM103)

---

## 🔧 修正模板

### Template 1: Vulnerability 創建

```python
# ❌ 錯誤
Vulnerability(
    type=VulnerabilityType.BOLA,
    description="...",
    cwe_id="CWE-285",
)

# ✅ 正確
Vulnerability(
    name=VulnerabilityType.BOLA,
    cwe="CWE-285",
    severity=Severity.HIGH,
    confidence=Confidence.FIRM,
)
```

### Template 2: FindingPayload 創建

```python
# ❌ 錯誤
FindingPayload(
    finding_id="...",
    task_id="...",
    severity=Severity.HIGH,
    vulnerability=vuln,
)

# ✅ 正確
FindingPayload(
    finding_id="...",
    task_id="...",
    scan_id="...",
    status="detected",
    vulnerability=vuln,
    target=FindingTarget(...),
    evidence=FindingEvidence(...),
    impact=FindingImpact(...),
    recommendation=FindingRecommendation(...),
)
```

### Template 3: Authentication 使用

```python
# ❌ 錯誤
if auth.bearer_token:
    headers["Authorization"] = f"Bearer {auth.bearer_token}"

# ✅ 正確
if auth.method == "bearer" and auth.credentials:
    token = auth.credentials.get("bearer_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
```

---

## 📝 下一步行動

1. **立即修正 bfla_tester.py** (30+ 錯誤)
2. **立即修正 mass_assignment_tester.py** (~24 錯誤)
3. **驗證所有修正** (執行 Pylance 檢查)
4. **更新文檔** (記錄正確的 API 用法)

---

**報告完成時間**: 2025-01-13  
**待修正檔案**: 2 個  
**預估修正時間**: 15-20 分鐘
