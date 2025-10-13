# AIVA 功能模組強化分析報告
**Function Module Enhancement Analysis**

版本：1.0.0  
分析日期：2025-10-13  
分析者：AIVA Architecture Team

---

## 📑 目錄

1. [現有架構分析](#現有架構分析)
2. [十二個建議模組可行性評估](#十二個建議模組可行性評估)
3. [模組歸屬分類](#模組歸屬分類)
4. [實施優先級與路線圖](#實施優先級與路線圖)

---

## 🏗️ 現有架構分析

### AIVA 四大模組架構

```
┌─────────────────────────────────────────────────────────────┐
│                     AIVA 系統架構                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │  Core    │   │  Scan    │   │ Function │   │Integration│ │
│  │  模組    │   │  模組    │   │  模組    │   │  模組     │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       ↓              ↓              ↓              ↓        │
│  智慧分析      資產發現      漏洞檢測      資料整合        │
│  協調調度      爬蟲引擎      攻擊測試      報告生成        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 現有 Function 模組清單

| 模組名稱 | 語言 | 架構模式 | 狀態 | 檢測能力 |
|---------|------|---------|------|---------|
| **function_sqli** | Python | 策略模式 | ✅ 完整 | 時間盲注、布林盲注、錯誤注入、UNION注入 |
| **function_xss** | Python | 組合模式 | ✅ 完整 | 反射型XSS、儲存型XSS、DOM XSS |
| **function_ssrf** | Python + Go | 雙語言混合 | ✅ 完整 | 內網探測、雲端元數據、協議利用 |
| **function_idor** | Python | 簡單模式 | ✅ 完整 | 橫向越權、縱向提權 |

### 技術債務與改進空間

#### ✅ 優勢
1. **統一數據合約**：所有模組遵循 Pydantic BaseModel
2. **多語言支持**：Python + Go 混合架構已驗證可行
3. **Topic-based 通訊**：RabbitMQ 訊息佇列解耦設計
4. **模組化設計**：各功能獨立部署、獨立擴展

#### ⚠️ 不足
1. **檢測廣度有限**：僅覆蓋 4 種漏洞類型（OWASP Top 10 覆蓋 40%）
2. **靜態分析缺失**：無 SAST 能力，無法分析原始碼漏洞
3. **依賴掃描缺失**：無 SCA 能力，無法檢測第三方庫漏洞
4. **後滲透能力弱**：發現漏洞後無深度利用驗證機制
5. **業務邏輯盲區**：無法檢測價格操縱、工作流程繞過等邏輯漏洞

---

## 🆕 十二個建議模組可行性評估

### 第一批：基礎掃描能力強化（立即可行）

#### ✅ 1. Function-SCA（軟體組成分析）

**目標**：檢測第三方依賴庫的已知漏洞

**可行性評估**：⭐⭐⭐⭐⭐ (5/5)

**現狀分析**：
- ✅ AIVA 已有檔案掃描能力（`info_gatherer/`）
- ✅ 已有 `package.json`, `pyproject.toml`, `go.mod` 解析經驗
- ❌ 尚未整合 OSV/GitHub Advisories API

**建議技術棧**：
- **主要語言**：Go（高併發、OSV-Scanner 整合）
- **次要語言**：Rust（檔案解析效能）
- **核心依賴**：
  - OSV-Scanner (Google)
  - `syft` (容器映像掃描)
  - `cyclonedx-go` (SBOM 生成)

**整合方式**：
```python
# 新增 Topic
Topic.TASK_FUNCTION_SCA = "tasks.function.sca"

# Core 模組觸發邏輯
if asset.type == AssetType.PACKAGE_FILE:
    task = FunctionTaskPayload(
        task_id=new_id("sca"),
        function_type=FunctionType.SCA,
        target=FunctionTaskTarget(url=asset.location)
    )
    publish(Topic.TASK_FUNCTION_SCA, task)
```

**數據合約設計**：
```python
class SCAFindingPayload(FindingPayload):
    package_name: str              # 套件名稱
    package_version: str           # 套件版本
    cve_id: str | None            # CVE 編號
    ghsa_id: str | None           # GitHub Advisory ID
    fixed_version: str | None     # 修復版本
    dependency_path: list[str]    # 依賴鏈路徑
```

---

#### ✅ 2. Function-SAST（靜態應用程式安全測試）

**目標**：分析原始碼 AST，發現資料流漏洞

**可行性評估**：⭐⭐⭐⭐ (4/5)

**現狀分析**：
- ✅ AIVA 已有 `javascript_source_analyzer.py`（簡化版 SAST）
- ✅ Core 模組的 `knowledge_base.py` 使用 Python `ast` 模組
- ❌ 尚未實現完整的污點分析（Taint Analysis）

**建議技術棧**：
- **主要語言**：Rust（效能、tree-sitter 整合）
- **核心依賴**：
  - `tree-sitter`（多語言 AST 解析）
  - `semgrep-core`（規則引擎）
  - `datalog`（資料流分析）

**整合方式**：
```python
# 擴展現有 info_gatherer
class StaticAnalysisEngine:
    async def analyze_source_code(self, file_path: str) -> list[FindingPayload]:
        ast = parse_source(file_path)
        sources = find_taint_sources(ast)  # HTTP 請求參數
        sinks = find_taint_sinks(ast)      # eval(), SQL query
        paths = trace_dataflow(sources, sinks)
        return [self._create_finding(path) for path in paths]
```

**現有程式碼複用**：
- `services/scan/aiva_scan/info_gatherer/javascript_source_analyzer.py`
  - 已有 Sink 檢測邏輯（`eval`, `innerHTML`）
  - 可擴展為完整污點分析

---

#### ✅ 3. Function-CSPM（雲端安全態勢管理）

**目標**：掃描 IaC 檔案與雲端設定錯誤

**可行性評估**：⭐⭐⭐⭐ (4/5)

**現狀分析**：
- ✅ AIVA 已有 Docker 環境（`docker-compose.yml`）
- ✅ 已有檔案掃描能力
- ❌ 尚未整合 Trivy/Checkov

**建議技術棧**：
- **主要語言**：Go（雲端 SDK 生態豐富）
- **核心依賴**：
  - `trivy`（容器與 IaC 掃描）
  - `checkov`（IaC 策略檢查）
  - AWS SDK for Go

**整合方式**：
```python
# Scan 模組發現 IaC 檔案時觸發
if asset.type in [AssetType.DOCKERFILE, AssetType.K8S_YAML]:
    task = FunctionTaskPayload(
        function_type=FunctionType.CSPM,
        target=FunctionTaskTarget(url=asset.location)
    )
```

---

### 第二批：智慧分析能力（需 AI 強化）

#### ⚠️ 4. Module-AttackPath（攻擊路徑分析）

**目標**：圖分析引擎，生成攻擊鏈

**可行性評估**：⭐⭐⭐⭐⭐ (5/5)

**現狀分析**：
- ✅ AIVA 已有 Neo4j（`docker-compose.yml` 包含）
- ✅ Core 模組有 `VulnerabilityCorrelationAnalyzer`（雛形）
- ✅ Integration 模組有完整 Finding 資料庫

**建議技術棧**：
- **主要語言**：Python（Neo4j driver、NetworkX）
- **核心依賴**：
  - `neo4j-driver`
  - `networkx`（圖演算法）
  - `pygraphviz`（視覺化）

**整合方式**：
```python
# Integration 模組新增分析器
class AttackPathEngine:
    def __init__(self, neo4j_session):
        self.graph = neo4j_session
    
    async def analyze_attack_paths(self) -> list[AttackPath]:
        # 1. 建立資產節點
        # 2. 建立漏洞節點
        # 3. 根據可利用性建立邊
        # 4. 計算最短攻擊路徑
        query = """
        MATCH path = (attacker:Node {type:'external'})
                     -[:EXPLOITS*]->(target:Node {type:'database'})
        RETURN path ORDER BY length(path) LIMIT 10
        """
        return self.graph.run(query)
```

**歸屬模組**：**Integration 模組**（資料分析層）

---

#### ✅ 5. Module-AuthN（認證安全測試）

**目標**：暴力破解、JWT 漏洞、MFA 繞過

**可行性評估**：⭐⭐⭐⭐ (4/5)

**現狀分析**：
- ✅ SSRF 模組已有高併發請求能力（Go 實現）
- ✅ Scan 模組能識別登入端點
- ❌ 尚無專門的認證測試邏輯

**建議技術棧**：
- **主要語言**：Go（高併發 HTTP 請求）
- **核心依賴**：
  - `jwt-go`（JWT 解析）
  - `colly`（爬蟲框架）

**整合方式**：
```python
# Core 模組識別認證端點
if asset.category == "authentication":
    task = FunctionTaskPayload(
        function_type=FunctionType.AUTHN,
        target=FunctionTaskTarget(url=asset.url)
    )
```

**歸屬模組**：**Function 模組**（新增 `function_authn_go/`）

---

#### ✅ 6. Module-APISec（API 安全攻擊）

**目標**：BOLA/BFLA/Mass Assignment

**可行性評估**：⭐⭐⭐⭐⭐ (5/5)

**現狀分析**：
- ✅ IDOR 模組已實現 BOLA 檢測
- ✅ Scan 模組能識別 `AJAX_ENDPOINT`, `API_CALL`
- ✅ 可直接擴展為完整 API 安全測試

**建議技術棧**：
- **主要語言**：Python（靈活的 HTTP 庫）
- **輔助語言**：Go（高併發測試）

**整合方式**：
```python
# 擴展現有 function_idor 模組
class APISecTester(CrossUserTester):
    async def test_bfla(self, endpoint: str):
        """測試破碎的函式級授權"""
        admin_methods = ["DELETE", "PUT", "PATCH"]
        for method in admin_methods:
            response = await self.send_as_user(method, endpoint)
            if response.status == 200:
                yield Finding(type=VulnerabilityType.BFLA)
    
    async def test_mass_assignment(self, endpoint: str):
        """測試巨量賦值"""
        payload = {"isAdmin": True, "role": "admin"}
        response = await self.send_as_user("POST", endpoint, json=payload)
```

**歸屬模組**：**Function 模組**（擴展 `function_idor/` 或新建 `function_apisec/`）

---

#### ✅ 7. Module-Secrets（憑證洩漏掃描）

**目標**：掃描 Git 歷史、高熵字串、容器映像

**可行性評估**：⭐⭐⭐⭐⭐ (5/5)

**現狀分析**：
- ✅ AIVA 已有 `SensitiveInfoDetector`（執行期檢測）
- ✅ Rust 模組 `info_gatherer_rust` 已實現正則掃描
- ✅ 可擴展為 Git 歷史掃描

**建議技術棧**：
- **主要語言**：Rust（高效能檔案 I/O）
- **核心依賴**：
  - `regex`, `aho-corasick`
  - `git2`（Git 操作）
  - `entropy`（熵值計算）

**整合方式**：
```rust
// 擴展現有 info_gatherer_rust
pub struct SecretScanner {
    high_entropy_threshold: f64,
    git_repo: Repository,
}

impl SecretScanner {
    pub async fn scan_git_history(&self) -> Vec<Finding> {
        let commits = self.git_repo.log();
        for commit in commits {
            let diff = commit.diff();
            let secrets = self.detect_secrets(diff.content());
            // ...
        }
    }
}
```

**歸屬模組**：**Scan 模組**（擴展 `info_gatherer_rust/`）

---

### 第三批：後滲透與深度利用（高風險）

#### ⚠️ 8. Module-PostEx（漏洞利用與後滲透）

**目標**：SQLi to RCE、SSRF to Port Scan、自動化 Exploit

**可行性評估**：⭐⭐⭐ (3/5) - **需謹慎設計**

**風險警告**：
- ⚠️ 可能造成目標系統實際損害
- ⚠️ 需嚴格的權限控制與審計
- ⚠️ 建議僅在授權環境（沙盒、測試環境）使用

**現狀分析**：
- ✅ SSRF 模組已有內網探測能力
- ✅ SQLi 模組已有時間盲注檢測
- ❌ 尚無實際 RCE 利用能力

**建議技術棧**：
- **主要語言**：Python（pwntools、Metasploit 整合）
- **核心依賴**：
  - `pwntools`
  - `impacket`
  - `nuclei`（PoC 驗證）

**整合方式**：
```python
# Core 模組條件觸發
async def process_function_results(finding: FindingPayload):
    if finding.severity == Severity.CRITICAL and finding.confidence == Confidence.CERTAIN:
        if user_config.enable_postex and target.is_sandbox:
            task = FunctionTaskPayload(
                function_type=FunctionType.POSTEX,
                target=finding.target,
                context={"original_finding": finding}
            )
```

**歸屬模組**：**Function 模組**（新增 `function_postex/`）

**必要控制**：
1. 預設關閉，需用戶明確啟用
2. 記錄所有利用行為到審計日誌
3. 限制僅在標記為「測試環境」的目標上執行

---

### 第四批：智慧與營運導向（需 AI 整合）

#### ⚠️ 9. Module-ThreatIntel（威脅情資整合）

**目標**：整合 CISA KEV、AlienVault OTX，動態調整風險評級

**可行性評估**：⭐⭐⭐⭐⭐ (5/5)

**現狀分析**：
- ✅ Core 模組有 `RiskAssessmentEngine`
- ✅ 已有 HTTP 請求能力
- ❌ 尚未整合外部情資 API

**建議技術棧**：
- **主要語言**：Python（API 整合、資料處理）
- **核心依賴**：
  - `httpx`
  - `pandas`

**整合方式**：
```python
# Core 模組新增情資服務
class ThreatIntelService:
    def __init__(self):
        self.cisa_kev_url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        self.otx_api = "https://otx.alienvault.com/api/v1"
    
    async def is_actively_exploited(self, cve_id: str) -> bool:
        kev_data = await self.fetch_cisa_kev()
        return cve_id in kev_data["vulnerabilities"]
    
    async def enrich_finding(self, finding: FindingPayload) -> FindingPayload:
        if finding.vulnerability.cve_id:
            if await self.is_actively_exploited(finding.vulnerability.cve_id):
                finding.severity = Severity.CRITICAL
                finding.tags.append("actively-exploited")
        return finding
```

**歸屬模組**：**Core 模組**（`analysis/` 目錄）

---

#### ⚠️ 10. Module-Remediation（自動化修復）

**目標**：生成程式碼補丁、建立 Pull Request

**可行性評估**：⭐⭐⭐ (3/5) - **需 LLM 整合**

**現狀分析**：
- ✅ Core 模組有 AI 引擎（`BioNeuronRAGAgent`）
- ❌ 尚無程式碼生成能力
- ❌ 需整合 GitHub API

**建議技術棧**：
- **主要語言**：Python（Git 操作、LLM API）
- **核心依賴**：
  - `gitpython`
  - OpenAI API / Anthropic Claude API

**整合方式**：
```python
# Integration 模組新增修復服務
class RemediationService:
    async def generate_patch(self, finding: FindingPayload) -> str:
        prompt = f"""
        漏洞類型: {finding.vulnerability.type}
        受影響程式碼:
        {finding.evidence.request}
        
        請生成安全的修復程式碼。
        """
        patch = await self.llm.generate(prompt)
        return patch
    
    async def create_pull_request(self, patch: str, finding: FindingPayload):
        repo = git.Repo(finding.target.repository)
        branch = repo.create_head(f"fix/{finding.finding_id}")
        repo.index.apply_patch(patch)
        # Push and create PR via GitHub API
```

**歸屬模組**：**Integration 模組**（`reporting/` 目錄）

---

#### ⚠️ 11. Module-BizLogic（業務邏輯濫用測試）

**目標**：價格操縱、工作流程繞過、推薦機制濫用

**可行性評估**：⭐⭐ (2/5) - **極度依賴 AI**

**現狀分析**：
- ✅ AI 引擎已能理解程式碼語意
- ❌ 需要 AI 理解業務邏輯（極高難度）
- ❌ 需要大量人工定義業務規則

**建議技術棧**：
- **主要語言**：Python（靈活的測試腳本）
- **核心依賴**：
  - LLM API（理解業務流程）
  - `locust`（併發測試）

**整合方式**：
```python
# 用戶自然語言描述 → AI 生成測試
user_input = "測試購物車是否能透過併發請求操縱價格"

test_script = await ai_agent.generate_test_script(user_input)
results = await execute_test_script(test_script)
```

**歸屬模組**：**Function 模組**（新增 `function_bizlogic/`）

**挑戰**：
1. AI 需理解複雜的業務邏輯（電商、金融、社交）
2. 需大量真實案例訓練
3. 誤報率可能極高

---

#### ✅ 12. Module-AuthZ（授權模型繪製）

**目標**：繪製權限矩陣，發現授權異常

**可行性評估**：⭐⭐⭐⭐ (4/5)

**現狀分析**：
- ✅ IDOR 模組已有多角色測試能力
- ✅ Scan 模組能發現所有端點
- ✅ 可擴展為系統化權限矩陣

**建議技術棧**：
- **主要語言**：Python（資料分析、視覺化）
- **核心依賴**：
  - `httpx`
  - `pandas`
  - `matplotlib`（權限矩陣視覺化）

**整合方式**：
```python
# Function 模組新增授權測試器
class AuthorizationMapper:
    def __init__(self, roles: list[Authentication]):
        self.roles = roles  # [admin, user, guest]
    
    async def map_permissions(self, endpoints: list[Asset]) -> PermissionMatrix:
        matrix = defaultdict(dict)
        for role in self.roles:
            for endpoint in endpoints:
                response = await self.test_access(role, endpoint)
                matrix[role.username][endpoint.url] = response.status
        return PermissionMatrix(data=matrix)
    
    def find_anomalies(self, matrix: PermissionMatrix) -> list[Finding]:
        # 檢測異常：普通用戶能訪問管理員端點
        anomalies = []
        for user, permissions in matrix.data.items():
            if user != "admin":
                admin_endpoints = [url for url, status in permissions.items() 
                                  if "/admin/" in url and status == 200]
                if admin_endpoints:
                    anomalies.append(Finding(...))
        return anomalies
```

**歸屬模組**：**Function 模組**（擴展 `function_idor/`）

---

## 📊 模組歸屬分類

### Core 模組（智慧分析與協調）

| 模組編號 | 模組名稱 | 職責 | 優先級 |
|---------|---------|------|--------|
| 4 | Module-AttackPath | 攻擊路徑分析（圖引擎） | P0 |
| 9 | Module-ThreatIntel | 威脅情資整合 | P1 |

**理由**：這些模組是分析與決策層，負責提升整體系統智慧，不直接執行掃描或檢測。

---

### Scan 模組（資產發現與資訊收集）

| 模組編號 | 模組名稱 | 職責 | 優先級 |
|---------|---------|------|--------|
| 2 | Function-SAST | 靜態原始碼分析 | P1 |
| 7 | Module-Secrets | 憑證洩漏掃描（Git 歷史） | P0 |

**理由**：這些模組是被動掃描，不發送攻擊流量，屬於資訊收集階段。

---

### Function 模組（主動漏洞檢測）

| 模組編號 | 模組名稱 | 職責 | 優先級 |
|---------|---------|------|--------|
| 1 | Function-SCA | 軟體組成分析 | P0 |
| 3 | Function-CSPM | IaC 與雲端設定掃描 | P1 |
| 5 | Module-AuthN | 認證安全測試 | P1 |
| 6 | Module-APISec | API 安全攻擊（擴展 IDOR） | P0 |
| 8 | Module-PostEx | 漏洞利用與後滲透 | P2 |
| 11 | Module-BizLogic | 業務邏輯濫用測試 | P3 |
| 12 | Module-AuthZ | 授權模型繪製 | P1 |

**理由**：這些模組主動發送測試請求，執行實際的安全檢測。

---

### Integration 模組（資料整合與報告）

| 模組編號 | 模組名稱 | 職責 | 優先級 |
|---------|---------|------|--------|
| 10 | Module-Remediation | 自動化修復建議 | P2 |

**理由**：這個模組處理檢測結果的後續行動，生成報告與補丁。

---

## 🎯 實施優先級與路線圖

### P0 級（立即實施，3 個月內完成）

#### ✅ Module-APISec（擴展現有 IDOR）
- **時程**：2 週
- **工作量**：小（基於現有程式碼擴展）
- **價值**：高（覆蓋 OWASP API Top 10）
- **技術風險**：低

**實施步驟**：
1. 擴展 `function_idor/cross_user_tester.py`
2. 新增 `bfla_tester.py`（函式級授權）
3. 新增 `mass_assignment_tester.py`
4. 更新數據合約，新增 `VulnerabilityType.BFLA`, `MASS_ASSIGNMENT`

---

#### ✅ Function-SCA（新建 Go 模組）
- **時程**：4 週
- **工作量**：中
- **價值**：極高（自動化依賴掃描）
- **技術風險**：低（整合 OSV-Scanner）

**實施步驟**：
1. 建立 `services/function/function_sca_go/`
2. 整合 Google OSV-Scanner
3. 解析 `package.json`, `pyproject.toml`, `go.mod`, `Cargo.toml`
4. 新增 Topic `TASK_FUNCTION_SCA`
5. Core 模組新增 SCA 任務生成邏輯

---

#### ✅ Module-Secrets（擴展 Rust 掃描器）
- **時程**：3 週
- **工作量**：中
- **價值**：高（防止憑證洩漏）
- **技術風險**：低

**實施步驟**：
1. 擴展 `services/scan/info_gatherer_rust/`
2. 新增 `git2` 依賴，實現 Git 歷史掃描
3. 新增熵值計算模組
4. 整合 `truffleHog` 規則庫

---

#### ✅ Module-AttackPath（新建 Python 分析引擎）
- **時程**：6 週
- **工作量**：大
- **價值**：極高（最有價值的差異化功能）
- **技術風險**：中

**實施步驟**：
1. 在 `services/integration/` 新增 `attack_path_analyzer/`
2. 設計 Neo4j 圖結構（資產節點、漏洞節點、攻擊邊）
3. 實現圖資料導入邏輯
4. 實現最短路徑演算法（Dijkstra）
5. 實現風險評分演算法
6. 建立 API 端點供 AI 代理查詢

---

### P1 級（6 個月內完成）

1. **Function-SAST**（Rust + tree-sitter）- 8 週
2. **Function-CSPM**（Go + Trivy）- 4 週
3. **Module-AuthN**（Go）- 4 週
4. **Module-ThreatIntel**（Python）- 3 週
5. **Module-AuthZ**（Python，擴展 IDOR）- 3 週

---

### P2 級（1 年內完成）

1. **Module-PostEx**（Python，需嚴格控制）- 8 週
2. **Module-Remediation**（需 LLM 整合）- 12 週

---

### P3 級（研究性質，暫緩）

1. **Module-BizLogic**（極度依賴 AI，技術不成熟）

---

## 📈 預期成果

### 完成 P0 級後（3 個月）
- ✅ 漏洞覆蓋率：從 4 種 → 10+ 種
- ✅ OWASP Top 10 覆蓋率：從 40% → 80%
- ✅ 新增能力：
  - 第三方庫漏洞檢測（SCA）
  - API 安全全面測試（APISec）
  - 憑證洩漏檢測（Secrets）
  - 攻擊路徑視覺化（AttackPath）

### 完成 P1 級後（6 個月）
- ✅ 漏洞覆蓋率：15+ 種
- ✅ 新增能力：
  - 原始碼靜態分析（SAST）
  - IaC 安全掃描（CSPM）
  - 認證系統測試（AuthN）
  - 威脅情資整合（ThreatIntel）
  - 授權模型分析（AuthZ）

### 完成 P2 級後（1 年）
- ✅ 深度利用能力（PostEx）
- ✅ 自動化修復建議（Remediation）

---

## 🔧 技術棧總結

### 各模組語言選擇

| 模組 | 語言 | 理由 |
|------|------|------|
| SCA | Go | 高併發 API 請求、OSV-Scanner 整合 |
| SAST | Rust | 效能要求極高、tree-sitter 整合 |
| CSPM | Go | 雲端 SDK 生態豐富 |
| AttackPath | Python | NetworkX、Neo4j driver |
| AuthN | Go | 高併發暴力破解 |
| APISec | Python | 擴展現有 IDOR（已是 Python） |
| Secrets | Rust | 擴展現有 info_gatherer_rust |
| PostEx | Python | pwntools、Metasploit 整合 |
| ThreatIntel | Python | API 整合、資料處理 |
| Remediation | Python | LLM API、Git 操作 |
| BizLogic | Python | 靈活的測試腳本 |
| AuthZ | Python | 擴展現有 IDOR |

### 語言分布統計
- **Python**：6 個模組（APISec, AttackPath, PostEx, ThreatIntel, Remediation, BizLogic, AuthZ）
- **Go**：3 個模組（SCA, CSPM, AuthN）
- **Rust**：2 個模組（SAST, Secrets）

---

## ✅ 結論

### 可立即實施（P0）
1. ✅ **Module-APISec** - 擴展現有 function_idor
2. ✅ **Function-SCA** - 新建 Go 模組
3. ✅ **Module-Secrets** - 擴展 Rust 掃描器
4. ✅ **Module-AttackPath** - 新建 Python 分析引擎

### 需適度調整（P1-P2）
5. ⚠️ **Function-SAST** - 建議先實現簡化版
6. ⚠️ **Module-PostEx** - 需嚴格權限控制
7. ⚠️ **Module-Remediation** - 需 LLM 整合

### 暫緩實施（P3）
8. ❌ **Module-BizLogic** - 技術不成熟，建議觀察業界進展

---

**下一步行動**：
1. 創建詳細的 P0 模組實施計劃
2. 設計各模組的數據合約擴展
3. 準備開發環境與依賴
4. 建立模組開發模板（Go/Rust/Python）

---

**文檔結束**
