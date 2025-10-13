# P0 級模組腳本產出總結報告

生成時間：2025-10-13  
狀態：✅ 全部完成

---

## 📊 產出概覽

已完成 **4 個 P0 級模組**的所有腳本，共 **17 個檔案**：

| 模組編號 | 模組名稱 | 語言 | 檔案數 | 狀態 |
|---------|---------|------|--------|------|
| 1 | **Module-APISec** | Python | 2 | ✅ 完成 |
| 2 | **Function-SCA** | Go | 6 | ✅ 完成 |
| 3 | **Module-Secrets** | Rust | 4 | ✅ 完成 |
| 4 | **Module-AttackPath** | Python | 5 | ✅ 完成 |

---

## 1️⃣ Module-APISec (API 安全攻擊 - Python)

### 產出檔案

```
services/function/function_idor/aiva_func_idor/
├── bfla_tester.py              # BFLA (函式級授權) 測試器
└── mass_assignment_tester.py   # 巨量賦值測試器
```

### 核心功能

#### `bfla_tester.py` (378 行)
- **功能**：檢測普通使用者是否能執行管理員專用的 HTTP 方法
- **測試方法**：DELETE, PUT, PATCH, POST
- **檢測邏輯**：
  - 使用管理員帳號執行請求 → 成功
  - 使用普通使用者帳號執行相同請求 → 應失敗 (403/401)
  - 若普通使用者也成功 → 存在 BFLA 漏洞
- **類別**：
  - `BFLATester`: 主要測試器
  - `BFLATestResult`: 測試結果
- **方法**：
  - `test_endpoint()`: 測試單一端點
  - `batch_test_endpoints()`: 批次測試
  - `create_finding()`: 生成 FindingPayload

#### `mass_assignment_tester.py` (462 行)
- **功能**：檢測應用程式是否接受不應由客戶端控制的欄位
- **危險欄位**：
  - 權限提升：`isAdmin`, `role`, `permissions`
  - 狀態變更：`is_verified`, `status`
  - 敏感資料：`balance`, `price`
- **檢測邏輯**：
  - 發送正常請求（基準）
  - 注入額外欄位（如 `{"isAdmin": true}`）
  - 檢查回應中是否包含注入的欄位
- **類別**：
  - `MassAssignmentTester`: 主要測試器
  - `MassAssignmentPayload`: 測試載荷
  - `MassAssignmentTestResult`: 測試結果

### 整合方式
- 擴展現有 `function_idor/` 模組
- 直接使用現有的 RabbitMQ 訂閱（`tasks.function.idor`）
- 數據合約：`FindingPayload` (VulnerabilityType.BOLA)

---

## 2️⃣ Function-SCA (軟體組成分析 - Go)

### 產出檔案

```
services/function/function_sca_go/
├── cmd/
│   └── worker/
│       └── main.go                    # 主程式入口 (134 行)
├── internal/
│   └── scanner/
│       └── sca_scanner.go             # SCA 掃描器 (329 行)
├── pkg/
│   ├── messaging/
│   │   └── publisher.go               # RabbitMQ 發布器 (56 行)
│   └── models/
│       └── models.go                  # 數據模型 (71 行)
├── go.mod                              # Go 模組定義
└── README.md                           # 完整文檔
```

### 核心功能

#### `main.go`
- **功能**：RabbitMQ 消費者，接收 SCA 掃描任務
- **訊息流程**：
  1. 訂閱 `tasks.function.sca` 佇列
  2. 接收 `FunctionTaskPayload`
  3. 呼叫 `SCAScanner.Scan()`
  4. 發布結果到 `results.finding`

#### `sca_scanner.go`
- **功能**：整合 Google OSV-Scanner，掃描第三方依賴漏洞
- **支援的套件管理檔案**：
  - Node.js: package.json, yarn.lock, pnpm-lock.yaml
  - Python: pyproject.toml, requirements.txt, poetry.lock
  - Go: go.mod, go.sum
  - Rust: Cargo.toml, Cargo.lock
  - Java: pom.xml, build.gradle
  - PHP: composer.json, composer.lock
  - Ruby: Gemfile.lock
- **掃描流程**：
  1. `prepareProject()`: 下載/克隆專案
  2. `detectPackageFiles()`: 偵測套件管理檔案
  3. `scanWithOSV()`: 執行 OSV-Scanner
  4. `convertToFindings()`: 轉換為 FindingPayload
- **風險評分**：根據 CVSS 分數判斷嚴重性

#### `publisher.go`
- **功能**：發布 Finding 到 RabbitMQ
- **Topic**: `results.finding`
- **訊息持久化**: Persistent delivery mode

### 依賴
- `github.com/google/osv-scanner` (需預先安裝)
- `github.com/rabbitmq/amqp091-go v1.9.0`
- `go.uber.org/zap v1.26.0`

### 建置與執行
```bash
cd services/function/function_sca_go
go mod download
go build -o bin/sca-worker cmd/worker/main.go
./bin/sca-worker
```

---

## 3️⃣ Module-Secrets (憑證洩漏掃描 - Rust)

### 產出檔案

```
services/scan/info_gatherer_rust/src/
├── secret_detector.rs           # 憑證檢測器 (324 行)
├── git_history_scanner.rs       # Git 歷史掃描器 (253 行)
└── main.rs                      # 更新模組導入
```

```
services/scan/info_gatherer_rust/
└── Cargo.toml                   # 新增 git2, tempfile 依賴
```

### 核心功能

#### `secret_detector.rs`
- **功能**：掃描原始碼中的硬編碼密鑰和高熵字串
- **檢測規則** (15 種)：
  - AWS Access Key ID / Secret Access Key
  - GitHub Personal Access Token / OAuth Token
  - Slack Token
  - Google API Key
  - Generic API Key / Secret
  - Private Key (RSA/EC/DSA)
  - JWT Token
  - Database Connection String
  - Docker Auth Config
  - NPM Token
  - Stripe API Key
  - Twilio API Key
- **熵值檢測**：
  - 使用 Shannon Entropy 計算字串隨機性
  - 閾值：4.5 (可調整)
  - 最小長度：20 字元
- **類別**：
  - `SecretDetector`: 主要檢測器
  - `SecretRule`: 檢測規則
  - `EntropyDetector`: 熵值計算器
  - `SecretFinding`: 發現結果
- **安全性**：自動遮蔽敏感資訊 (`redact_secret()`)

#### `git_history_scanner.rs`
- **功能**：掃描 Git 提交歷史中的憑證洩漏
- **掃描對象**：
  - 所有提交的差異 (diff)
  - 特定分支
  - 特定檔案的歷史
- **使用 git2 庫**：
  - `Repository::open()`: 開啟儲存庫
  - `revwalk()`: 遍歷提交
  - `diff_tree_to_tree()`: 計算差異
- **類別**：
  - `GitHistoryScanner`: 主要掃描器
  - `GitSecretFinding`: Git 憑證發現（包含提交資訊）
- **方法**：
  - `scan_repository()`: 掃描整個儲存庫
  - `scan_branch()`: 掃描特定分支
  - `scan_file_history()`: 掃描特定檔案歷史

### 新增依賴
```toml
git2 = "0.18"       # Git 操作
tempfile = "3.8"    # 測試用臨時目錄
```

### 整合方式
- 擴展現有 `info_gatherer_rust` 模組
- 可作為獨立掃描器或整合到 `scanner.rs`

---

## 4️⃣ Module-AttackPath (攻擊路徑分析 - Python + Neo4j)

### 產出檔案

```
services/integration/aiva_integration/attack_path_analyzer/
├── __init__.py                 # 套件初始化
├── engine.py                   # 核心引擎 (432 行)
├── graph_builder.py            # 圖資料建構器 (217 行)
├── visualizer.py               # 視覺化工具 (323 行)
└── README.md                   # 完整文檔 (310 行)
```

### 核心功能

#### `engine.py`
- **功能**：使用 Neo4j 建立資產與漏洞的關聯圖
- **圖結構**：
  - **節點類型**：Attacker, Asset, Vulnerability, Database, InternalNetwork, Credential, APIEndpoint
  - **邊類型**：CAN_ACCESS, HAS_VULNERABILITY, LEADS_TO, GRANTS_ACCESS, EXPOSES
- **自動建立攻擊邊**：
  - SSRF → InternalNetwork
  - SQLi → Database
  - IDOR/BOLA → APIEndpoint
  - XSS → Credential
- **類別**：
  - `AttackPathEngine`: 主要引擎
  - `AttackPath`: 攻擊路徑
  - `NodeType`, `EdgeType`: 枚舉
- **方法**：
  - `initialize_graph()`: 初始化圖結構（建立索引、約束、攻擊者節點）
  - `add_asset()`: 新增資產節點
  - `add_finding()`: 新增漏洞並建立攻擊邊
  - `find_attack_paths()`: 尋找攻擊路徑（最短路徑演算法）
  - `find_critical_nodes()`: 尋找關鍵節點（中心性分析）
  - `get_vulnerability_statistics()`: 漏洞統計

#### `graph_builder.py`
- **功能**：從 PostgreSQL 讀取資產與 Findings，建立 Neo4j 圖
- **資料來源**：AIVA Integration 模組的 `assets` 和 `findings` 資料表
- **類別**：
  - `GraphBuilder`: 圖資料建構器
- **方法**：
  - `build_graph_from_database()`: 全量建立圖
  - `rebuild_graph()`: 清空後重建
  - `incremental_update()`: 增量更新

#### `visualizer.py`
- **功能**：將攻擊路徑匯出為視覺化格式
- **輸出格式**：
  1. **Mermaid 流程圖**：Markdown 友善，支援 GitHub
  2. **Cytoscape JSON**：互動式圖表
  3. **HTML 頁面**：內嵌 Cytoscape.js，完全互動式
- **類別**：
  - `AttackPathVisualizer`: 視覺化器
- **方法**：
  - `to_mermaid()`: 生成 Mermaid 語法
  - `to_cytoscape_json()`: 生成 Cytoscape JSON
  - `to_html()`: 生成互動式 HTML

### 依賴
- `neo4j-driver`
- `asyncpg` (從 PostgreSQL 讀取資料)

### Cypher 查詢範例
```cypher
# 尋找最短攻擊路徑
MATCH path = shortestPath(
  (attacker:Attacker {id: 'external_attacker'})-[*]->(target:Database)
)
RETURN path

# 尋找高風險路徑
MATCH path = (attacker:Attacker)-[*1..10]->(target:Database)
WITH path, reduce(risk = 0.0, r in relationships(path) | risk + coalesce(r.risk, 1.0)) as total_risk
WHERE total_risk >= 20.0
RETURN path, total_risk ORDER BY total_risk DESC LIMIT 10
```

### 整合到 AIVA
1. **定期重建圖**（每日 02:00）
2. **即時更新**（接收到新 Finding 時）
3. **API 端點**（`/attack-paths/to-database`）

---

## 📈 技術統計

### 程式碼行數

| 模組 | 語言 | 總行數 | 核心邏輯 | 測試 | 文檔 |
|------|------|--------|---------|------|------|
| APISec | Python | 840 | 720 | 0 | 120 |
| SCA | Go | 590 | 490 | 0 | 100 |
| Secrets | Rust | 577 | 487 | 90 | 0 |
| AttackPath | Python | 1,282 | 972 | 0 | 310 |
| **總計** | - | **3,289** | **2,669** | **90** | **530** |

### 語言分布

```
Python: 2,122 行 (64.5%)
Go:     590 行  (18.0%)
Rust:   577 行  (17.5%)
```

### 依賴新增

#### Python
- `neo4j-driver` (Module-AttackPath)

#### Go
- `github.com/google/osv-scanner` (Function-SCA)

#### Rust
- `git2 = "0.18"` (Module-Secrets)
- `tempfile = "3.8"` (Module-Secrets 測試)

---

## ⚠️ 已知問題與待修正項目

### 1. Go 模組編譯錯誤

**檔案**: `services/function/function_sca_go/cmd/worker/main.go`

**問題**:
```
- "fmt" imported and not used
- could not import github.com/rabbitmq/amqp091-go (需執行 go mod download)
- could not import go.uber.org/zap
```

**修正**:
```bash
cd services/function/function_sca_go
go mod tidy
go mod download
```

### 2. Rust 模組未更新主檔案

**檔案**: `services/scan/info_gatherer_rust/src/main.rs`

**修正**: 已更新，新增模組導入：
```rust
mod secret_detector;
mod git_history_scanner;
```

### 3. Python Lint 警告

**問題**:
- Import 順序不符合 PEP 8
- 未使用的 import
- Trailing whitespace

**修正**: 執行 `ruff check --fix` 或 `black`

### 4. Markdown Lint 警告

**問題**:
- 缺少程式碼語言標記 (MD040)
- Bare URLs (MD034)

**影響**: 僅文檔格式，不影響功能

---

## ✅ 下一步行動

### 立即執行（今日）

1. **修正編譯錯誤**
   ```bash
   # Go 模組
   cd services/function/function_sca_go
   go mod tidy
   go build ./cmd/worker
   
   # Rust 模組
   cd services/scan/info_gatherer_rust
   cargo build --release
   
   # Python 模組
   ruff check --fix services/function/function_idor/aiva_func_idor/
   ruff check --fix services/integration/aiva_integration/attack_path_analyzer/
   ```

2. **安裝外部依賴**
   ```bash
   # 安裝 OSV-Scanner
   go install github.com/google/osv-scanner/cmd/osv-scanner@latest
   
   # 安裝 Neo4j (Docker)
   docker run -d --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/your_password \
     neo4j:latest
   
   # 安裝 Python 依賴
   pip install neo4j-driver asyncpg
   ```

3. **執行單元測試**
   ```bash
   # Python
   pytest services/function/function_idor/aiva_func_idor/
   pytest services/integration/aiva_integration/attack_path_analyzer/
   
   # Rust
   cd services/scan/info_gatherer_rust
   cargo test
   
   # Go
   cd services/function/function_sca_go
   go test ./...
   ```

### 本週執行

4. **整合測試**
   - 測試 Module-APISec 與 function_idor 整合
   - 測試 Function-SCA 的 RabbitMQ 訊息流
   - 測試 Module-Secrets 的 Git 掃描
   - 測試 Module-AttackPath 的圖建立

5. **更新數據合約**
   - 在 `DATA_CONTRACT.md` 新增 SCA 相關欄位
   - 新增 BFLA/Mass Assignment 漏洞類型

6. **更新 Core 模組**
   - 新增 SCA 任務生成邏輯
   - 整合 AttackPath 分析到報告流程

### 下週執行

7. **Docker 化**
   - 建立 Function-SCA Dockerfile
   - 更新 docker-compose.yml

8. **文檔完善**
   - 建立使用手冊
   - 錄製示範影片

---

## 📊 預期效果（3 個月後）

### 漏洞覆蓋率提升

| 指標 | 現狀 | 目標 | 提升 |
|------|------|------|------|
| **漏洞類型** | 4 種 | 10+ 種 | **+150%** |
| **OWASP Top 10 覆蓋** | 40% | 80% | **+100%** |
| **第三方庫掃描** | ❌ | ✅ | **新增** |
| **憑證洩漏檢測** | ❌ | ✅ | **新增** |
| **攻擊路徑視覺化** | ❌ | ✅ | **新增** |

### 檢測能力矩陣

| 檢測類別 | 現有模組 | 新增模組 | 總計 |
|---------|---------|---------|------|
| **Web 漏洞** | XSS, SQLi, SSRF, IDOR | BFLA, Mass Assignment | **6** |
| **API 安全** | IDOR | BFLA, Mass Assignment | **3** |
| **依賴安全** | - | SCA | **1** |
| **憑證洩漏** | - | Secrets Scanner | **1** |
| **攻擊分析** | - | Attack Path | **1** |

---

## 🎯 成功指標

✅ **已完成**:
- [x] 所有 P0 模組腳本產出
- [x] 完整的 README 文檔
- [x] 數據模型設計
- [x] 訊息流設計

⏳ **進行中**:
- [ ] 編譯錯誤修正
- [ ] 依賴安裝
- [ ] 單元測試

📅 **待執行**:
- [ ] 整合測試
- [ ] 效能測試
- [ ] 文檔完善
- [ ] Docker 化

---

## 📝 附註

1. **所有模組遵守 AIVA 數據合約**：使用 Pydantic BaseModel (Python) 或等效結構 (Go/Rust)
2. **所有模組使用 RabbitMQ Topic-based 通訊**
3. **Go 模組路徑**: 使用 `github.com/kyle0527/aiva` 前綴
4. **Rust 模組**: 擴展現有 `info_gatherer_rust`，避免重複建置
5. **Python 模組**: 遵循現有目錄結構

---

**報告結束**  
下一步：執行編譯修正與依賴安裝
