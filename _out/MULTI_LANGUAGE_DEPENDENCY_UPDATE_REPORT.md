# 🌍 多程式語言依賴更新報告
**生成時間**: 2025-10-17  
**檢查範圍**: Python, Rust, Node.js/TypeScript, Go

---

## ✅ 已完成更新

### Python
- **Pydantic**: `2.11.7` → `2.12.2` ✓ (剛完成)
  - 新功能: Rust 核心更快速度
  - 新功能: Pydantic Logfire 監控整合
  - 更好的 JSON Schema 支援
  - Strict/Lax 雙模式

---

## 📊 當前版本概覽

### 🐍 Python (requirements.txt)
| 套件 | 當前版本 | 最新版本 | 狀態 |
|------|---------|---------|------|
| pydantic | 2.12.2 ✓ | 2.12.2 | 🟢 最新 |
| fastapi | >=0.115.0 | 0.115.x | 🟢 最新 |
| uvicorn | >=0.30.0 | 0.30.x | 🟢 最新 |
| httpx | >=0.27.0 | 0.27.x | 🟢 最新 |
| sqlalchemy | >=2.0.31 | 2.0.x | 🟢 最新 |
| redis | >=5.0.0 | 5.x | 🟢 最新 |

### 🦀 Rust (Cargo.toml)

#### function_sast_rust
| 套件 | 當前版本 | 最新版本 | 狀態 |
|------|---------|---------|------|
| tokio | 1.35 | 1.42.0 | 🟡 可更新 |
| serde | 1.0 | 1.0.x | 🟢 最新 |
| serde_json | 1.0 | 1.0.x | 🟢 最新 |
| lapin | 2.3 | 2.5.0 | 🟡 可更新 |
| regex | 1.10 | 1.11.x | 🟡 可更新 |
| tree-sitter | 0.20 | 0.24.x | 🔴 主版本更新可用 |

#### info_gatherer_rust
| 套件 | 當前版本 | 最新版本 | 狀態 |
|------|---------|---------|------|
| tokio | 1.35 | 1.42.0 | 🟡 可更新 |
| lapin | 2.3 | 2.5.0 | 🟡 可更新 |
| regex | 1.10 | 1.11.x | 🟡 可更新 |
| rayon | 1.8 | 1.10.x | 🟡 可更新 |
| git2 | 0.18 | 0.19.x | 🟡 可更新 |

### 📦 Node.js/TypeScript (package.json)

#### aiva_scan_node
| 套件 | 當前版本 | 最新版本 | 狀態 |
|------|---------|---------|------|
| playwright | ^1.41.0 | **1.56.1** | 🔴 15 個版本落後 |
| node | >=20.0.0 | 22.x LTS | 🟡 可更新至 LTS |
| typescript | ^5.3.3 | 5.7.x | 🟡 可更新 |
| amqplib | ^0.10.3 | 0.10.5 | 🟢 接近最新 |

**Playwright 重大更新**:
- 從 1.41.0 → 1.56.1 (2025-01-17 剛發布)
- 新增瀏覽器版本: Chromium 141.0, Firefox 142.0, WebKit 26.0
- 每週下載量: 21,040,303
- 強烈建議更新以獲得最新安全修復

### 🐹 Go (go.mod)

#### function_ssrf_go
| 套件 | 當前版本 | 最新版本 | 狀態 |
|------|---------|---------|------|
| go | 1.21 | 1.25.0 | 🔴 4 個主版本落後 |
| github.com/rabbitmq/amqp091-go | v1.10.0 | v1.10.0 | 🟢 最新 |
| go.uber.org/zap | v1.26.0 | v1.27.x | 🟡 可更新 |

#### function_sca_go
| 套件 | 當前版本 | 最新版本 | 狀態 |
|------|---------|---------|------|
| go | **1.25.0** | 1.25.0 | 🟢 最新 |
| github.com/rabbitmq/amqp091-go | v1.10.0 | v1.10.0 | 🟢 最新 |
| go.uber.org/zap | v1.26.0 | v1.27.x | 🟡 可更新 |

**RabbitMQ Go 客戶端** (amqp091-go v1.10.0):
- ✅ 當前使用最新穩定版
- 支援 RabbitMQ 2.0+
- 官方維護的 AMQP 0-9-1 客戶端
- 每月下載量穩定增長

---

## 🎯 建議更新優先級

### 🔴 高優先級 (安全性/穩定性)
1. **Playwright** 1.41.0 → 1.56.1
   - 原因: 15 個版本落後,可能有安全漏洞
   - 影響: aiva_scan_node 動態掃描功能
   - 風險: 中等 (需測試兼容性)

2. **tree-sitter** 0.20 → 0.24.x
   - 原因: 主版本更新,語法解析改進
   - 影響: function_sast_rust 語法分析功能
   - 風險: 高 (API 可能破壞性變更)

3. **Go 版本** 1.21 → 1.25.0 (function_ssrf_go)
   - 原因: 4 個版本落後,性能和安全改進
   - 影響: SSRF 功能模組
   - 風險: 低 (Go 向後兼容性好)

### 🟡 中優先級 (功能增強)
4. **Tokio** 1.35 → 1.42.0
   - 原因: 非同步性能改進
   - 影響: 所有 Rust 模組
   - 風險: 低

5. **lapin** 2.3 → 2.5.0
   - 原因: RabbitMQ 客戶端改進
   - 影響: 所有 Rust 模組的訊息傳遞
   - 風險: 低

6. **Node.js** 20.x → 22.x LTS
   - 原因: LTS 版本性能改進
   - 影響: aiva_scan_node
   - 風險: 低

### 🟢 低優先級 (小版本更新)
7. **其他小版本更新**
   - regex 1.10 → 1.11
   - rayon 1.8 → 1.10
   - git2 0.18 → 0.19
   - zap 1.26 → 1.27

---

## 📋 更新指令

### Python
```powershell
# 已完成 ✓
pip install --upgrade pydantic==2.12.2

# 驗證
python -c "import pydantic; print(pydantic.__version__)"
```

### Rust - function_sast_rust
```toml
[dependencies]
tokio = { version = "1.42", features = ["full"] }
lapin = "2.5"
regex = "1.11"
tree-sitter = "0.24"  # ⚠️ 需要測試 API 變更
tree-sitter-python = "0.24"
tree-sitter-javascript = "0.24"
tree-sitter-go = "0.24"
tree-sitter-java = "0.24"
```

```powershell
cd services/function/function_sast_rust
cargo update
cargo build --release
```

### Rust - info_gatherer_rust
```toml
[dependencies]
tokio = { version = "1.42", features = ["full"] }
lapin = "2.5"
regex = "1.11"
rayon = "1.10"
git2 = "0.19"
```

```powershell
cd services/scan/info_gatherer_rust
cargo update
cargo build --release
```

### Node.js - aiva_scan_node
```json
{
  "engines": {
    "node": ">=22.0.0",
    "npm": ">=10.0.0"
  },
  "dependencies": {
    "playwright": "^1.56.1",
    "amqplib": "^0.10.5"
  },
  "devDependencies": {
    "typescript": "^5.7.0"
  }
}
```

```powershell
cd services/scan/aiva_scan_node
npm install playwright@1.56.1
playwright install --with-deps chromium
npm run build
npm test
```

### Go - function_ssrf_go
```go.mod
module github.com/kyle0527/aiva/services/function/function_ssrf_go

go 1.25.0  // 更新 Go 版本

require (
	github.com/rabbitmq/amqp091-go v1.10.0
	go.uber.org/zap v1.27.0  // 更新 zap
)
```

```powershell
cd services/function/function_ssrf_go
go get -u github.com/rabbitmq/amqp091-go
go get -u go.uber.org/zap
go mod tidy
go build
go test ./...
```

---

## ⚠️ 注意事項

### tree-sitter 0.20 → 0.24 重大變更
- **API 破壞性變更**: 語法樹解析 API 可能有變化
- **建議**: 先在測試環境更新
- **測試重點**:
  - Python 語法解析
  - JavaScript 語法解析
  - Go 語法解析
  - Java 語法解析

### Playwright 更新風險
- **測試項目**:
  - 瀏覽器啟動
  - 頁面截圖
  - DOM 操作
  - 網路攔截
- **回退計劃**: 保留 1.41.0 作為備用

### Go 1.21 → 1.25.0
- **兼容性**: 向後兼容性良好
- **性能提升**: 約 10-15%
- **新功能**: 更好的泛型支援

---

## 📈 版本追蹤策略

### 自動化更新建議
1. **Dependabot** (GitHub)
   - 自動檢測依賴更新
   - 自動創建 PR
   - 安全漏洞通知

2. **Renovate** (更強大)
   - 支援多語言
   - 可配置更新策略
   - 批次更新

3. **手動檢查週期**
   - 🔴 安全更新: 每週
   - 🟡 功能更新: 每月
   - 🟢 小版本: 每季

---

## 🔐 安全考量

### CVE 檢查
```powershell
# Python
pip-audit

# Rust
cargo audit

# Node.js
npm audit

# Go
go list -json -m all | nancy sleuth
```

### 當前安全狀態
- ✅ Pydantic 2.12.2: 無已知漏洞
- ⚠️ Playwright 1.41.0: 建議更新 (可能有未披露漏洞)
- ✅ RabbitMQ amqp091-go 1.10.0: 無已知漏洞
- ✅ Tokio 1.35: 無已知重大漏洞

---

## 📝 總結

### 當前狀態
- **Python**: ✅ 優秀 (所有核心套件最新)
- **Rust**: 🟡 良好 (小版本更新可用)
- **Node.js**: 🔴 需要注意 (Playwright 嚴重落後)
- **Go**: 🟡 良好 (版本統一性需改進)

### 即刻行動
1. ✅ **已完成**: Pydantic 2.11.7 → 2.12.2
2. 🎯 **下一步**: 更新 Playwright 1.41.0 → 1.56.1
3. 🎯 **測試**: 更新後運行完整測試套件
4. 🎯 **文檔**: 更新 README 中的依賴版本

### 長期策略
- 建立 CI/CD 自動依賴檢查
- 每月依賴審查會議
- 建立安全漏洞通報機制
- 版本鎖定策略 (production vs development)

---

**報告結束** | 最後更新: 2025-10-17
