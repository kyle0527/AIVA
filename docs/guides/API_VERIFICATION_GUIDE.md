# API 驗證功能文檔

## 概述

API 驗證功能參考 [TruffleHog](https://github.com/trufflesecurity/trufflehog) 的實現，用於驗證檢測到的密鑰和憑證是否有效。這有助於：

1. **減少誤報**: 確認檢測到的密鑰是真實有效的
2. **優先級排序**: 將有效密鑰標記為高優先級處理
3. **風險評估**: 了解哪些洩露的密鑰構成實際安全威脅

## 架構設計

### 核心組件

```
verifier.rs
├── VerificationStatus    # 驗證狀態枚舉
├── VerificationResult    # 驗證結果結構
├── CacheEntry           # 緩存條目
└── Verifier             # 驗證器主類
    ├── verify()         # 主驗證方法
    ├── verify_aws()     # AWS 憑證驗證
    ├── verify_github()  # GitHub Token 驗證
    ├── verify_slack()   # Slack Token 驗證
    └── ... (其他服務)
```

### 驗證流程

```mermaid
graph TD
    A[SecretDetector 檢測到密鑰] --> B{是否為高優先級?}
    B -->|CRITICAL/HIGH| C[檢查緩存]
    B -->|MEDIUM/LOW| D[跳過驗證]
    C -->|緩存命中| E[返回緩存結果]
    C -->|緩存未命中| F[調用 API 驗證]
    F --> G{API 響應}
    G -->|成功 200| H[標記為 Valid]
    G -->|失敗 401| I[標記為 Invalid]
    G -->|其他錯誤| J[標記為 Unknown]
    H --> K[存入緩存]
    I --> K
    J --> K
    K --> L[返回驗證結果]
    E --> L
    D --> L
    L --> M[附加到 Finding]
```

## 驗證狀態

### VerificationStatus 枚舉

| 狀態 | 說明 | 示例 |
|------|------|------|
| `Valid` | 密鑰有效，可正常使用 | API 返回 200 OK |
| `Invalid` | 密鑰無效或已撤銷 | API 返回 401 Unauthorized |
| `Unknown` | 無法驗證（網絡錯誤等） | 連接超時、DNS 錯誤 |
| `NotVerified` | 未進行驗證 | 不支持的密鑰類型、低優先級 |

## 支持的服務

### 已實現驗證 (10 個服務)

| 服務 | API 端點 | 驗證方式 | 元數據 |
|------|----------|----------|--------|
| **GitHub** | `GET /user` | Bearer Token | username |
| **Slack** | `POST /api/auth.test` | Bearer Token | team, user |
| **Stripe** | `GET /v1/balance` | Basic Auth | - |
| **SendGrid** | `GET /v3/scopes` | Bearer Token | scopes |
| **DigitalOcean** | `GET /v2/account` | Bearer Token | email |
| **Cloudflare** | `GET /client/v4/user/tokens/verify` | Bearer Token | status |
| **Datadog** | `GET /api/v1/validate` | Custom Header | - |
| **AWS** | *未完整實現* | STS GetCallerIdentity | account_id, arn |
| **Twilio** | *未完整實現* | - | - |
| **Mailgun** | *未完整實現* | - | - |

### 驗證示例

#### GitHub Token 驗證

```rust
// 檢測到 GitHub Token
let token = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";

// 驗證
let result = verifier.verify("GitHub Personal Access Token", token).await;

// 結果
match result.status {
    VerificationStatus::Valid => {
        println!("✓ 有效 Token");
        println!("用戶: {}", result.metadata.get("username").unwrap());
    }
    VerificationStatus::Invalid => {
        println!("✗ 無效 Token (已撤銷或錯誤)");
    }
    VerificationStatus::Unknown => {
        println!("? 無法驗證 (網絡問題)");
    }
}
```

#### Slack Token 驗證

```rust
let token = "xoxb-xxxxxxxxxxxx-xxxxxxxxxxxx-xxxxxxxxxxxxxxxxxxxxxxxx";
let result = verifier.verify("Slack Bot Token", token).await;

if result.status == VerificationStatus::Valid {
    println!("Team: {}", result.metadata.get("team").unwrap());
    println!("User: {}", result.metadata.get("user").unwrap());
}
```

## 緩存機制

### 目的

- **避免重複驗證**: 相同密鑰在緩存期內不重複調用 API
- **提高性能**: 減少網絡請求延遲
- **降低 API 成本**: 減少計費 API 調用次數

### 實現細節

```rust
// 緩存配置
let verifier = Verifier::new()
    .with_cache_ttl(Duration::from_secs(3600)); // 1 小時

// 緩存鍵計算
fn hash_secret(&self, secret_type: &str, secret: &str) -> String {
    // 使用 DefaultHasher 計算哈希
    // 格式: hash(secret_type + secret)
}

// 緩存清理
verifier.cleanup_cache(); // 手動清理過期條目
```

### 緩存策略

| 參數 | 默認值 | 說明 |
|------|--------|------|
| TTL | 3600 秒 (1 小時) | 緩存過期時間 |
| 鍵格式 | `hash(type + secret)` | 基於密鑰類型和值的哈希 |
| 存儲 | `HashMap` + `Mutex` | 線程安全的內存緩存 |

## 驗證結果結構

```rust
pub struct VerificationResult {
    pub status: VerificationStatus,           // 驗證狀態
    pub verified_at: u64,                    // 驗證時間戳 (Unix)
    pub message: String,                     // 驗證訊息
    pub metadata: HashMap<String, String>,   // 額外資訊
}
```

### 元數據示例

**GitHub**:
```json
{
  "username": "octocat"
}
```

**Slack**:
```json
{
  "team": "ACME Corp",
  "user": "U01234567"
}
```

**DigitalOcean**:
```json
{
  "email": "user@example.com"
}
```

## 整合到掃描流程

### Finding 結構擴展

```rust
struct Finding {
    // ... 原有欄位
    verified: Option<bool>,                           // 驗證狀態
    verification_message: Option<String>,             // 驗證訊息
    verification_metadata: Option<HashMap<String, String>>, // 元數據
}
```

### 掃描邏輯

```rust
// 僅驗證高優先級密鑰
let should_verify = matches!(
    finding.severity.as_str(),
    "CRITICAL" | "HIGH"
);

if should_verify {
    let result = verifier.verify(&finding.rule_name, &finding.matched_text).await;
    
    finding.verified = match result.status {
        VerificationStatus::Valid => Some(true),
        VerificationStatus::Invalid => Some(false),
        _ => None,
    };
    
    finding.verification_message = Some(result.message);
    finding.verification_metadata = Some(result.metadata);
}
```

## 性能考量

### 驗證策略

1. **選擇性驗證**: 僅驗證 `CRITICAL` 和 `HIGH` 級別密鑰
2. **異步執行**: 使用 `tokio` 異步運行時，不阻塞掃描流程
3. **超時控制**: HTTP 請求 10 秒超時
4. **緩存優化**: 1 小時緩存減少重複請求

### 性能指標

| 指標 | 值 | 說明 |
|------|------|------|
| API 超時 | 10 秒 | reqwest 客戶端配置 |
| 緩存 TTL | 3600 秒 | 1 小時過期 |
| 並發度 | 異步 | 使用 tokio runtime |
| 內存開銷 | ~1KB/密鑰 | 緩存條目大小估算 |

## 錯誤處理

### 網絡錯誤

```rust
Err(e) => VerificationResult::new(
    VerificationStatus::Unknown,
    format!("Network error: {}", e),
)
```

### API 錯誤

```rust
if response.status().as_u16() == 401 {
    VerificationStatus::Invalid
} else {
    VerificationStatus::Unknown
}
```

### 不支持的類型

```rust
_ => VerificationResult::new(
    VerificationStatus::NotVerified,
    format!("Verification not supported for type: {}", secret_type),
)
```

## 擴展指南

### 添加新服務驗證

1. **在 `verify()` 方法中添加匹配分支**:

```rust
"New Service Token" => {
    self.verify_new_service(secret).await
}
```

2. **實現驗證方法**:

```rust
async fn verify_new_service(&self, token: &str) -> VerificationResult {
    let url = "https://api.newservice.com/verify";
    
    match self.client
        .get(url)
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await
    {
        Ok(response) => {
            if response.status().is_success() {
                VerificationResult::new(
                    VerificationStatus::Valid,
                    "Valid token".to_string(),
                )
            } else {
                VerificationResult::new(
                    VerificationStatus::Invalid,
                    "Invalid token".to_string(),
                )
            }
        }
        Err(e) => VerificationResult::new(
            VerificationStatus::Unknown,
            format!("Network error: {}", e),
        ),
    }
}
```

3. **添加測試**:

```rust
#[tokio::test]
async fn test_verify_new_service() {
    let verifier = Verifier::new();
    let result = verifier.verify("New Service Token", "test_token").await;
    // 斷言...
}
```

## 測試

### 單元測試覆蓋

| 測試項 | 測試數量 | 說明 |
|--------|----------|------|
| 基礎功能 | 5 | 結構創建、哈希、緩存操作 |
| 驗證邏輯 | 3 | 不支持類型、緩存命中、異步驗證 |
| **總計** | **8** | 覆蓋核心功能 |

### 運行測試

```bash
cd services/scan/info_gatherer_rust
cargo test verifier
```

### 測試示例

```rust
#[tokio::test]
async fn test_verify_with_cache() {
    let verifier = Verifier::new();
    
    // 第一次驗證
    let result1 = verifier.verify("Unknown", "secret123").await;
    
    // 第二次驗證應該從緩存取得
    let result2 = verifier.verify("Unknown", "secret123").await;
    
    // 時間戳應該相同 (來自緩存)
    assert_eq!(result1.verified_at, result2.verified_at);
}
```

## 最佳實踐

### 1. 選擇性驗證

僅驗證高風險密鑰，避免過多 API 調用：

```rust
let should_verify = matches!(
    finding.severity.as_str(),
    "CRITICAL" | "HIGH"
);
```

### 2. 緩存管理

定期清理過期緩存：

```rust
// 在適當的時機調用
tokio::spawn(async move {
    let mut interval = tokio::time::interval(Duration::from_secs(3600));
    loop {
        interval.tick().await;
        verifier.cleanup_cache();
    }
});
```

### 3. 錯誤處理

區分 Invalid (確定無效) 和 Unknown (無法確定)：

```rust
match result.status {
    VerificationStatus::Invalid => {
        // 確定無效，但可能仍需處理 (如報告)
        log_invalid_secret(finding);
    }
    VerificationStatus::Unknown => {
        // 無法驗證，保守處理 (視為潛在威脅)
        log_unverified_secret(finding);
    }
}
```

### 4. 元數據利用

使用元數據提供更多上下文：

```rust
if let Some(username) = result.metadata.get("username") {
    println!("Alert: Token for user '{}' exposed!", username);
}
```

## 安全考量

### 1. 密鑰處理

- ❌ **不要記錄完整密鑰**: 日誌中使用遮罩 (如 `ghp_****...****`)
- ✓ **使用哈希存儲**: 緩存鍵使用哈希而非明文

### 2. 網絡安全

- ✓ **HTTPS Only**: 所有 API 調用使用 HTTPS
- ✓ **超時控制**: 10 秒超時防止掛起
- ✓ **錯誤處理**: 不洩露敏感資訊

### 3. 緩存安全

- ✓ **內存緩存**: 不持久化到磁碟
- ✓ **自動過期**: 1 小時 TTL 自動清理
- ✓ **線程安全**: 使用 `Arc<Mutex<>>` 保護

## 依賴項

### Cargo.toml

```toml
[dependencies]
# HTTP 客戶端
reqwest = { version = "0.11", features = ["json"] }

# 異步運行時
tokio = { version = "1.35", features = ["full"] }

# 序列化
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## 未來改進

### 短期 (1-2 週)

- [ ] 完成 AWS STS GetCallerIdentity 驗證
- [ ] 實現 Twilio 完整驗證
- [ ] 實現 Mailgun 完整驗證
- [ ] 添加 Azure 服務主體驗證
- [ ] 添加 GitLab Token 驗證

### 中期 (1 個月)

- [ ] 實現批量驗證 API (減少網絡往返)
- [ ] 添加 Redis 分佈式緩存支持
- [ ] 實現驗證結果持久化
- [ ] 添加驗證統計和指標

### 長期 (3 個月)

- [ ] 機器學習模型預測密鑰有效性
- [ ] 主動撤銷檢測 (定期重新驗證)
- [ ] 整合更多服務 (npm, Docker Hub, PyPI 等)
- [ ] 實現自定義驗證規則

## 參考資源

- [TruffleHog](https://github.com/trufflesecurity/trufflehog) - 原始靈感來源
- [GitHub API Docs](https://docs.github.com/en/rest) - GitHub API 文檔
- [Slack API Docs](https://api.slack.com/docs) - Slack API 文檔
- [Stripe API Docs](https://stripe.com/docs/api) - Stripe API 文檔

## 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| 1.0.0 | 2025-10-25 | 初始版本，支持 10 個服務驗證 |

## 聯絡資訊

問題或建議請聯繫 AIVA 開發團隊。
