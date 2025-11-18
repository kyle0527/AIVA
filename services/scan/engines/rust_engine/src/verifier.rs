// API 驗證器模塊 - 驗證檢測到的密鑰是否有效
// 參考 TruffleHog 的驗證機制

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// 驗證結果
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VerificationStatus {
    /// 驗證成功 - 密鑰有效
    Valid,
    /// 驗證失敗 - 密鑰無效
    Invalid,
    /// 無法驗證 - 網絡錯誤或其他問題
    Unknown,
    /// 尚未驗證
    NotVerified,
}

/// 驗證結果詳情
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// 驗證狀態
    pub status: VerificationStatus,
    /// 驗證時間戳
    pub verified_at: u64,
    /// 驗證訊息 (如帳戶名稱、錯誤訊息等)
    pub message: String,
    /// 額外資訊 (如權限範圍、帳戶 ID 等)
    pub metadata: HashMap<String, String>,
}

impl VerificationResult {
    pub fn new(status: VerificationStatus, message: String) -> Self {
        Self {
            status,
            verified_at: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            message,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// 緩存條目
#[derive(Debug, Clone)]
struct CacheEntry {
    result: VerificationResult,
    expires_at: SystemTime,
}

/// API 驗證器
pub struct Verifier {
    /// 驗證結果緩存 (密鑰哈希 -> 驗證結果)
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    /// 緩存過期時間 (秒)
    cache_ttl: Duration,
    /// HTTP 客戶端
    client: reqwest::Client,
}

impl Verifier {
    /// 創建新的驗證器
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            cache_ttl: Duration::from_secs(3600), // 1小時
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
        }
    }

    /// 設置緩存過期時間
    #[allow(dead_code)] // Reserved for future cache configuration
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// 計算密鑰的哈希值 (用於緩存鍵)
    fn hash_secret(&self, secret_type: &str, secret: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        secret_type.hash(&mut hasher);
        secret.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// 從緩存獲取驗證結果
    fn get_cached(&self, secret_type: &str, secret: &str) -> Option<VerificationResult> {
        let hash = self.hash_secret(secret_type, secret);
        let cache = self.cache.lock().unwrap();

        if let Some(entry) = cache.get(&hash) {
            if entry.expires_at > SystemTime::now() {
                return Some(entry.result.clone());
            }
        }
        None
    }

    /// 將驗證結果存入緩存
    fn cache_result(&self, secret_type: &str, secret: &str, result: VerificationResult) {
        let hash = self.hash_secret(secret_type, secret);
        let mut cache = self.cache.lock().unwrap();

        cache.insert(
            hash,
            CacheEntry {
                result: result.clone(),
                expires_at: SystemTime::now() + self.cache_ttl,
            },
        );
    }

    /// 清理過期的緩存條目
    #[allow(dead_code)] // Reserved for future cache management
    pub fn cleanup_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        let now = SystemTime::now();
        cache.retain(|_, entry| entry.expires_at > now);
    }

    /// 驗證密鑰
    pub async fn verify(&self, secret_type: &str, secret: &str) -> VerificationResult {
        // 檢查緩存
        if let Some(cached) = self.get_cached(secret_type, secret) {
            return cached;
        }

        // 根據密鑰類型選擇驗證方法
        let result = match secret_type {
            "AWS Access Key ID" | "AWS Secret Access Key" => self.verify_aws(secret).await,
            "GitHub Personal Access Token" | "GitHub OAuth Access Token" => {
                self.verify_github(secret).await
            }
            "Slack Token" | "Slack Bot Token" | "Slack Webhook URL" => {
                self.verify_slack(secret).await
            }
            "Stripe API Key" | "Stripe Restricted API Key" => self.verify_stripe(secret).await,
            "Twilio API Key" => self.verify_twilio(secret).await,
            "SendGrid API Token" => self.verify_sendgrid(secret).await,
            "Mailgun API Key" => self.verify_mailgun(secret).await,
            "DigitalOcean Access Token" => self.verify_digitalocean(secret).await,
            "Cloudflare API Token" => self.verify_cloudflare(secret).await,
            "Datadog API Key" => self.verify_datadog(secret).await,
            _ => {
                // 不支持的密鑰類型
                VerificationResult::new(
                    VerificationStatus::NotVerified,
                    format!("Verification not supported for type: {}", secret_type),
                )
            }
        };

        // 緩存結果
        self.cache_result(secret_type, secret, result.clone());
        result
    }

    /// 驗證 AWS 憑證
    async fn verify_aws(&self, _secret: &str) -> VerificationResult {
        // AWS STS GetCallerIdentity API
        // 注意: 這需要同時有 Access Key ID 和 Secret Access Key
        // 這裡簡化處理,實際應該傳入完整憑證

        VerificationResult::new(
            VerificationStatus::NotVerified,
            "AWS verification requires both Access Key ID and Secret Access Key".to_string(),
        )
    }

    /// 驗證 GitHub Token
    async fn verify_github(&self, token: &str) -> VerificationResult {
        let url = "https://api.github.com/user";

        match self
            .client
            .get(url)
            .header("Authorization", format!("token {}", token))
            .header("User-Agent", "AIVA-Security-Scanner")
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    // 嘗試解析響應獲取用戶資訊
                    if let Ok(body) = response.json::<serde_json::Value>().await {
                        let username = body["login"].as_str().unwrap_or("unknown");
                        VerificationResult::new(
                            VerificationStatus::Valid,
                            format!("Valid GitHub token for user: {}", username),
                        )
                        .with_metadata("username".to_string(), username.to_string())
                    } else {
                        VerificationResult::new(
                            VerificationStatus::Valid,
                            "Valid GitHub token".to_string(),
                        )
                    }
                } else if response.status().as_u16() == 401 {
                    VerificationResult::new(
                        VerificationStatus::Invalid,
                        "Invalid GitHub token (401 Unauthorized)".to_string(),
                    )
                } else {
                    VerificationResult::new(
                        VerificationStatus::Unknown,
                        format!("GitHub API returned status: {}", response.status()),
                    )
                }
            }
            Err(e) => VerificationResult::new(
                VerificationStatus::Unknown,
                format!("Network error: {}", e),
            ),
        }
    }

    /// 驗證 Slack Token
    async fn verify_slack(&self, token: &str) -> VerificationResult {
        let url = "https://slack.com/api/auth.test";

        match self
            .client
            .post(url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
        {
            Ok(response) => {
                if let Ok(body) = response.json::<serde_json::Value>().await {
                    let ok = body["ok"].as_bool().unwrap_or(false);
                    if ok {
                        let team = body["team"].as_str().unwrap_or("unknown");
                        let user = body["user"].as_str().unwrap_or("unknown");
                        VerificationResult::new(
                            VerificationStatus::Valid,
                            format!("Valid Slack token for team: {}, user: {}", team, user),
                        )
                        .with_metadata("team".to_string(), team.to_string())
                        .with_metadata("user".to_string(), user.to_string())
                    } else {
                        let error = body["error"].as_str().unwrap_or("unknown");
                        VerificationResult::new(
                            VerificationStatus::Invalid,
                            format!("Invalid Slack token: {}", error),
                        )
                    }
                } else {
                    VerificationResult::new(
                        VerificationStatus::Unknown,
                        "Failed to parse Slack API response".to_string(),
                    )
                }
            }
            Err(e) => VerificationResult::new(
                VerificationStatus::Unknown,
                format!("Network error: {}", e),
            ),
        }
    }

    /// 驗證 Stripe API Key
    async fn verify_stripe(&self, key: &str) -> VerificationResult {
        let url = "https://api.stripe.com/v1/balance";

        match self.client.get(url).basic_auth(key, Some("")).send().await {
            Ok(response) => {
                if response.status().is_success() {
                    VerificationResult::new(
                        VerificationStatus::Valid,
                        "Valid Stripe API key".to_string(),
                    )
                } else if response.status().as_u16() == 401 {
                    VerificationResult::new(
                        VerificationStatus::Invalid,
                        "Invalid Stripe API key (401 Unauthorized)".to_string(),
                    )
                } else {
                    VerificationResult::new(
                        VerificationStatus::Unknown,
                        format!("Stripe API returned status: {}", response.status()),
                    )
                }
            }
            Err(e) => VerificationResult::new(
                VerificationStatus::Unknown,
                format!("Network error: {}", e),
            ),
        }
    }

    /// 驗證 Twilio API Key
    async fn verify_twilio(&self, _key: &str) -> VerificationResult {
        // Twilio 需要 Account SID 和 Auth Token
        // 這裡簡化處理
        VerificationResult::new(
            VerificationStatus::NotVerified,
            "Twilio verification requires Account SID and Auth Token".to_string(),
        )
    }

    /// 驗證 SendGrid API Token
    async fn verify_sendgrid(&self, token: &str) -> VerificationResult {
        let url = "https://api.sendgrid.com/v3/scopes";

        match self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    if let Ok(body) = response.json::<serde_json::Value>().await {
                        let scopes = body["scopes"].as_array().map(|arr| arr.len()).unwrap_or(0);
                        VerificationResult::new(
                            VerificationStatus::Valid,
                            format!("Valid SendGrid token with {} scopes", scopes),
                        )
                    } else {
                        VerificationResult::new(
                            VerificationStatus::Valid,
                            "Valid SendGrid token".to_string(),
                        )
                    }
                } else if response.status().as_u16() == 401 {
                    VerificationResult::new(
                        VerificationStatus::Invalid,
                        "Invalid SendGrid token (401 Unauthorized)".to_string(),
                    )
                } else {
                    VerificationResult::new(
                        VerificationStatus::Unknown,
                        format!("SendGrid API returned status: {}", response.status()),
                    )
                }
            }
            Err(e) => VerificationResult::new(
                VerificationStatus::Unknown,
                format!("Network error: {}", e),
            ),
        }
    }

    /// 驗證 Mailgun API Key
    async fn verify_mailgun(&self, _key: &str) -> VerificationResult {
        // Mailgun 需要域名資訊
        // 這裡簡化處理
        VerificationResult::new(
            VerificationStatus::NotVerified,
            "Mailgun verification requires domain information".to_string(),
        )
    }

    /// 驗證 DigitalOcean Access Token
    async fn verify_digitalocean(&self, token: &str) -> VerificationResult {
        let url = "https://api.digitalocean.com/v2/account";

        match self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
        {
            Ok(response) => {
                if response.status().is_success() {
                    if let Ok(body) = response.json::<serde_json::Value>().await {
                        let email = body["account"]["email"].as_str().unwrap_or("unknown");
                        VerificationResult::new(
                            VerificationStatus::Valid,
                            format!("Valid DigitalOcean token for account: {}", email),
                        )
                        .with_metadata("email".to_string(), email.to_string())
                    } else {
                        VerificationResult::new(
                            VerificationStatus::Valid,
                            "Valid DigitalOcean token".to_string(),
                        )
                    }
                } else if response.status().as_u16() == 401 {
                    VerificationResult::new(
                        VerificationStatus::Invalid,
                        "Invalid DigitalOcean token (401 Unauthorized)".to_string(),
                    )
                } else {
                    VerificationResult::new(
                        VerificationStatus::Unknown,
                        format!("DigitalOcean API returned status: {}", response.status()),
                    )
                }
            }
            Err(e) => VerificationResult::new(
                VerificationStatus::Unknown,
                format!("Network error: {}", e),
            ),
        }
    }

    /// 驗證 Cloudflare API Token
    async fn verify_cloudflare(&self, token: &str) -> VerificationResult {
        let url = "https://api.cloudflare.com/client/v4/user/tokens/verify";

        match self
            .client
            .get(url)
            .header("Authorization", format!("Bearer {}", token))
            .send()
            .await
        {
            Ok(response) => {
                if let Ok(body) = response.json::<serde_json::Value>().await {
                    let success = body["success"].as_bool().unwrap_or(false);
                    if success {
                        let status = body["result"]["status"].as_str().unwrap_or("unknown");
                        VerificationResult::new(
                            VerificationStatus::Valid,
                            format!("Valid Cloudflare token (status: {})", status),
                        )
                    } else {
                        VerificationResult::new(
                            VerificationStatus::Invalid,
                            "Invalid Cloudflare token".to_string(),
                        )
                    }
                } else {
                    VerificationResult::new(
                        VerificationStatus::Unknown,
                        "Failed to parse Cloudflare API response".to_string(),
                    )
                }
            }
            Err(e) => VerificationResult::new(
                VerificationStatus::Unknown,
                format!("Network error: {}", e),
            ),
        }
    }

    /// 驗證 Datadog API Key
    async fn verify_datadog(&self, key: &str) -> VerificationResult {
        let url = "https://api.datadoghq.com/api/v1/validate";

        match self.client.get(url).header("DD-API-KEY", key).send().await {
            Ok(response) => {
                if let Ok(body) = response.json::<serde_json::Value>().await {
                    let valid = body["valid"].as_bool().unwrap_or(false);
                    if valid {
                        VerificationResult::new(
                            VerificationStatus::Valid,
                            "Valid Datadog API key".to_string(),
                        )
                    } else {
                        VerificationResult::new(
                            VerificationStatus::Invalid,
                            "Invalid Datadog API key".to_string(),
                        )
                    }
                } else {
                    VerificationResult::new(
                        VerificationStatus::Unknown,
                        "Failed to parse Datadog API response".to_string(),
                    )
                }
            }
            Err(e) => VerificationResult::new(
                VerificationStatus::Unknown,
                format!("Network error: {}", e),
            ),
        }
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result_creation() {
        let result = VerificationResult::new(VerificationStatus::Valid, "Test message".to_string());
        assert_eq!(result.status, VerificationStatus::Valid);
        assert_eq!(result.message, "Test message");
        assert!(result.verified_at > 0);
    }

    #[test]
    fn test_verification_result_with_metadata() {
        let result = VerificationResult::new(VerificationStatus::Valid, "Test".to_string())
            .with_metadata("key1".to_string(), "value1".to_string())
            .with_metadata("key2".to_string(), "value2".to_string());

        assert_eq!(result.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(result.metadata.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_verifier_hash_secret() {
        let verifier = Verifier::new();
        let hash1 = verifier.hash_secret("GitHub", "token123");
        let hash2 = verifier.hash_secret("GitHub", "token123");
        let hash3 = verifier.hash_secret("GitHub", "token456");

        // 相同輸入應產生相同哈希
        assert_eq!(hash1, hash2);
        // 不同輸入應產生不同哈希
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_cache_operations() {
        let verifier = Verifier::new();
        let result =
            VerificationResult::new(VerificationStatus::Valid, "Cached result".to_string());

        // 初始應該沒有緩存
        assert!(verifier.get_cached("GitHub", "token123").is_none());

        // 存入緩存
        verifier.cache_result("GitHub", "token123", result.clone());

        // 應該能取得緩存
        let cached = verifier.get_cached("GitHub", "token123");
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().message, "Cached result");
    }

    #[tokio::test]
    async fn test_verify_unsupported_type() {
        let verifier = Verifier::new();
        let result = verifier.verify("Unknown Type", "secret123").await;
        assert_eq!(result.status, VerificationStatus::NotVerified);
        assert!(result.message.contains("not supported"));
    }

    #[tokio::test]
    async fn test_verify_with_cache() {
        let verifier = Verifier::new();

        // 第一次驗證 (不支援的類型,會被緩存)
        let result1 = verifier.verify("Unknown", "secret123").await;

        // 第二次驗證應該從緩存取得
        let result2 = verifier.verify("Unknown", "secret123").await;

        // 時間戳應該相同 (來自緩存)
        assert_eq!(result1.verified_at, result2.verified_at);
    }
}
