// Default Credential Spraying
// OWASP A07:2021 - Identification and Authentication Failures
// 實現高並發默認憑證測試

use futures::future::join_all;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{debug, info, warn};

/// 憑證測試結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialResult {
    /// 用戶名
    pub username: String,
    /// 密碼
    pub password: String,
    /// 是否成功
    pub success: bool,
    /// 響應時間（毫秒）
    pub response_time_ms: u64,
    /// 響應狀態碼
    pub status_code: Option<u16>,
    /// 響應體片段
    pub response_body: Option<String>,
}

/// 認證類型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthType {
    /// HTTP Basic Authentication
    BasicAuth,
    /// Form-based Authentication
    FormBased,
    /// JSON API Authentication
    JsonApi,
}

/// Credential Sprayer
pub struct CredentialSprayer {
    /// 並發數量
    concurrency: usize,
    /// 默認憑證列表
    default_credentials: Vec<(String, String)>,
    /// HTTP 客戶端
    client: Client,
    /// 認證類型
    auth_type: AuthType,
}

impl CredentialSprayer {
    /// 創建新的 Credential Sprayer
    pub fn new(concurrency: usize) -> Self {
        Self {
            concurrency,
            default_credentials: Self::load_default_credentials(),
            client: Client::builder()
                .timeout(Duration::from_secs(10))
                .danger_accept_invalid_certs(true) // 測試環境
                .build()
                .unwrap(),
            auth_type: AuthType::BasicAuth,
        }
    }

    /// 設置認證類型
    pub fn with_auth_type(mut self, auth_type: AuthType) -> Self {
        self.auth_type = auth_type;
        self
    }

    /// 加載默認憑證
    fn load_default_credentials() -> Vec<(String, String)> {
        vec![
            // 通用默認憑證
            ("admin".to_string(), "admin".to_string()),
            ("admin".to_string(), "password".to_string()),
            ("admin".to_string(), "123456".to_string()),
            ("administrator".to_string(), "administrator".to_string()),
            ("root".to_string(), "root".to_string()),
            ("root".to_string(), "toor".to_string()),
            ("root".to_string(), "password".to_string()),
            ("user".to_string(), "user".to_string()),
            ("test".to_string(), "test".to_string()),
            ("guest".to_string(), "guest".to_string()),
            
            // 數據庫默認憑證
            ("postgres".to_string(), "postgres".to_string()),
            ("mysql".to_string(), "mysql".to_string()),
            ("oracle".to_string(), "oracle".to_string()),
            ("sa".to_string(), "sa".to_string()),
            
            // 應用默認憑證
            ("tomcat".to_string(), "tomcat".to_string()),
            ("jenkins".to_string(), "jenkins".to_string()),
            ("nexus".to_string(), "nexus".to_string()),
            ("sonar".to_string(), "sonar".to_string()),
            
            // IoT 設備默認憑證
            ("admin".to_string(), "admin123".to_string()),
            ("admin".to_string(), "".to_string()),  // 空密碼
            ("support".to_string(), "support".to_string()),
            ("ubnt".to_string(), "ubnt".to_string()),
            
            // 路由器默認憑證
            ("admin".to_string(), "1234".to_string()),
            ("admin".to_string(), "admin1234".to_string()),
            ("cisco".to_string(), "cisco".to_string()),
            ("default".to_string(), "default".to_string()),
        ]
    }

    /// 執行憑證噴射
    pub async fn spray(
        &self,
        target: &str,
        custom_credentials: Option<Vec<(String, String)>>,
    ) -> Vec<CredentialResult> {
        let creds = custom_credentials.unwrap_or(self.default_credentials.clone());
        info!("Starting credential spraying against {} with {} credentials", 
              target, creds.len());

        let semaphore = Arc::new(Semaphore::new(self.concurrency));
        let client = self.client.clone();
        let target = target.to_string();
        let auth_type = self.auth_type;

        // 創建並發任務
        let tasks: Vec<_> = creds
            .into_iter()
            .map(|(user, pass)| {
                let sem = Arc::clone(&semaphore);
                let client = client.clone();
                let target = target.clone();

                tokio::spawn(async move {
                    let _permit = sem.acquire().await.unwrap();
                    Self::test_credential(&client, &target, &user, &pass, auth_type).await
                })
            })
            .collect();

        // 等待所有任務完成
        let results = join_all(tasks).await;

        // 收集成功的結果
        let successful: Vec<_> = results
            .into_iter()
            .filter_map(Result::ok)
            .filter(|r| r.success)
            .collect();

        if !successful.is_empty() {
            info!("✓ Found {} valid credentials", successful.len());
        } else {
            info!("No valid credentials found");
        }

        successful
    }

    /// 測試單個憑證
    async fn test_credential(
        client: &Client,
        target: &str,
        username: &str,
        password: &str,
        auth_type: AuthType,
    ) -> CredentialResult {
        let start = Instant::now();
        debug!("Testing {}:{}", username, "***");

        let result = match auth_type {
            AuthType::BasicAuth => {
                Self::test_basic_auth(client, target, username, password).await
            }
            AuthType::FormBased => {
                Self::test_form_based(client, target, username, password).await
            }
            AuthType::JsonApi => {
                Self::test_json_api(client, target, username, password).await
            }
        };

        let response_time_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok((success, status_code, body)) => {
                if success {
                    info!("✓ Valid credential found: {}:{}", username, password);
                }
                CredentialResult {
                    username: username.to_string(),
                    password: password.to_string(),
                    success,
                    response_time_ms,
                    status_code: Some(status_code),
                    response_body: Some(body),
                }
            }
            Err(e) => {
                warn!("Error testing {}:{} - {}", username, "***", e);
                CredentialResult {
                    username: username.to_string(),
                    password: password.to_string(),
                    success: false,
                    response_time_ms,
                    status_code: None,
                    response_body: Some(e.to_string()),
                }
            }
        }
    }

    /// 測試 HTTP Basic Authentication
    async fn test_basic_auth(
        client: &Client,
        target: &str,
        username: &str,
        password: &str,
    ) -> Result<(bool, u16, String), Box<dyn std::error::Error>> {
        let response = client
            .get(target)
            .basic_auth(username, Some(password))
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        // 2xx 或 3xx 表示認證成功
        let success = status.is_success() || status.is_redirection();

        Ok((success, status.as_u16(), body))
    }

    /// 測試 Form-based Authentication
    async fn test_form_based(
        client: &Client,
        target: &str,
        username: &str,
        password: &str,
    ) -> Result<(bool, u16, String), Box<dyn std::error::Error>> {
        // 通用表單字段
        let form = [
            ("username", username),
            ("password", password),
            ("user", username),
            ("pass", password),
            ("email", username),
            ("login", username),
        ];

        let response = client.post(target).form(&form).send().await?;

        let status = response.status();
        let body = response.text().await?;

        // 檢測成功標誌
        let success = status.is_success()
            && !body.contains("error")
            && !body.contains("invalid")
            && !body.contains("incorrect")
            && (body.contains("dashboard") || body.contains("welcome") || body.contains("logout"));

        Ok((success, status.as_u16(), body))
    }

    /// 測試 JSON API Authentication
    async fn test_json_api(
        client: &Client,
        target: &str,
        username: &str,
        password: &str,
    ) -> Result<(bool, u16, String), Box<dyn std::error::Error>> {
        let json_body = serde_json::json!({
            "username": username,
            "password": password,
        });

        let response = client
            .post(target)
            .header("Content-Type", "application/json")
            .json(&json_body)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        // 檢測成功標誌
        let success = status.is_success()
            && (body.contains("token") || body.contains("session") || body.contains("success"));

        Ok((success, status.as_u16(), body))
    }

    /// 智能認證類型檢測
    pub async fn detect_auth_type(&self, target: &str) -> AuthType {
        // 發送探測請求
        match self.client.get(target).send().await {
            Ok(response) => {
                // 先克隆 headers，再獲取 body
                let headers = response.headers().clone();
                let body = response.text().await.unwrap_or_default();

                // 檢測 WWW-Authenticate 標頭
                if headers.contains_key("www-authenticate") {
                    return AuthType::BasicAuth;
                }

                // 檢測表單登錄
                if body.contains("<form") && body.contains("password") {
                    return AuthType::FormBased;
                }

                // 檢測 JSON API
                if headers
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .map(|v| v.contains("application/json"))
                    .unwrap_or(false)
                {
                    return AuthType::JsonApi;
                }

                AuthType::FormBased // 默認
            }
            Err(_) => AuthType::BasicAuth, // 默認
        }
    }
}
