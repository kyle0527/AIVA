// HTTP Request Smuggling Detector
// OWASP A05:2021 - Security Misconfiguration
// 實現 CL.TE 和 TE.CL 攻擊檢測

use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tracing::{debug, info, warn};

/// Smuggling 檢測結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmugglingResult {
    /// 目標主機
    pub target_host: String,
    /// 目標端口
    pub target_port: u16,
    /// 是否檢測到 TE.CL 漏洞
    pub te_cl_vulnerable: bool,
    /// 是否檢測到 CL.TE 漏洞
    pub cl_te_vulnerable: bool,
    /// 檢測證據
    pub evidence: Vec<String>,
    /// 風險等級
    pub risk_level: String,
}

/// HTTP Request Smuggling 檢測器
pub struct SmugglingDetector {
    target_host: String,
    target_port: u16,
    timeout_duration: Duration,
}

impl SmugglingDetector {
    /// 創建新的檢測器實例
    pub fn new(target_host: String, target_port: u16) -> Self {
        Self {
            target_host,
            target_port,
            timeout_duration: Duration::from_secs(10),
        }
    }

    /// 設置超時時間
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_duration = timeout;
        self
    }

    /// 執行完整的 Smuggling 檢測
    pub async fn detect(&self) -> Result<SmugglingResult, Box<dyn std::error::Error>> {
        info!("Starting HTTP Request Smuggling detection on {}:{}", 
              self.target_host, self.target_port);

        let mut result = SmugglingResult {
            target_host: self.target_host.clone(),
            target_port: self.target_port,
            te_cl_vulnerable: false,
            cl_te_vulnerable: false,
            evidence: Vec::new(),
            risk_level: "Safe".to_string(),
        };

        // 測試 TE.CL 漏洞
        match self.test_te_cl().await {
            Ok(vulnerable) => {
                result.te_cl_vulnerable = vulnerable;
                if vulnerable {
                    result.evidence.push("TE.CL desync detected".to_string());
                    info!("✓ TE.CL vulnerability confirmed");
                }
            }
            Err(e) => {
                warn!("TE.CL test failed: {}", e);
            }
        }

        // 測試 CL.TE 漏洞
        match self.test_cl_te().await {
            Ok(vulnerable) => {
                result.cl_te_vulnerable = vulnerable;
                if vulnerable {
                    result.evidence.push("CL.TE desync detected".to_string());
                    info!("✓ CL.TE vulnerability confirmed");
                }
            }
            Err(e) => {
                warn!("CL.TE test failed: {}", e);
            }
        }

        // 確定風險等級
        result.risk_level = if result.te_cl_vulnerable || result.cl_te_vulnerable {
            "Critical".to_string()
        } else {
            "Safe".to_string()
        };

        Ok(result)
    }

    /// 測試 TE.CL 漏洞
    /// Transfer-Encoding 優先於 Content-Length
    pub async fn test_te_cl(&self) -> Result<bool, Box<dyn std::error::Error>> {
        debug!("Testing TE.CL vulnerability...");

        // 構造 TE.CL 攻擊向量
        // 前端處理 Transfer-Encoding (chunked)
        // 後端處理 Content-Length
        let payload = format!(
            "POST / HTTP/1.1\r\n\
             Host: {}\r\n\
             Transfer-Encoding: chunked\r\n\
             Content-Length: 4\r\n\
             \r\n\
             1\r\n\
             Z\r\n\
             Q",
            self.target_host
        );

        let response = self.send_payload(&payload).await?;
        
        // 分析響應判斷是否存在漏洞
        // 檢測異常響應碼或超時
        Ok(self.analyze_response(&response))
    }

    /// 測試 CL.TE 漏洞
    /// Content-Length 優先於 Transfer-Encoding
    pub async fn test_cl_te(&self) -> Result<bool, Box<dyn std::error::Error>> {
        debug!("Testing CL.TE vulnerability...");

        // 構造 CL.TE 攻擊向量
        // 前端處理 Content-Length
        // 後端處理 Transfer-Encoding
        let payload = format!(
            "POST / HTTP/1.1\r\n\
             Host: {}\r\n\
             Content-Length: 6\r\n\
             Transfer-Encoding: chunked\r\n\
             \r\n\
             0\r\n\
             \r\n\
             G",
            self.target_host
        );

        let response = self.send_payload(&payload).await?;
        
        // 分析響應
        Ok(self.analyze_response(&response))
    }

    /// 發送 payload 到目標
    async fn send_payload(&self, payload: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let addr = format!("{}:{}", self.target_host, self.target_port);
        
        // 連接到目標，帶超時
        let mut stream = timeout(
            self.timeout_duration,
            TcpStream::connect(&addr)
        ).await??;

        debug!("Connected to {}", addr);

        // 發送 payload
        stream.write_all(payload.as_bytes()).await?;
        debug!("Payload sent ({} bytes)", payload.len());

        // 讀取響應
        let mut response = Vec::new();
        timeout(
            self.timeout_duration,
            stream.read_to_end(&mut response)
        ).await??;

        debug!("Received response ({} bytes)", response.len());
        Ok(response)
    }

    /// 分析響應判斷是否存在漏洞
    fn analyze_response(&self, response: &[u8]) -> bool {
        let response_str = String::from_utf8_lossy(response);
        
        // 檢測異常狀態碼
        let suspicious_codes = ["400", "502", "408", "413"];
        for code in suspicious_codes {
            if response_str.contains(code) {
                debug!("Suspicious response code detected: {}", code);
                return true;
            }
        }

        // 檢測多個 HTTP 響應（desync 的明顯標誌）
        let http_response_count = response_str.matches("HTTP/").count();
        if http_response_count > 1 {
            debug!("Multiple HTTP responses detected: {}", http_response_count);
            return true;
        }

        // 檢測響應長度異常
        if response.is_empty() {
            debug!("Empty response (possible timeout)");
            return true;
        }

        // 檢測分塊編碼錯誤
        if response_str.contains("Transfer-Encoding") && response_str.contains("Content-Length") {
            debug!("Conflicting headers detected");
            return true;
        }

        false
    }

    /// 執行差異測試
    /// 發送正常請求和 smuggling 請求，比較響應差異
    pub async fn differential_test(&self) -> Result<bool, Box<dyn std::error::Error>> {
        // 正常請求
        let normal_payload = format!(
            "GET / HTTP/1.1\r\n\
             Host: {}\r\n\
             \r\n",
            self.target_host
        );

        let normal_response = self.send_payload(&normal_payload).await?;

        // Smuggling 請求
        let smuggling_payload = format!(
            "POST / HTTP/1.1\r\n\
             Host: {}\r\n\
             Content-Length: 4\r\n\
             Transfer-Encoding: chunked\r\n\
             \r\n\
             1\r\n\
             Z\r\n\
             0\r\n\
             \r\n",
            self.target_host
        );

        let smuggling_response = self.send_payload(&smuggling_payload).await?;

        // 比較響應差異
        let has_difference = normal_response.len() != smuggling_response.len() ||
                            normal_response != smuggling_response;

        if has_difference {
            debug!("Differential response detected (normal: {} bytes, smuggling: {} bytes)",
                  normal_response.len(), smuggling_response.len());
        }

        Ok(has_difference)
    }
}
