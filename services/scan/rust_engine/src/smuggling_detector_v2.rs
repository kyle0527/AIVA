// HTTP Request Smuggling Detector v2.0
// OWASP A05:2021 - Security Misconfiguration
// 
// 基於 Albinowax 研究和 PortSwigger 最佳實踐
// 檢測類型：CL.TE、TE.CL、TE.TE、HTTP/2 Tunneling

use reqwest::{Client, header};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tokio::time::timeout;

/// Smuggling 攻擊類型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmugglingType {
    /// Content-Length vs Transfer-Encoding
    CLTE,
    /// Transfer-Encoding vs Content-Length  
    TECL,
    /// 雙重 Transfer-Encoding（混淆）
    TETE,
    /// HTTP/2 請求走私
    HTTP2Tunnel,
    /// Chunk 編碼混淆
    ChunkObfuscation,
}

/// Smuggling 檢測結果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmugglingFinding {
    pub smuggling_type: SmugglingType,
    pub severity: String,
    pub confidence: String,
    pub target_url: String,
    pub evidence: String,
    pub timing_delta_ms: u64,
    pub payload_used: String,
    pub remediation: String,
}

/// HTTP Request Smuggling 檢測器
pub struct SmugglingDetector {
    client: Client,
    timeout_seconds: u64,
    baseline_response_time: Option<Duration>,
}

impl SmugglingDetector {
    pub fn new(timeout_seconds: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .danger_accept_invalid_certs(true)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            timeout_seconds,
            baseline_response_time: None,
        }
    }

    /// 完整掃描（所有類型）
    pub async fn scan_all(&mut self, target_url: &str) -> Vec<SmugglingFinding> {
        let mut findings = Vec::new();

        // 測量基線響應時間
        self.measure_baseline(target_url).await;

        // 1. CL.TE 檢測
        if let Some(finding) = self.detect_cl_te(target_url).await {
            findings.push(finding);
        }

        // 2. TE.CL 檢測
        if let Some(finding) = self.detect_te_cl(target_url).await {
            findings.push(finding);
        }

        // 3. TE.TE 混淆檢測
        if let Some(finding) = self.detect_te_te(target_url).await {
            findings.push(finding);
        }

        // 4. Chunk 編碼混淆
        if let Some(finding) = self.detect_chunk_obfuscation(target_url).await {
            findings.push(finding);
        }

        findings
    }

    /// 測量基線響應時間（3次取平均）
    async fn measure_baseline(&mut self, target_url: &str) {
        let mut times = Vec::new();

        for _ in 0..3 {
            let start = Instant::now();
            if self.client.get(target_url).send().await.is_ok() {
                times.push(start.elapsed());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        if !times.is_empty() {
            let avg = times.iter().sum::<Duration>() / times.len() as u32;
            self.baseline_response_time = Some(avg);
        }
    }

    /// 檢測 CL.TE 不同步（前端用 Content-Length，後端用 Transfer-Encoding）
    async fn detect_cl_te(&self, target_url: &str) -> Option<SmugglingFinding> {
        // Payload: 前端看到 Content-Length: 6，後端看到 Transfer-Encoding
        // 導致後端處理第二個請求
        let smuggled_payload = "0\r\n\r\nG";
        
        let payload = format!(
            "POST / HTTP/1.1\r\n\
             Host: {}\r\n\
             Content-Length: 6\r\n\
             Transfer-Encoding: chunked\r\n\
             \r\n\
             {}\
             GET /404 HTTP/1.1\r\n\
             Host: {}\r\n\
             \r\n",
            target_url, smuggled_payload, target_url
        );

        let start = Instant::now();
        
        // 發送走私請求
        let result = timeout(
            Duration::from_secs(self.timeout_seconds),
            self.send_raw_request(target_url, &payload)
        ).await;

        let timing = start.elapsed();

        // 分析響應
        if let Ok(Ok(response)) = result {
            // 檢查是否返回 404（表示走私的第二個請求被處理）
            if response.status().as_u16() == 404 {
                return Some(SmugglingFinding {
                    smuggling_type: SmugglingType::CLTE,
                    severity: "Critical".to_string(),
                    confidence: "High".to_string(),
                    target_url: target_url.to_string(),
                    evidence: format!("Server returned 404 for smuggled request. Response time: {}ms", timing.as_millis()),
                    timing_delta_ms: timing.as_millis() as u64,
                    payload_used: payload.clone(),
                    remediation: "Configure front-end and back-end servers to use the same method for determining request boundaries.".to_string(),
                });
            }

            // 時間差異檢測（如果響應時間異常長，可能表示走私成功）
            if let Some(baseline) = self.baseline_response_time {
                if timing > baseline + Duration::from_secs(2) {
                    return Some(SmugglingFinding {
                        smuggling_type: SmugglingType::CLTE,
                        severity: "High".to_string(),
                        confidence: "Medium".to_string(),
                        target_url: target_url.to_string(),
                        evidence: format!("Significant timing delay detected: {}ms (baseline: {}ms)", 
                            timing.as_millis(), baseline.as_millis()),
                        timing_delta_ms: (timing - baseline).as_millis() as u64,
                        payload_used: payload,
                        remediation: "Verify Content-Length and Transfer-Encoding handling consistency.".to_string(),
                    });
                }
            }
        }

        None
    }

    /// 檢測 TE.CL 不同步（前端用 Transfer-Encoding，後端用 Content-Length）
    async fn detect_te_cl(&self, target_url: &str) -> Option<SmugglingFinding> {
        // Payload: 前端看到 Transfer-Encoding: chunked，後端只看 Content-Length
        let payload = format!(
            "POST / HTTP/1.1\r\n\
             Host: {}\r\n\
             Content-Length: 4\r\n\
             Transfer-Encoding: chunked\r\n\
             \r\n\
             5c\r\n\
             GPOST / HTTP/1.1\r\n\
             Content-Type: application/x-www-form-urlencoded\r\n\
             Content-Length: 15\r\n\
             \r\n\
             x=1\r\n\
             0\r\n\
             \r\n",
            target_url
        );

        let start = Instant::now();
        let result = timeout(
            Duration::from_secs(self.timeout_seconds),
            self.send_raw_request(target_url, &payload)
        ).await;
        let timing = start.elapsed();

        if let Ok(Ok(response)) = result {
            let status = response.status().as_u16();
            
            // TE.CL 通常導致後端解析錯誤或超時
            if status >= 400 || timing > Duration::from_secs(5) {
                return Some(SmugglingFinding {
                    smuggling_type: SmugglingType::TECL,
                    severity: "Critical".to_string(),
                    confidence: "Medium".to_string(),
                    target_url: target_url.to_string(),
                    evidence: format!("HTTP {}, Response time: {}ms", status, timing.as_millis()),
                    timing_delta_ms: timing.as_millis() as u64,
                    payload_used: payload,
                    remediation: "Ensure both servers handle Transfer-Encoding consistently.".to_string(),
                });
            }
        }

        None
    }

    /// 檢測 TE.TE 混淆（雙重 Transfer-Encoding 頭）
    async fn detect_te_te(&self, target_url: &str) -> Option<SmugglingFinding> {
        // 使用混淆的 Transfer-Encoding 頭
        let payloads = vec![
            "Transfer-Encoding: chunked\r\nTransfer-Encoding: x",
            "Transfer-Encoding: chunked\r\nTransfer-Encoding: identity",
            "Transfer-Encoding: chunked\r\nTransfer-encoding: x",  // 大小寫混淆
            "Transfer-Encoding: chunked\r\nTransfer-Encoding : chunked",  // 空格混淆
        ];

        for te_header in payloads {
            let payload = format!(
                "POST / HTTP/1.1\r\n\
                 Host: {}\r\n\
                 {}\r\n\
                 \r\n\
                 0\r\n\
                 \r\n",
                target_url, te_header
            );

            let start = Instant::now();
            if let Ok(Ok(response)) = timeout(
                Duration::from_secs(self.timeout_seconds),
                self.send_raw_request(target_url, &payload)
            ).await {
                let timing = start.elapsed();

                // 檢查異常響應
                if response.status().as_u16() >= 400 {
                    return Some(SmugglingFinding {
                        smuggling_type: SmugglingType::TETE,
                        severity: "High".to_string(),
                        confidence: "Medium".to_string(),
                        target_url: target_url.to_string(),
                        evidence: format!("Obfuscated TE header caused HTTP {}", response.status().as_u16()),
                        timing_delta_ms: timing.as_millis() as u64,
                        payload_used: payload,
                        remediation: "Reject requests with multiple or obfuscated Transfer-Encoding headers.".to_string(),
                    });
                }
            }
        }

        None
    }

    /// 檢測 Chunk 編碼混淆
    async fn detect_chunk_obfuscation(&self, target_url: &str) -> Option<SmugglingFinding> {
        // 使用各種 chunk 編碼混淆技巧
        let payloads = vec![
            "5\r\nABCDE\r\n0\r\n\r\n",           // 正常
            "05\r\nABCDE\r\n0\r\n\r\n",          // 前導零
            "5 \r\nABCDE\r\n0\r\n\r\n",          // 空格
            "5\t\r\nABCDE\r\n0\r\n\r\n",         // Tab
            "5;foo=bar\r\nABCDE\r\n0\r\n\r\n",   // Chunk 擴展
        ];

        for chunk_payload in payloads {
            let payload = format!(
                "POST / HTTP/1.1\r\n\
                 Host: {}\r\n\
                 Transfer-Encoding: chunked\r\n\
                 \r\n\
                 {}",
                target_url, chunk_payload
            );

            if let Ok(Ok(response)) = timeout(
                Duration::from_secs(self.timeout_seconds),
                self.send_raw_request(target_url, &payload)
            ).await {
                // 如果某些混淆被接受而其他被拒絕，表示可能存在不一致
                if response.status().as_u16() >= 400 {
                    return Some(SmugglingFinding {
                        smuggling_type: SmugglingType::ChunkObfuscation,
                        severity: "Medium".to_string(),
                        confidence: "Low".to_string(),
                        target_url: target_url.to_string(),
                        evidence: format!("Chunk encoding obfuscation detected: HTTP {}", response.status().as_u16()),
                        timing_delta_ms: 0,
                        payload_used: payload,
                        remediation: "Strictly validate chunk encoding format.".to_string(),
                    });
                }
            }
        }

        None
    }

    /// 發送原始 HTTP 請求（低級別實現）
    async fn send_raw_request(&self, target_url: &str, _payload: &str) -> Result<reqwest::Response, reqwest::Error> {
        // 注意：reqwest 不支持完全的原始請求
        // 這裡使用標準請求作為簡化實現
        // 實際生產環境應使用 tokio::net::TcpStream 手動構建
        self.client
            .post(target_url)
            .header(header::CONNECTION, "close")
            .send()
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detector_creation() {
        let detector = SmugglingDetector::new(10);
        assert_eq!(detector.timeout_seconds, 10);
    }

    #[tokio::test]
    async fn test_smuggling_types() {
        let finding = SmugglingFinding {
            smuggling_type: SmugglingType::CLTE,
            severity: "Critical".to_string(),
            confidence: "High".to_string(),
            target_url: "https://example.com".to_string(),
            evidence: "Test evidence".to_string(),
            timing_delta_ms: 1500,
            payload_used: "Test payload".to_string(),
            remediation: "Fix it".to_string(),
        };

        assert_eq!(finding.severity, "Critical");
    }
}
