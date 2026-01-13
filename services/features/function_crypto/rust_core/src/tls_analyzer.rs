// TLS/SSL 配置分析器
// 從網路握手分析 TLS 配置安全性

use crate::Finding;
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio_rustls::rustls::{ClientConfig, RootCertStore};
use tokio_rustls::TlsConnector;

pub async fn analyze_tls(target: &str, port: u16) -> Vec<Finding> {
    let mut findings = Vec::new();
    let addr = format!("{}:{}", target, port);

    // 1. 準備 TLS 配置 - 使用系統根證書
    let mut root_store = RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    
    let config = ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    
    let connector = TlsConnector::from(Arc::new(config));
    
    // 2. 解析域名（對於 localhost/IP，使用特殊處理）
    let server_name = if target == "localhost" || target.parse::<std::net::IpAddr>().is_ok() {
        match rustls::pki_types::ServerName::try_from("localhost") {
            Ok(name) => name.to_owned(),
            Err(_) => {
                findings.push(Finding {
                    issue_type: "INVALID_TARGET_FORMAT".to_string(),
                    severity: "INFO".to_string(),
                    title: "Target is not a valid DNS name".to_string(),
                    description: "TLS SNI requires a valid domain name.".to_string(),
                    recommendation: "Use domain name instead of IP if possible.".to_string(),
                    location: Some(target.to_string()),
                    evidence: None,
                });
                return findings;
            }
        }
    } else {
        match rustls::pki_types::ServerName::try_from(target.to_string()) {
            Ok(name) => name.to_owned(),
            Err(_) => {
                findings.push(Finding {
                    issue_type: "INVALID_TARGET_FORMAT".to_string(),
                    severity: "INFO".to_string(),
                    title: "Invalid domain name format".to_string(),
                    description: format!("Cannot parse '{}' as valid DNS name", target),
                    recommendation: "Use proper domain name format.".to_string(),
                    location: Some(target.to_string()),
                    evidence: None,
                });
                return findings;
            }
        }
    };

    // 3. 嘗試 TCP 連線
    let stream = match TcpStream::connect(&addr).await {
        Ok(s) => s,
        Err(e) => {
            findings.push(Finding {
                issue_type: "CONNECTION_REFUSED".to_string(),
                severity: "INFO".to_string(),
                title: "Port Closed or Unreachable".to_string(),
                description: format!("Could not connect to {}: {}", addr, e),
                recommendation: "Check if the service is running.".to_string(),
                location: Some(addr),
                evidence: None,
            });
            return findings;
        }
    };

    // 4. 嘗試 TLS 握手
    match connector.connect(server_name, stream).await {
        Ok(_tls_stream) => {
            // 握手成功，表示支援 TLS
            // 這裡可以進一步分析協議版本和加密套件
            // 但目前的 rustls API 不直接暴露這些信息
        },
        Err(e) => {
            let err_msg = e.to_string();
            if err_msg.contains("received fatal alert") 
                || err_msg.contains("unexpected message")
                || err_msg.contains("invalid peer certificate")
                || err_msg.contains("UnexpectedMessage") {
                // 這些錯誤表明目標可能不支援 TLS
                findings.push(Finding {
                    issue_type: "NO_TLS_DETECTED".to_string(),
                    severity: "HIGH".to_string(),
                    title: "Plain HTTP Service Detected".to_string(),
                    description: "The service appears to be running plain HTTP, not HTTPS.".to_string(),
                    recommendation: "Enable TLS/SSL (HTTPS) to encrypt communications.".to_string(),
                    location: Some(addr.clone()),
                    evidence: Some(format!("TLS handshake error: {}", err_msg)),
                });
            } else {
                findings.push(Finding {
                    issue_type: "TLS_HANDSHAKE_FAILED".to_string(),
                    severity: "MEDIUM".to_string(),
                    title: "TLS Handshake Failed".to_string(),
                    description: format!("Failed to establish TLS connection: {}", err_msg),
                    recommendation: "Check SSL/TLS configuration and certificate validity.".to_string(),
                    location: Some(addr),
                    evidence: Some(err_msg),
                });
            }
        }
    }

    findings
}
