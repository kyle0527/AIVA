// AIVA Crypto Scanner - 獨立 CLI 模式
// 日期: 2025-12-11
// 功能: 高性能密碼學弱點掃描器，獨立運行

use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "crypto-scanner")]
#[command(version = "0.1.0")]
#[command(about = "AIVA Cryptography Scanner - 密碼學弱點掃描器", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 掃描代碼中的密碼學弱點
    Scan {
        /// 要掃描的代碼（字符串）
        #[arg(long)]
        code: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct Finding {
    issue_type: String,
    detail: String,
}

fn scan_crypto_weaknesses(code: &str) -> Vec<Finding> {
    let mut findings = Vec::new();
    let lower = code.to_lowercase();
    
    // ===== 弱算法檢測 =====
    if lower.contains("md5") {
        findings.push(Finding {
            issue_type: "WEAK_ALGORITHM".to_string(),
            detail: "MD5 is cryptographically broken - use SHA-256 or SHA-3".to_string(),
        });
    }
    if lower.contains("sha1") || lower.contains("sha-1") {
        findings.push(Finding {
            issue_type: "WEAK_ALGORITHM".to_string(),
            detail: "SHA-1 is deprecated - use SHA-256 or higher".to_string(),
        });
    }
    
    // ===== 弱加密檢測 =====
    if lower.contains(" des ") || lower.contains("des(") || lower.contains("_des_") {
        findings.push(Finding {
            issue_type: "WEAK_CIPHER".to_string(),
            detail: "DES has 56-bit key (broken) - use AES-256".to_string(),
        });
    }
    if lower.contains("3des") || lower.contains("tripledes") {
        findings.push(Finding {
            issue_type: "WEAK_CIPHER".to_string(),
            detail: "3DES is deprecated (NIST SP 800-131A) - use AES".to_string(),
        });
    }
    if lower.contains("rc4") || lower.contains("arcfour") {
        findings.push(Finding {
            issue_type: "WEAK_CIPHER".to_string(),
            detail: "RC4 has known biases - use AES-GCM or ChaCha20".to_string(),
        });
    }
    if lower.contains("blowfish") {
        findings.push(Finding {
            issue_type: "WEAK_CIPHER".to_string(),
            detail: "Blowfish has 64-bit block (birthday attack) - use AES".to_string(),
        });
    }
    
    // ===== 加密模式檢測 =====
    if lower.contains("ecb") {
        findings.push(Finding {
            issue_type: "WEAK_MODE".to_string(),
            detail: "ECB mode reveals patterns - use GCM, CBC, or CTR".to_string(),
        });
    }
    if lower.contains("cbc") && !lower.contains("hmac") {
        findings.push(Finding {
            issue_type: "WEAK_MODE".to_string(),
            detail: "CBC without HMAC vulnerable to padding oracle - use GCM".to_string(),
        });
    }
    
    // ===== TLS/SSL 安全 =====
    if lower.contains("verify=false") || lower.contains("verify_ssl=false") || lower.contains("sslverify=false") {
        findings.push(Finding {
            issue_type: "INSECURE_TLS".to_string(),
            detail: "TLS certificate verification disabled - MITM risk".to_string(),
        });
    }
    if lower.contains("sslv2") || lower.contains("sslv3") {
        findings.push(Finding {
            issue_type: "INSECURE_TLS".to_string(),
            detail: "SSLv2/v3 are broken (POODLE, DROWN) - use TLS 1.2+".to_string(),
        });
    }
    if lower.contains("tls1.0") || lower.contains("tlsv1.0") || lower.contains("tls 1.0") {
        findings.push(Finding {
            issue_type: "INSECURE_TLS".to_string(),
            detail: "TLS 1.0 deprecated (RFC 8996) - use TLS 1.2+".to_string(),
        });
    }
    if lower.contains("tls1.1") || lower.contains("tlsv1.1") {
        findings.push(Finding {
            issue_type: "INSECURE_TLS".to_string(),
            detail: "TLS 1.1 deprecated (RFC 8996) - use TLS 1.2+".to_string(),
        });
    }
    
    // ===== 硬編碼密鑰 =====
    if lower.contains("private key") || (lower.contains("-----begin") && lower.contains("private key-----")) {
        findings.push(Finding {
            issue_type: "HARDCODED_KEY".to_string(),
            detail: "Private key embedded in code - use key management system".to_string(),
        });
    }
    if lower.contains("api_key") || lower.contains("apikey") || lower.contains("secret_key") {
        // 簡單檢查是否是硬編碼（包含等號和引號）
        if (lower.contains("=\"") || lower.contains("='")) && !lower.contains("env") {
            findings.push(Finding {
                issue_type: "HARDCODED_KEY".to_string(),
                detail: "API/Secret key hardcoded - use environment variables".to_string(),
            });
        }
    }
    
    // ===== 弱隨機數生成器 =====
    if lower.contains("random(") || lower.contains("math.random") {
        findings.push(Finding {
            issue_type: "WEAK_RANDOM".to_string(),
            detail: "Insecure RNG for security - use secrets/crypto.randomBytes".to_string(),
        });
    }
    if lower.contains("srand(") || lower.contains("rand(") {
        findings.push(Finding {
            issue_type: "WEAK_RANDOM".to_string(),
            detail: "C rand() is predictable - use /dev/urandom or BCryptGenRandom".to_string(),
        });
    }
    if lower.contains("mt_rand") || lower.contains("mt_srand") {
        findings.push(Finding {
            issue_type: "WEAK_RANDOM".to_string(),
            detail: "Mersenne Twister not cryptographically secure - use random_bytes()".to_string(),
        });
    }
    
    // ===== 密碼存儲 =====
    if lower.contains("password") && (lower.contains("md5") || lower.contains("sha1") || lower.contains("sha256")) {
        findings.push(Finding {
            issue_type: "WEAK_PASSWORD_HASH".to_string(),
            detail: "Fast hash for passwords vulnerable to brute-force - use bcrypt/Argon2".to_string(),
        });
    }
    
    // ===== JWT 安全 =====
    if lower.contains("jwt") && lower.contains("none") {
        findings.push(Finding {
            issue_type: "INSECURE_JWT".to_string(),
            detail: "JWT 'none' algorithm bypasses signature - enforce HS256/RS256".to_string(),
        });
    }
    
    findings
}

fn main() {
    let cli = Cli::parse();
    
    match &cli.command {
        Commands::Scan { code } => {
            // 執行掃描
            let findings = scan_crypto_weaknesses(code);
            
            // 輸出 JSON
            let json_output = serde_json::to_string_pretty(&findings)
                .expect("Failed to serialize findings");
            println!("{}", json_output);
        }
    }
}
