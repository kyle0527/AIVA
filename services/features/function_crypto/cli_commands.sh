# AIVA Rust Analysis CLI Commands

## Category: ANALYSIS
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\cookie_analyzer.rs --func analyze_cookies
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\header_analyzer.rs --func analyze_headers
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\tls_analyzer.rs --func analyze_tls

## Category: OTHER
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\cookie_analyzer.rs --func extract_cookie_name
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\cookie_analyzer.rs --func is_sensitive_cookie
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\header_analyzer.rs --func extract_max_age
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\js_crypto_analyzer.rs --func find_line_number
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\js_crypto_analyzer.rs --func mask_key
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\main.rs --func Summary::from_findings
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\main.rs --func main

## Category: RECONNAISSANCE
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\js_crypto_analyzer.rs --func scan_javascript
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\js_crypto_analyzer.rs --func detect_hardcoded_keys
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\js_crypto_analyzer.rs --func detect_weak_crypto_usage
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\js_crypto_analyzer.rs --func detect_weak_random
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\js_crypto_analyzer.rs --func detect_jwt_issues
cargo run --bin rs2mermaid -- --file C:\D\fold7\AIVA-git\services\features\function_crypto\rust_core\src\js_crypto_analyzer.rs --func detect_insecure_storage

