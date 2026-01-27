"""
function_crypto - 密碼學配置掃描器（純 Rust CLI）

架構: 獨立 Rust CLI
版本: 2.0.0 - 網路層密碼學配置掃描
日期: 2025-12-12

設計理念:
- Rust CLI: crypto-scanner（獨立可執行文件）
- AI Commander: 直接生成並執行 CLI 指令
- 無 Python 包裝層: AI 自己處理 JSON 輸出
- 黑盒測試: 僅分析網路層可觀察的密碼學配置

使用方式:
    # AI Commander 動態生成指令
    cmd = [
        "crypto-scanner",
        "scan-js",
        "--content", js_content,
        "--url", "https://target.com"
    ]
    
    # 執行並解析 JSON
    process = await asyncio.create_subprocess_exec(*cmd, ...)
    stdout, _ = await process.communicate()
    result = json.loads(stdout.decode())

CLI 命令:
    # JavaScript 分析
    crypto-scanner scan-js --content "<JS>" --url "<URL>"
    
    # TLS 配置分析
    crypto-scanner analyze-tls --target <DOMAIN> --port <PORT>
    
    # Cookie 安全分析
    crypto-scanner analyze-cookies --cookies-json '<JSON>' --url "<URL>"
    
    # 安全標頭分析
    crypto-scanner analyze-headers --headers-json '<JSON>' --url "<URL>"

v1 備份位置:
    C:\\Users\\User\\Downloads\\新增資料夾 (3)\\function_crypto_v1_source_code_analysis

架構文檔: 
- services/features/SIMPLE_ARCHITECTURE.md
- services/core/aiva_core/internal_exploration/CLI_COMMANDS_ARCHITECTURE_ANALYSIS.md

__version__ = "3.0.0"
__status__ = "production"
__architecture__ = "rust-cli-only"
__last_updated__ = "2026-01-23"

# 整合狀態 (2026-01-23)
# ✅ Rust CLI 架構完成
# ✅ crypto-scanner 二進制可用
- ⏳ AI Commander 整合待開發
"""

__all__ = []
__version__ = "2.0.0"
__status__ = "ready_for_integration"
__last_updated__ = "2025-12-17"

__version__ = "2.0.0"
__status__ = "production"
__architecture__ = "rust-cli-only"
__cli_binary__ = "rust_core/target/release/crypto-scanner"
