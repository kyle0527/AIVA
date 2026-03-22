# function_crypto - 密碼學分析模組

> **版本**: v1.0.0 | **狀態**: ⬜ Rust 核心完成，Python binding 缺失 | **語言**: Rust | **能力登錄**: ⬜ 待登錄

## 模組概述

基於 Rust 的密碼學分析工具，透過 CLI binary 執行，提供加密強度分析、弱加密偵測、密鑰長度驗證等功能。

### 功能完成狀態

| 功能 | 狀態 | 說明 |
|------|------|------|
| Rust CLI binary | ✅ 完成 | `rust_core/` 下的 Rust 實作 |
| Python binding | ⬜ 未完成 | 缺少 Python wrapper |
| AIVA 執行器整合 | ⬜ 未完成 | `external_classification.json` 中僅有 1 個 flow |

## 架構

```
function_crypto/
├── rust_core/          # Rust 實作（crypto-scanner binary）
│   ├── src/
│   ├── Cargo.toml
│   └── ...
├── clap_analysis.json  # CLI 參數分析（clap 框架）
├── clap_cli_reference.md # CLI 參考文件
├── cli_commands.sh     # CLI 指令範例
└── batch_verify.ps1    # 批次驗證腳本（Windows）
```

## 執行方式

### 直接呼叫 Rust binary

```bash
# 先編譯
cd services/features/function_crypto/rust_core
cargo build --release

# 執行（參考 cli_commands.sh）
./target/release/crypto-scanner --help
```

### 透過 AIVA 執行器

```bash
# Rust 能力透過 external_classification.json 呼叫
python services/core/aiva_core/internal_exploration/aiva_external_executor.py \
    --lang rust --func crypto-scanner
```

## 待完成工作

- 建立 Python wrapper（`crypto_wrapper.py`）以接通 subprocess 呼叫
- 將 `sast_scan` 或 `secret_detection` 能力對應至此模組
- 補全 `CAPABILITY_CONFIGS` 中的 `sast_scan` entry

## 注意事項

- 需先執行 `cargo build --release` 編譯 Rust binary
- 目前無法透過 Python 直接 import，必須使用 subprocess 呼叫 binary
- `batch_verify.ps1` 為 Windows 腳本，Linux 環境請使用 `cli_commands.sh`
