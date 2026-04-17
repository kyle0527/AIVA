# AIVA Rust AST 分析工具

> **版本**: v3.0  
> **最後更新**: 2026-01-20  
> **狀態**: ✅ 生產就緒（語法解析限制已知）  
> **核心文件**: src/main.rs  
> **代碼行數**: 864 行  
> **執行檔**: rs2mermaid.exe

## ⚠️ 已知限制 (2026-01-20)

### Clap CLI 模式的分析限制

**背景**：部分 Rust 模組（如 `function_crypto`）使用 Clap 框架定義 CLI 參數

**技術細節**：
- 使用 `#[derive(Parser)]` 和 `#[arg(long)]` 宏定義參數
- 參數通過命令行傳遞，不是 stdin JSON
- 例如：`crypto-scanner scan-js --content 'code' --url 'url'`

**分析限制**：
- ✅ **語法解析**：`syn` crate 可以解析 AST
- ❌ **語義分析**：無法提取 derive macro 展開後的參數類型
- ❌ **參數提取**：`#[arg]` 屬性需要語義分析才能解讀

**對比**：
| 工具 | 能力 | 限制 |
|------|------|------|
| Go `go/ast` | ✅ 語義分析 | 標準庫完整支援 |
| Rust `syn` | ⚠️ 語法解析 | 無類型推導 |
| Python `ast` | ✅ 語義分析 | 動態類型 |

**結果**：
- `function_crypto`: 0 flows（Clap CLI 模式）
- `rust_engine`: 1 flow（stdin JSON 模式）

**解決方案選項**（未實作）：
1. 手動解析 `#[arg]` 屬性（複雜度高）
2. 整合 rust-analyzer LSP（工程量大）
3. 接受現狀，專注 Python + Go 支援

**決策**：採用方案 3，將資源聚焦於主要語言支援

## 📂 子模組 (Submodules)

- [src](./src/README.md)

