# AST 分析工具 - 文件索引

## 📑 目錄

- [📑 快速導航](#-快速導航)
  - [🚀 開始使用](#-開始使用)
  - [📖 完整文檔](#-完整文檔)
  - [📝 實施總結](#-實施總結)
- [🛠️ 工具文件](#-工具文件)
  - [核心工具](#核心工具)
  - [配置文件](#配置文件)
  - [測試腳本](#測試腳本)
- [🎯 使用場景對照表](#-使用場景對照表)
  - [想要分析 TypeScript Engine？](#想要分析-typescript-engine)
  - [想要分析 Rust Engine？](#想要分析-rust-engine)
  - [想要了解工具原理？](#想要了解工具原理)
  - [想要自定義分析？](#想要自定義分析)
  - [遇到問題？](#遇到問題)
- [📊 工具比較](#-工具比較)
- [🎓 學習路徑](#-學習路徑)
  - [新手路徑](#新手路徑)
  - [進階路徑](#進階路徑)
  - [專家路徑](#專家路徑)
- [📋 檢查清單](#-檢查清單)
  - [第一次使用](#第一次使用)
  - [分析 TypeScript Engine](#分析-typescript-engine)
  - [分析 Rust Engine](#分析-rust-engine)
- [🔗 快速連結](#-快速連結)
  - [工具執行](#工具執行)
  - [查看結果](#查看結果)
  - [線上工具](#線上工具)
- [💡 常見使用模式](#-常見使用模式)
  - [快速驗證工具](#快速驗證工具)
  - [分析特定目錄](#分析特定目錄)
  - [生成橫向流程圖](#生成橫向流程圖)
  - [限制處理檔案數](#限制處理檔案數)
- [🎯 推薦工作流程](#-推薦工作流程)
- [📞 獲取幫助](#-獲取幫助)
- [🎉 開始使用](#-開始使用)

---

## 📑 快速導航

### 🚀 開始使用
- **[快速開始指南](QUICKSTART.md)** ⭐ 推薦新手閱讀
  - 最簡單的使用方式
  - 實際應用案例
  - 常見問題解答

### 📖 完整文檔
- **[完整工具文檔](AST_ANALYSIS_TOOLS_README.md)**
  - 四種工具的詳細說明
  - 技術細節
  - 支援的語言特性

### 📝 實施總結
- **[實施總結](IMPLEMENTATION_SUMMARY.md)**
  - 已創建文件清單
  - 工具對應關係
  - 技術實現細節

## 🛠️ 工具文件

### 核心工具
| 文件 | 語言 | 功能 | 目標引擎 |
|-----|------|------|---------|
| [py2mermaid.py](py2mermaid.py) | Python | Python AST → Mermaid | Python Engine |
| [go2mermaid.go](go2mermaid.go) | Go | Go AST → Mermaid | Go Engine |
| [ts2mermaid.ts](ts2mermaid.ts) | TypeScript | TS AST → Mermaid | **TypeScript Engine** |
| [rs2mermaid.rs](rs2mermaid.rs) | Rust | Rust AST → Mermaid | **Rust Engine** |

### 配置文件
| 文件 | 用途 |
|-----|------|
| [ts2mermaid-package.json](ts2mermaid-package.json) | TypeScript 工具依賴 |
| [rs2mermaid-Cargo.toml](rs2mermaid-Cargo.toml) | Rust 工具依賴 |

### 測試腳本
| 文件 | 用途 |
|-----|------|
| [test_ast_tools.ps1](test_ast_tools.ps1) | 自動化測試所有工具 |

## 🎯 使用場景對照表

### 想要分析 TypeScript Engine？
```bash
# 方法 1: 使用測試腳本（推薦）
.\test_ast_tools.ps1

# 方法 2: 直接使用工具
cd tools/common/development
cp ts2mermaid-package.json package.json
npm install
npx ts-node ts2mermaid.ts \
  -i ../../../services/scan/engines/typescript_engine \
  -o ../../../docs/diagrams/typescript
```
👉 詳見: [QUICKSTART.md](QUICKSTART.md#typescript-engine-分析)

### 想要分析 Rust Engine？
```bash
# 方法 1: 使用測試腳本（推薦）
.\test_ast_tools.ps1

# 方法 2: 直接使用工具
cd tools/common/development
cp rs2mermaid-Cargo.toml Cargo.toml
cargo run --bin rs2mermaid -- \
  -i ../../../services/scan/engines/rust_engine \
  -o ../../../docs/diagrams/rust
```
👉 詳見: [QUICKSTART.md](QUICKSTART.md#rust-engine-分析)

### 想要了解工具原理？
👉 閱讀: [AST_ANALYSIS_TOOLS_README.md](AST_ANALYSIS_TOOLS_README.md#技術細節)

### 想要自定義分析？
👉 參考: [AST_ANALYSIS_TOOLS_README.md](AST_ANALYSIS_TOOLS_README.md#最佳實踐)

### 遇到問題？
👉 查看: [QUICKSTART.md](QUICKSTART.md#疑難排解)

## 📊 工具比較

| 特性 | py2mermaid | go2mermaid | ts2mermaid | rs2mermaid |
|-----|-----------|-----------|-----------|-----------|
| 解析速度 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 記憶體使用 | 低 | 低 | 中 | 低 |
| 依賴安裝 | 無需 | 無需 | npm | cargo |
| 學習曲線 | 簡單 | 簡單 | 中等 | 中等 |
| 適合檔案數 | 1000+ | 2000+ | 500 | 5000+ |

## 🎓 學習路徑

### 新手路徑
1. 閱讀 [QUICKSTART.md](QUICKSTART.md)
2. 運行 `test_ast_tools.ps1` 體驗工具
3. 查看生成的 `.mmd` 檔案
4. 嘗試分析自己的程式碼

### 進階路徑
1. 閱讀 [AST_ANALYSIS_TOOLS_README.md](AST_ANALYSIS_TOOLS_README.md) 了解原理
2. 查看 [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) 了解實作細節
3. 自定義參數和輸出格式
4. 整合到 CI/CD 流程

### 專家路徑
1. 研究工具源碼
2. 擴展支援更多語言特性
3. 優化效能
4. 貢獻改進

## 📋 檢查清單

### 第一次使用
- [ ] 閱讀 [QUICKSTART.md](QUICKSTART.md)
- [ ] 確認環境需求（Python, Go, Node.js, Rust）
- [ ] 執行 `test_ast_tools.ps1`
- [ ] 檢查輸出目錄 `docs/diagrams/`

### 分析 TypeScript Engine
- [ ] 安裝 TypeScript 依賴
- [ ] 運行 ts2mermaid 工具
- [ ] 檢視生成的流程圖
- [ ] 整合到文檔

### 分析 Rust Engine
- [ ] 設置 Cargo 項目
- [ ] 運行 rs2mermaid 工具
- [ ] 檢視生成的流程圖
- [ ] 整合到文檔

## 🔗 快速連結

### 工具執行
- TypeScript 分析: `npx ts-node ts2mermaid.ts -i <input> -o <output>`
- Rust 分析: `cargo run --bin rs2mermaid -- -i <input> -o <output>`
- 完整測試: `.\test_ast_tools.ps1`

### 查看結果
- 輸出目錄: `docs/diagrams/`
- TypeScript 圖表: `docs/diagrams/typescript/`
- Rust 圖表: `docs/diagrams/rust/`

### 線上工具
- [Mermaid Live Editor](https://mermaid.live/) - 在線預覽
- [VS Code Mermaid Extension](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-mermaid) - VS Code 預覽

## 💡 常見使用模式

### 快速驗證工具
```bash
.\test_ast_tools.ps1
```

### 分析特定目錄
```bash
# TypeScript
npx ts-node ts2mermaid.ts -i ./src/services -o ./diagrams

# Rust
cargo run --bin rs2mermaid -- -i ./src/core -o ./diagrams
```

### 生成橫向流程圖
```bash
# TypeScript
npx ts-node ts2mermaid.ts -i ./src -o ./diagrams -d LR

# Rust
cargo run --bin rs2mermaid -- -i ./src -o ./diagrams -d LR
```

### 限制處理檔案數
```bash
# TypeScript
npx ts-node ts2mermaid.ts -i ./src -o ./diagrams -m 20

# Rust
cargo run --bin rs2mermaid -- -i ./src -o ./diagrams -m 20
```

## 🎯 推薦工作流程

1. **初次使用**: 運行測試腳本確認所有工具正常
2. **定期分析**: 在重大更新後重新生成流程圖
3. **文檔整合**: 將流程圖嵌入技術文檔
4. **程式碼審查**: 使用流程圖輔助審查
5. **重構參考**: 基於流程圖優化程式結構

## 📞 獲取幫助

- **快速問題**: 查看 [QUICKSTART.md](QUICKSTART.md#疑難排解)
- **深入問題**: 閱讀 [AST_ANALYSIS_TOOLS_README.md](AST_ANALYSIS_TOOLS_README.md#常見問題)
- **技術細節**: 參考 [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md#技術實現細節)

## 🎉 開始使用

最簡單的方式：

```bash
cd tools/common/development
.\test_ast_tools.ps1
```

這會自動完成所有設置並生成示例流程圖！

---

**更新日期**: 2025-11-22  
**版本**: 1.0.0  
**維護**: AIVA Development Team
