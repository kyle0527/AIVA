# 🦀 AIVA Rust Engine - 高性能掃描引擎

> **版本**: v3.0 | **狀態**: ✅ Production Ready | **更新**: 2026-01-23

---

## 📋 概述

**Rust Engine** 是 AIVA 的高性能掃描引擎，專注於需要極致性能和並發的安全檢測任務。

### 🎯 核心能力

- ✅ **端口掃描** - 高速網絡探測
- ✅ **HTTP Request Smuggling 檢測** - CL.TE / TE.CL / TE.TE
- ✅ **信息收集** - Web 資產枚舉
- ✅ **認證爆破** - 智能速率控制

---

## 🏗️ 架構設計

```
rust_engine/
├── src/
│   ├── main.rs                    # CLI 入口
│   ├── scanner.rs                 # 核心掃描器
│   ├── smuggling_detector_v2.rs   # HTTP Smuggling 檢測
│   └── auth_bruteforcer.rs        # 認證爆破
├── Cargo.toml                     # 依賴配置
└── target/release/                # 編譯輸出
    └── aiva-info-gatherer         # 可執行檔案
```

---

## 🚀 快速開始

### 1️⃣ 編譯

```bash
cd services/scan/rust_engine
cargo build --release
```

### 2️⃣ 運行

```bash
# 端口掃描
./target/release/aiva-info-gatherer scan --target example.com --ports 80,443

# HTTP Smuggling 檢測
./target/release/aiva-info-gatherer smuggling --url https://example.com

# 信息收集
./target/release/aiva-info-gatherer gather --url https://example.com
```

---

## 🔧 主要模組

### 1. HTTP Request Smuggling 檢測器

**文件**: `src/smuggling_detector_v2.rs`

**檢測類型**:
- **CL.TE**: Content-Length vs Transfer-Encoding
- **TE.CL**: Transfer-Encoding vs Content-Length  
- **TE.TE**: 雙重 Transfer-Encoding 混淆
- **Chunk Obfuscation**: Chunk 編碼混淆

**特性**:
- 基線響應比對（MD5 + 長度）
- 時間差異檢測
- 智能 WAF 繞過

### 2. 端口掃描器

**特性**:
- 多線程並發掃描
- SYN/Connect 掃描模式
- 服務指紋識別

### 3. 認證爆破器

**特性**:
- 智能速率限制
- 自動失敗檢測
- 多協議支持（HTTP/SSH/FTP）

---

## 📊 性能特點

| 特性 | 指標 |
|------|------|
| 並發能力 | 10,000+ 連接/秒 |
| 記憶體佔用 | < 50MB |
| CPU 使用率 | 高效多核利用 |
| 編譯產物大小 | ~5MB（release） |

---

## 🔗 相關文檔

- [主掃描模組 README](../README.md)
- [HTTP Smuggling 報告](../SCAN_ENGINE_ENHANCEMENT_REPORT.md)
- [Cargo 官方文檔](https://doc.rust-lang.org/cargo/)

---

## 📝 許可證

MIT License - 詳見主專案 LICENSE 文件
