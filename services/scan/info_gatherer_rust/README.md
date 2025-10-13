# AIVA Sensitive Info Gatherer (Rust)

使用 Rust 實現的超高性能敏感資訊掃描器。

## 特性

- ✅ **極致性能**: Rust 零成本抽象,正則引擎比 Python 快 10-100 倍
- ✅ **並行處理**: Rayon 並行掃描,充分利用多核 CPU
- ✅ **低記憶體**: 單次掃描僅需 ~5 MB 記憶體
- ✅ **多種模式**: 支援 10+ 種敏感資訊檢測

## 安裝

```powershell
# 安裝 Rust (如果還沒有)
# https://www.rust-lang.org/tools/install

# 編譯 (釋出版本)
cargo build --release
```

## 運行

```powershell
# 開發模式
cargo run

# 釋出模式 (優化編譯)
.\target\release\aiva-info-gatherer.exe
```

## 支援的敏感資訊類型

1. **AWS Access Key** - `AKIA[0-9A-Z]{16}`
2. **AWS Secret Key** - `aws(.{0,20})?['"][0-9a-zA-Z/+]{40}['\"]`
3. **GitHub Token** - `ghp_[0-9a-zA-Z]{36}`
4. **Generic API Key** - `api_key = "..."`
5. **Private Key** - `-----BEGIN PRIVATE KEY-----`
6. **Email** - `user@example.com`
7. **IP Address** - `192.168.1.1`
8. **JWT Token** - `eyJ...`
9. **Password in Code** - `password = "..."`
10. **Database Connection String** - `mysql://...`

## 任務格式

```json
{
  "task_id": "task_xxx",
  "content": "const api_key = 'sk_test_1234567890';",
  "source_url": "https://example.com/config.js"
}
```

## 結果格式

```json
{
  "task_id": "task_xxx",
  "info_type": "Generic API Key",
  "value": "sk_t**********7890",
  "confidence": 0.75,
  "location": "https://example.com/config.js:15"
}
```

## 性能基準

在 AMD Ryzen 5 5600 上測試:

- **小文件 (10 KB)**: ~0.5 ms
- **中文件 (100 KB)**: ~2 ms
- **大文件 (1 MB)**: ~15 ms
- **記憶體佔用**: ~5 MB

**對比 Python re 模組**: 快 10-100 倍

## 測試

```powershell
# 運行單元測試
cargo test

# 運行基準測試
cargo bench
```

## 優化細節

1. **Aho-Corasick 算法**: 快速關鍵字過濾
2. **Rayon 並行**: 多模式並行匹配
3. **零拷貝**: 使用字串切片避免記憶體分配
4. **LTO 優化**: Link-Time Optimization
5. **Strip 符號**: 移除除錯符號減小體積
