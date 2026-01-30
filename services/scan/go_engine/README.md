# 🔵 AIVA Go Engine - 快速並發掃描引擎

> **版本**: v3.0 | **狀態**: ✅ Production Ready | **更新**: 2026-01-23

---

## 📋 概述

**Go Engine** 是 AIVA 的快速並發掃描引擎，專注於需要大量並發和快速響應的安全檢測任務。

### 🎯 核心能力

- ✅ **SSRF 檢測** - 服務器端請求偽造檢測
- ✅ **SCA (軟體組成分析)** - 依賴漏洞掃描
- ✅ **CSPM (雲安全態勢管理)** - 雲配置檢測
- ✅ **參數模糊測試** - HTTP 參數注入測試

---

## 🏗️ 架構設計

```
go_engine/
├── cmd/
│   └── scanner/
│       └── main.go              # CLI 入口
├── internal/
│   ├── ssrf/
│   │   └── detector.go          # SSRF 檢測器
│   ├── sca/
│   │   └── scanner.go           # SCA 掃描器
│   └── cspm/
│       └── checker.go           # CSPM 檢查器
├── pkg/
│   ├── http/
│   │   └── client.go            # HTTP 客戶端
│   └── payload/
│       └── generator.go         # Payload 生成器
├── go.mod                       # Go 模組定義
└── Makefile                     # 編譯腳本
```

---

## 🚀 快速開始

### 1️⃣ 編譯

```bash
cd services/scan/go_engine
make build

# 或手動編譯
go build -o bin/scanner cmd/scanner/main.go
```

### 2️⃣ 運行

```bash
# SSRF 檢測
./bin/scanner ssrf --url https://example.com/api?url=

# SCA 掃描
./bin/scanner sca --target ./package.json

# CSPM 檢查
./bin/scanner cspm --provider aws --region us-east-1
```

---

## 🔧 主要模組

### 1. SSRF 檢測器

**文件**: `internal/ssrf/detector.go`

**檢測方法**:
- **基線比對** - MD5 hash + 響應長度
- **OOB 回調** - Burp Collaborator 集成
- **雲端 Metadata** - AWS/Azure/GCP 掃描
- **繞過技巧** - IP 進制轉換、URL 語義攻擊

**特性**:
- DNS Rebinding 檢測
- 內部服務探測（127.0.0.1、localhost）
- 協議走私（file://、gopher://、dict://）

**符合標準**:
- ✅ OWASP WSTG-INPV-19
- ✅ PortSwigger SSRF 測試方法

### 2. SCA 掃描器

**功能**:
- NPM / Composer / Maven 依賴分析
- CVE 數據庫比對
- 漏洞版本檢測
- 修復建議生成

### 3. CSPM 檢查器

**支援雲平台**:
- AWS（S3、IAM、EC2、RDS）
- Azure（Storage、RBAC、VM）
- GCP（Cloud Storage、IAM）

**檢查項目**:
- 公開訪問檢測
- 權限過度分配
- 加密配置缺失
- 日誌記錄不完整

---

## 📊 性能特點

| 特性 | 指標 |
|------|------|
| 並發能力 | 5,000+ goroutines |
| 記憶體佔用 | < 100MB |
| 啟動速度 | < 1秒 |
| 編譯產物大小 | ~8MB（靜態編譯） |

---

## 🔗 相關文檔

- [主掃描模組 README](../README.md)
- [SSRF 檢測報告](../SCAN_ENGINE_ENHANCEMENT_REPORT.md)
- [Go 官方文檔](https://go.dev/)

---

## 📝 許可證

MIT License - 詳見主專案 LICENSE 文件
