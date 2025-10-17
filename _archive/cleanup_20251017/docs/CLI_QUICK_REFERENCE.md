# AIVA CLI 速查表 🚀

## 🎯 快速開始

```bash
# 安裝
pip install -e .

# 驗證
aiva --help
```

## 📋 命令總覽

| 類別 | 命令 | 說明 |
|------|------|------|
| 掃描 | `aiva scan start <url>` | 啟動網站掃描 |
| 檢測 | `aiva detect sqli <url> --param <name>` | SQL 注入檢測 |
| 檢測 | `aiva detect xss <url> --param <name>` | XSS 檢測 |
| AI | `aiva ai train --mode <mode>` | 訓練 AI 模型 |
| AI | `aiva ai status` | 查看 AI 狀態 |
| 報告 | `aiva report generate <scan_id>` | 生成報告 |
| 系統 | `aiva system status` | 查看系統狀態 |
| 工具 | `aiva tools schemas` | 導出 JSON Schema |
| 工具 | `aiva tools typescript` | 導出 TS 型別 |
| 工具 | `aiva tools export-all` | 一鍵導出全部 |

## ⚡ 常用命令

### 掃描網站
```bash
# 基本掃描
aiva scan start https://example.com

# 深度掃描
aiva scan start https://example.com --max-depth 5 --max-pages 200 --wait
```

### 漏洞檢測
```bash
# SQL 注入
aiva detect sqli https://example.com/login --param username --wait

# XSS 檢測
aiva detect xss https://example.com/search --param q --type reflected --wait
```

### AI 訓練
```bash
# 實時訓練
aiva ai train --mode realtime --epochs 10

# 模擬訓練
aiva ai train --mode simulation --scenarios 500 --epochs 15
```

### 報告生成
```bash
# HTML 報告
aiva report generate scan_xxx --format html --output report.html

# PDF 報告
aiva report generate scan_xxx --format pdf --output report.pdf

# JSON 報告
aiva report generate scan_xxx --format json --output data.json
```

### 型別導出
```bash
# 導出 JSON Schema
aiva tools schemas --out contracts/schemas.json

# 導出 TypeScript
aiva tools typescript --out types/aiva.d.ts

# 一鍵全部導出
aiva tools export-all --out-dir contracts
```

## 🔑 關鍵選項

| 選項 | 說明 | 預設值 |
|------|------|--------|
| `--max-depth` | 爬取深度 | 3 |
| `--max-pages` | 最大頁數 | 100 |
| `--wait` | 等待完成 | false |
| `--format` | 輸出格式 (human/json) | human |
| `--timeout` | 逾時秒數 | 30 |
| `--concurrency` | 併發數 | 8 |

## 🎨 輸出格式

大部分命令支援 `--format` 選項：

```bash
# 人類可讀（預設）
aiva tools schemas --format human

# JSON 格式（用於自動化）
aiva tools schemas --format json
```

## 🔄 完整工作流程

```bash
# 1️⃣ 掃描
aiva scan start https://example.com --wait

# 2️⃣ 檢測
aiva detect sqli https://example.com/login --param user --wait
aiva detect xss https://example.com/search --param q --wait

# 3️⃣ 生成報告
aiva report generate scan_xxx --format pdf --output final.pdf

# 4️⃣ 導出協定（供其他語言使用）
aiva tools export-all --out-dir contracts
```

## 🐛 除錯技巧

```bash
# 查看詳細幫助
aiva <command> --help

# 檢查系統狀態
aiva system status

# 查看 AI 狀態
aiva ai status

# 使用 JSON 輸出便於除錯
aiva tools schemas --format json | jq .
```

## 📦 跨模組整合

```bash
# 導出 JSON Schema（用於 Go/Rust/TypeScript）
aiva tools schemas --out _out/schemas.json

# 導出 TypeScript 型別
aiva tools typescript --out _out/types.d.ts

# 這些檔案可用於：
# - 前端開發
# - Go/Rust 生成對應型別
# - API 文件
# - 測試資料驗證
```

## 🔧 環境變數（計劃中）

```bash
# 設定預設值
export AIVA_MAX_DEPTH=5
export AIVA_TIMEOUT=60
export AIVA_CONCURRENCY=16

# 優先級：命令列 > 環境變數 > 配置檔
```

## 📊 退出碼

```bash
0   # 成功
1   # 使用錯誤
2   # 系統錯誤
10+ # 業務錯誤
```

## 🔗 另請參閱

- [完整命令參考](./CLI_COMMAND_REFERENCE.md)
- [安裝指南](./CLI_UNIFIED_SETUP_GUIDE.md)
- [快速開始](./QUICK_START.md)

---

💡 **提示**: 使用 `aiva <command> --help` 查看任何命令的詳細說明！
