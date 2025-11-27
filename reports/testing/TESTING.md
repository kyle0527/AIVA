# 🧪 AIVA 測試工具

## 📑 目錄

- [主要測試工具](#主要測試工具)
  - [1. 📦 `aiva_test.py` - 完整測試套件](#1-aivatestpy-完整測試套件)
  - [2. 🚀 `quick_test.py` - 快速驗證](#2-quicktestpy-快速驗證)
  - [3. 🔍 `diagnose.py` - 系統診斷](#3-diagnosepy-系統診斷)
- [測試工具對比](#測試工具對比)
- [推薦使用流程](#推薦使用流程)
  - [1. 初始設置](#1-初始設置)
  - [2. 日常開發](#2-日常開發)
  - [3. 問題排查](#3-問題排查)
  - [4. CI/CD 流程](#4-cicd-流程)
- [整合測試](#整合測試)
- [已歸檔腳本](#已歸檔腳本)
- [測試目標](#測試目標)
- [常見問題](#常見問題)
  - [Q: TypeScript 引擎不可用？](#q-typescript-引擎不可用)
  - [Q: Go 引擎不可用？](#q-go-引擎不可用)
  - [Q: Docker 靶場無法連接？](#q-docker-靶場無法連接)
  - [Q: HTTP 請求失敗？](#q-http-請求失敗)
- [開發建議](#開發建議)

---

## 主要測試工具

AIVA 項目提供三個主要測試工具，涵蓋所有測試需求：

### 1. 📦 `aiva_test.py` - 完整測試套件

全功能測試工具，支持詳細的測試場景和配置。

**用法:**
```bash
# 測試引擎可用性
python aiva_test.py engines

# 測試 HTTP 連接
python aiva_test.py http

# 快速掃描單一目標
python aiva_test.py scan http://localhost:3000

# 使用 Playwright 動態掃描
python aiva_test.py dynamic http://localhost:3000

# 測試所有 Docker 靶場
python aiva_test.py all-targets

# 完整測試套件
python aiva_test.py full
```

**功能:**
- ✅ 引擎可用性檢查 (Python, TypeScript, Rust, Go)
- ✅ HTTP 客戶端測試
- ✅ 單目標快速掃描
- ✅ Playwright 動態掃描
- ✅ 多靶場測試
- ✅ 完整測試套件

### 2. 🚀 `quick_test.py` - 快速驗證

一鍵式快速測試，用於驗證系統基本功能。

**用法:**
```bash
# 運行所有測試
python quick_test.py

# 跳過掃描測試 (更快)
python quick_test.py --skip-scan
```

**測試項目:**
1. 引擎可用性 (3/4 引擎)
2. HTTP 連接測試
3. 命令處理器創建
4. 快速掃描測試 (可選)

**特點:**
- ⚡ 快速執行 (< 10 秒)
- 📊 清晰的測試結果摘要
- ✅ 返回正確的退出碼

### 3. 🔍 `diagnose.py` - 系統診斷

診斷工具，用於檢查系統配置和環境問題。

**用法:**
```bash
# 完整系統診斷
python diagnose.py

# 僅檢查引擎
python diagnose.py engines

# 檢查 Docker 靶場
python diagnose.py docker

# 測試 HTTP 連接
python diagnose.py http
```

**診斷項目:**
- 🐳 Docker 靶場狀態 (4 個容器)
- 🔧 引擎可用性 (包含錯誤提示)
- 🌐 HTTP 連接測試
- 💡 問題修復建議

---

## 測試工具對比

| 工具 | 用途 | 執行時間 | 適用場景 |
|------|------|----------|----------|
| `aiva_test.py` | 完整測試 | 1-5 分鐘 | 開發、調試、詳細測試 |
| `quick_test.py` | 快速驗證 | < 10 秒 | CI/CD、快速檢查 |
| `diagnose.py` | 系統診斷 | < 5 秒 | 問題排查、環境驗證 |

---

## 推薦使用流程

### 1. 初始設置
```bash
# 第一次設置時，診斷系統
python diagnose.py
```

### 2. 日常開發
```bash
# 快速驗證系統狀態
python quick_test.py

# 如果需要詳細測試
python aiva_test.py engines
python aiva_test.py scan http://localhost:3000
```

### 3. 問題排查
```bash
# 診斷問題
python diagnose.py

# 檢查特定引擎
python diagnose.py engines

# 檢查 Docker 靶場
python diagnose.py docker
```

### 4. CI/CD 流程
```bash
# 快速驗證 (不執行掃描)
python quick_test.py --skip-scan

# 或完整驗證
python quick_test.py
```

---

## 整合測試

除了上述三個工具外，還有 **4 個整合測試腳本** 用於測試複雜業務流程：

**位置**: `tests/integration/`

| 測試腳本 | 用途 | 執行時間 |
|---------|------|----------|
| `test_ai_command_scan.py` | AI 命令中心整合 | 2-5 分鐘 |
| `test_dual_loop_juice_shop.py` | 雙閉環系統實戰 | 5-15 分鐘 |
| `test_two_phase_scan.py` | 兩階段掃描編排 | 15-30 分鐘 |
| `test_multi_language_analysis.py` | 多語言能力分析 | 1-3 分鐘 |

**詳細說明**: 請參考 `tests/integration/README.md`

---

## 已歸檔腳本

功能重複的舊測試腳本已移動到 `_archive/old_tests/` 目錄（13 個）。這些腳本的功能已整合到上述三個主要工具中。

---

## 測試目標

所有測試都使用合法的 OWASP 安全靶場：

| 靶場 | 端口 | 容器名稱 |
|------|------|----------|
| OWASP Juice Shop | 3000 | juice-shop-live |
| OWASP Juice Shop | 3001 | ecstatic_ritchie |
| OWASP Juice Shop | 3003 | vigilant_shockley |
| OWASP WebGoat | 8080 | laughing_jang |

**啟動靶場:**
```bash
# 如果靶場未運行，請使用 Docker Compose
cd docker
docker-compose up -d
```

---

## 常見問題

### Q: TypeScript 引擎不可用？
**A:** TypeScript 引擎需要編譯：
```bash
cd services/scan/engines/typescript_engine
npm install
npm run build
```

### Q: Go 引擎不可用？
**A:** Go 掃描器應該已經編譯完成 (`scanner.exe`)。如果沒有：
```bash
cd services/scan/engines/go_engine
go build -o scanner.exe ./cmd/ssrf-scanner
```

### Q: Docker 靶場無法連接？
**A:** 確保 Docker 容器正在運行：
```bash
docker ps
# 或使用診斷工具
python diagnose.py docker
```

### Q: HTTP 請求失敗？
**A:** 檢查防火牆和網絡設置：
```bash
python diagnose.py http
```

---

## 開發建議

- ✅ 提交代碼前運行 `python quick_test.py`
- ✅ 修改引擎後運行 `python diagnose.py engines`
- ✅ 修改 HTTP 相關代碼後運行 `python aiva_test.py http`
- ✅ 添加新功能時使用 `aiva_test.py` 進行詳細測試

---

最後更新: 2025-11-22
