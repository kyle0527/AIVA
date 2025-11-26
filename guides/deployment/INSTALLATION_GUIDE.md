# AIVA 專案安裝指南

## 📋 目錄

- [📑 目錄](#目錄)
- [📋 安裝狀態](#安裝狀態)
- [🚀 快速開始 (已安裝用戶)](#快速開始-已安裝用戶)
- [📦 完整安裝步驟 (新環境)](#完整安裝步驟-新環境)
  - [前置需求](#前置需求)
  - [步驟 1: 切換到專案目錄](#步驟-1-切換到專案目錄)
  - [步驟 2: 升級 pip 工具](#步驟-2-升級-pip-工具)
  - [步驟 3: 安裝專案 (可編輯模式)](#步驟-3-安裝專案-可編輯模式)
  - [步驟 4: 生成 Protocol Buffers 代碼](#步驟-4-生成-protocol-buffers-代碼)
  - [步驟 4: 驗證安裝](#步驟-4-驗證安裝)
- [🔧 安裝方式詳解](#安裝方式詳解)
  - [Option A: 可編輯安裝 (推薦用於開發)](#option-a-可編輯安裝-推薦用於開發)
  - [Option B: requirements.txt 安裝](#option-b-requirementstxt-安裝)
  - [Option C: 使用官方腳本](#option-c-使用官方腳本)
- [📂 專案結構](#專案結構)
- [🎯 導入方式說明](#導入方式說明)
  - [✅ 正確的導入方式](#正確的導入方式)
  - [❌ 錯誤的導入方式](#錯誤的導入方式)
- [🧪 執行測試](#執行測試)
  - [執行單一測試](#執行單一測試)
  - [執行所有測試](#執行所有測試)
  - [測試配置](#測試配置)
- [🔍 常見問題](#常見問題)
  - [Q1: `ModuleNotFoundError: No module named 'services'`](#q1-modulenotfounderror-no-module-named-services)
  - [Q2: `ModuleNotFoundError: No module named 'XXX'`](#q2-modulenotfounderror-no-module-named-xxx)
  - [Q3: 代碼修改後沒有生效](#q3-代碼修改後沒有生效)
  - [Q4: IDE 無法自動完成或找不到模組](#q4-ide-無法自動完成或找不到模組)
- [📦 依賴管理](#依賴管理)
  - [核心依賴 (pyproject.toml)](#核心依賴-pyprojecttoml)
  - [可選依賴](#可選依賴)
  - [完整依賴清單](#完整依賴清單)
- [🛠️ 開發工具](#開發工具)
  - [代碼格式化](#代碼格式化)
  - [類型檢查](#類型檢查)
  - [Pre-commit Hooks](#pre-commit-hooks)
- [🚀 生產部署](#生產部署)
  - [使用 Docker](#使用-docker)
  - [使用標準安裝](#使用標準安裝)
- [📚 相關文件](#相關文件)
- [🆘 獲取幫助](#獲取幫助)
- [🔗 相關資源](#相關資源)
  - [生產環境部署](#生產環境部署)
  - [開發環境](#開發環境)
  - [故障排除](#故障排除)
  - [使用者手冊](#使用者手冊)

> **重要**: 本專案已完成初始安裝設定,本文件提供完整的安裝步驟說明。

## 📑 目錄

- [📋 安裝狀態](#安裝狀態)
- [🚀 快速開始 (已安裝用戶)](#快速開始-已安裝用戶)
- [⚙️ 系統要求](#系統要求)
- [📍 詳細安裝步驟](#詳細安裝步驟)
  - [步驟 1: 克隆專案](#步驟-1-克隆專案)
  - [步驟 2: 設定 Python 環境](#步驟-2-設定-python-環境)
  - [步驟 3: 安裝依賴](#步驟-3-安裝依賴)
  - [步驟 4: 配置環境變數](#步驟-4-配置環境變數)
  - [步驟 5: 驗證安裝](#步驟-5-驗證安裝)
- [🐛 常見問題與解決](#常見問題與解決)
- [🚀 進階配置](#進階配置)
- [📞 支援](#支援)

---

## 📋 安裝狀態

✅ **已完成全域安裝** (2025-11-25)

- ✅ Python 套件: 全域環境
- ✅ 套件已安裝: `aiva-platform-integrated 1.0.0`
- ✅ 可編輯模式 (editable install)
- ✅ 所有依賴已安裝至全域

**驗證方式**:
```powershell
# 檢查安裝狀態
python -m pip list | Select-String "aiva"

# 預期輸出:
# aiva-platform-integrated 1.0.0     C:\D\fold7\AIVA-git
```

---

## 🚀 快速開始 (已安裝用戶)

專案已使用全域 Python 環境，直接使用即可:

```powershell
# 驗證 Python 版本
python --version
# Python 3.13.x

# 開始使用
python -m pytest services/core/tests/ -v
```

---

## 📦 完整安裝步驟 (新環境)

### 前置需求

- **Python**: 3.13.9 或更高版本 (全域安裝)
- **Git**: 用於版本控制
- **安裝策略**: 使用全域 Python 環境，不使用虛擬環境

### 步驟 1: 切換到專案目錄

```powershell
# 切換到專案目錄
cd C:\D\fold7\AIVA-git
```

### 步驟 2: 升級 pip 工具

```powershell
# 升級 pip, setuptools, wheel
python -m pip install --upgrade pip setuptools wheel
```

### 步驟 3: 安裝專案 (可編輯模式)

```powershell
# 安裝所有依賴到全域環境
pip install -r requirements.txt

# 將專案本身以可編輯模式安裝到全域
pip install -e .
```

### 步驟 4: 生成 Protocol Buffers 代碼

```powershell
# 進入 protocols 目錄
cd services/aiva_common/protocols

# 執行 protobuf 編譯腳本
python generate_proto.py

# 返回專案根目錄
cd ../../..

# 驗證生成結果
python -c "from services.aiva_common.protocols import aiva_services_pb2; print('Protobuf OK')"
```

**說明**:
- `-e` 表示可編輯模式 (editable install)
- 代碼修改會立即生效,無需重新安裝
- 支援跨模組導入 (`from services.xxx import ...`)

### 步驟 4: 驗證安裝

```powershell
# 檢查已安裝的套件
pip list | Select-String "aiva"

# 預期輸出:
# aiva-platform-integrated 1.0.0     C:\D\fold7\AIVA-git

# 測試導入
python -c "import services; print('✓ services 套件正常')"

# 執行測試
pytest services/core/tests/ -v
```

---

## 🔧 安裝方式詳解

### Option A: 可編輯安裝 (推薦用於開發)

```powershell
pip install -e .
```

**優點**:
- ✅ 代碼修改立即生效
- ✅ 支援所有導入模式
- ✅ IDE 自動完成正常
- ✅ 符合 Python 標準 (PEP 517/518)

**適用場景**:
- 日常開發
- 功能開發與測試
- 調試與除錯

---

### Option B: requirements.txt 安裝

```powershell
pip install -r requirements.txt
```

**優點**:
- ✅ 快速安裝所有外部依賴
- ✅ 固定版本號
- ✅ 適合生產環境

**注意**:
- ⚠️ 不會安裝內部 aiva 套件
- ⚠️ 需要額外執行 `pip install -e .`

**適用場景**:
- CI/CD 環境
- Docker 容器
- 生產部署

---

### Option C: 使用官方腳本

```powershell
.\scripts\common\setup\setup_multilang.ps1
```

**功能**:
- ✅ 自動升級 pip
- ✅ 執行 `pip install -e .`
- ✅ 安裝 Node.js 依賴 (如有)
- ✅ 安裝 Playwright (如需要)
- ✅ 處理跨語言依賴

**適用場景**:
- 首次設定開發環境
- 需要多語言支援 (Python, Node.js, Go, Rust)
- 完整環境初始化

---

## 📂 專案結構

```
AIVA-git/
├── pyproject.toml                   # 主專案配置
├── requirements.txt                 # Python 依賴清單
│
├── services/                        # 服務層 (Python 套件)
│   ├── __init__.py
│   ├── pyproject.toml               # 服務層配置
│   │
│   ├── aiva_common/                 # 共用模組
│   │   ├── __init__.py
│   │   ├── pyproject.toml
│   │   └── requirements.txt
│   │
│   ├── core/                        # 核心服務
│   │   ├── __init__.py
│   │   ├── aiva_core/               # AIVA 核心引擎
│   │   ├── tests/                   # 測試檔案
│   │   └── requirements.txt
│   │
│   ├── integration/                 # 整合服務
│   ├── features/                    # 功能服務
│   └── scan/                        # 掃描服務
│
└── scripts/                         # 工具腳本
    └── common/setup/
        └── setup_multilang.ps1      # 自動化設定腳本
```

---

## 🎯 導入方式說明

### ✅ 正確的導入方式

```python
# 方式 1: 直接導入 (推薦)
from aiva_common import Config
from aiva_common.enums import Severity, Confidence

# 方式 2: 使用 services 前綴 (可編輯安裝後支援)
from services.core import models
from services.integration.capability import CapabilityRegistry
```

### ❌ 錯誤的導入方式

```python
# 錯誤 1: 使用三點相對導入跨越套件邊界
from ...aiva_common import Config  # ❌

# 錯誤 2: 混用導入風格
from services.aiva_common import Config  # ❌ (舊式,已禁用)
```

**說明**:
- 使用 `pip install -e .` 後,Python 會自動處理所有導入
- 無需手動修改 `sys.path`
- 符合 DEVELOPMENT_STANDARDS.md 規範

---

## 🧪 執行測試

### 執行單一測試

```powershell
# 執行特定測試
pytest services/core/tests/test_module_explorer.py -v
```

### 執行所有測試

```powershell
# 執行核心服務測試
pytest services/core/tests/ -v

# 含覆蓋率報告
pytest services/core/tests/ --cov=services.core --cov-report=html
```

### 測試配置

測試配置位於 `services/core/pytest.ini`:

```ini
[pytest]
pythonpath = ..
asyncio_mode = auto
testpaths = tests
addopts = -v --tb=short
```

---

## 🔍 常見問題

### Q1: `ModuleNotFoundError: No module named 'services'`

**原因**: 專案未安裝或套件缺失

**解決方式**:
```powershell
# 確認是否已安裝
pip list | Select-String "aiva"

# 如未安裝,執行:
pip install -e .
```
pip install -e .
```

---

### Q2: `ModuleNotFoundError: No module named 'XXX'`

**原因**: 缺少外部依賴

**解決方式**:
```powershell
# 安裝完整依賴
pip install -r requirements.txt

# 或安裝特定套件
pip install XXX
```

---

### Q3: 代碼修改後沒有生效

**原因**: 可能使用了標準安裝 (`pip install .`) 而非可編輯安裝

**解決方式**:
```powershell
# 重新安裝為可編輯模式
pip uninstall aiva-platform-integrated
pip install -e .
```

---

### Q4: IDE 無法自動完成或找不到模組

**原因**: IDE 未正確識別 Python 解譯器

**解決方式**:
```
1. 在 VS Code 中按 Ctrl+Shift+P
2. 輸入 "Python: Select Interpreter"
3. 選擇全域 Python (`python.exe`)
4. 重新載入視窗
```

---

## 📦 依賴管理

### 核心依賴 (pyproject.toml)

- `fastapi>=0.115.0` - Web 框架
- `pydantic>=2.7.0` - 數據驗證
- `sqlalchemy>=2.0.31` - ORM
- `redis>=5.0.0` - 快取
- `neo4j>=5.23.0` - 圖數據庫
- ...等 13 個核心依賴

### 可選依賴

```powershell
# 開發工具
pip install -e ".[dev]"

# RabbitMQ 支援
pip install -e ".[rabbit]"

# PDF 生成
pip install -e ".[pdf]"

# 監控工具
pip install -e ".[monitoring]"
```

### 完整依賴清單

詳見:
- `requirements.txt` - 完整 Python 依賴 (60+ 套件)
- `services/core/requirements.txt` - 核心服務依賴
- `services/aiva_common/requirements.txt` - 共用模組依賴

---

## 🛠️ 開發工具

### 代碼格式化

```powershell
# Black (代碼格式化)
black services/ --line-length 88

# Ruff (快速 Linting)
ruff check services/ --fix
```

### 類型檢查

```powershell
# MyPy (靜態類型檢查)
mypy services/core/
```

### Pre-commit Hooks

```powershell
# 安裝 pre-commit hooks
pre-commit install

# 手動執行
pre-commit run --all-files
```

---

## 🚀 生產部署

### 使用 Docker

```powershell
# 建置映像
docker-compose build

# 啟動服務
docker-compose up -d
```

### 使用標準安裝

```powershell
# 安裝 (非可編輯模式)
pip install .

# 或使用 requirements.txt
pip install -r requirements.txt
```

---

## 📚 相關文件

- [README.md](./README.md) - 專案概述
- [USAGE_GUIDE.md](./services/core/aiva_core/USAGE_GUIDE.md) - 使用指南
- [DEVELOPMENT_STANDARDS.md](./docs/DEVELOPMENT_STANDARDS.md) - 開發規範
- [IMPORT_FIX_PROGRESS.md](./services/core/IMPORT_FIX_PROGRESS.md) - 導入修復記錄
- [DEPENDENCY_ANALYSIS.md](./services/core/DEPENDENCY_ANALYSIS.md) - 依賴分析

---

## 🆘 獲取幫助

如遇到安裝問題:

1. 檢查 Python 版本: `python --version` (需要 3.13+)
2. 確認 Python 在系統 PATH 中: `python --version`
3. 查看錯誤日誌: `pip install -e . --verbose`
4. 參考文件: [DEPENDENCY_ANALYSIS.md](./services/core/DEPENDENCY_ANALYSIS.md)

---

**最後更新**: 2025-11-13  
**版本**: 1.0.0  
**狀態**: ✅ 已完成安裝


---

## 🔗 相關資源

### 生產環境部署
- 📖 [系統安裝指南](./SYSTEM_INSTALLATION_GUIDE.md) - **生產環境完整安裝** (Python/Node/Go/Rust/PostgreSQL/Redis)
- 📖 [生產環境故障排除指南](../troubleshooting/PRODUCTION_TROUBLESHOOTING_GUIDE.md) - 運行時問題解決
- 📖 [構建指南](./BUILD_GUIDE.md) - 項目構建
- 📖 [Docker/K8s 指南](./DOCKER_KUBERNETES_GUIDE.md) - 容器化部署
- 📖 [部署檢查清單](../../docs/DEPLOYMENT_CHECKLIST.md) - 發布前修復項目

### 開發環境
- 📍 [當前文件](./INSTALLATION_GUIDE.md) - **開發環境** Python 全域安裝
- 📖 [開發快速指南](../development/DEVELOPMENT_QUICK_START_GUIDE.md)
- 📖 [依賴管理指南](../development/DEPENDENCY_MANAGEMENT_GUIDE.md)

### 故障排除
- 📖 [性能優化指南](../troubleshooting/PERFORMANCE_OPTIMIZATION_GUIDE.md)
- 📖 [測試重現指南](../troubleshooting/TESTING_REPRODUCTION_GUIDE.md)

### 使用者手冊
- 📚 [AIVA 使用者手冊](../../docs/user_guides/00_general/AIVA_USER_MANUAL.md)

