# AIVA 專案安裝指南

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

✅ **已完成安裝** (2025-11-13)

- ✅ Python 虛擬環境: `.venv/`
- ✅ 套件已安裝: `aiva-platform-integrated 1.0.0`
- ✅ 可編輯模式 (editable install)
- ✅ 所有依賴已安裝

**驗證方式**:
```powershell
# 檢查安裝狀態
C:/D/fold7/AIVA-git/.venv/Scripts/python.exe -m pip list | Select-String "aiva"

# 預期輸出:
# aiva-platform-integrated 1.0.0     C:\D\fold7\AIVA-git
```

---

## 🚀 快速開始 (已安裝用戶)

如果專案已完成安裝,只需激活虛擬環境:

```powershell
# 激活虛擬環境
& C:/D/fold7/AIVA-git/.venv/Scripts/Activate.ps1

# 驗證 Python 版本
python --version
# Python 3.13.9

# 開始使用
python -m pytest services/core/tests/ -v
```

---

## 📦 完整安裝步驟 (新環境)

### 前置需求

- **Python**: 3.13.9 或更高版本
- **Git**: 用於版本控制
- **虛擬環境**: 推薦使用 venv

### 步驟 1: 建立虛擬環境

```powershell
# 切換到專案目錄
cd C:\D\fold7\AIVA-git

# 建立虛擬環境
python -m venv .venv

# 激活虛擬環境
& .venv\Scripts\Activate.ps1
```

### 步驟 2: 升級 pip 工具

```powershell
# 升級 pip, setuptools, wheel
python -m pip install --upgrade pip setuptools wheel
```

### 步驟 3: 安裝專案 (可編輯模式)

```powershell
# 方案 A: 基礎安裝
pip install -e .

# 方案 B: 含開發工具 (推薦)
pip install -e ".[dev]"

# 方案 C: 完整安裝 (包含所有依賴)
pip install -e .
pip install -r requirements.txt
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
├── .venv/                           # Python 虛擬環境
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
# 激活虛擬環境
& .venv\Scripts\Activate.ps1

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

**原因**: 專案未安裝或虛擬環境未激活

**解決方式**:
```powershell
# 激活虛擬環境
& .venv\Scripts\Activate.ps1

# 確認是否已安裝
pip list | Select-String "aiva"

# 如未安裝,執行:
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

**原因**: IDE 未正確識別虛擬環境

**解決方式**:
```
1. 在 VS Code 中按 Ctrl+Shift+P
2. 輸入 "Python: Select Interpreter"
3. 選擇 ".venv\Scripts\python.exe"
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
2. 確認虛擬環境已激活: `which python` (應指向 `.venv/`)
3. 查看錯誤日誌: `pip install -e . --verbose`
4. 參考文件: [DEPENDENCY_ANALYSIS.md](./services/core/DEPENDENCY_ANALYSIS.md)

---

**最後更新**: 2025-11-13  
**版本**: 1.0.0  
**狀態**: ✅ 已完成安裝
