# 🧪 AIVA 測試套件

本目錄包含 AIVA 專案的各種測試腳本和驗證工具。

## 📁 測試文件

### 🧠 AI 相關測試
- `test_ai_integration.py` - AI 整合測試
- `verify_ai_working.py` - AI 功能驗證

### 🏗️ 架構測試
- `test_architecture_improvements.py` - 架構改進測試
- `test_improvements_simple.py` - 簡化改進測試

### 🔗 整合測試
- `test_integration.py` - 系統整合測試
- `test_module_imports.py` - 模組導入測試

### 🔍 系統測試
- `test_complete_system.py` - 完整系統測試
- `test_scan.ps1` - 掃描功能測試

## 🚀 執行測試

### 完整測試套件
```bash
# Python 測試
python -m pytest tests/ -v

# 或者單獨執行
python tests/test_complete_system.py
```

### AI 功能測試
```bash
# AI 整合測試
python tests/test_ai_integration.py

# AI 功能驗證
python tests/verify_ai_working.py
```

### 架構測試
```bash
# 架構改進測試
python tests/test_architecture_improvements.py

# 簡化測試
python tests/test_improvements_simple.py
```

### 掃描測試
```powershell
# PowerShell 掃描測試
.\tests\test_scan.ps1
```

## 📊 測試報告

測試執行後會生成以下報告：
- `test_results.json` - JSON 格式測試結果
- `coverage_report.html` - 程式碼覆蓋率報告
- `performance_metrics.json` - 性能指標

## 🛠️ 測試環境

### 前置需求
- Python 3.11+
- pytest 7.0+
- Docker & Docker Compose
- 所有服務正常運行

### 環境變數
```bash
export AIVA_TEST_MODE=true
export AIVA_LOG_LEVEL=DEBUG
export AIVA_DB_TEST_URL=postgresql://test:test@localhost:5432/aiva_test
```

### Docker 測試環境
```bash
# 啟動測試服務
docker-compose -f docker/docker-compose.test.yml up -d

# 執行測試
python -m pytest tests/

# 清理測試環境
docker-compose -f docker/docker-compose.test.yml down
```

## 📋 測試策略

### 單元測試
- 個別函數和類別測試
- Mock 外部依賴
- 快速執行（< 1 秒）

### 整合測試
- 模組間互動測試
- 真實資料庫連接
- 中等執行時間（< 30 秒）

### 系統測試
- 端到端功能測試
- 完整工作流程驗證
- 長時間執行（< 5 分鐘）

### 性能測試
- 負載測試
- 壓力測試
- 效能基準測試

## 🔍 測試覆蓋率

目標覆蓋率：
- **單元測試**: > 90%
- **整合測試**: > 70%
- **系統測試**: > 60%

查看覆蓋率報告：
```bash
pytest --cov=services --cov-report=html tests/
open htmlcov/index.html
```

## 🚨 CI/CD 整合

### GitHub Actions
測試會在以下情況自動執行：
- Pull Request 提交
- 主分支推送
- 定時執行（每日）

### 測試階段
1. **快速測試** - 單元測試和語法檢查
2. **整合測試** - 模組整合驗證
3. **系統測試** - 完整功能測試
4. **部署測試** - 部署後驗證

---

**維護者**: QA Team  
**最後更新**: 2025-10-16