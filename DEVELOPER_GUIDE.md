# AIVA 開發者指南

## 🛠️ 開發環境設置

### 1. 環境要求
- Python 3.8+
- Git
- VS Code (推薦)

### 2. 專案設置
```bash
# 克隆專案
git clone <repository-url>
cd AIVA-git

# 設置虛擬環境
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 安裝依賴
pip install -r requirements.txt

# 複製環境配置
cp .env.example .env
```

## 📝 開發規範

### 程式碼風格
- 使用 `ruff` 進行格式化
- 使用 `mypy` 進行型別檢查
- 遵循 PEP 8 規範

### 提交規範
```bash
# 執行預提交檢查
pre-commit run --all-files

# 提交格式
git commit -m "feat: 添加新功能"
git commit -m "fix: 修復bug"
git commit -m "docs: 更新文件"
```

## 🏗️ 模組開發

### 新增功能檢測模組
```python
# services/features/function_newattack/
# ├── __init__.py
# ├── detector.py
# ├── payload_generator.py
# └── validator.py

from services.aiva_common import BaseDetector

class NewAttackDetector(BaseDetector):
    def detect(self, target):
        # 實現檢測邏輯
        pass
```

### 新增 AI 引擎組件
```python
# services/core/aiva_core/ai_engine/
# └── new_ai_component.py

from services.aiva_common import AIComponent

class NewAIComponent(AIComponent):
    def process(self, data):
        # 實現 AI 處理邏輯
        pass
```

## 🧪 測試指南

### 單元測試
```bash
# 執行所有測試
pytest tests/

# 執行特定模組測試
pytest tests/test_core/

# 測試覆蓋率
pytest --cov=services
```

### 整合測試
```bash
# API 測試
python api/test_api.py

# 系統整合測試
python services/core/aiva_core/ai_integration_test.py
```

## 📊 監控與除錯

### 日誌系統
```python
import logging
from services.aiva_common.logging import get_logger

logger = get_logger(__name__)
logger.info("處理開始")
logger.error("發生錯誤: %s", error_msg)
```

### 效能監控
```python
from services.integration.aiva_integration.system_performance_monitor import monitor

@monitor
def your_function():
    # 自動監控函數效能
    pass
```

## 🔧 常見開發任務

### 1. 新增 API 端點
```python
# api/routers/new_router.py
from fastapi import APIRouter

router = APIRouter()

@router.post("/new-endpoint")
async def new_endpoint(data: dict):
    return {"result": "success"}
```

### 2. 新增資料庫模型
```python
# services/integration/models.py
from sqlalchemy import Column, String, Integer

class NewModel(Base):
    __tablename__ = "new_table"
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
```

### 3. 新增配置選項
```python
# config/settings.py
NEW_FEATURE_ENABLED = True
NEW_FEATURE_CONFIG = {
    "timeout": 30,
    "retries": 3
}
```

## 📦 部署指南

### Docker 部署
```bash
# 構建映像
docker-compose build

# 啟動服務
docker-compose up -d

# 生產環境
docker-compose -f docker-compose.production.yml up -d
```

### 本地部署
```bash
# 啟動所有服務
python scripts/launcher/aiva_launcher.py

# 或分別啟動
python api/start_api.py &
python services/integration/aiva_integration/trigger_ai_continuous_learning.py &
```

## 🐛 故障排除

### 常見問題
1. **導入錯誤**: 檢查 `sys.path` 設置
2. **資料庫連接**: 檢查 `.env` 配置
3. **端口衝突**: 修改配置檔案中的端口設置

### 除錯工具
```bash
# 檢查套件狀態
python aiva_package_validator.py

# 檢查系統狀態
python -c "from services.integration.aiva_integration.system_performance_monitor import check_system; check_system()"
```

## 📚 參考資源

- [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) - 完整專案結構
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - 快速參考
- [API 文件](api/README.md) - API 使用說明

## 🤝 貢獻指南

1. Fork 專案
2. 創建功能分支: `git checkout -b feature/new-feature`
3. 提交變更: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 提交 Pull Request

---

*開發愉快！有問題請參考文件或聯繫開發團隊*